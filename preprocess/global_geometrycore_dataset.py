"""
基于 GeometryCore 的文件级全局数据入口。

设计目标：
1. 取代旧版 preprocess/sgp.py 的 ShapefileGlobalDataset
2. 使用新的 GeometryCore 前端：
   - 工作投影 CRS
   - flatten Multi* / GeometryCollection
   - family-aware tokenization
3. 尽量兼容现有 train_global.py 的输入接口

当前最小兼容输出格式仍然是：
    views: [n_views, max_geometries, fixed_points, 2]
    masks: [n_views, max_geometries]

这样可以先复用现有 global trainer / aggregator / signature 路线，
把新前端接上去，后面再决定是否进一步升级模型结构。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess.geometry_core.crs import ensure_working_projected_crs
from preprocess.geometry_core.flatten import flatten_geometry
from preprocess.geometry_core.tokenize_line import tokenize_line_family
from preprocess.geometry_core.tokenize_point import tokenize_point_family
from preprocess.geometry_core.tokenize_polygon import tokenize_polygon_family
from preprocess.geometry_core.types import FlattenedPrimitive, PrimitiveTokens
from utils.seed import worker_init_fn


def _resolve_data_dir(data_dir: str) -> Path:
    """兼容命令行和 IDE 的相对路径解析。"""
    raw = Path(data_dir)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = Path(__file__).resolve().parent.parent / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def _collect_shapefiles(data_dir: str, file_pattern: str = "*.shp") -> List[Path]:
    """递归收集真实的 shapefile 文件。"""
    base = _resolve_data_dir(data_dir)
    return sorted(path for path in base.glob(f"**/{file_pattern}") if path.is_file())


def _normalize_token_coords(coords: np.ndarray) -> np.ndarray:
    """
    对 token 坐标做全局平移与尺度归一化。

    这里使用所有点的 RMS 半径归一化，保持与现有新几何主线的思路一致。
    """
    if len(coords) == 0:
        return coords

    center = coords.mean(axis=0, keepdims=True)
    centered = coords - center
    scale = np.sqrt((centered * centered).sum(axis=1).mean()) + 1e-6
    return (centered / scale).astype(np.float32)


def _sample_or_pad_tokens(
    token_arrays: List[np.ndarray],
    max_geometries: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将文件级若干 geometry token 组织成固定几何数。

    输出：
    - geometries: [max_geometries, fixed_points, 2]
    - mask: [max_geometries]
    """
    n_total = len(token_arrays)
    if n_total == 0:
        raise ValueError("没有任何可入模的 geometry token。")

    if n_total <= max_geometries:
        sampled = token_arrays.copy()
        mask = np.zeros(max_geometries, dtype=bool)
        mask[:n_total] = True
        while len(sampled) < max_geometries:
            sampled.append(token_arrays[len(sampled) % n_total])
    else:
        # 先做一个确定性子采样，避免调试时每次结果都漂。
        indices = np.linspace(0, n_total - 1, max_geometries).astype(int)
        sampled = [token_arrays[i] for i in indices]
        mask = np.ones(max_geometries, dtype=bool)

    return np.stack(sampled[:max_geometries]).astype(np.float32), mask


class GeometryCoreAugmentor:
    """
    GeometryCore 最小增强器。

    这里只保留对现有 global 对比学习最关键的几何扰动：
    - 旋转
    - 加噪
    """

    def __init__(
        self,
        rotation_angles: Optional[List[float]] = None,
        noise_std: float = 0.001,
    ):
        self.rotation_angles = rotation_angles or [0, 90, 180, 270]
        self.noise_std = noise_std

    def augment(self, coords: np.ndarray) -> np.ndarray:
        """对单个 token 序列做几何扰动。"""
        out = coords.copy()

        angle = random.choice(self.rotation_angles)
        if angle != 0:
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            x = out[:, 0]
            y = out[:, 1]
            out[:, 0] = x * cos_a - y * sin_a
            out[:, 1] = x * sin_a + y * cos_a

        if self.noise_std > 0:
            noise = np.random.randn(*out.shape).astype(np.float32) * self.noise_std
            out = out + noise

        return out.astype(np.float32)


class GeometryCoreGlobalDataset(Dataset):
    """
    基于 GeometryCore 的文件级全局数据集。

    重要说明：
    - 每个 shp 文件作为一个样本
    - 文件内部每个 feature 会先展开为 point / line / polygon 原语
    - 再按几何家族分别 token 化
    - 最后对整个文件组织成固定数量的 geometry token
    """

    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*.shp",
        max_files: Optional[int] = None,
        source_crs_if_missing: Optional[str] = None,
        working_crs: Optional[str] = None,
        max_geometries: int = 500,
        min_geometries: int = 10,
        fixed_points: int = 128,
        include_holes: bool = True,
        point_token_budget: int = 32,
        line_token_budget: Optional[int] = None,
        polygon_token_budget: Optional[int] = None,
        point_enabled: bool = True,
        line_enabled: bool = True,
        polygon_enabled: bool = True,
        line_eps_len: float = 1e-6,
        polygon_eps_area: float = 1e-8,
        line_densify_max_segment_length: Optional[float] = None,
        verbose: bool = True,
    ):
        self.data_dir = _resolve_data_dir(data_dir)
        self.file_pattern = file_pattern
        self.max_files = max_files
        self.source_crs_if_missing = source_crs_if_missing
        self.working_crs = working_crs
        self.max_geometries = max_geometries
        self.min_geometries = min_geometries
        self.fixed_points = fixed_points
        self.include_holes = include_holes

        self.point_token_budget = point_token_budget
        self.line_token_budget = line_token_budget or fixed_points
        self.polygon_token_budget = polygon_token_budget or fixed_points

        self.point_enabled = point_enabled
        self.line_enabled = line_enabled
        self.polygon_enabled = polygon_enabled
        self.line_eps_len = line_eps_len
        self.polygon_eps_area = polygon_eps_area
        self.line_densify_max_segment_length = line_densify_max_segment_length
        self.verbose = verbose

        self.shp_files: List[str] = []
        self.file_tokens: List[List[np.ndarray]] = []
        self.file_metadata: List[Dict] = []

        self._load_files()

    def _tokenize_primitive(self, primitive: FlattenedPrimitive) -> Optional[PrimitiveTokens]:
        """根据几何家族选择对应的 tokenizer。"""
        if primitive.family == "point":
            if not self.point_enabled:
                return None
            return tokenize_point_family(primitive, max_tokens=self.point_token_budget)

        if primitive.family == "line":
            if not self.line_enabled:
                return None
            return tokenize_line_family(
                primitive,
                max_tokens=self.line_token_budget,
                eps_len=self.line_eps_len,
                densify_max_segment_length=self.line_densify_max_segment_length,
            )

        if primitive.family == "polygon":
            if not self.polygon_enabled:
                return None
            if primitive.is_hole and not self.include_holes:
                return None
            return tokenize_polygon_family(
                primitive,
                max_tokens=self.polygon_token_budget,
                eps_area=self.polygon_eps_area,
            )

        return None

    def _load_single_file(self, shp_path: Path) -> Tuple[List[np.ndarray], Dict]:
        """
        将单个 shp 文件转换为 geometry token 列表。

        返回：
        - token_arrays: 每个 geometry 的 [fixed_points, 2] 数组
        - metadata: 文件级统计信息
        """
        gdf = gpd.read_file(shp_path)
        gdf, working_crs = ensure_working_projected_crs(
            gdf,
            source_crs_if_missing=self.source_crs_if_missing,
            working_crs=self.working_crs,
        )

        token_arrays: List[np.ndarray] = []
        family_counts: Dict[str, int] = {"point": 0, "line": 0, "polygon": 0}
        rejected_primitives = 0

        for feature_id, geom in enumerate(gdf.geometry):
            primitives = flatten_geometry(geom, parent_id=feature_id)
            for primitive in primitives:
                try:
                    token = self._tokenize_primitive(primitive)
                except Exception:
                    token = None

                if token is None:
                    rejected_primitives += 1
                    continue

                coords = _normalize_token_coords(token.coords)

                # 统一到 old global trainer 所需的 fixed_points
                if len(coords) != self.fixed_points:
                    if len(coords) > self.fixed_points:
                        indices = np.linspace(0, len(coords) - 1, self.fixed_points).astype(int)
                        coords = coords[indices]
                    else:
                        padded = np.zeros((self.fixed_points, 2), dtype=np.float32)
                        padded[: len(coords)] = coords
                        padded[len(coords):] = coords[-1]
                        coords = padded

                token_arrays.append(coords.astype(np.float32))
                family_counts[primitive.family] += 1

        metadata = {
            "working_crs": working_crs,
            "valid_token_geometries": len(token_arrays),
            "rejected_primitives": rejected_primitives,
            "family_counts": family_counts,
        }
        return token_arrays, metadata

    def _load_files(self) -> None:
        """扫描并加载所有 shapefile。"""
        shp_files = _collect_shapefiles(str(self.data_dir), self.file_pattern)
        total_found = len(shp_files)
        if self.max_files is not None:
            shp_files = shp_files[: self.max_files]

        if self.verbose:
            if self.max_files is None:
                print(f"[GeometryCoreGlobalDataset] Found {total_found} shapefiles, loading...")
            else:
                print(
                    f"[GeometryCoreGlobalDataset] Found {total_found} shapefiles, "
                    f"loading first {len(shp_files)} files..."
                )

        for shp_path in shp_files:
            try:
                token_arrays, metadata = self._load_single_file(shp_path)
                if len(token_arrays) < self.min_geometries:
                    if self.verbose:
                        print(
                            f"  Skipped {shp_path.name}: only {len(token_arrays)} valid geometry tokens"
                        )
                    continue

                self.shp_files.append(str(shp_path))
                self.file_tokens.append(token_arrays)
                self.file_metadata.append(metadata)
                if self.verbose:
                    print(
                        f"  Loaded {shp_path.name}: "
                        f"{len(token_arrays)} tokens "
                        f"(point={metadata['family_counts']['point']}, "
                        f"line={metadata['family_counts']['line']}, "
                        f"polygon={metadata['family_counts']['polygon']})"
                    )
            except Exception as exc:
                if self.verbose:
                    print(f"  Error loading {shp_path}: {exc}")

        if self.verbose:
            print(f"[GeometryCoreGlobalDataset] Total loaded shapefiles: {len(self.shp_files)}")

    def __len__(self) -> int:
        return len(self.shp_files)

    def __getitem__(self, idx: int) -> Dict:
        """
        返回与旧版 global dataset 兼容的单样本结构。
        """
        token_arrays = self.file_tokens[idx]
        geometries, mask = _sample_or_pad_tokens(token_arrays, self.max_geometries)
        return {
            "geometries": torch.FloatTensor(geometries),
            "mask": torch.BoolTensor(mask),
            "n_geometries": len(token_arrays),
            "file_path": self.shp_files[idx],
            "file_idx": idx,
            "metadata": self.file_metadata[idx],
        }


class GeometryCoreGlobalContrastiveDataset(Dataset):
    """
    基于 GeometryCore file-level token 的对比学习数据集。

    为了最小兼容 train_global.py，这里仍然输出：
    - views: [n_views, max_geometries, fixed_points, 2]
    - masks: [n_views, max_geometries]
    """

    def __init__(
        self,
        base_dataset: GeometryCoreGlobalDataset,
        n_views: int = 2,
        augmentor: Optional[GeometryCoreAugmentor] = None,
    ):
        self.base_dataset = base_dataset
        self.n_views = n_views
        self.augmentor = augmentor or GeometryCoreAugmentor()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        original = self.base_dataset[idx]
        geometries_np = original["geometries"].numpy()
        mask_np = original["mask"].numpy()

        views = [geometries_np]
        masks = [mask_np]

        for _ in range(self.n_views - 1):
            aug_geometries = np.stack(
                [self.augmentor.augment(geom) for geom in geometries_np]
            ).astype(np.float32)
            views.append(aug_geometries)
            masks.append(mask_np)

        return {
            "views": torch.FloatTensor(np.stack(views)),
            "masks": torch.BoolTensor(np.stack(masks)),
            "n_geometries": original["n_geometries"],
            "file_path": original["file_path"],
            "file_idx": original["file_idx"],
            "metadata": original["metadata"],
        }


def create_geometrycore_global_dataloader(
    data_dir: str,
    batch_size_files: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    n_views: int = 2,
    max_geometries: int = 500,
    min_geometries: int = 10,
    fixed_points: int = 128,
    max_files: Optional[int] = None,
    source_crs_if_missing: Optional[str] = None,
    working_crs: Optional[str] = None,
    point_enabled: bool = True,
    line_enabled: bool = True,
    polygon_enabled: bool = True,
    verbose: bool = True,
) -> DataLoader:
    """
    创建基于 GeometryCore 的 file-level global dataloader。
    """
    base_dataset = GeometryCoreGlobalDataset(
        data_dir=data_dir,
        max_geometries=max_geometries,
        min_geometries=min_geometries,
        fixed_points=fixed_points,
        max_files=max_files,
        source_crs_if_missing=source_crs_if_missing,
        working_crs=working_crs,
        point_enabled=point_enabled,
        line_enabled=line_enabled,
        polygon_enabled=polygon_enabled,
        verbose=verbose,
    )
    dataset = GeometryCoreGlobalContrastiveDataset(
        base_dataset=base_dataset,
        n_views=n_views,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size_files,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
