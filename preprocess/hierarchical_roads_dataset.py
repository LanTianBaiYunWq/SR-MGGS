"""
roads / line-only / hierarchical signature 数据入口。

当前文件不追求一次把所有细节做完，而是提供新的主线骨架：

1. 文件级 canonical normalization
2. quadtree + overlap tile 切分
3. 每个 tile 内 polyline chunk 化
4. 局部 road subgraph 构建
5. 输出 tile 级样本，供后续局部 GNN + Set Transformer 使用

注意：
- 这个数据入口是新主线，不再沿用“固定少量 token + 一次全局池化”。
- 当前实现先提供最小可运行骨架，便于后续逐步替换细节策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, box
from torch.utils.data import Dataset

from preprocess.geometry_core.crs import ensure_working_projected_crs


@dataclass
class RoadChunk:
    """单个 tile 内的一段局部道路片段。"""

    file_id: str
    feature_id: int
    tile_id: str
    chunk_id: str
    coords: np.ndarray
    length: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]


@dataclass
class TileSample:
    """单个 tile 的局部 road subgraph 样本。"""

    file_id: str
    tile_id: str
    tile_bbox: Tuple[float, float, float, float]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    chunk_ids: List[str]
    chunk_coords_cache: List[np.ndarray]
    num_chunks: int


@dataclass
class FileSample:
    """单个文件的分层输出。"""

    file_id: str
    file_path: str
    family: str
    tiles: List[TileSample]
    metadata: Dict[str, object]


def _resolve_data_dir(data_dir: str) -> Path:
    """兼容相对路径和绝对路径。"""
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


def collect_shapefiles(data_dir: str, file_pattern: str = "*.shp") -> List[Path]:
    """递归收集真实 shapefile。"""
    base = _resolve_data_dir(data_dir)
    return sorted(path for path in base.glob(f"**/{file_pattern}") if path.is_file())


def iter_lines(geometry: object) -> Iterable[LineString]:
    """仅保留 line 家族，展开 MultiLineString。"""
    if geometry is None:
        return

    if geometry.is_empty:
        return

    if isinstance(geometry, LineString):
        yield geometry
        return

    if isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            if not part.is_empty:
                yield part


def canonicalize_line(line: LineString) -> Optional[LineString]:
    """
    对道路折线做最小规范化。

    当前仅做：
    - 去重相邻重复点
    - 保证至少两个唯一坐标
    - 稳定方向：按端点字典序排序，必要时反转
    """
    coords = np.asarray(line.coords, dtype=np.float64)
    if len(coords) < 2:
        return None

    deduped = [coords[0]]
    for point in coords[1:]:
        if not np.allclose(point, deduped[-1]):
            deduped.append(point)

    deduped_arr = np.asarray(deduped, dtype=np.float64)
    if len(np.unique(deduped_arr, axis=0)) < 2:
        return None

    start = tuple(deduped_arr[0].tolist())
    end = tuple(deduped_arr[-1].tolist())
    if end < start:
        deduped_arr = deduped_arr[::-1].copy()

    return LineString(deduped_arr)


def build_regular_tiles(
    bounds: Tuple[float, float, float, float],
    max_depth: int,
    overlap_ratio: float,
) -> List[Tuple[str, shapely.Geometry]]:
    """
    使用规则四叉划分生成 tile。

    当前是最小实现：
    - 用固定 depth 的规则 quadtree 网格
    - overlap 通过扩张 bbox 实现
    后续可替换为自适应 quadtree。
    """
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)

    nx = 2 ** max_depth
    ny = 2 ** max_depth
    dx = width / nx
    dy = height / ny
    overlap_x = dx * overlap_ratio
    overlap_y = dy * overlap_ratio

    tiles: List[Tuple[str, shapely.Geometry]] = []
    for ix in range(nx):
        for iy in range(ny):
            x0 = minx + ix * dx - overlap_x
            x1 = minx + (ix + 1) * dx + overlap_x
            y0 = miny + iy * dy - overlap_y
            y1 = miny + (iy + 1) * dy + overlap_y
            tile_id = f"tile_{ix}_{iy}"
            tiles.append((tile_id, box(x0, y0, x1, y1)))
    return tiles


def clip_line_to_tile(line: LineString, tile_geom: shapely.Geometry) -> List[LineString]:
    """将 line 裁剪到 tile 内，返回有效线段列表。"""
    inter = line.intersection(tile_geom)
    if inter.is_empty:
        return []

    segments: List[LineString] = []
    if isinstance(inter, LineString):
        segments.append(inter)
    elif isinstance(inter, MultiLineString):
        segments.extend(seg for seg in inter.geoms if not seg.is_empty)
    elif hasattr(inter, "geoms"):
        for geom in inter.geoms:
            if isinstance(geom, LineString) and not geom.is_empty:
                segments.append(geom)
            elif isinstance(geom, MultiLineString):
                segments.extend(seg for seg in geom.geoms if not seg.is_empty)
    return segments


def split_line_into_chunks(line: LineString, max_points_per_chunk: int) -> List[np.ndarray]:
    """
    将 tile 内的折线切成局部 chunks。

    当前先按点数上限切分，后续再换成按弧长和路网结构切分。
    """
    coords = np.asarray(line.coords, dtype=np.float32)
    if len(coords) < 2:
        return []

    if len(coords) <= max_points_per_chunk:
        return [coords]

    chunks: List[np.ndarray] = []
    start = 0
    while start < len(coords) - 1:
        end = min(start + max_points_per_chunk, len(coords))
        chunk = coords[start:end]
        if len(chunk) >= 2:
            chunks.append(chunk)
        start = max(end - 1, start + 1)
    return chunks


def make_chunk_feature(coords: np.ndarray) -> np.ndarray:
    """构造单个 chunk 的节点特征。"""
    line = LineString(coords)
    minx, miny, maxx, maxy = line.bounds
    dx = coords[-1, 0] - coords[0, 0]
    dy = coords[-1, 1] - coords[0, 1]
    angle = np.arctan2(dy, dx)
    return np.asarray(
        [
            float(line.length),
            float(maxx - minx),
            float(maxy - miny),
            float(angle),
            float(coords.shape[0]),
        ],
        dtype=np.float32,
    )


def build_local_road_subgraph(chunks: List[RoadChunk], connect_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建局部 road subgraph。

    当前使用最小版本：
    - 若两个 chunk 的中心点距离小于阈值，则连边
    - 边特征包括中心距和方向差的近似量
    """
    if not chunks:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    centers = np.asarray([chunk.centroid for chunk in chunks], dtype=np.float32)
    edges: List[Tuple[int, int]] = []
    edge_features: List[np.ndarray] = []

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist > connect_radius:
                continue

            edges.append((i, j))
            edges.append((j, i))
            feat = np.asarray([dist, 1.0 / (dist + 1e-6)], dtype=np.float32)
            edge_features.append(feat)
            edge_features.append(feat)

    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_attr = np.asarray(edge_features, dtype=np.float32)
    return edge_index, edge_attr


class HierarchicalRoadsDataset(Dataset):
    """roads / line-only / hierarchical signature 的文件级数据集。"""

    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*roads*.shp",
        max_files: Optional[int] = None,
        source_crs_if_missing: Optional[str] = None,
        working_crs: Optional[str] = None,
        quadtree_depth: int = 2,
        tile_overlap_ratio: float = 0.1,
        max_points_per_chunk: int = 64,
        min_chunks_per_tile: int = 2,
        connect_radius: float = 50.0,
        verbose: bool = True,
    ):
        self.data_dir = _resolve_data_dir(data_dir)
        self.file_pattern = file_pattern
        self.max_files = max_files
        self.source_crs_if_missing = source_crs_if_missing
        self.working_crs = working_crs
        self.quadtree_depth = quadtree_depth
        self.tile_overlap_ratio = tile_overlap_ratio
        self.max_points_per_chunk = max_points_per_chunk
        self.min_chunks_per_tile = min_chunks_per_tile
        self.connect_radius = connect_radius
        self.verbose = verbose

        self.file_samples: List[FileSample] = []
        self._load_files()

    def _load_single_file(self, shp_path: Path, file_idx: int) -> Optional[FileSample]:
        """读取单个 roads 文件并构造成分层样本。"""
        gdf = gpd.read_file(shp_path)
        gdf, working_crs = ensure_working_projected_crs(
            gdf,
            source_crs_if_missing=self.source_crs_if_missing,
            working_crs=self.working_crs,
        )

        canonical_lines: List[Tuple[int, LineString]] = []
        for feature_id, geom in enumerate(gdf.geometry):
            for line in iter_lines(geom):
                normalized = canonicalize_line(line)
                if normalized is not None:
                    canonical_lines.append((feature_id, normalized))

        if not canonical_lines:
            return None

        file_geom = shapely.union_all([line for _, line in canonical_lines])
        tiles = build_regular_tiles(file_geom.bounds, self.quadtree_depth, self.tile_overlap_ratio)

        tile_samples: List[TileSample] = []
        for tile_id, tile_geom in tiles:
            chunks: List[RoadChunk] = []
            for feature_id, line in canonical_lines:
                clipped_segments = clip_line_to_tile(line, tile_geom)
                for seg_id, segment in enumerate(clipped_segments):
                    for local_idx, coords in enumerate(split_line_into_chunks(segment, self.max_points_per_chunk)):
                        line_obj = LineString(coords)
                        chunks.append(
                            RoadChunk(
                                file_id=f"file_{file_idx:04d}",
                                feature_id=feature_id,
                                tile_id=tile_id,
                                chunk_id=f"{tile_id}_f{feature_id}_s{seg_id}_c{local_idx}",
                                coords=coords,
                                length=float(line_obj.length),
                                bbox=line_obj.bounds,
                                centroid=(float(line_obj.centroid.x), float(line_obj.centroid.y)),
                            )
                        )

            if len(chunks) < self.min_chunks_per_tile:
                continue

            node_features = np.stack([make_chunk_feature(chunk.coords) for chunk in chunks]).astype(np.float32)
            edge_index, edge_features = build_local_road_subgraph(chunks, self.connect_radius)
            tile_samples.append(
                TileSample(
                    file_id=f"file_{file_idx:04d}",
                    tile_id=tile_id,
                    tile_bbox=tile_geom.bounds,
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_features=edge_features,
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    chunk_coords_cache=[chunk.coords for chunk in chunks],
                    num_chunks=len(chunks),
                )
            )

        if not tile_samples:
            return None

        metadata = {
            "working_crs": working_crs,
            "num_canonical_lines": len(canonical_lines),
            "num_tiles": len(tile_samples),
            "quadtree_depth": self.quadtree_depth,
        }
        return FileSample(
            file_id=f"file_{file_idx:04d}",
            file_path=str(shp_path),
            family="line",
            tiles=tile_samples,
            metadata=metadata,
        )

    def _load_files(self) -> None:
        """扫描并加载文件。"""
        shp_files = collect_shapefiles(str(self.data_dir), self.file_pattern)
        if self.max_files is not None:
            shp_files = shp_files[: self.max_files]

        if self.verbose:
            print(f"[HierarchicalRoadsDataset] Loading {len(shp_files)} shapefiles...")

        for idx, shp_path in enumerate(shp_files):
            try:
                sample = self._load_single_file(shp_path, idx)
            except Exception as exc:
                sample = None
                if self.verbose:
                    print(f"  Error loading {shp_path.name}: {exc}")

            if sample is None:
                if self.verbose:
                    print(f"  Skipped {shp_path.name}: no valid tile sample")
                continue

            self.file_samples.append(sample)
            if self.verbose:
                print(
                    f"  Loaded {shp_path.name}: "
                    f"tiles={len(sample.tiles)}, "
                    f"canonical_lines={sample.metadata['num_canonical_lines']}"
                )

    def __len__(self) -> int:
        return len(self.file_samples)

    def __getitem__(self, idx: int) -> FileSample:
        return self.file_samples[idx]
