"""
tile_builder

职责：
1. 读取 roads shapefile
2. 统一 CRS
3. 规范化 line geometry
4. 构建自适应 quadtree + overlap tiles
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, box

from roads_hier_stage1.types import CanonicalLineRecord, TileRecord


def resolve_path(path_str: str) -> Path:
    """兼容相对路径和绝对路径。"""
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = Path(__file__).resolve().parents[1] / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def collect_shapefiles(data_dir: str, file_pattern: str = "*roads*.shp") -> List[Path]:
    """递归收集目标 shapefile，支持逗号分隔的多个 glob pattern。"""
    base = resolve_path(data_dir)
    patterns = [pattern.strip() for pattern in file_pattern.split(",") if pattern.strip()]
    if not patterns:
        patterns = ["*.shp"]

    results = set()
    for pattern in patterns:
        results.update(path for path in base.glob(f"**/{pattern}") if path.is_file())
    return sorted(results)


def prepare_working_gdf(
    shp_path: Path,
    source_crs_if_missing: Optional[str] = None,
    working_crs: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, Optional[str]]:
    """
    读取并转换到工作投影 CRS。

    规则：
    - 原始无 CRS 且来源已知时，允许 set_crs 补标签
    - 真正重投影必须使用 to_crs
    """
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None and source_crs_if_missing:
        gdf = gdf.set_crs(source_crs_if_missing)

    if gdf.crs is None:
        return gdf, None

    if working_crs:
        gdf = gdf.to_crs(working_crs)
        return gdf, str(gdf.crs)

    # 默认优先使用本地 UTM，确保长度和面积计算在投影坐标中进行。
    try:
        estimated = gdf.estimate_utm_crs()
        if estimated is not None:
            gdf = gdf.to_crs(estimated)
            return gdf, str(gdf.crs)
    except Exception:
        pass

    if gdf.crs.is_geographic:
        return gdf.to_crs("EPSG:3857"), "EPSG:3857"
    return gdf, str(gdf.crs)


def iter_lines(geometry: object) -> Iterable[LineString]:
    """仅保留 line 家族，并展开 MultiLineString。"""
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            if not part.is_empty:
                yield part


def canonicalize_line(line: LineString) -> Optional[LineString]:
    """
    规范化道路折线。

    当前只做最小稳定化：
    - 去掉相邻重复点
    - 保证至少两个唯一坐标
    - 方向按端点字典序稳定
    """
    try:
        coords = np.asarray(line.coords, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if coords.ndim != 2 or coords.shape[0] < 2:
        return None
    coords = coords[:, :2]
    if not np.isfinite(coords).all():
        return None

    deduped = [coords[0]]
    for point in coords[1:]:
        if not np.allclose(point, deduped[-1]):
            deduped.append(point)

    deduped_arr = np.asarray(deduped, dtype=np.float64)
    if deduped_arr.shape[0] < 2:
        return None
    if float(np.max(np.linalg.norm(deduped_arr - deduped_arr[0], axis=1))) <= 1e-12:
        return None

    start = tuple(deduped_arr[0].tolist())
    end = tuple(deduped_arr[-1].tolist())
    if end < start:
        deduped_arr = deduped_arr[::-1].copy()

    return LineString(deduped_arr)


def extract_canonical_lines(gdf: gpd.GeoDataFrame) -> List[CanonicalLineRecord]:
    """从 GeoDataFrame 提取规范化后的 line records。"""
    line_records: List[CanonicalLineRecord] = []

    for feature_id, geom in enumerate(gdf.geometry):
        local_idx = 0
        for line in iter_lines(geom):
            normalized = canonicalize_line(line)
            if normalized is None:
                continue

            line_id = f"f{feature_id}_l{local_idx}"
            line_records.append(
                CanonicalLineRecord(
                    feature_id=feature_id,
                    line_id=line_id,
                    coords=np.asarray(normalized.coords, dtype=np.float64).tolist(),
                    length=float(normalized.length),
                    bbox=normalized.bounds,
                )
            )
            local_idx += 1

    return line_records


def bbox_diagonal(bounds: Sequence[float]) -> float:
    """计算 bbox 对角线长度。"""
    minx, miny, maxx, maxy = bounds
    return float(math.hypot(maxx - minx, maxy - miny))


def expand_bounds(bounds: Sequence[float], overlap_ratio: float) -> Tuple[float, float, float, float]:
    """按比例扩展 bbox，形成 overlap tile。"""
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    ox = width * overlap_ratio
    oy = height * overlap_ratio
    return (minx - ox, miny - oy, maxx + ox, maxy + oy)


def line_intersects_bounds(line: CanonicalLineRecord, bounds: Sequence[float]) -> bool:
    """快速判断 line bbox 是否与 tile bbox 相交。"""
    lminx, lminy, lmaxx, lmaxy = line.bbox
    tminx, tminy, tmaxx, tmaxy = bounds
    return not (lmaxx < tminx or lminx > tmaxx or lmaxy < tminy or lminy > tmaxy)


def split_bounds_quadrants(bounds: Sequence[float]) -> List[Tuple[float, float, float, float]]:
    """将 bbox 分成四个子 bbox。"""
    minx, miny, maxx, maxy = bounds
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0
    return [
        (minx, miny, midx, midy),
        (midx, miny, maxx, midy),
        (minx, midy, midx, maxy),
        (midx, midy, maxx, maxy),
    ]


def build_adaptive_quadtree_tiles(
    line_records: Sequence[CanonicalLineRecord],
    *,
    root_bounds: Sequence[float],
    max_depth: int,
    overlap_ratio: float,
    max_primitives_per_tile: int,
) -> List[TileRecord]:
    """
    构建自适应 quadtree + overlap tiles。

    停止条件：
    - primitive 数 <= max_primitives_per_tile
    - 或达到最大深度
    """
    tiles: List[TileRecord] = []

    def recurse(bounds: Sequence[float], depth: int, prefix: str) -> None:
        assigned = [line.line_id for line in line_records if line_intersects_bounds(line, bounds)]
        if not assigned:
            return

        if len(assigned) <= max_primitives_per_tile or depth >= max_depth:
            overlap_bounds = expand_bounds(bounds, overlap_ratio)
            tiles.append(
                TileRecord(
                    tile_id=prefix,
                    depth=depth,
                    bbox=overlap_bounds,
                    raw_line_count=len(assigned),
                    assigned_line_ids=assigned,
                )
            )
            return

        child_bounds = split_bounds_quadrants(bounds)
        for idx, child in enumerate(child_bounds):
            recurse(child, depth + 1, f"{prefix}_{idx}")

    recurse(root_bounds, 0, "tile")
    return tiles
