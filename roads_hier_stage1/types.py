"""
Gate 1 共享数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CanonicalLineRecord:
    """规范化后的单条道路。"""

    feature_id: int
    line_id: str
    coords: List[List[float]]
    length: float
    bbox: Tuple[float, float, float, float]


@dataclass
class TileRecord:
    """单个 tile 的描述信息。"""

    tile_id: str
    depth: int
    bbox: Tuple[float, float, float, float]
    raw_line_count: int
    assigned_line_ids: List[str]


@dataclass
class ChunkRecord:
    """单个 polyline chunk 的描述信息。"""

    chunk_id: str
    tile_id: str
    source_line_id: str
    num_vertices: int
    length: float
    bbox: Tuple[float, float, float, float]
    coords: List[List[float]]


@dataclass
class StageTiming:
    """单阶段耗时统计。"""

    stage: str
    seconds: float


def build_file_manifest(
    *,
    file_id: str,
    file_path: str,
    family: str,
    working_crs: str | None,
    file_bbox: Tuple[float, float, float, float],
    file_bbox_diagonal: float,
    num_raw_features: int,
    num_canonical_lines: int,
    num_tiles: int,
    num_chunks: int,
    tile_chunk_count_stats: Dict[str, float],
    timings: List[Dict[str, float]],
    tile_manifest_path: str,
    chunk_store_dir: str,
    render_dir: str | None,
    tile_selection: Dict[str, object] | None = None,
    chunk_feature_names: List[str] | None = None,
) -> Dict[str, object]:
    """构造文件级 manifest。"""
    return {
        "file_id": file_id,
        "file_path": file_path,
        "family": family,
        "working_crs": working_crs,
        "file_bbox": list(file_bbox),
        "file_bbox_diagonal": file_bbox_diagonal,
        "num_raw_features": num_raw_features,
        "num_canonical_lines": num_canonical_lines,
        "num_tiles": num_tiles,
        "num_chunks": num_chunks,
        "tile_chunk_count_stats": tile_chunk_count_stats,
        "timings": timings,
        "tile_manifest_path": tile_manifest_path,
        "chunk_store_dir": chunk_store_dir,
        "render_dir": render_dir,
        "tile_selection": tile_selection or {},
        "chunk_feature_names": chunk_feature_names or [],
    }
