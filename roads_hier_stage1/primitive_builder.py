"""
primitive_builder

职责：
1. 将进入 tile 的 line 裁剪为局部片段
2. 按顶点上限和弧长上限切分为 polyline chunks
3. 执行局部 budget，避免 tile 内 primitive 爆炸
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, MultiLineString, box

from roads_hier_stage1.types import CanonicalLineRecord, ChunkRecord, TileRecord


def clip_line_to_tile(line_coords: Sequence[Sequence[float]], tile_bounds: Sequence[float]) -> List[LineString]:
    """将单条 line 裁剪到 tile 内。"""
    line = LineString(line_coords)
    tile_geom = box(*tile_bounds)
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


def chunk_line_segment(
    segment: LineString,
    *,
    max_vertices_per_chunk: int,
    max_arc_length: float,
) -> List[np.ndarray]:
    """
    按顶点上限和弧长上限切分单段 line。

    当前第一版采用保守规则：
    - 超过任一阈值就继续切
    - 分段时保持端点重叠，避免断裂
    """
    coords = np.asarray(segment.coords, dtype=np.float64)
    if len(coords) < 2:
        return []

    if len(coords) <= max_vertices_per_chunk and float(segment.length) <= max_arc_length:
        return [coords.astype(np.float32)]

    chunks: List[np.ndarray] = []
    start = 0
    while start < len(coords) - 1:
        end = min(start + max_vertices_per_chunk, len(coords))
        candidate = coords[start:end]
        if len(candidate) < 2:
            break

        # 如果局部长度仍超限，则继续向前缩减。
        while len(candidate) > 2 and float(LineString(candidate).length) > max_arc_length:
            candidate = candidate[:-1]

        if len(candidate) >= 2:
            chunks.append(candidate.astype(np.float32))

        if end >= len(coords):
            break
        start = max(start + len(candidate) - 1, start + 1)

    return chunks


def build_tile_chunks(
    tile: TileRecord,
    line_lookup: Dict[str, CanonicalLineRecord],
    *,
    max_vertices_per_chunk: int,
    max_arc_length: float,
    k_line_per_tile: int,
) -> Tuple[List[ChunkRecord], Dict[str, int]]:
    """
    为单个 tile 构建 chunk。

    先生成全部 chunks，再按局部 budget 截断。
    第一版使用长度优先保留策略，确保长道路优先留下。
    """
    all_chunks: List[ChunkRecord] = []

    for line_id in tile.assigned_line_ids:
        record = line_lookup[line_id]
        clipped_segments = clip_line_to_tile(record.coords, tile.bbox)
        for seg_idx, segment in enumerate(clipped_segments):
            chunk_arrays = chunk_line_segment(
                segment,
                max_vertices_per_chunk=max_vertices_per_chunk,
                max_arc_length=max_arc_length,
            )
            for local_idx, coords in enumerate(chunk_arrays):
                chunk_line = LineString(coords)
                all_chunks.append(
                    ChunkRecord(
                        chunk_id=f"{tile.tile_id}_{line_id}_s{seg_idx}_c{local_idx}",
                        tile_id=tile.tile_id,
                        source_line_id=line_id,
                        num_vertices=int(coords.shape[0]),
                        length=float(chunk_line.length),
                        bbox=chunk_line.bounds,
                        coords=coords.tolist(),
                    )
                )

    before_budget = len(all_chunks)
    if before_budget > k_line_per_tile:
        all_chunks.sort(key=lambda item: item.length, reverse=True)
        all_chunks = all_chunks[:k_line_per_tile]

    stats = {
        "raw_chunk_count": before_budget,
        "budgeted_chunk_count": len(all_chunks),
    }
    return all_chunks, stats

