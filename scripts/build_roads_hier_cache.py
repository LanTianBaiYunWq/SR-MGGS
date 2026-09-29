"""
离线构建 roads / line-only / hierarchical Gate 1 缓存。

产物：
1. tile
2. chunk
3. manifest
4. 可选 render

注意：
- 当前脚本先不构图
- 当前脚本先不训练
- Gate 1 的通过标准是产物完整、耗时可解释、规模可控
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roads_hier_stage1.geometrycore_hier_dataset import resolve_path
from roads_hier_stage1.primitive_builder import build_tile_chunks
from roads_hier_stage1.render_multiscale_teacher_targets import render_polyline_chunks
from roads_hier_stage1.tile_builder import (
    bbox_diagonal,
    build_adaptive_quadtree_tiles,
    collect_shapefiles,
    extract_canonical_lines,
    prepare_working_gdf,
)
from roads_hier_stage1.types import build_file_manifest


CHUNK_FEATURE_NAMES = [
    "start_x_tile_norm",
    "start_y_tile_norm",
    "end_x_tile_norm",
    "end_y_tile_norm",
    "center_x_tile_norm",
    "center_y_tile_norm",
    "direction_sin",
    "direction_cos",
    "length_norm_file_diag",
    "local_density",
]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build roads hierarchical cache.")
    parser.add_argument("--data_dir", type=str, required=True, help="原始数据目录。")
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*roads*.shp",
        help="文件过滤模式，支持逗号分隔多个 glob，例如 *roads*.shp,*railways*.shp,*waterways*.shp 。",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="缓存输出目录。")
    parser.add_argument("--max_files", type=int, default=None, help="最多处理多少个文件。")
    parser.add_argument("--source_crs_if_missing", type=str, default=None, help="缺失 CRS 时补的原始 CRS。")
    parser.add_argument("--working_crs", type=str, default=None, help="工作投影 CRS。")
    parser.add_argument("--max_depth", type=int, default=3, help="quadtree 最大深度。")
    parser.add_argument("--overlap_ratio", type=float, default=0.1, help="tile overlap 比例。")
    parser.add_argument("--max_primitives_per_tile", type=int, default=256, help="tile 停止分裂的 primitive 上限。")
    parser.add_argument("--max_vertices_per_chunk", type=int, default=16, help="每个 chunk 顶点上限。")
    parser.add_argument("--max_arc_length_ratio", type=float, default=0.02, help="chunk 最大弧长占文件 bbox 对角线的比例。")
    parser.add_argument("--k_line_per_tile", type=int, default=64, help="每个 tile 的局部 budget。")
    parser.add_argument(
        "--tile_score_key",
        type=str,
        default="chunk_length_sum",
        choices=["chunk_length_sum", "budgeted_chunk_count", "raw_chunk_count", "raw_line_count"],
        help="file-level top-M tile 选择时使用的主评分字段。",
    )
    parser.add_argument("--train_max_tiles", type=int, default=24, help="训练阶段默认保留的 tile 数。")
    parser.add_argument("--eval_max_tiles", type=int, default=32, help="评估阶段默认保留的 tile 数。")
    parser.add_argument("--render", action="store_true", help="是否额外离线渲染 tile/file 教师目标。")
    parser.add_argument("--render_image_size", type=int, default=224, help="render 图像尺寸。")
    return parser


def percentile(values: List[int], q: float) -> float:
    """简化版分位数统计。"""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def print_stage_timing(prefix: str, seconds: float) -> None:
    """统一打印阶段耗时。"""
    print(f"  [{prefix}] {seconds:.3f}s")


def tile_score_value(tile_row: Dict[str, object], score_key: str) -> float:
    """读取 tile 的主评分。"""
    value = tile_row.get(score_key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_point_to_tile(x: float, y: float, tile_bbox: List[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    """把投影坐标归一化到 tile bbox 下的 [0, 1]^2。"""
    minx, miny, maxx, maxy = tile_bbox
    width = max(float(maxx - minx), 1e-9)
    height = max(float(maxy - miny), 1e-9)
    nx = (float(x) - float(minx)) / width
    ny = (float(y) - float(miny)) / height
    return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))


def build_chunk_feature_vector(
    coords_arr: np.ndarray,
    *,
    chunk_length: float,
    tile_bbox: List[float] | tuple[float, float, float, float],
    file_diag: float,
    tile_area: float,
) -> np.ndarray:
    """构造归一化几何特征版 chunk feature。"""
    start_x, start_y = coords_arr[0].tolist()
    end_x, end_y = coords_arr[-1].tolist()
    center = np.mean(coords_arr, axis=0)
    center_x, center_y = float(center[0]), float(center[1])

    start_x_norm, start_y_norm = normalize_point_to_tile(start_x, start_y, tile_bbox)
    end_x_norm, end_y_norm = normalize_point_to_tile(end_x, end_y, tile_bbox)
    center_x_norm, center_y_norm = normalize_point_to_tile(center_x, center_y, tile_bbox)

    dx = float(end_x - start_x)
    dy = float(end_y - start_y)
    direction_norm = max(float(np.hypot(dx, dy)), 1e-9)
    direction_cos = dx / direction_norm
    direction_sin = dy / direction_norm

    length_norm = float(chunk_length) / max(float(file_diag), 1e-9)
    local_density = float(chunk_length) / max(float(tile_area), 1e-9)

    return np.asarray(
        [
            start_x_norm,
            start_y_norm,
            end_x_norm,
            end_y_norm,
            center_x_norm,
            center_y_norm,
            direction_sin,
            direction_cos,
            length_norm,
            local_density,
        ],
        dtype=np.float32,
    )


def rank_tiles_for_file(
    tile_rows: List[Dict[str, object]],
    *,
    score_key: str,
    train_max_tiles: int,
    eval_max_tiles: int,
) -> Dict[str, object]:
    """为单文件 tiles 写入排序和 train/eval top-M 选择结果。"""
    ranked = sorted(
        tile_rows,
        key=lambda row: (
            tile_score_value(row, score_key),
            float(row.get("budgeted_chunk_count", 0)),
            float(row.get("raw_chunk_count", 0)),
            float(row.get("raw_line_count", 0)),
            -float(row.get("depth", 0)),
        ),
        reverse=True,
    )

    selected_train_ids = {row["tile_id"] for row in ranked[:train_max_tiles]}
    selected_eval_ids = {row["tile_id"] for row in ranked[:eval_max_tiles]}

    for rank_idx, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank_idx
        row["selection_score"] = tile_score_value(row, score_key)
        row["selected_for_train"] = row["tile_id"] in selected_train_ids
        row["selected_for_eval"] = row["tile_id"] in selected_eval_ids

    return {
        "score_key": score_key,
        "train_max_tiles": train_max_tiles,
        "eval_max_tiles": eval_max_tiles,
        "train_selected_count": min(train_max_tiles, len(ranked)),
        "eval_selected_count": min(eval_max_tiles, len(ranked)),
        "train_selected_tile_ids": [row["tile_id"] for row in ranked[:train_max_tiles]],
        "eval_selected_tile_ids": [row["tile_id"] for row in ranked[:eval_max_tiles]],
    }


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shp_files = collect_shapefiles(args.data_dir, args.file_pattern)
    if args.max_files is not None:
        shp_files = shp_files[: args.max_files]

    global_index: List[Dict[str, object]] = []

    for file_idx, shp_path in enumerate(shp_files):
        print("=" * 100)
        print(f"Processing file {file_idx + 1}/{len(shp_files)}: {shp_path.name}")
        print("=" * 100)
        file_id = f"file_{file_idx:04d}"
        file_dir = output_dir / file_id
        chunk_dir = file_dir / "chunks"
        render_dir = file_dir / "renders"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        if args.render:
            render_dir.mkdir(parents=True, exist_ok=True)

        timings: List[Dict[str, float]] = []

        t0 = time.perf_counter()
        gdf, working_crs = prepare_working_gdf(
            shp_path,
            source_crs_if_missing=args.source_crs_if_missing,
            working_crs=args.working_crs,
        )
        read_reproject_seconds = time.perf_counter() - t0
        timings.append({"stage": "read_reproject", "seconds": read_reproject_seconds})
        print_stage_timing("read_reproject", read_reproject_seconds)

        t1 = time.perf_counter()
        canonical_lines = extract_canonical_lines(gdf)
        canonicalize_seconds = time.perf_counter() - t1
        timings.append({"stage": "canonicalize", "seconds": canonicalize_seconds})
        print_stage_timing("canonicalize", canonicalize_seconds)

        if not canonical_lines:
            print(f"Skipped {shp_path.name}: no canonical lines")
            continue

        minx = min(line.bbox[0] for line in canonical_lines)
        miny = min(line.bbox[1] for line in canonical_lines)
        maxx = max(line.bbox[2] for line in canonical_lines)
        maxy = max(line.bbox[3] for line in canonical_lines)
        file_bounds = (minx, miny, maxx, maxy)
        file_diag = bbox_diagonal(file_bounds)
        max_arc_length = args.max_arc_length_ratio * file_diag
        line_lookup = {line.line_id: line for line in canonical_lines}

        t2 = time.perf_counter()
        tiles = build_adaptive_quadtree_tiles(
            canonical_lines,
            root_bounds=file_bounds,
            max_depth=args.max_depth,
            overlap_ratio=args.overlap_ratio,
            max_primitives_per_tile=args.max_primitives_per_tile,
        )
        tile_split_seconds = time.perf_counter() - t2
        timings.append({"stage": "tile_split", "seconds": tile_split_seconds})
        print_stage_timing("tile_split", tile_split_seconds)

        tile_rows: List[Dict[str, object]] = []
        tile_chunk_counts: List[int] = []
        tile_edge_counts: List[int] = []
        total_chunks = 0
        tile_build_seconds_list: List[float] = []
        compression_ratios: List[float] = []
        budget_hit_count = 0

        for tile in tiles:
            t3 = time.perf_counter()
            chunks, chunk_stats = build_tile_chunks(
                tile,
                line_lookup,
                max_vertices_per_chunk=args.max_vertices_per_chunk,
                max_arc_length=max_arc_length,
                k_line_per_tile=args.k_line_per_tile,
            )
            build_seconds = time.perf_counter() - t3
            tile_build_seconds_list.append(build_seconds)

            chunk_coords = []
            chunk_features = []
            chunk_meta = []
            chunk_length_sum = 0.0
            tile_minx, tile_miny, tile_maxx, tile_maxy = tile.bbox
            tile_width = float(tile_maxx - tile_minx)
            tile_height = float(tile_maxy - tile_miny)
            tile_area = max(tile_width * tile_height, 1e-9)
            for chunk in chunks:
                coords_arr = np.asarray(chunk.coords, dtype=np.float32)
                chunk_coords.append(coords_arr)
                chunk_length_sum += float(chunk.length)
                chunk_features.append(
                    build_chunk_feature_vector(
                        coords_arr,
                        chunk_length=float(chunk.length),
                        tile_bbox=tile.bbox,
                        file_diag=file_diag,
                        tile_area=tile_area,
                    )
                )
                chunk_meta.append([chunk.chunk_id, chunk.source_line_id])

            if chunk_coords:
                np.savez_compressed(
                    chunk_dir / f"{tile.tile_id}.npz",
                    chunk_coords=np.asarray(chunk_coords, dtype=object),
                    chunk_features=np.asarray(chunk_features, dtype=np.float32),
                    chunk_meta=np.asarray(chunk_meta, dtype=object),
                )

            if args.render and chunk_coords:
                render_polyline_chunks(
                    [np.asarray(coords, dtype=np.float32) for coords in chunk_coords],
                    render_dir / f"{tile.tile_id}.png",
                    image_size=args.render_image_size,
                )
            compression_ratio = (
                float(chunk_stats["budgeted_chunk_count"]) / float(chunk_stats["raw_chunk_count"])
                if chunk_stats["raw_chunk_count"] > 0
                else 0.0
            )
            chunk_density = float(chunk_length_sum) / tile_area

            tile_rows.append(
                {
                    "tile_id": tile.tile_id,
                    "depth": tile.depth,
                    "bbox": list(tile.bbox),
                    "raw_line_count": tile.raw_line_count,
                    "assigned_line_count": len(tile.assigned_line_ids),
                    "raw_chunk_count": chunk_stats["raw_chunk_count"],
                    "budgeted_chunk_count": chunk_stats["budgeted_chunk_count"],
                    "chunk_length_sum": chunk_length_sum,
                    "tile_area": tile_area,
                    "chunk_density": chunk_density,
                    "compression_ratio": compression_ratio,
                    "edge_count": 0,
                    "timings": {
                        "chunk_split": build_seconds,
                    },
                }
            )
            tile_chunk_counts.append(chunk_stats["budgeted_chunk_count"])
            tile_edge_counts.append(0)
            total_chunks += chunk_stats["budgeted_chunk_count"]
            if chunk_stats["raw_chunk_count"] > 0:
                compression_ratios.append(compression_ratio)
            if chunk_stats["budgeted_chunk_count"] == args.k_line_per_tile:
                budget_hit_count += 1

            print(
                f"  [tile] id={tile.tile_id} depth={tile.depth} "
                f"raw_lines={tile.raw_line_count} "
                f"raw_chunks={chunk_stats['raw_chunk_count']} "
                f"budgeted_chunks={chunk_stats['budgeted_chunk_count']} "
                f"edge_count=0 "
                f"chunk_split={build_seconds:.3f}s"
            )

        t4 = time.perf_counter()
        if args.render:
            file_chunks = []
            for tile in tiles:
                npz_path = chunk_dir / f"{tile.tile_id}.npz"
                if not npz_path.exists():
                    continue
                data = np.load(npz_path, allow_pickle=True)
                file_chunks.extend([np.asarray(item, dtype=np.float32) for item in data["chunk_coords"]])
            if file_chunks:
                render_polyline_chunks(
                    file_chunks,
                    render_dir / "file.png",
                    image_size=args.render_image_size,
                )
        render_seconds = time.perf_counter() - t4
        timings.append({"stage": "render", "seconds": render_seconds})
        print_stage_timing("render", render_seconds)

        tile_selection = rank_tiles_for_file(
            tile_rows,
            score_key=args.tile_score_key,
            train_max_tiles=args.train_max_tiles,
            eval_max_tiles=args.eval_max_tiles,
        )

        tiles_path = file_dir / "tiles.json"
        with tiles_path.open("w", encoding="utf-8") as f:
            json.dump(tile_rows, f, indent=2, ensure_ascii=False)

        tile_chunk_stats = {
            "p50": percentile(tile_chunk_counts, 50),
            "p95": percentile(tile_chunk_counts, 95),
            "max": max(tile_chunk_counts) if tile_chunk_counts else 0,
            "edge_p50": percentile(tile_edge_counts, 50),
            "edge_p95": percentile(tile_edge_counts, 95),
            "edge_max": max(tile_edge_counts) if tile_edge_counts else 0,
            "chunk_split_seconds_p50": percentile([int(x * 1000) for x in tile_build_seconds_list], 50) / 1000.0 if tile_build_seconds_list else 0.0,
            "chunk_split_seconds_p95": percentile([int(x * 1000) for x in tile_build_seconds_list], 95) / 1000.0 if tile_build_seconds_list else 0.0,
            "chunk_split_seconds_max": max(tile_build_seconds_list) if tile_build_seconds_list else 0.0,
            "compression_ratio_p50": percentile([int(x * 1000000) for x in compression_ratios], 50) / 1000000.0 if compression_ratios else 0.0,
            "compression_ratio_p95": percentile([int(x * 1000000) for x in compression_ratios], 95) / 1000000.0 if compression_ratios else 0.0,
            "compression_ratio_min": min(compression_ratios) if compression_ratios else 0.0,
            "budget_hit_count": budget_hit_count,
            "budget_hit_ratio": (float(budget_hit_count) / float(len(tile_rows))) if tile_rows else 0.0,
        }

        manifest = build_file_manifest(
            file_id=file_id,
            file_path=str(shp_path),
            family="line",
            working_crs=working_crs,
            file_bbox=file_bounds,
            file_bbox_diagonal=file_diag,
            num_raw_features=len(gdf),
            num_canonical_lines=len(canonical_lines),
            num_tiles=len(tile_rows),
            num_chunks=total_chunks,
            tile_chunk_count_stats=tile_chunk_stats,
            timings=timings,
            tile_manifest_path=str(tiles_path),
            chunk_store_dir=str(chunk_dir),
            render_dir=str(render_dir) if args.render else None,
            tile_selection=tile_selection,
            chunk_feature_names=CHUNK_FEATURE_NAMES,
        )
        manifest_path = file_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        global_index.append(
            {
                "file_id": file_id,
                "file_path": str(shp_path),
                "family": "line",
                "manifest_path": str(manifest_path),
            }
        )

        print(
            f"{shp_path.name}: "
            f"lines={len(canonical_lines)}, "
            f"tiles={len(tile_rows)}, "
            f"chunks={total_chunks}, "
            f"chunk_p95={tile_chunk_stats['p95']:.1f}, "
            f"chunk_split_p95={tile_chunk_stats['chunk_split_seconds_p95']:.3f}s, "
            f"budget_hit_ratio={tile_chunk_stats['budget_hit_ratio']:.3f}, "
            f"compression_p50={tile_chunk_stats['compression_ratio_p50']:.6f}, "
            f"train_top_m={tile_selection['train_selected_count']}, "
            f"eval_top_m={tile_selection['eval_selected_count']}, "
            f"tile_score_key={tile_selection['score_key']}"
        )

    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"files": global_index}, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("roads hierarchical Gate 1 cache build finished.")
    print(f"Index: {index_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
