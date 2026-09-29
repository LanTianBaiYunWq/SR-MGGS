"""
Build offline hierarchical cache for point-family files.

Outputs:
1. tiles
2. point stores
3. manifest
4. global index
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.geometry_core.flatten import flatten_geometry
from roads_hier_stage1.geometrycore_hier_dataset import resolve_path
from roads_hier_stage1.tile_builder import collect_shapefiles, prepare_working_gdf
from roads_hier_stage1.types import build_file_manifest


POINT_FEATURE_NAMES = [
    "x_tile_norm",
    "y_tile_norm",
    "x_global_norm",
    "y_global_norm",
    "nn_dist_norm_file_diag",
    "tile_point_density",
    "radial_norm_tile",
    "k3_nn_mean_norm_file_diag",
    "k5_nn_mean_norm_file_diag",
    "k8_nn_mean_norm_file_diag",
    "x_rank_tile",
    "y_rank_tile",
    "dx_to_tile_center_norm",
    "dy_to_tile_center_norm",
]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build point hierarchical cache.")
    parser.add_argument("--data_dir", type=str, required=True, help="Raw data directory.")
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.shp",
        help="Glob pattern, comma separated patterns are allowed.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Cache output directory.")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files.")
    parser.add_argument(
        "--balance_subtypes",
        action="store_true",
        help="Select files by round-robin point subtype before applying max_files.",
    )
    parser.add_argument(
        "--exclude_area_free",
        action="store_true",
        help="Skip Geofabrik *_a_free_1.shp files, which are area layers rather than point layers.",
    )
    parser.add_argument("--source_crs_if_missing", type=str, default=None)
    parser.add_argument("--working_crs", type=str, default=None)
    parser.add_argument("--max_depth", type=int, default=3, help="Adaptive quadtree max depth.")
    parser.add_argument("--overlap_ratio", type=float, default=0.1, help="Tile overlap ratio.")
    parser.add_argument("--max_points_per_tile", type=int, default=256, help="Split stop threshold.")
    parser.add_argument("--k_points_per_tile", type=int, default=64, help="Per-tile budget.")
    parser.add_argument(
        "--tile_score_key",
        type=str,
        default="budgeted_point_count",
        choices=["budgeted_point_count", "raw_point_count", "point_density"],
        help="Tile ranking score.",
    )
    parser.add_argument("--train_max_tiles", type=int, default=24)
    parser.add_argument("--eval_max_tiles", type=int, default=32)
    return parser


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def print_stage_timing(prefix: str, seconds: float) -> None:
    print(f"  [{prefix}] {seconds:.3f}s")


def infer_point_cache_subtype(path: Path) -> str:
    name = path.name.lower()
    if "pois" in name or name == "pois.shp":
        return "pois"
    if "places" in name or name == "places.shp":
        return "places"
    if "traffic" in name or name == "traffic.shp":
        return "traffic"
    if "transport" in name or name == "transport.shp":
        return "transport"
    if "pofw" in name or name == "pofw.shp":
        return "pofw"
    return "point"


def select_balanced_subtype_files(shp_files: Sequence[Path], max_files: int | None) -> List[Path]:
    subtype_order = ["pois", "places", "traffic", "transport", "pofw", "point"]
    grouped: Dict[str, List[Path]] = {subtype: [] for subtype in subtype_order}
    for path in shp_files:
        subtype = infer_point_cache_subtype(path)
        grouped.setdefault(subtype, []).append(path)

    selected: List[Path] = []
    cursor = {subtype: 0 for subtype in grouped}
    while True:
        progressed = False
        for subtype in subtype_order:
            files = grouped.get(subtype, [])
            idx = cursor.get(subtype, 0)
            if idx >= len(files):
                continue
            selected.append(files[idx])
            cursor[subtype] = idx + 1
            progressed = True
            if max_files is not None and len(selected) >= max_files:
                return selected
        if not progressed:
            break
    return selected


def tile_score_value(tile_row: Dict[str, object], score_key: str) -> float:
    value = tile_row.get(score_key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def bbox_diagonal(bounds: Sequence[float]) -> float:
    minx, miny, maxx, maxy = bounds
    return float(math.hypot(maxx - minx, maxy - miny))


def expand_bounds(bounds: Sequence[float], overlap_ratio: float) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    ox = width * overlap_ratio
    oy = height * overlap_ratio
    return (minx - ox, miny - oy, maxx + ox, maxy + oy)


def split_bounds_quadrants(bounds: Sequence[float]) -> List[Tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bounds
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0
    return [
        (minx, miny, midx, midy),
        (midx, miny, maxx, midy),
        (minx, midy, midx, maxy),
        (midx, midy, maxx, maxy),
    ]


def normalize_to_bbox(x: float, y: float, bbox: Sequence[float]) -> tuple[float, float]:
    minx, miny, maxx, maxy = bbox
    width = max(float(maxx - minx), 1e-9)
    height = max(float(maxy - miny), 1e-9)
    nx = (float(x) - float(minx)) / width
    ny = (float(y) - float(miny)) / height
    return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))


def infer_point_family(file_path: Path) -> str:
    normalized = str(file_path).lower()
    if "poi" in normalized:
        return "point_poi"
    if "facility" in normalized:
        return "point_facility"
    if "station" in normalized:
        return "point_station"
    return "point"


def extract_canonical_points(gdf) -> List[Dict[str, object]]:
    point_records: List[Dict[str, object]] = []
    for feature_id, geom in enumerate(gdf.geometry):
        primitives = flatten_geometry(geom, feature_id)
        local_idx = 0
        for primitive in primitives:
            if primitive.family != "point":
                continue
            coords = np.asarray(primitive.coords, dtype=np.float64)
            for coord_idx, coord in enumerate(coords):
                x, y = float(coord[0]), float(coord[1])
                point_id = f"f{feature_id}_p{local_idx}_{coord_idx}"
                point_records.append(
                    {
                        "feature_id": feature_id,
                        "point_id": point_id,
                        "x": x,
                        "y": y,
                        "bbox": (x, y, x, y),
                    }
                )
            local_idx += 1
    return point_records


def point_in_bounds(point: Dict[str, object], bounds: Sequence[float]) -> bool:
    x = float(point["x"])
    y = float(point["y"])
    minx, miny, maxx, maxy = bounds
    return (minx <= x <= maxx) and (miny <= y <= maxy)


def build_adaptive_quadtree_tiles_for_points(
    point_records: Sequence[Dict[str, object]],
    *,
    root_bounds: Sequence[float],
    max_depth: int,
    overlap_ratio: float,
    max_points_per_tile: int,
) -> List[Dict[str, object]]:
    tiles: List[Dict[str, object]] = []

    def recurse(bounds: Sequence[float], depth: int, prefix: str) -> None:
        assigned = [point["point_id"] for point in point_records if point_in_bounds(point, bounds)]
        if not assigned:
            return
        if len(assigned) <= max_points_per_tile or depth >= max_depth:
            overlap_bounds = expand_bounds(bounds, overlap_ratio)
            tiles.append(
                {
                    "tile_id": prefix,
                    "depth": depth,
                    "bbox": overlap_bounds,
                    "raw_point_count": len(assigned),
                    "assigned_point_ids": assigned,
                }
            )
            return
        for idx, child in enumerate(split_bounds_quadrants(bounds)):
            recurse(child, depth + 1, f"{prefix}_{idx}")

    recurse(root_bounds, 0, "tile")
    return tiles


def select_points_for_tile(
    tile: Dict[str, object],
    point_lookup: Dict[str, Dict[str, object]],
    *,
    k_points_per_tile: int,
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    assigned = [point_lookup[point_id] for point_id in tile["assigned_point_ids"]]
    raw_count = len(assigned)
    if raw_count == 0:
        return [], {"raw_point_count": 0, "budgeted_point_count": 0}

    if raw_count <= k_points_per_tile:
        selected = assigned
    else:
        selected = assigned[:k_points_per_tile]
    return selected, {
        "raw_point_count": raw_count,
        "budgeted_point_count": len(selected),
    }


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] <= 1:
        return np.zeros((coords.shape[0], coords.shape[0]), dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist.astype(np.float32)


def build_point_feature_matrix(
    selected_points: List[Dict[str, object]],
    *,
    tile_bbox: Sequence[float],
    file_bbox: Sequence[float],
    file_diag: float,
    tile_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray([[float(point["x"]), float(point["y"])] for point in selected_points], dtype=np.float32)
    pairwise = compute_pairwise_distances(coords)
    if coords.shape[0] > 1:
        np.fill_diagonal(pairwise, np.inf)
        nn_dist = np.min(pairwise, axis=1)
        sorted_d = np.sort(pairwise, axis=1)
        k3 = np.mean(sorted_d[:, : min(3, sorted_d.shape[1])], axis=1)
        k5 = np.mean(sorted_d[:, : min(5, sorted_d.shape[1])], axis=1)
        k8 = np.mean(sorted_d[:, : min(8, sorted_d.shape[1])], axis=1)
        nn_dist = np.where(np.isfinite(nn_dist), nn_dist, 0.0)
        k3 = np.where(np.isfinite(k3), k3, 0.0)
        k5 = np.where(np.isfinite(k5), k5, 0.0)
        k8 = np.where(np.isfinite(k8), k8, 0.0)
    else:
        nn_dist = np.zeros((coords.shape[0],), dtype=np.float32)
        k3 = np.zeros((coords.shape[0],), dtype=np.float32)
        k5 = np.zeros((coords.shape[0],), dtype=np.float32)
        k8 = np.zeros((coords.shape[0],), dtype=np.float32)

    tile_density = float(coords.shape[0]) / max(float(tile_area), 1e-9)
    tile_diag = max(bbox_diagonal(tile_bbox), 1e-9)
    tile_center = np.asarray(
        [float(tile_bbox[0] + tile_bbox[2]) / 2.0, float(tile_bbox[1] + tile_bbox[3]) / 2.0],
        dtype=np.float32,
    )
    denom_rank = max(coords.shape[0] - 1, 1)
    x_order = np.argsort(coords[:, 0], kind="mergesort")
    y_order = np.argsort(coords[:, 1], kind="mergesort")
    x_rank = np.zeros((coords.shape[0],), dtype=np.float32)
    y_rank = np.zeros((coords.shape[0],), dtype=np.float32)
    x_rank[x_order] = np.arange(coords.shape[0], dtype=np.float32) / float(denom_rank)
    y_rank[y_order] = np.arange(coords.shape[0], dtype=np.float32) / float(denom_rank)

    features: List[np.ndarray] = []
    meta: List[List[str]] = []
    for idx, point in enumerate(selected_points):
        x, y = float(point["x"]), float(point["y"])
        x_tile, y_tile = normalize_to_bbox(x, y, tile_bbox)
        x_global, y_global = normalize_to_bbox(x, y, file_bbox)
        radial = float(np.linalg.norm(coords[idx] - tile_center) / tile_diag)
        dx_to_center = float((coords[idx, 0] - tile_center[0]) / tile_diag)
        dy_to_center = float((coords[idx, 1] - tile_center[1]) / tile_diag)
        features.append(
            np.asarray(
                [
                    x_tile,
                    y_tile,
                    x_global,
                    y_global,
                    float(nn_dist[idx]) / max(float(file_diag), 1e-9),
                    tile_density,
                    radial,
                    float(k3[idx]) / max(float(file_diag), 1e-9),
                    float(k5[idx]) / max(float(file_diag), 1e-9),
                    float(k8[idx]) / max(float(file_diag), 1e-9),
                    float(x_rank[idx]),
                    float(y_rank[idx]),
                    dx_to_center,
                    dy_to_center,
                ],
                dtype=np.float32,
            )
        )
        meta.append([str(point["point_id"]), str(point["feature_id"])])

    return np.asarray(features, dtype=np.float32), np.asarray(meta, dtype=object)


def rank_tiles_for_file(
    tile_rows: List[Dict[str, object]],
    *,
    score_key: str,
    train_max_tiles: int,
    eval_max_tiles: int,
) -> Dict[str, object]:
    ranked = sorted(
        tile_rows,
        key=lambda row: (
            tile_score_value(row, score_key),
            float(row.get("budgeted_point_count", 0)),
            float(row.get("raw_point_count", 0)),
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
    if args.exclude_area_free:
        shp_files = [path for path in shp_files if "_a_free_1" not in path.name.lower()]
    if args.balance_subtypes:
        shp_files = select_balanced_subtype_files(shp_files, args.max_files)
    elif args.max_files is not None:
        shp_files = shp_files[: args.max_files]

    global_index: List[Dict[str, object]] = []

    for file_idx, shp_path in enumerate(shp_files):
        print("=" * 100)
        print(f"Processing file {file_idx + 1}/{len(shp_files)}: {shp_path.name}")
        print("=" * 100)

        file_id = f"file_{file_idx:04d}"
        file_dir = output_dir / file_id
        point_dir = file_dir / "points"
        point_dir.mkdir(parents=True, exist_ok=True)

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
        canonical_points = extract_canonical_points(gdf)
        canonicalize_seconds = time.perf_counter() - t1
        timings.append({"stage": "canonicalize", "seconds": canonicalize_seconds})
        print_stage_timing("canonicalize", canonicalize_seconds)

        if not canonical_points:
            print(f"Skipped {shp_path.name}: no canonical points")
            continue

        minx = min(point["x"] for point in canonical_points)
        miny = min(point["y"] for point in canonical_points)
        maxx = max(point["x"] for point in canonical_points)
        maxy = max(point["y"] for point in canonical_points)
        file_bounds = (minx, miny, maxx, maxy)
        file_diag = bbox_diagonal(file_bounds)
        point_lookup = {str(point["point_id"]): point for point in canonical_points}

        t2 = time.perf_counter()
        tiles = build_adaptive_quadtree_tiles_for_points(
            canonical_points,
            root_bounds=file_bounds,
            max_depth=args.max_depth,
            overlap_ratio=args.overlap_ratio,
            max_points_per_tile=args.max_points_per_tile,
        )
        tile_split_seconds = time.perf_counter() - t2
        timings.append({"stage": "tile_split", "seconds": tile_split_seconds})
        print_stage_timing("tile_split", tile_split_seconds)

        tile_rows: List[Dict[str, object]] = []
        tile_point_counts: List[int] = []
        tile_build_seconds_list: List[float] = []
        densities: List[float] = []
        budget_hit_count = 0
        total_budgeted_points = 0

        for tile in tiles:
            t3 = time.perf_counter()
            selected_points, point_stats = select_points_for_tile(
                tile,
                point_lookup,
                k_points_per_tile=args.k_points_per_tile,
            )
            build_seconds = time.perf_counter() - t3
            tile_build_seconds_list.append(build_seconds)

            tile_bbox = tile["bbox"]
            tile_width = float(tile_bbox[2] - tile_bbox[0])
            tile_height = float(tile_bbox[3] - tile_bbox[1])
            tile_area = max(tile_width * tile_height, 1e-9)

            if selected_points:
                point_features, point_meta = build_point_feature_matrix(
                    selected_points,
                    tile_bbox=tile_bbox,
                    file_bbox=file_bounds,
                    file_diag=file_diag,
                    tile_area=tile_area,
                )
                np.savez_compressed(
                    point_dir / f"{tile['tile_id']}.npz",
                    point_features=point_features,
                    point_meta=point_meta,
                )
            else:
                point_features = np.zeros((0, len(POINT_FEATURE_NAMES)), dtype=np.float32)

            point_density = float(point_stats["budgeted_point_count"]) / tile_area
            tile_rows.append(
                {
                    "tile_id": tile["tile_id"],
                    "depth": tile["depth"],
                    "bbox": list(tile_bbox),
                    "raw_point_count": point_stats["raw_point_count"],
                    "assigned_point_count": len(tile["assigned_point_ids"]),
                    "budgeted_point_count": point_stats["budgeted_point_count"],
                    "tile_area": tile_area,
                    "point_density": point_density,
                    "timings": {"point_select": build_seconds},
                }
            )
            tile_point_counts.append(point_stats["budgeted_point_count"])
            total_budgeted_points += point_stats["budgeted_point_count"]
            if point_stats["budgeted_point_count"] > 0:
                densities.append(point_density)
            if point_stats["budgeted_point_count"] == args.k_points_per_tile:
                budget_hit_count += 1

            print(
                f"  [tile] id={tile['tile_id']} depth={tile['depth']} "
                f"raw_points={point_stats['raw_point_count']} "
                f"budgeted_points={point_stats['budgeted_point_count']} "
                f"point_select={build_seconds:.3f}s"
            )

        tile_selection = rank_tiles_for_file(
            tile_rows,
            score_key=args.tile_score_key,
            train_max_tiles=args.train_max_tiles,
            eval_max_tiles=args.eval_max_tiles,
        )

        tiles_path = file_dir / "tiles.json"
        with tiles_path.open("w", encoding="utf-8") as f:
            json.dump(tile_rows, f, indent=2, ensure_ascii=False)

        tile_point_stats = {
            "p50": percentile(tile_point_counts, 50),
            "p95": percentile(tile_point_counts, 95),
            "max": max(tile_point_counts) if tile_point_counts else 0,
            "point_select_seconds_p50": percentile([int(x * 1000) for x in tile_build_seconds_list], 50) / 1000.0 if tile_build_seconds_list else 0.0,
            "point_select_seconds_p95": percentile([int(x * 1000) for x in tile_build_seconds_list], 95) / 1000.0 if tile_build_seconds_list else 0.0,
            "point_select_seconds_max": max(tile_build_seconds_list) if tile_build_seconds_list else 0.0,
            "point_density_p50": median(densities) if densities else 0.0,
            "point_density_p95": percentile([int(x * 1000000) for x in densities], 95) / 1000000.0 if densities else 0.0,
            "budget_hit_count": budget_hit_count,
            "budget_hit_ratio": (float(budget_hit_count) / float(len(tile_rows))) if tile_rows else 0.0,
        }

        manifest = build_file_manifest(
            file_id=file_id,
            file_path=str(shp_path),
            family="point",
            working_crs=working_crs,
            file_bbox=file_bounds,
            file_bbox_diagonal=file_diag,
            num_raw_features=len(gdf),
            num_canonical_lines=len(canonical_points),
            num_tiles=len(tile_rows),
            num_chunks=total_budgeted_points,
            tile_chunk_count_stats=tile_point_stats,
            timings=timings,
            tile_manifest_path=str(tiles_path),
            chunk_store_dir=str(point_dir),
            render_dir=None,
            tile_selection=tile_selection,
            chunk_feature_names=POINT_FEATURE_NAMES,
        )
        manifest_path = file_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        global_index.append(
            {
                "file_id": file_id,
                "file_path": str(shp_path),
                "family": infer_point_family(shp_path),
                "manifest_path": str(manifest_path),
            }
        )

        print(
            f"{shp_path.name}: "
            f"points={len(canonical_points)}, "
            f"tiles={len(tile_rows)}, "
            f"budgeted_points={total_budgeted_points}, "
            f"point_p95={tile_point_stats['p95']:.1f}, "
            f"point_select_p95={tile_point_stats['point_select_seconds_p95']:.3f}s, "
            f"budget_hit_ratio={tile_point_stats['budget_hit_ratio']:.3f}, "
            f"train_top_m={tile_selection['train_selected_count']}, "
            f"eval_top_m={tile_selection['eval_selected_count']}, "
            f"tile_score_key={tile_selection['score_key']}"
        )

    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"files": global_index}, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("point hierarchical cache build finished.")
    print(f"Index: {index_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
