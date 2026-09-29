"""Build offline hierarchical cache for polygon-family files."""

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


POLYGON_FEATURE_NAMES = [
    "cx_tile_norm",
    "cy_tile_norm",
    "cx_global_norm",
    "cy_global_norm",
    "area_norm_file_bbox",
    "perimeter_norm_file_diag",
    "compactness",
    "is_hole",
    "radial_norm_tile",
    "bbox_aspect",
]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build polygon hierarchical cache.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.shp")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--source_crs_if_missing", type=str, default=None)
    parser.add_argument("--working_crs", type=str, default=None)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--overlap_ratio", type=float, default=0.1)
    parser.add_argument("--max_polygons_per_tile", type=int, default=256)
    parser.add_argument("--k_polygons_per_tile", type=int, default=64)
    parser.add_argument(
        "--tile_score_key",
        type=str,
        default="budgeted_polygon_count",
        choices=["budgeted_polygon_count", "raw_polygon_count", "polygon_density"],
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


def infer_polygon_family(file_path: Path) -> str:
    normalized = file_path.name.lower()
    if "building" in normalized:
        return "polygon_building"
    if "landuse" in normalized:
        return "polygon_landuse"
    if "natural" in normalized:
        return "polygon_natural"
    if "water" in normalized:
        return "polygon_water"
    return "polygon"


def signed_area(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() * 0.5)


def ring_perimeter(coords: np.ndarray) -> float:
    closed = np.vstack([coords, coords[:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def extract_canonical_polygons(gdf) -> List[Dict[str, object]]:
    polygon_records: List[Dict[str, object]] = []
    for feature_id, geom in enumerate(gdf.geometry):
        primitives = flatten_geometry(geom, feature_id)
        for primitive in primitives:
            if primitive.family != "polygon":
                continue
            coords = np.asarray(primitive.coords, dtype=np.float64)
            if coords.shape[0] < 3:
                continue
            minx = float(np.min(coords[:, 0]))
            miny = float(np.min(coords[:, 1]))
            maxx = float(np.max(coords[:, 0]))
            maxy = float(np.max(coords[:, 1]))
            area = abs(signed_area(coords))
            perimeter = ring_perimeter(coords)
            centroid = np.mean(coords, axis=0)
            polygon_records.append(
                {
                    "feature_id": feature_id,
                    "polygon_id": f"f{feature_id}_poly{primitive.part_id}",
                    "coords": coords.tolist(),
                    "bbox": (minx, miny, maxx, maxy),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "is_hole": bool(primitive.is_hole),
                }
            )
    return polygon_records


def polygon_in_bounds(record: Dict[str, object], bounds: Sequence[float]) -> bool:
    x = float(record["centroid_x"])
    y = float(record["centroid_y"])
    minx, miny, maxx, maxy = bounds
    return (minx <= x <= maxx) and (miny <= y <= maxy)


def build_adaptive_quadtree_tiles_for_polygons(
    polygon_records: Sequence[Dict[str, object]],
    *,
    root_bounds: Sequence[float],
    max_depth: int,
    overlap_ratio: float,
    max_polygons_per_tile: int,
) -> List[Dict[str, object]]:
    tiles: List[Dict[str, object]] = []

    def recurse(bounds: Sequence[float], depth: int, prefix: str) -> None:
        assigned = [record["polygon_id"] for record in polygon_records if polygon_in_bounds(record, bounds)]
        if not assigned:
            return
        if len(assigned) <= max_polygons_per_tile or depth >= max_depth:
            tiles.append(
                {
                    "tile_id": prefix,
                    "depth": depth,
                    "bbox": expand_bounds(bounds, overlap_ratio),
                    "raw_polygon_count": len(assigned),
                    "assigned_polygon_ids": assigned,
                }
            )
            return
        for idx, child in enumerate(split_bounds_quadrants(bounds)):
            recurse(child, depth + 1, f"{prefix}_{idx}")

    recurse(root_bounds, 0, "tile")
    return tiles


def select_polygons_for_tile(
    tile: Dict[str, object],
    polygon_lookup: Dict[str, Dict[str, object]],
    *,
    k_polygons_per_tile: int,
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    assigned = [polygon_lookup[poly_id] for poly_id in tile["assigned_polygon_ids"]]
    raw_count = len(assigned)
    if raw_count == 0:
        return [], {"raw_polygon_count": 0, "budgeted_polygon_count": 0}
    if raw_count <= k_polygons_per_tile:
        selected = assigned
    else:
        selected = assigned[:k_polygons_per_tile]
    return selected, {"raw_polygon_count": raw_count, "budgeted_polygon_count": len(selected)}


def build_polygon_feature_matrix(
    selected_polygons: List[Dict[str, object]],
    *,
    tile_bbox: Sequence[float],
    file_bbox: Sequence[float],
    file_diag: float,
    file_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    tile_diag = max(bbox_diagonal(tile_bbox), 1e-9)
    tile_center = np.asarray(
        [float(tile_bbox[0] + tile_bbox[2]) / 2.0, float(tile_bbox[1] + tile_bbox[3]) / 2.0],
        dtype=np.float32,
    )
    features: List[np.ndarray] = []
    meta: List[List[str]] = []
    for polygon in selected_polygons:
        cx = float(polygon["centroid_x"])
        cy = float(polygon["centroid_y"])
        x_tile, y_tile = normalize_to_bbox(cx, cy, tile_bbox)
        x_global, y_global = normalize_to_bbox(cx, cy, file_bbox)
        area = float(polygon["area"])
        perimeter = float(polygon["perimeter"])
        compactness = float(4.0 * math.pi * area / (perimeter ** 2 + 1e-8))
        radial = float(np.linalg.norm(np.asarray([cx, cy], dtype=np.float32) - tile_center) / tile_diag)
        bbox = polygon["bbox"]
        bbox_w = max(float(bbox[2] - bbox[0]), 1e-9)
        bbox_h = max(float(bbox[3] - bbox[1]), 1e-9)
        features.append(
            np.asarray(
                [
                    x_tile,
                    y_tile,
                    x_global,
                    y_global,
                    area / max(file_area, 1e-9),
                    perimeter / max(file_diag, 1e-9),
                    compactness,
                    float(bool(polygon["is_hole"])),
                    radial,
                    bbox_w / bbox_h,
                ],
                dtype=np.float32,
            )
        )
        meta.append([str(polygon["polygon_id"]), str(polygon["feature_id"])])
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
            float(row.get("budgeted_polygon_count", 0)),
            float(row.get("raw_polygon_count", 0)),
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
        polygon_dir = file_dir / "polygons"
        polygon_dir.mkdir(parents=True, exist_ok=True)
        timings: List[Dict[str, float]] = []

        t0 = time.perf_counter()
        gdf, working_crs = prepare_working_gdf(
            shp_path,
            source_crs_if_missing=args.source_crs_if_missing,
            working_crs=args.working_crs,
        )
        seconds = time.perf_counter() - t0
        timings.append({"stage": "read_reproject", "seconds": seconds})
        print_stage_timing("read_reproject", seconds)

        t1 = time.perf_counter()
        canonical_polygons = extract_canonical_polygons(gdf)
        seconds = time.perf_counter() - t1
        timings.append({"stage": "canonicalize", "seconds": seconds})
        print_stage_timing("canonicalize", seconds)
        if not canonical_polygons:
            print(f"Skipped {shp_path.name}: no canonical polygons")
            continue

        minx = min(record["bbox"][0] for record in canonical_polygons)
        miny = min(record["bbox"][1] for record in canonical_polygons)
        maxx = max(record["bbox"][2] for record in canonical_polygons)
        maxy = max(record["bbox"][3] for record in canonical_polygons)
        file_bounds = (minx, miny, maxx, maxy)
        file_diag = bbox_diagonal(file_bounds)
        file_area = max(float((maxx - minx) * (maxy - miny)), 1e-9)
        polygon_lookup = {str(record["polygon_id"]): record for record in canonical_polygons}

        t2 = time.perf_counter()
        tiles = build_adaptive_quadtree_tiles_for_polygons(
            canonical_polygons,
            root_bounds=file_bounds,
            max_depth=args.max_depth,
            overlap_ratio=args.overlap_ratio,
            max_polygons_per_tile=args.max_polygons_per_tile,
        )
        seconds = time.perf_counter() - t2
        timings.append({"stage": "tile_split", "seconds": seconds})
        print_stage_timing("tile_split", seconds)

        tile_rows: List[Dict[str, object]] = []
        tile_counts: List[int] = []
        build_seconds_list: List[float] = []
        densities: List[float] = []
        budget_hit_count = 0
        total_budgeted_polygons = 0

        for tile in tiles:
            t3 = time.perf_counter()
            selected_polygons, poly_stats = select_polygons_for_tile(
                tile,
                polygon_lookup,
                k_polygons_per_tile=args.k_polygons_per_tile,
            )
            build_seconds = time.perf_counter() - t3
            build_seconds_list.append(build_seconds)

            tile_bbox = tile["bbox"]
            tile_width = float(tile_bbox[2] - tile_bbox[0])
            tile_height = float(tile_bbox[3] - tile_bbox[1])
            tile_area = max(tile_width * tile_height, 1e-9)

            if selected_polygons:
                polygon_features, polygon_meta = build_polygon_feature_matrix(
                    selected_polygons,
                    tile_bbox=tile_bbox,
                    file_bbox=file_bounds,
                    file_diag=file_diag,
                    file_area=file_area,
                )
                np.savez_compressed(
                    polygon_dir / f"{tile['tile_id']}.npz",
                    polygon_features=polygon_features,
                    polygon_meta=polygon_meta,
                )
            polygon_density = float(poly_stats["budgeted_polygon_count"]) / tile_area
            tile_rows.append(
                {
                    "tile_id": tile["tile_id"],
                    "depth": tile["depth"],
                    "bbox": list(tile_bbox),
                    "raw_polygon_count": poly_stats["raw_polygon_count"],
                    "assigned_polygon_count": len(tile["assigned_polygon_ids"]),
                    "budgeted_polygon_count": poly_stats["budgeted_polygon_count"],
                    "tile_area": tile_area,
                    "polygon_density": polygon_density,
                    "timings": {"polygon_select": build_seconds},
                }
            )
            tile_counts.append(poly_stats["budgeted_polygon_count"])
            total_budgeted_polygons += poly_stats["budgeted_polygon_count"]
            if poly_stats["budgeted_polygon_count"] > 0:
                densities.append(polygon_density)
            if poly_stats["budgeted_polygon_count"] == args.k_polygons_per_tile:
                budget_hit_count += 1

            print(
                f"  [tile] id={tile['tile_id']} depth={tile['depth']} "
                f"raw_polygons={poly_stats['raw_polygon_count']} "
                f"budgeted_polygons={poly_stats['budgeted_polygon_count']} "
                f"polygon_select={build_seconds:.3f}s"
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

        tile_stats = {
            "p50": percentile(tile_counts, 50),
            "p95": percentile(tile_counts, 95),
            "max": max(tile_counts) if tile_counts else 0,
            "polygon_select_seconds_p50": percentile([int(x * 1000) for x in build_seconds_list], 50) / 1000.0 if build_seconds_list else 0.0,
            "polygon_select_seconds_p95": percentile([int(x * 1000) for x in build_seconds_list], 95) / 1000.0 if build_seconds_list else 0.0,
            "polygon_select_seconds_max": max(build_seconds_list) if build_seconds_list else 0.0,
            "polygon_density_p50": median(densities) if densities else 0.0,
            "polygon_density_p95": percentile([int(x * 1000000) for x in densities], 95) / 1000000.0 if densities else 0.0,
            "budget_hit_count": budget_hit_count,
            "budget_hit_ratio": (float(budget_hit_count) / float(len(tile_rows))) if tile_rows else 0.0,
        }

        manifest = build_file_manifest(
            file_id=file_id,
            file_path=str(shp_path),
            family="polygon",
            working_crs=working_crs,
            file_bbox=file_bounds,
            file_bbox_diagonal=file_diag,
            num_raw_features=len(gdf),
            num_canonical_lines=len(canonical_polygons),
            num_tiles=len(tile_rows),
            num_chunks=total_budgeted_polygons,
            tile_chunk_count_stats=tile_stats,
            timings=timings,
            tile_manifest_path=str(tiles_path),
            chunk_store_dir=str(polygon_dir),
            render_dir=None,
            tile_selection=tile_selection,
            chunk_feature_names=POLYGON_FEATURE_NAMES,
        )
        manifest_path = file_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        global_index.append(
            {
                "file_id": file_id,
                "file_path": str(shp_path),
                "family": infer_polygon_family(shp_path),
                "manifest_path": str(manifest_path),
            }
        )

        print(
            f"{shp_path.name}: polygons={len(canonical_polygons)}, tiles={len(tile_rows)}, "
            f"budgeted_polygons={total_budgeted_polygons}, polygon_p95={tile_stats['p95']:.1f}, "
            f"polygon_select_p95={tile_stats['polygon_select_seconds_p95']:.3f}s, "
            f"budget_hit_ratio={tile_stats['budget_hit_ratio']:.3f}, "
            f"train_top_m={tile_selection['train_selected_count']}, "
            f"eval_top_m={tile_selection['eval_selected_count']}, tile_score_key={tile_selection['score_key']}"
        )

    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"files": global_index}, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("polygon hierarchical cache build finished.")
    print(f"Index: {index_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
