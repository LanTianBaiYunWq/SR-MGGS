"""
Audit positive-pair invariance for roads_hier_stage1 file-level views.

Goal:
1. Separate static file-level stats from augmented-view stats.
2. Measure how much two genuine views of the same file still preserve coarse identity.
3. Provide per-metric diff summaries and a few largest-difference examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset, resolve_path
from roads_hier_stage1.hard_negative_mining import infer_line_subtype
from train_roads_hier_stage1_minimal import load_config
from utils.seed import set_seed


def to_jsonable(obj: Any) -> Any:
    """Convert numpy-heavy structures into JSON-serializable objects."""
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def build_argument_parser() -> argparse.ArgumentParser:
    """Define CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit positive-pair invariance on roads hierarchical cache.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/roads_hier_stage1_minimal.yaml",
        help="Training config path.",
    )
    parser.add_argument("--cache_root", type=str, default=None, help="Override cache root.")
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="eval",
        choices=["train", "eval", "all"],
        help="Tile selection mode used to form per-file inputs.",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to audit.")
    parser.add_argument("--max_tiles", type=int, default=None, help="Override max tiles per file.")
    parser.add_argument("--seed", type=int, default=42, help="Audit RNG seed.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser


def infer_city(file_path: str) -> str:
    """Infer city folder from raw dataset path."""
    path = Path(file_path)
    try:
        return path.parents[1].name
    except IndexError:
        return "unknown"


def percentile(values: List[float], q: float) -> float:
    """Safe percentile helper."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    """Summarize a scalar distribution."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def summarize_pair_diffs(view1_values: List[float], view2_values: List[float], *, eps: float = 1e-8) -> Dict[str, float]:
    """Summarize pairwise differences between two same-file views."""
    arr1 = np.asarray(view1_values, dtype=np.float64)
    arr2 = np.asarray(view2_values, dtype=np.float64)
    abs_diff = np.abs(arr1 - arr2)
    rel_diff = abs_diff / np.maximum(np.maximum(np.abs(arr1), np.abs(arr2)), eps)
    return {
        "view1_mean": float(np.mean(arr1)) if arr1.size else 0.0,
        "view2_mean": float(np.mean(arr2)) if arr2.size else 0.0,
        "abs_diff_mean": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "abs_diff_p50": float(np.percentile(abs_diff, 50)) if abs_diff.size else 0.0,
        "abs_diff_p95": float(np.percentile(abs_diff, 95)) if abs_diff.size else 0.0,
        "abs_diff_max": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "rel_diff_mean": float(np.mean(rel_diff)) if rel_diff.size else 0.0,
        "rel_diff_p50": float(np.percentile(rel_diff, 50)) if rel_diff.size else 0.0,
        "rel_diff_p95": float(np.percentile(rel_diff, 95)) if rel_diff.size else 0.0,
        "rel_diff_max": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "fraction_exact": float(np.mean(abs_diff == 0.0)) if abs_diff.size else 0.0,
        "fraction_near_1pct": float(np.mean(rel_diff <= 0.01)) if rel_diff.size else 0.0,
        "fraction_near_5pct": float(np.mean(rel_diff <= 0.05)) if rel_diff.size else 0.0,
    }


def summarize_tile_rows(tile_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Summarize selected tile rows before any random augmentation."""
    if not tile_rows:
        return {
            "selected_tile_count": 0.0,
            "selected_budgeted_chunk_sum": 0.0,
            "selected_raw_chunk_sum": 0.0,
            "selected_chunk_length_sum": 0.0,
            "selected_chunk_density_mean": 0.0,
            "selected_chunk_density_p95": 0.0,
            "selected_compression_mean": 0.0,
            "selected_compression_p95": 0.0,
            "selected_tile_area_sum": 0.0,
        }

    chunk_densities = [float(row.get("chunk_density", 0.0)) for row in tile_rows]
    compression_ratios = [float(row.get("compression_ratio", 0.0)) for row in tile_rows]
    return {
        "selected_tile_count": float(len(tile_rows)),
        "selected_budgeted_chunk_sum": float(np.sum([float(row.get("budgeted_chunk_count", 0.0)) for row in tile_rows])),
        "selected_raw_chunk_sum": float(np.sum([float(row.get("raw_chunk_count", 0.0)) for row in tile_rows])),
        "selected_chunk_length_sum": float(np.sum([float(row.get("chunk_length_sum", 0.0)) for row in tile_rows])),
        "selected_chunk_density_mean": float(np.mean(chunk_densities)),
        "selected_chunk_density_p95": percentile(chunk_densities, 95),
        "selected_compression_mean": float(np.mean(compression_ratios)),
        "selected_compression_p95": percentile(compression_ratios, 95),
        "selected_tile_area_sum": float(np.sum([float(row.get("tile_area", 0.0)) for row in tile_rows])),
    }


def build_augmented_view_stats(
    sample: Dict[str, Any],
    selected_tile_rows: List[Dict[str, Any]],
    *,
    length_idx: int,
    density_idx: int,
    tile_keep_prob: float,
    chunk_keep_prob: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Mirror the random view protocol and collect view-level coarse stats."""
    tiles = sample["tiles"]
    if not tiles:
        raise ValueError(f"File {sample['file_id']} has no selected tiles.")

    tile_keep_mask = rng.random(len(tiles)) < tile_keep_prob
    if not np.any(tile_keep_mask):
        tile_keep_mask[rng.integers(0, len(tiles))] = True

    retained_tile_count = 0
    retained_chunk_count = 0
    retained_budgeted_chunk_sum = 0.0
    retained_raw_chunk_sum = 0.0
    retained_chunk_length_sum_base = 0.0
    retained_tile_area_sum = 0.0
    retained_compression_values: List[float] = []
    retained_length_norm_values: List[float] = []
    retained_density_values: List[float] = []

    for keep_tile, tile_payload, tile_row in zip(tile_keep_mask, tiles, selected_tile_rows):
        if not keep_tile:
            continue

        chunk_features = np.asarray(tile_payload["chunk_features"], dtype=np.float32)
        if chunk_features.ndim != 2 or chunk_features.shape[0] == 0:
            continue

        chunk_keep_mask = rng.random(chunk_features.shape[0]) < chunk_keep_prob
        if not np.any(chunk_keep_mask):
            chunk_keep_mask[rng.integers(0, chunk_features.shape[0])] = True

        retained = chunk_features[chunk_keep_mask]
        retained_tile_count += 1
        retained_chunk_count += int(retained.shape[0])
        retained_budgeted_chunk_sum += float(tile_row.get("budgeted_chunk_count", 0.0))
        retained_raw_chunk_sum += float(tile_row.get("raw_chunk_count", 0.0))
        retained_chunk_length_sum_base += float(tile_row.get("chunk_length_sum", 0.0))
        retained_tile_area_sum += float(tile_row.get("tile_area", 0.0))
        retained_compression_values.append(float(tile_row.get("compression_ratio", 0.0)))
        retained_length_norm_values.extend(retained[:, length_idx].astype(np.float64).tolist())
        retained_density_values.extend(retained[:, density_idx].astype(np.float64).tolist())

    if retained_tile_count == 0:
        first_tile = tiles[0]
        first_row = selected_tile_rows[0]
        fallback = np.asarray(first_tile["chunk_features"], dtype=np.float32)
        retained_tile_count = 1
        retained_chunk_count = int(fallback.shape[0])
        retained_budgeted_chunk_sum = float(first_row.get("budgeted_chunk_count", 0.0))
        retained_raw_chunk_sum = float(first_row.get("raw_chunk_count", 0.0))
        retained_chunk_length_sum_base = float(first_row.get("chunk_length_sum", 0.0))
        retained_tile_area_sum = float(first_row.get("tile_area", 0.0))
        retained_compression_values = [float(first_row.get("compression_ratio", 0.0))]
        retained_length_norm_values = fallback[:, length_idx].astype(np.float64).tolist()
        retained_density_values = fallback[:, density_idx].astype(np.float64).tolist()

    return {
        "retained_tile_count": float(retained_tile_count),
        "retained_chunk_count": float(retained_chunk_count),
        "retained_budgeted_chunk_sum_base": float(retained_budgeted_chunk_sum),
        "retained_raw_chunk_sum_base": float(retained_raw_chunk_sum),
        "retained_chunk_length_sum_base": float(retained_chunk_length_sum_base),
        "retained_tile_area_sum": float(retained_tile_area_sum),
        "retained_compression_mean_base": float(np.mean(retained_compression_values)) if retained_compression_values else 0.0,
        "retained_compression_p95_base": percentile(retained_compression_values, 95),
        "retained_length_norm_sum": float(np.sum(retained_length_norm_values)),
        "retained_length_norm_mean": float(np.mean(retained_length_norm_values)) if retained_length_norm_values else 0.0,
        "retained_length_norm_p95": percentile(retained_length_norm_values, 95),
        "retained_local_density_mean": float(np.mean(retained_density_values)) if retained_density_values else 0.0,
        "retained_local_density_p95": percentile(retained_density_values, 95),
    }


def main() -> None:
    """Script entry."""
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cache_root:
        config["data"]["cache_root"] = args.cache_root
    if args.max_tiles is not None:
        config["data"]["max_tiles"] = args.max_tiles
    if args.max_files is not None:
        config["data"]["max_files"] = args.max_files

    set_seed(args.seed)
    dataset = GeometryCoreHierDataset(
        config["data"]["cache_root"],
        mode=args.dataset_mode,
        max_tiles=config["data"].get("max_tiles"),
        tile_selector=config["data"].get("tile_selector", "manifest_default"),
    )
    num_files = len(dataset) if args.max_files is None else min(len(dataset), args.max_files)

    tile_keep_prob = float(config["augmentation"].get("tile_keep_prob", 0.85))
    chunk_keep_prob = float(config["augmentation"].get("chunk_keep_prob", 0.85))

    file_rows: List[Dict[str, Any]] = []
    subtype_counts: Dict[str, int] = {}
    city_counts: Dict[str, int] = {}

    static_metrics = [
        "file_bbox_diagonal",
        "manifest_num_tiles",
        "manifest_num_chunks",
        "manifest_budget_hit_ratio",
        "manifest_chunk_p95",
        "manifest_compression_p50",
    ]
    base_metrics = [
        "selected_tile_count",
        "selected_budgeted_chunk_sum",
        "selected_raw_chunk_sum",
        "selected_chunk_length_sum",
        "selected_chunk_density_mean",
        "selected_chunk_density_p95",
        "selected_compression_mean",
        "selected_compression_p95",
        "selected_tile_area_sum",
    ]
    view_metrics = [
        "retained_tile_count",
        "retained_chunk_count",
        "retained_budgeted_chunk_sum_base",
        "retained_raw_chunk_sum_base",
        "retained_chunk_length_sum_base",
        "retained_tile_area_sum",
        "retained_compression_mean_base",
        "retained_compression_p95_base",
        "retained_length_norm_sum",
        "retained_length_norm_mean",
        "retained_length_norm_p95",
        "retained_local_density_mean",
        "retained_local_density_p95",
    ]

    for idx in range(num_files):
        sample = dataset[idx]
        manifest = sample["manifest"]
        file_path = str(sample["file_path"])
        subtype = infer_line_subtype(file_path)
        city = infer_city(file_path)
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        city_counts[city] = city_counts.get(city, 0) + 1

        feature_names = list(manifest.get("chunk_feature_names", []))
        if "length_norm_file_diag" not in feature_names or "local_density" not in feature_names:
            raise ValueError("Cache chunk_feature_names missing length_norm_file_diag/local_density.")
        length_idx = feature_names.index("length_norm_file_diag")
        density_idx = feature_names.index("local_density")

        tile_manifest_path = resolve_path(str(manifest["tile_manifest_path"]))
        with tile_manifest_path.open("r", encoding="utf-8") as f:
            all_tile_rows = json.load(f)
        tile_row_by_id = {row["tile_id"]: row for row in all_tile_rows}
        selected_tile_rows = [tile_row_by_id[tile["tile_id"]] for tile in sample["tiles"]]

        static_stats = {
            "file_bbox_diagonal": float(manifest.get("file_bbox_diagonal", 0.0)),
            "manifest_num_tiles": float(manifest.get("num_tiles", 0.0)),
            "manifest_num_chunks": float(manifest.get("num_chunks", 0.0)),
            "manifest_budget_hit_ratio": float(manifest.get("tile_chunk_count_stats", {}).get("budget_hit_ratio", 0.0)),
            "manifest_chunk_p95": float(manifest.get("tile_chunk_count_stats", {}).get("p95", 0.0)),
            "manifest_compression_p50": float(manifest.get("tile_chunk_count_stats", {}).get("compression_ratio_p50", 0.0)),
        }
        base_stats = summarize_tile_rows(selected_tile_rows)

        rng1 = np.random.default_rng(args.seed + idx * 2)
        rng2 = np.random.default_rng(args.seed + idx * 2 + 1)
        view1_stats = build_augmented_view_stats(
            sample,
            selected_tile_rows,
            length_idx=length_idx,
            density_idx=density_idx,
            tile_keep_prob=tile_keep_prob,
            chunk_keep_prob=chunk_keep_prob,
            rng=rng1,
        )
        view2_stats = build_augmented_view_stats(
            sample,
            selected_tile_rows,
            length_idx=length_idx,
            density_idx=density_idx,
            tile_keep_prob=tile_keep_prob,
            chunk_keep_prob=chunk_keep_prob,
            rng=rng2,
        )

        view_diffs = {
            metric: float(abs(view1_stats[metric] - view2_stats[metric]))
            for metric in view_metrics
        }
        base_vs_view_mean = {
            "tile_keep_ratio_view1": float(view1_stats["retained_tile_count"] / max(base_stats["selected_tile_count"], 1.0)),
            "tile_keep_ratio_view2": float(view2_stats["retained_tile_count"] / max(base_stats["selected_tile_count"], 1.0)),
            "chunk_keep_ratio_view1": float(view1_stats["retained_chunk_count"] / max(base_stats["selected_budgeted_chunk_sum"], 1.0)),
            "chunk_keep_ratio_view2": float(view2_stats["retained_chunk_count"] / max(base_stats["selected_budgeted_chunk_sum"], 1.0)),
        }

        file_rows.append(
            {
                "file_idx": idx,
                "file_id": sample["file_id"],
                "file_path": file_path,
                "city": city,
                "subtype": subtype,
                "num_selected_tiles_dataset": int(sample["num_selected_tiles"]),
                "num_skipped_empty_tiles": int(sample["num_skipped_empty_tiles"]),
                "static_stats": static_stats,
                "base_selected_stats": base_stats,
                "view1_stats": view1_stats,
                "view2_stats": view2_stats,
                "view_abs_diffs": view_diffs,
                "base_vs_view_ratios": base_vs_view_mean,
            }
        )

    static_summary = {
        metric: summarize_distribution([row["static_stats"][metric] for row in file_rows])
        for metric in static_metrics
    }
    base_summary = {
        metric: summarize_distribution([row["base_selected_stats"][metric] for row in file_rows])
        for metric in base_metrics
    }
    view_diff_summary = {
        metric: summarize_pair_diffs(
            [row["view1_stats"][metric] for row in file_rows],
            [row["view2_stats"][metric] for row in file_rows],
        )
        for metric in view_metrics
    }

    largest_chunk_diff = sorted(
        file_rows,
        key=lambda row: row["view_abs_diffs"]["retained_chunk_count"],
        reverse=True,
    )[:10]
    largest_length_diff = sorted(
        file_rows,
        key=lambda row: row["view_abs_diffs"]["retained_length_norm_sum"],
        reverse=True,
    )[:10]
    largest_density_diff = sorted(
        file_rows,
        key=lambda row: row["view_abs_diffs"]["retained_local_density_mean"],
        reverse=True,
    )[:10]

    report = {
        "arguments": {
            "config": args.config,
            "cache_root": config["data"]["cache_root"],
            "dataset_mode": args.dataset_mode,
            "max_files": args.max_files,
            "max_tiles": config["data"].get("max_tiles"),
            "seed": args.seed,
            "tile_keep_prob": tile_keep_prob,
            "chunk_keep_prob": chunk_keep_prob,
            "feature_noise_std": float(config["augmentation"].get("feature_noise_std", 0.0)),
        },
        "n_files": len(file_rows),
        "subtype_counts": subtype_counts,
        "city_counts": city_counts,
        "notes": {
            "static_file_stats": "These are full-file manifest stats and are identical across positive views by construction.",
            "base_selected_stats": "These are post tile-selection but pre-random-augmentation stats.",
            "augmented_view_stats": "These are after random tile_keep and chunk_keep, before adding feature noise.",
        },
        "static_file_stats_summary": static_summary,
        "base_selected_stats_summary": base_summary,
        "augmented_view_diff_summary": view_diff_summary,
        "largest_chunk_count_diff_examples": largest_chunk_diff,
        "largest_length_sum_diff_examples": largest_length_diff,
        "largest_density_mean_diff_examples": largest_density_diff,
    }

    print("=" * 90)
    print("Roads Hier Positive Invariance Audit")
    print("=" * 90)
    print(f"Files: {report['n_files']}")
    print(f"Subtype counts: {report['subtype_counts']}")
    print(f"tile_keep_prob={tile_keep_prob:.2f}, chunk_keep_prob={chunk_keep_prob:.2f}")
    print("Key augmented-view diff metrics:")
    for metric in [
        "retained_tile_count",
        "retained_chunk_count",
        "retained_length_norm_sum",
        "retained_local_density_mean",
        "retained_compression_mean_base",
    ]:
        summary = view_diff_summary[metric]
        print(
            f"  {metric}: "
            f"abs_p50={summary['abs_diff_p50']:.6f}, "
            f"abs_p95={summary['abs_diff_p95']:.6f}, "
            f"rel_mean={summary['rel_diff_mean']:.6f}, "
            f"<=5%={summary['fraction_near_5pct']:.3f}"
        )
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
