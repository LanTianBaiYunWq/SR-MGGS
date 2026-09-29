"""
Sensitivity analysis for the current global preprocessing thresholds.

Why this script exists
----------------------
The current file-level global watermark pipeline relies on preprocess/sgp.py.
The earlier support report already showed that:

- Point is currently not supported at all.
- LineString and Polygon have low preprocessing pass rates.

At this stage, the most important follow-up question is:

    "Are LineString / Polygon poorly supported because the method is inherently
    incompatible with them, or because the current preprocessing thresholds are
    too strict?"

This script focuses on that exact question by sweeping preprocessing thresholds,
especially `min_points`, and measuring how much the pass rate changes.

What this script reports
------------------------
For each threshold setting, the script reports:

1. Feature-level pass rate by geometry type
2. File-level qualification count
   - A file qualifies if it has at least `min_geometries` valid geometries
3. Delta relative to the baseline setting

Recommended first use
---------------------
Use this script before changing the model. If pass rates improve sharply when
`min_points` is reduced, then the current bottleneck is primarily preprocessing,
not the file-level global encoder itself.

Example
-------
python scripts/sweep_preprocess_thresholds.py --data_dir data/raw
python scripts/sweep_preprocess_thresholds.py --data_dir data/raw --min_points_list 4 8 12 16 --output reports/preprocess_sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd

from preprocess.sgp import AdaptiveGeometryNormalizer


# The script lives under scripts/, so the repository root is one level above.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_input_path(path_str: str) -> Path:
    """
    Resolve an input path robustly across terminal and IDE execution modes.

    Resolution order:
    1. Absolute path
    2. Relative to current working directory
    3. Relative to repository root
    """
    raw_path = Path(path_str)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = Path.cwd() / raw_path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = PROJECT_ROOT / raw_path
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def collect_shapefiles(data_dir: str) -> List[Path]:
    """
    Discover all actual shapefile files recursively.

    This explicitly filters with is_file() because this project stores raw data
    in directories whose names may also end with ".shp".
    """
    data_path = resolve_input_path(data_dir)
    return sorted(path for path in data_path.glob("**/*.shp") if path.is_file())


def geometry_type_name(geom) -> str:
    """Return a stable type label, including null and empty geometries."""
    if geom is None:
        return "None"
    if geom.is_empty:
        return "Empty"
    return geom.geom_type


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI for threshold sweep experiments."""
    parser = argparse.ArgumentParser(
        description="Sweep preprocessing thresholds to diagnose geometry-type support bottlenecks."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing shapefiles.",
    )
    parser.add_argument(
        "--target_crs",
        type=str,
        default="EPSG:4326",
        help="Target CRS used before normalization.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=128,
        help="Fixed max point count used by the current global pipeline.",
    )
    parser.add_argument(
        "--min_points_list",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16],
        help="List of min_points values to sweep.",
    )
    parser.add_argument(
        "--min_geometries",
        type=int,
        default=10,
        help="Minimum valid geometry count required for a file to qualify for the current global pipeline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def analyze_file_with_normalizer(
    shp_path: Path,
    normalizer: AdaptiveGeometryNormalizer,
    target_crs: str,
    min_geometries: int,
) -> Dict:
    """
    Analyze one shapefile under one threshold setting.

    This mirrors the current global preprocessing path and measures:
    - raw geometry type counts
    - how many features survive normalization
    - whether the file still qualifies for the global pipeline
    """
    gdf = gpd.read_file(shp_path)

    if gdf.crs is not None and str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    raw_type_counts: Counter = Counter()
    success_counts: Counter = Counter()
    failure_counts: Counter = Counter()

    valid_geometries = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        geom_type = geometry_type_name(geom)
        raw_type_counts[geom_type] += 1

        if geom is None or geom.is_empty:
            continue

        coords = normalizer.normalize(geom)
        if coords is None:
            failure_counts[geom_type] += 1
            continue

        success_counts[geom_type] += 1
        valid_geometries += 1

    return {
        "file_path": str(shp_path),
        "raw_type_counts": dict(raw_type_counts),
        "success_counts": dict(success_counts),
        "failure_counts": dict(failure_counts),
        "valid_geometries": valid_geometries,
        "qualifies": valid_geometries >= min_geometries,
    }


def aggregate_results(per_file_results: List[Dict], min_geometries: int, min_points: int) -> Dict:
    """
    Aggregate per-file results into a compact experiment summary.

    The output is designed to answer:
    - How much feature-level support changes when min_points changes
    - How many files become usable for the current global pipeline
    """
    raw_counter: Counter = Counter()
    success_counter: Counter = Counter()
    failure_counter: Counter = Counter()

    qualified_files = 0
    total_files = len(per_file_results)

    for result in per_file_results:
        raw_counter.update(result["raw_type_counts"])
        success_counter.update(result["success_counts"])
        failure_counter.update(result["failure_counts"])
        if result["qualifies"]:
            qualified_files += 1

    all_types = sorted(set(raw_counter.keys()) | set(success_counter.keys()) | set(failure_counter.keys()))
    type_metrics = {}
    for geom_type in all_types:
        raw = raw_counter.get(geom_type, 0)
        success = success_counter.get(geom_type, 0)
        failure = failure_counter.get(geom_type, 0)
        pass_rate = (success / raw) if raw > 0 else 0.0
        type_metrics[geom_type] = {
            "raw_feature_count": int(raw),
            "success_count": int(success),
            "failure_count": int(failure),
            "success_rate": round(pass_rate, 6),
        }

    return {
        "setting": {
            "min_points": min_points,
            "min_geometries": min_geometries,
        },
        "overview": {
            "total_files": total_files,
            "qualified_files": qualified_files,
            "qualified_file_ratio": round(qualified_files / total_files, 6) if total_files > 0 else 0.0,
        },
        "type_metrics": type_metrics,
    }


def add_deltas_against_baseline(summaries: List[Dict]) -> None:
    """
    Add delta metrics relative to the first threshold setting.

    This makes it easy to answer questions like:
    - How many more files qualify if min_points drops from 16 to 8?
    - How much does LineString pass rate change?
    """
    if not summaries:
        return

    baseline = summaries[0]
    baseline_qualified = baseline["overview"]["qualified_files"]
    baseline_types = baseline["type_metrics"]

    for summary in summaries:
        summary["delta_vs_baseline"] = {
            "qualified_files_delta": summary["overview"]["qualified_files"] - baseline_qualified,
            "type_success_rate_delta": {},
        }

        all_types = sorted(set(summary["type_metrics"].keys()) | set(baseline_types.keys()))
        for geom_type in all_types:
            current_rate = summary["type_metrics"].get(geom_type, {}).get("success_rate", 0.0)
            baseline_rate = baseline_types.get(geom_type, {}).get("success_rate", 0.0)
            summary["delta_vs_baseline"]["type_success_rate_delta"][geom_type] = round(
                current_rate - baseline_rate, 6
            )


def print_summary_table(summaries: List[Dict]) -> None:
    """Print a concise human-readable table for rapid diagnosis."""
    print("=" * 90)
    print("Preprocessing Threshold Sensitivity Report")
    print("=" * 90)

    for summary in summaries:
        min_points = summary["setting"]["min_points"]
        overview = summary["overview"]
        delta = summary.get("delta_vs_baseline", {})
        qualified_delta = delta.get("qualified_files_delta", 0)

        print(
            f"min_points={min_points:<3d} "
            f"qualified_files={overview['qualified_files']:<4d}/{overview['total_files']:<4d} "
            f"ratio={overview['qualified_file_ratio']:<8.4f} "
            f"delta_vs_baseline={qualified_delta:+d}"
        )

        # Print the geometry types that matter most for the current project.
        for geom_type in ["Point", "LineString", "Polygon", "MultiPolygon"]:
            metrics = summary["type_metrics"].get(geom_type)
            if metrics is None:
                continue
            rate_delta = summary["delta_vs_baseline"]["type_success_rate_delta"].get(geom_type, 0.0)
            print(
                f"  {geom_type:<14} "
                f"raw={metrics['raw_feature_count']:<10d} "
                f"success={metrics['success_count']:<10d} "
                f"rate={metrics['success_rate']:<8.4f} "
                f"delta={rate_delta:+.4f}"
            )
        print()

    print("=" * 90)


def main() -> None:
    """Run the sweep."""
    parser = build_argument_parser()
    args = parser.parse_args()

    shp_files = collect_shapefiles(args.data_dir)
    if not shp_files:
        raise ValueError(f"No shapefiles found under: {args.data_dir}")

    print(f"Found {len(shp_files)} shapefiles to analyze.")
    print(f"Sweeping min_points over: {args.min_points_list}")

    all_summaries: List[Dict] = []
    start_time = time.time()

    for min_points in args.min_points_list:
        print(f"Running sweep setting: min_points={min_points}")
        normalizer = AdaptiveGeometryNormalizer(
            max_points=args.max_points,
            min_points=min_points,
        )

        per_file_results: List[Dict] = []
        setting_start = time.time()
        for index, shp_path in enumerate(shp_files, start=1):
            result = analyze_file_with_normalizer(
                shp_path=shp_path,
                normalizer=normalizer,
                target_crs=args.target_crs,
                min_geometries=args.min_geometries,
            )
            per_file_results.append(result)

            if index == 1 or index % 20 == 0 or index == len(shp_files):
                elapsed = time.time() - setting_start
                qualified_so_far = sum(1 for item in per_file_results if item["qualifies"])
                print(
                    f"  [{index}/{len(shp_files)}] elapsed={elapsed:.1f}s "
                    f"qualified_files={qualified_so_far}"
                )

        summary = aggregate_results(
            per_file_results=per_file_results,
            min_geometries=args.min_geometries,
            min_points=min_points,
        )
        summary["files"] = per_file_results
        all_summaries.append(summary)

    add_deltas_against_baseline(all_summaries)
    print_summary_table(all_summaries)

    total_elapsed = time.time() - start_time
    print(f"Total elapsed time: {total_elapsed:.1f}s")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"sweeps": all_summaries}, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
