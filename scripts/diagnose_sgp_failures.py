"""
旧 preprocess/sgp.py 流水线的阶段级故障诊断。为什么这个脚本存在 ----------------------
当前的全局水印路线仍依赖于旧的 SGP 归一化器。我们已经知道文件级的合格性有限，并且降低 `min_points` 本身并不能解决问题。
下一步正确的做法不是更多盲目的阈值调节。下一步正确的做法是识别旧预处理链中特征失败的具体位置。
这个脚本以分阶段的方式重放旧的 GeometryNormalizer / AdaptiveGeometryNormalizer 逻辑，
并记录：
- 哪个阶段拒绝了特征
- 阶段前后的几何类型
- 阶段前后的坐标数量
- 阶段前后的有效性
- 阶段前后的最小间隙 输出可用于构建透视表，
例如： - raw_type x fail_stage - raw_type x fail_reason - file_path x fail_stage

Typical use
-----------
python scripts/diagnose_sgp_failures.py --data_dir data/raw
python scripts/diagnose_sgp_failures.py --data_dir data/raw --detail_output reports/sgp_failure_details.csv --summary_output reports/sgp_failure_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
from shapely.validation import explain_validity

from preprocess.sgp import AdaptiveGeometryNormalizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_input_path(path_str: str) -> Path:
    """Resolve input paths robustly for terminal and IDE usage."""
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
    """Collect actual shapefile files recursively."""
    base = resolve_input_path(data_dir)
    return sorted(path for path in base.glob("**/*.shp") if path.is_file())


def geometry_type_name(geom) -> str:
    """Return a stable geometry type label including null and empty cases."""
    if geom is None:
        return "None"
    if geom.is_empty:
        return "Empty"
    return geom.geom_type


def count_coordinates_from_geometry(geom) -> int:
    """Count coordinates recursively for geometry diagnostics."""
    if geom is None or geom.is_empty:
        return 0

    geom_type = geom.geom_type
    if geom_type == "Point":
        return 1
    if geom_type in {"LineString", "LinearRing"}:
        return len(geom.coords)
    if geom_type == "Polygon":
        total = len(geom.exterior.coords)
        for ring in geom.interiors:
            total += len(ring.coords)
        return total
    if hasattr(geom, "geoms"):
        return sum(count_coordinates_from_geometry(part) for part in geom.geoms)
    return 0


def safe_is_valid(geom) -> Optional[bool]:
    """Return validity when meaningful, otherwise None."""
    if geom is None or geom.is_empty:
        return None
    try:
        return bool(geom.is_valid)
    except Exception:
        return None


def safe_valid_reason(geom) -> Optional[str]:
    """Explain invalidity when possible."""
    if geom is None or geom.is_empty:
        return None
    try:
        return explain_validity(geom)
    except Exception:
        return None


def safe_minimum_clearance(geom) -> Optional[float]:
    """Expose shapely minimum_clearance when available."""
    if geom is None or geom.is_empty:
        return None
    try:
        return float(geom.minimum_clearance)
    except Exception:
        return None


def coord_count_from_array(coords: Optional[np.ndarray]) -> int:
    """Count coordinates in a normalized coordinate array."""
    if coords is None:
        return 0
    return int(len(coords))


def append_stage_row(
    rows: List[Dict],
    *,
    file_path: str,
    feature_id: int,
    raw_type: str,
    stage: str,
    before_type: Optional[str],
    after_type: Optional[str],
    n_coords_before: Optional[int],
    n_coords_after: Optional[int],
    is_valid_before: Optional[bool],
    is_valid_after: Optional[bool],
    valid_reason_before: Optional[str],
    valid_reason_after: Optional[str],
    minimum_clearance_before: Optional[float],
    minimum_clearance_after: Optional[float],
    fail_reason: Optional[str],
    accepted: bool,
) -> None:
    """
    Append one long-form stage record.

    The long format is intentional: it is easier to aggregate later using pivot
    tables than a wide per-feature format with many optional columns.
    """
    rows.append(
        {
            "file_path": file_path,
            "feature_id": feature_id,
            "raw_type": raw_type,
            "stage": stage,
            "before_type": before_type,
            "after_type": after_type,
            "n_coords_before": n_coords_before,
            "n_coords_after": n_coords_after,
            "is_valid_before": is_valid_before,
            "is_valid_after": is_valid_after,
            "valid_reason_before": valid_reason_before,
            "valid_reason_after": valid_reason_after,
            "minimum_clearance_before": minimum_clearance_before,
            "minimum_clearance_after": minimum_clearance_after,
            "fail_reason": fail_reason,
            "accepted": accepted,
        }
    )


DETAIL_FIELDNAMES = [
    "file_path",
    "feature_id",
    "raw_type",
    "stage",
    "before_type",
    "after_type",
    "n_coords_before",
    "n_coords_after",
    "is_valid_before",
    "is_valid_after",
    "valid_reason_before",
    "valid_reason_after",
    "minimum_clearance_before",
    "minimum_clearance_after",
    "fail_reason",
    "accepted",
]


def diagnose_feature(
    geom,
    feature_id: int,
    file_path: str,
    normalizer: AdaptiveGeometryNormalizer,
) -> List[Dict]:
    """
    Replay the old SGP normalization stages with explicit logging.

    This function mirrors the old path as closely as possible:
    - extract coords
    - min_points gate
    - simplify when coord count is too large
    - adaptive resample
    - normalize coords
    - align deterministic start point
    """
    rows: List[Dict] = []
    raw_type = geometry_type_name(geom)
    before_type = raw_type
    before_count = count_coordinates_from_geometry(geom)
    before_valid = safe_is_valid(geom)
    before_reason = safe_valid_reason(geom)
    before_clearance = safe_minimum_clearance(geom)

    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="read_ok",
        before_type=before_type,
        after_type=before_type,
        n_coords_before=before_count,
        n_coords_after=before_count,
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason=None,
        accepted=False,
    )

    if geom is None:
        append_stage_row(
            rows,
            file_path=file_path,
            feature_id=feature_id,
            raw_type=raw_type,
            stage="rejected",
            before_type=raw_type,
            after_type=raw_type,
            n_coords_before=0,
            n_coords_after=0,
            is_valid_before=None,
            is_valid_after=None,
            valid_reason_before=None,
            valid_reason_after=None,
            minimum_clearance_before=None,
            minimum_clearance_after=None,
            fail_reason="geometry_is_none",
            accepted=False,
        )
        return rows

    if geom.is_empty:
        append_stage_row(
            rows,
            file_path=file_path,
            feature_id=feature_id,
            raw_type=raw_type,
            stage="rejected",
            before_type=raw_type,
            after_type=raw_type,
            n_coords_before=0,
            n_coords_after=0,
            is_valid_before=None,
            is_valid_after=None,
            valid_reason_before=None,
            valid_reason_after=None,
            minimum_clearance_before=None,
            minimum_clearance_after=None,
            fail_reason="geometry_is_empty",
            accepted=False,
        )
        return rows

    # Stage 1: old _extract_coords()
    coords = normalizer._extract_coords(geom)
    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="extract_coords",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=before_count,
        n_coords_after=coord_count_from_array(coords),
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason="extract_coords_returned_none" if coords is None else None,
        accepted=False,
    )
    if coords is None:
        append_stage_row(
            rows,
            file_path=file_path,
            feature_id=feature_id,
            raw_type=raw_type,
            stage="rejected",
            before_type=raw_type,
            after_type=raw_type,
            n_coords_before=before_count,
            n_coords_after=0,
            is_valid_before=before_valid,
            is_valid_after=before_valid,
            valid_reason_before=before_reason,
            valid_reason_after=before_reason,
            minimum_clearance_before=before_clearance,
            minimum_clearance_after=before_clearance,
            fail_reason="extract_coords_returned_none",
            accepted=False,
        )
        return rows

    # Stage 2: min_points gate
    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="min_points_gate",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=len(coords),
        n_coords_after=len(coords),
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason=(
            f"coord_count_below_min_points:{len(coords)}<{normalizer.min_points}"
            if len(coords) < normalizer.min_points
            else None
        ),
        accepted=False,
    )
    if len(coords) < normalizer.min_points:
        append_stage_row(
            rows,
            file_path=file_path,
            feature_id=feature_id,
            raw_type=raw_type,
            stage="rejected",
            before_type=raw_type,
            after_type=raw_type,
            n_coords_before=len(coords),
            n_coords_after=len(coords),
            is_valid_before=before_valid,
            is_valid_after=before_valid,
            valid_reason_before=before_reason,
            valid_reason_after=before_reason,
            minimum_clearance_before=before_clearance,
            minimum_clearance_after=before_clearance,
            fail_reason=f"coord_count_below_min_points:{len(coords)}<{normalizer.min_points}",
            accepted=False,
        )
        return rows

    # Stage 3: adaptive target point calculation
    target_points = normalizer._compute_adaptive_points(geom, coords)
    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="adaptive_target_points",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=len(coords),
        n_coords_after=target_points,
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason=None,
        accepted=False,
    )

    # Stage 4: simplify when coord count is large.
    coords_after_simplify = coords
    if len(coords) > target_points * 2:
        coords_after_simplify = normalizer._simplify(coords)
    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="simplify",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=len(coords),
        n_coords_after=coord_count_from_array(coords_after_simplify),
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason="simplify_returned_none" if coords_after_simplify is None else None,
        accepted=False,
    )
    if coords_after_simplify is None:
        append_stage_row(
            rows,
            file_path=file_path,
            feature_id=feature_id,
            raw_type=raw_type,
            stage="rejected",
            before_type=raw_type,
            after_type=raw_type,
            n_coords_before=len(coords),
            n_coords_after=0,
            is_valid_before=before_valid,
            is_valid_after=before_valid,
            valid_reason_before=before_reason,
            valid_reason_after=before_reason,
            minimum_clearance_before=before_clearance,
            minimum_clearance_after=before_clearance,
            fail_reason="simplify_returned_none",
            accepted=False,
        )
        return rows

    # Stage 5: resample to target_points.
    original_max_points = normalizer.max_points
    normalizer.max_points = target_points
    try:
        coords_after_resample = normalizer._resample(coords_after_simplify)
    finally:
        normalizer.max_points = original_max_points

    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="resample",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=len(coords_after_simplify),
        n_coords_after=coord_count_from_array(coords_after_resample),
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason="resample_returned_none" if coords_after_resample is None else None,
        accepted=False,
    )
    if coords_after_resample is None:
        append_stage_row(
            rows,
            file_path=file_path,
            feature_id=feature_id,
            raw_type=raw_type,
            stage="rejected",
            before_type=raw_type,
            after_type=raw_type,
            n_coords_before=len(coords_after_simplify),
            n_coords_after=0,
            is_valid_before=before_valid,
            is_valid_after=before_valid,
            valid_reason_before=before_reason,
            valid_reason_after=before_reason,
            minimum_clearance_before=before_clearance,
            minimum_clearance_after=before_clearance,
            fail_reason="resample_returned_none",
            accepted=False,
        )
        return rows

    # Stage 6: normalize coordinates.
    coords_after_normalize = normalizer._normalize_coords(coords_after_resample)
    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="normalize_coords",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=len(coords_after_resample),
        n_coords_after=coord_count_from_array(coords_after_normalize),
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason=None,
        accepted=False,
    )

    # Stage 7: deterministic start-point alignment.
    coords_after_align = normalizer._align_start_point(coords_after_normalize)
    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="align_start_point",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=len(coords_after_normalize),
        n_coords_after=coord_count_from_array(coords_after_align),
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason=None,
        accepted=False,
    )

    append_stage_row(
        rows,
        file_path=file_path,
        feature_id=feature_id,
        raw_type=raw_type,
        stage="accepted",
        before_type=raw_type,
        after_type=raw_type,
        n_coords_before=len(coords_after_align),
        n_coords_after=len(coords_after_align),
        is_valid_before=before_valid,
        is_valid_after=before_valid,
        valid_reason_before=before_reason,
        valid_reason_after=before_reason,
        minimum_clearance_before=before_clearance,
        minimum_clearance_after=before_clearance,
        fail_reason=None,
        accepted=True,
    )
    return rows


def open_detail_writer(output_path: Path):
    """Open a CSV writer for streaming detail rows to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_handle = output_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(file_handle, fieldnames=DETAIL_FIELDNAMES)
    writer.writeheader()
    return file_handle, writer


class StreamingSummary:
    """Maintain aggregate counters online to avoid storing all detail rows."""

    def __init__(self):
        self.fail_stage_counter: Counter = Counter()
        self.fail_reason_counter: Counter = Counter()
        self.fail_stage_by_type: Dict[str, Counter] = defaultdict(Counter)
        self.accepted_by_type: Counter = Counter()
        self.rejected_by_type: Counter = Counter()

    def update(self, rows: List[Dict]) -> None:
        """Update counters from one feature's stage rows."""
        for row in rows:
            stage = row["stage"]
            raw_type = row["raw_type"]
            fail_reason = row["fail_reason"]

            if stage == "accepted":
                self.accepted_by_type[raw_type] += 1
            elif stage == "rejected":
                self.rejected_by_type[raw_type] += 1
                if fail_reason is not None:
                    self.fail_reason_counter[fail_reason] += 1
                    root = fail_reason.split(":")[0]
                    self.fail_stage_counter[root] += 1
                    self.fail_stage_by_type[raw_type][root] += 1

    def to_dict(self) -> Dict:
        """Serialize current counters into a summary dictionary."""
        return {
            "accepted_by_type": dict(self.accepted_by_type),
            "rejected_by_type": dict(self.rejected_by_type),
            "top_fail_reasons": dict(self.fail_reason_counter.most_common(20)),
            "top_fail_stage_roots": dict(self.fail_stage_counter.most_common(20)),
            "fail_stage_by_type": {
                geom_type: dict(counter.most_common())
                for geom_type, counter in self.fail_stage_by_type.items()
            },
        }


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI for the old SGP failure-diagnosis tool."""
    parser = argparse.ArgumentParser(
        description="Diagnose feature-level failures inside the old preprocess/sgp.py pipeline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing raw shapefiles.",
    )
    parser.add_argument(
        "--target_crs",
        type=str,
        default="EPSG:4326",
        help="Target CRS used before feature-level diagnosis.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=128,
        help="max_points fed into AdaptiveGeometryNormalizer.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=16,
        help="min_points fed into AdaptiveGeometryNormalizer.",
    )
    parser.add_argument(
        "--detail_output",
        type=str,
        default=None,
        help="Optional CSV path for long-form stage rows.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=None,
        help="Optional JSON path for aggregate summary.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional file limit for debugging or partial diagnosis.",
    )
    return parser


def main() -> None:
    """Entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    shp_files = collect_shapefiles(args.data_dir)
    if not shp_files:
        raise ValueError(f"No shapefiles found under: {args.data_dir}")
    if args.max_files is not None:
        shp_files = shp_files[: args.max_files]

    print(f"Found {len(shp_files)} shapefiles to diagnose.")
    normalizer = AdaptiveGeometryNormalizer(
        max_points=args.max_points,
        min_points=args.min_points,
    )

    summary_state = StreamingSummary()
    detail_file_handle = None
    detail_writer = None
    if args.detail_output:
        detail_file_handle, detail_writer = open_detail_writer(Path(args.detail_output))

    start_time = time.time()
    total_rows_written = 0

    try:
        for index, shp_path in enumerate(shp_files, start=1):
            gdf = gpd.read_file(shp_path)
            if gdf.crs is not None and str(gdf.crs) != args.target_crs:
                gdf = gdf.to_crs(args.target_crs)

            for feature_id, row in enumerate(gdf.itertuples(index=False), start=0):
                geom = getattr(row, "geometry", None)
                feature_rows = diagnose_feature(
                    geom=geom,
                    feature_id=feature_id,
                    file_path=str(shp_path),
                    normalizer=normalizer,
                )
                summary_state.update(feature_rows)
                if detail_writer is not None:
                    detail_writer.writerows(feature_rows)
                    total_rows_written += len(feature_rows)

            if index == 1 or index % 10 == 0 or index == len(shp_files):
                elapsed = time.time() - start_time
                print(
                    f"[{index}/{len(shp_files)}] elapsed={elapsed:.1f}s "
                    f"detail_rows_written={total_rows_written}"
                )
    finally:
        if detail_file_handle is not None:
            detail_file_handle.close()

    summary = summary_state.to_dict()

    print("=" * 90)
    print("Old SGP Failure Diagnosis Summary")
    print("=" * 90)
    print("Accepted by type:")
    for geom_type, count in summary["accepted_by_type"].items():
        print(f"  {geom_type:<18} {count}")
    print("Rejected by type:")
    for geom_type, count in summary["rejected_by_type"].items():
        print(f"  {geom_type:<18} {count}")
    print("Top fail reasons:")
    for reason, count in summary["top_fail_reasons"].items():
        print(f"  {reason:<40} {count}")
    print("=" * 90)
    print(f"Total elapsed time: {time.time() - start_time:.1f}s")

    if args.detail_output:
        print(f"Saved detail CSV to: {Path(args.detail_output)}")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
