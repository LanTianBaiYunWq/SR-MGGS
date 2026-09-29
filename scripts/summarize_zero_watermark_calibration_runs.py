"""Summarize zero-watermark calibration diagnostics across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def load_report(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get(data: Dict[str, Any], path: str) -> float:
    current: Any = data
    for key in path.split("."):
        current = current[key]
    return float(current)


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(np.mean(arr)) if arr.size else 0.0, "std": float(np.std(arr)) if arr.size else 0.0}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize zero-watermark calibration diagnostics.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = [load_report(path) for path in args.reports]

    operating_points: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in ("raw", "z_norm", "cohort_row_norm", "affine", "raw_eer_threshold"):
        operating_points[method] = {
            metric: mean_std(get(report, f"operating_points.{method}.{metric}") for report in reports)
            for metric in ("tar", "far", "frr", "threshold")
        }

    component_shift: Dict[str, Dict[str, Dict[str, float]]] = {}
    for component in ("lp", "point", "subtype", "generic", "final"):
        component_shift[component] = {
            metric: mean_std(get(report, f"components.{component}.shift.{metric}") for report in reports)
            for metric in (
                "genuine_mean_test_minus_calibration",
                "impostor_mean_test_minus_calibration",
                "separability_test_minus_calibration",
                "auc_test_minus_calibration",
                "eer_test_minus_calibration",
            )
        }

    summary = {
        "n_runs": len(reports),
        "reports": args.reports,
        "operating_points": operating_points,
        "component_shift": component_shift,
    }
    print("=" * 90)
    print("Zero-Watermark Calibration Multi-Seed Summary")
    print("=" * 90)
    for method, metrics in operating_points.items():
        print(
            f"{method}: TAR={metrics['tar']['mean']:.6f}±{metrics['tar']['std']:.6f}, "
            f"FAR={metrics['far']['mean']:.6f}±{metrics['far']['std']:.6f}, "
            f"FRR={metrics['frr']['mean']:.6f}±{metrics['frr']['std']:.6f}"
        )
    print("-" * 90)
    for component, metrics in component_shift.items():
        print(
            f"{component}: genuine_shift={metrics['genuine_mean_test_minus_calibration']['mean']:.6f}±"
            f"{metrics['genuine_mean_test_minus_calibration']['std']:.6f}, "
            f"impostor_shift={metrics['impostor_mean_test_minus_calibration']['mean']:.6f}±"
            f"{metrics['impostor_mean_test_minus_calibration']['std']:.6f}, "
            f"sep_shift={metrics['separability_test_minus_calibration']['mean']:.6f}±"
            f"{metrics['separability_test_minus_calibration']['std']:.6f}"
        )
    print("=" * 90)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {out}")


if __name__ == "__main__":
    main()
