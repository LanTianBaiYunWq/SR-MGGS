"""Analyze calibration-to-heldout score shift for authentication reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def load_report(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)) if array.size else 0.0,
        "std": float(np.std(array)) if array.size else 0.0,
    }


def get(report: Dict[str, Any], path: str) -> float:
    current: Any = report
    for key in path.split("."):
        current = current[key]
    return float(current)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze authentication score distribution shift.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = [load_report(path) for path in args.reports]

    fields = {
        "calibration_auc": "calibration.metrics.pairwise.auc",
        "test_auc": "test.metrics.pairwise.auc",
        "calibration_eer": "calibration.metrics.pairwise.eer",
        "test_eer": "test.metrics.pairwise.eer",
        "calibration_genuine_mean": "calibration.metrics.pairwise.genuine_mean",
        "test_genuine_mean": "test.metrics.pairwise.genuine_mean",
        "calibration_impostor_mean": "calibration.metrics.pairwise.impostor_mean",
        "test_impostor_mean": "test.metrics.pairwise.impostor_mean",
        "calibration_separability": "calibration.metrics.pairwise.separability",
        "test_separability": "test.metrics.pairwise.separability",
        "eer_threshold": "calibration.calibrated_thresholds.eer",
        "accept_threshold": "calibration.calibrated_thresholds.accept",
        "reject_threshold": "calibration.calibrated_thresholds.reject",
    }
    per_field = {name: [get(report, path) for report in reports] for name, path in fields.items()}
    summary = {
        "n_runs": len(reports),
        "reports": args.reports,
        "metrics": {name: mean_std(values) for name, values in per_field.items()},
        "deltas": {
            "genuine_mean_test_minus_calibration": mean_std(
                test - cal
                for test, cal in zip(per_field["test_genuine_mean"], per_field["calibration_genuine_mean"])
            ),
            "impostor_mean_test_minus_calibration": mean_std(
                test - cal
                for test, cal in zip(per_field["test_impostor_mean"], per_field["calibration_impostor_mean"])
            ),
            "separability_test_minus_calibration": mean_std(
                test - cal
                for test, cal in zip(per_field["test_separability"], per_field["calibration_separability"])
            ),
            "auc_test_minus_calibration": mean_std(
                test - cal for test, cal in zip(per_field["test_auc"], per_field["calibration_auc"])
            ),
            "eer_test_minus_calibration": mean_std(
                test - cal for test, cal in zip(per_field["test_eer"], per_field["calibration_eer"])
            ),
        },
    }
    print("=" * 90)
    print("Zero-Watermark Score Shift Analysis")
    print("=" * 90)
    for name, stats in summary["metrics"].items():
        print(f"{name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    print("-" * 90)
    for name, stats in summary["deltas"].items():
        print(f"{name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    print("=" * 90)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
