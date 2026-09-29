"""Summarize zero-watermark authentication reports across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def nested_get(data: Dict[str, Any], path: str) -> float:
    current: Any = data
    for key in path.split("."):
        current = current[key]
    return float(current)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize zero-watermark authentication seed runs.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = [load_report(path) for path in args.reports]
    fields = {
        "auc": "test.metrics.pairwise.auc",
        "eer": "test.metrics.pairwise.eer",
        "separability": "test.metrics.pairwise.separability",
        "tar_at_far": "test.metrics.single_threshold_far_calibrated.tar",
        "far_at_accept": "test.metrics.single_threshold_far_calibrated.far",
        "frr_at_accept": "test.metrics.single_threshold_far_calibrated.frr",
        "genuine_accept": "test.metrics.dual_threshold.genuine.accept",
        "genuine_uncertain": "test.metrics.dual_threshold.genuine.uncertain",
        "genuine_reject": "test.metrics.dual_threshold.genuine.reject",
        "impostor_accept": "test.metrics.dual_threshold.impostor.accept",
        "impostor_uncertain": "test.metrics.dual_threshold.impostor.uncertain",
        "impostor_reject": "test.metrics.dual_threshold.impostor.reject",
        "top1_attribution": "test.metrics.attribution.top1_attribution_accuracy",
        "top5_attribution": "test.metrics.attribution.top5_hit_rate",
        "attribution_mrr": "test.metrics.attribution.mrr",
        "top1_over_threshold": "test.metrics.attribution.top1_correct_over_accept_threshold_rate",
    }
    summary = {
        "n_runs": len(reports),
        "reports": args.reports,
        "arguments": reports[0].get("arguments", {}) if reports else {},
        "metrics": {
            name: mean_std(nested_get(report, field_path) for report in reports)
            for name, field_path in fields.items()
        },
    }
    print("=" * 90)
    print("Zero-Watermark Authentication Multi-Seed Summary")
    print("=" * 90)
    print(f"Runs: {summary['n_runs']}")
    for name, stats in summary["metrics"].items():
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
