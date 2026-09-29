"""Summarize zero-watermark verification-head seed reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_roads_hier_stage1_minimal import resolve_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize verification-head multi-seed reports.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}


def extract_metrics(report: Dict[str, Any], section: str) -> Dict[str, float]:
    metrics = report[section]["test"]
    pairwise = metrics["pairwise"]
    threshold = metrics["single_threshold_far_calibrated"]
    attribution = metrics["attribution"]
    return {
        "auc": float(pairwise["auc"]),
        "eer": float(pairwise["eer"]),
        "separability": float(pairwise["separability"]),
        "tar_at_far": float(threshold["tar"]),
        "far_at_accept": float(threshold["far"]),
        "frr_at_accept": float(threshold["frr"]),
        "top1_attribution": float(attribution["top1_attribution_accuracy"]),
        "top5_attribution": float(attribution["top5_hit_rate"]),
        "attribution_mrr": float(attribution["mrr"]),
        "top1_over_threshold": float(attribution["top1_correct_over_accept_threshold_rate"]),
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = []
    for report_path in args.reports:
        with Path(report_path).open("r", encoding="utf-8") as f:
            reports.append(json.load(f))

    summary: Dict[str, Any] = {"runs": len(reports), "level0": {}, "registered": {}}
    for section in ("level0", "registered"):
        rows = [extract_metrics(report, section) for report in reports]
        for key in rows[0]:
            summary[section][key] = mean_std([row[key] for row in rows])

    print("=" * 90)
    print("Zero-Watermark Verification Head Multi-Seed Summary")
    print("=" * 90)
    print(f"Runs: {summary['runs']}")
    for section in ("level0", "registered"):
        print(section)
        for key, value in summary[section].items():
            print(f"  {key}: mean={value['mean']:.6f}, std={value['std']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
