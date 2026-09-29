"""Summarize SR-MSGS Level-0 seed reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize SR-MSGS Level-0 seed runs.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def load_report(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = [load_report(path) for path in args.reports]
    metrics = {
        "auc": [float(report["pairwise"]["auc"]) for report in reports],
        "eer": [float(report["pairwise"]["eer"]) for report in reports],
        "separability": [float(report["pairwise"]["separability"]) for report in reports],
        "recall_at_1": [float(report["retrieval"]["recall_at_1"]) for report in reports],
        "recall_at_5": [float(report["retrieval"]["recall_at_5"]) for report in reports],
        "mrr": [float(report["retrieval"]["mrr"]) for report in reports],
    }
    far_keys = sorted(reports[0].get("operating_points", {}).keys())
    for key in far_keys:
        metrics[f"{key}_tar"] = [float(report["operating_points"][key]["tar"]) for report in reports]
        metrics[f"{key}_far"] = [float(report["operating_points"][key]["far"]) for report in reports]

    summary = {
        "runs": len(reports),
        "reports": [str(path) for path in args.reports],
        "metrics": {key: mean_std(values) for key, values in metrics.items()},
    }

    print("=" * 90)
    print("SR-MSGS Level-0 Multi-Seed Summary")
    print("=" * 90)
    for key, value in summary["metrics"].items():
        print(f"{key}: mean={value['mean']:.6f}, std={value['std']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
