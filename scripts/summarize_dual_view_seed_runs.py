"""
Summarize multiple dual-view evaluation JSON reports into mean/std tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def load_report(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def collect_metric(reports: List[Dict[str, Any]], section: str, key: str) -> List[float]:
    return [float(report[section][key]) for report in reports]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize multi-seed dual-view evaluation reports.")
    parser.add_argument(
        "--reports",
        type=str,
        nargs="+",
        required=True,
        help="List of evaluation JSON paths.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    reports = [load_report(path) for path in args.reports]
    summary = {
        "n_runs": len(reports),
        "report_paths": args.reports,
        "pairwise_metrics": {},
        "retrieval_metrics": {},
        "overlap_metrics": {},
    }

    for key in ["auc", "eer", "separability", "genuine_mean", "impostor_mean"]:
        values = collect_metric(reports, "pairwise_metrics", key)
        summary["pairwise_metrics"][key] = {"mean": mean(values), "std": std(values), "values": values}

    for key in ["recall_at_1", "recall_at_5", "mrr"]:
        values = collect_metric(reports, "retrieval_metrics", key)
        summary["retrieval_metrics"][key] = {"mean": mean(values), "std": std(values), "values": values}

    for key in ["tile_overlap_mean", "chunk_overlap_mean"]:
        values = collect_metric(reports, "overlap_metrics", key)
        summary["overlap_metrics"][key] = {"mean": mean(values), "std": std(values), "values": values}

    print("=" * 90)
    print("Dual-View Multi-Seed Summary")
    print("=" * 90)
    print(f"Runs: {len(reports)}")
    for key in ["auc", "eer", "separability"]:
        metric = summary["pairwise_metrics"][key]
        print(f"Pairwise {key}: mean={metric['mean']:.6f}, std={metric['std']:.6f}")
    for key in ["recall_at_1", "recall_at_5", "mrr"]:
        metric = summary["retrieval_metrics"][key]
        print(f"Retrieval {key}: mean={metric['mean']:.6f}, std={metric['std']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
