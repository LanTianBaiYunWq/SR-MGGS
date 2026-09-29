"""Summarize dual-condition accept reports across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def load(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dual-condition accept runs.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    reports = [load(path) for path in args.reports]
    rules = ("final_only", "final_and_subtype", "final_and_lp")
    summary = {
        "n_runs": len(reports),
        "reports": args.reports,
        "decision_rules": {
            rule: {
                metric: mean_std(report["decision_rules"][rule][metric] for report in reports)
                for metric in ("tar", "far", "frr")
            }
            for rule in rules
        },
    }
    print("=" * 90)
    print("Zero-Watermark Dual-Condition Accept Summary")
    print("=" * 90)
    for rule, metrics in summary["decision_rules"].items():
        print(
            f"{rule}: TAR={metrics['tar']['mean']:.6f}±{metrics['tar']['std']:.6f}, "
            f"FAR={metrics['far']['mean']:.6f}±{metrics['far']['std']:.6f}, "
            f"FRR={metrics['frr']['mean']:.6f}±{metrics['frr']['std']:.6f}"
        )
    print("=" * 90)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {output}")


if __name__ == "__main__":
    main()
