"""Summarize multi-seed generic/subtype-aware ablation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


METHOD_KEYS = (
    "lp_only",
    "subtype_aware_masked",
    "generic_only",
)


def to_jsonable(obj: Any) -> Any:
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


def collect_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    pairwise = metrics["pairwise_metrics"]
    retrieval = metrics["retrieval_metrics"]
    return {
        "auc": float(pairwise["auc"]),
        "eer": float(pairwise["eer"]),
        "separability": float(pairwise["separability"]),
        "recall_at_1": float(retrieval["recall_at_1"]),
        "recall_at_5": float(retrieval["recall_at_5"]),
        "mrr": float(retrieval["mrr"]),
    }


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }


def lookup_beta_report(report: Dict[str, Any], beta: str) -> Dict[str, Any]:
    reports = report["subtype_plus_generic"]
    candidates = [beta]
    try:
        beta_float = float(beta)
        candidates.extend([str(beta_float), f"{beta_float:g}"])
    except ValueError:
        pass
    for candidate in candidates:
        if candidate in reports:
            return reports[candidate]
    raise KeyError(f"Cannot find beta={beta}; available={list(reports.keys())}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize mixed generic ablation seed reports.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--generic_betas", type=str, default="0.5,1.0")
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    reports = []
    for path_str in args.reports:
        with Path(path_str).open("r", encoding="utf-8") as f:
            reports.append(json.load(f))

    method_rows: Dict[str, List[Dict[str, float]]] = {key: [] for key in METHOD_KEYS}
    for report in reports:
        for key in METHOD_KEYS:
            method_rows[key].append(collect_metrics(report[key]))

    for beta in [value.strip() for value in args.generic_betas.split(",") if value.strip()]:
        key = f"subtype_plus_{beta}_generic"
        method_rows[key] = []
        for report in reports:
            method_rows[key].append(collect_metrics(lookup_beta_report(report, beta)))

    summary = {
        "n_runs": len(reports),
        "reports": args.reports,
        "methods": {key: summarize_rows(rows) for key, rows in method_rows.items()},
    }

    print("=" * 90)
    print("Mixed Generic Ablation Multi-Seed Summary")
    print("=" * 90)
    print(f"Runs: {summary['n_runs']}")
    for method, metrics in summary["methods"].items():
        print(method)
        for metric in ("auc", "eer", "separability", "recall_at_1", "recall_at_5", "mrr"):
            value = metrics[metric]
            print(f"  {metric}: mean={value['mean']:.6f}, std={value['std']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
