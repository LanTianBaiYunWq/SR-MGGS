"""
Summarize multiple dual-view fold error audit JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_roads_hier_stage1_minimal import resolve_path


DEFAULT_OVERLAP_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 1.01]
DEFAULT_RANK_BANDS = [
    ("top1", 1, 1),
    ("rank2_3", 2, 3),
    ("rank4_5", 4, 5),
    ("rank_gt5", 6, None),
]


def load_json(path: str) -> Dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": math.nan, "std": math.nan}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def overlap_bucket_name(left: float, right: float) -> str:
    right_text = "1.0" if right >= 1.0 else f"{right:.1f}"
    return f"[{left:.1f}, {right_text})"


def assign_bucket(value: float, bins: List[float]) -> str:
    for left, right in zip(bins[:-1], bins[1:]):
        if left <= value < right:
            return overlap_bucket_name(left, right)
    return overlap_bucket_name(bins[-2], bins[-1])


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "model_rank_mean": math.nan,
            "baseline_rank_mean": math.nan,
            "model_top1_rate": math.nan,
            "baseline_top1_rate": math.nan,
            "tile_overlap_mean": math.nan,
            "chunk_overlap_mean": math.nan,
        }
    model_ranks = [int(row["model_rank"]) for row in rows]
    baseline_ranks = [int(row["baseline_rank"]) for row in rows]
    model_top1 = [1.0 if bool(row["model_hit_at_1"]) else 0.0 for row in rows]
    baseline_top1 = [1.0 if bool(row["baseline_hit_at_1"]) else 0.0 for row in rows]
    tile_overlap = [float(row["tile_overlap_ratio"]) for row in rows]
    chunk_overlap = [float(row["chunk_overlap_ratio"]) for row in rows]
    return {
        "count": len(rows),
        "model_rank_mean": float(np.mean(model_ranks)),
        "baseline_rank_mean": float(np.mean(baseline_ranks)),
        "model_rank_median": float(np.median(model_ranks)),
        "baseline_rank_median": float(np.median(baseline_ranks)),
        "model_top1_rate": float(np.mean(model_top1)),
        "baseline_top1_rate": float(np.mean(baseline_top1)),
        "model_top3_rate": float(np.mean([1.0 if int(row["model_rank"]) <= 3 else 0.0 for row in rows])),
        "baseline_top3_rate": float(np.mean([1.0 if int(row["baseline_rank"]) <= 3 else 0.0 for row in rows])),
        "model_top5_rate": float(np.mean([1.0 if int(row["model_rank"]) <= 5 else 0.0 for row in rows])),
        "baseline_top5_rate": float(np.mean([1.0 if int(row["baseline_rank"]) <= 5 else 0.0 for row in rows])),
        "tile_overlap_mean": float(np.mean(tile_overlap)),
        "chunk_overlap_mean": float(np.mean(chunk_overlap)),
    }


def aggregate_subtype_stats(audits: List[Dict[str, Any]]) -> Dict[str, Any]:
    subtype_rows: Dict[str, List[Dict[str, Any]]] = {}
    for audit in audits:
        for row in audit["all_queries"]:
            subtype_rows.setdefault(str(row["subtype"]), []).append(row)

    return {
        subtype: {
            "all": summarize_rows(rows),
            "model_only_top1": summarize_rows(
                [row for row in rows if bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
            "baseline_only_top1": summarize_rows(
                [row for row in rows if bool(row["baseline_hit_at_1"]) and not bool(row["model_hit_at_1"])]
            ),
            "both_correct": summarize_rows(
                [row for row in rows if bool(row["model_hit_at_1"]) and bool(row["baseline_hit_at_1"])]
            ),
            "both_wrong": summarize_rows(
                [row for row in rows if not bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
        }
        for subtype, rows in sorted(subtype_rows.items())
    }


def aggregate_overlap_stats(audits: List[Dict[str, Any]], key: str, bins: List[float]) -> Dict[str, Any]:
    rows_by_bucket: Dict[str, List[Dict[str, Any]]] = {
        overlap_bucket_name(left, right): [] for left, right in zip(bins[:-1], bins[1:])
    }
    for audit in audits:
        for row in audit["all_queries"]:
            bucket = assign_bucket(float(row[key]), bins)
            rows_by_bucket[bucket].append(row)

    return {
        bucket: {
            "all": summarize_rows(rows),
            "model_only_top1": summarize_rows(
                [row for row in rows if bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
            "baseline_only_top1": summarize_rows(
                [row for row in rows if bool(row["baseline_hit_at_1"]) and not bool(row["model_hit_at_1"])]
            ),
            "both_wrong": summarize_rows(
                [row for row in rows if not bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
        }
        for bucket, rows in rows_by_bucket.items()
    }


def summarize_rank_bands(rows: List[Dict[str, Any]], rank_key: str) -> Dict[str, Any]:
    total = len(rows)
    summary: Dict[str, Any] = {}
    for label, left, right in DEFAULT_RANK_BANDS:
        if right is None:
            band_rows = [row for row in rows if int(row[rank_key]) >= left]
        else:
            band_rows = [row for row in rows if left <= int(row[rank_key]) <= right]
        summary[label] = {
            "count": len(band_rows),
            "rate": float(len(band_rows) / total) if total else math.nan,
            "tile_overlap_mean": float(np.mean([float(row["tile_overlap_ratio"]) for row in band_rows]))
            if band_rows
            else math.nan,
            "chunk_overlap_mean": float(np.mean([float(row["chunk_overlap_ratio"]) for row in band_rows]))
            if band_rows
            else math.nan,
        }
    return summary


def summarize_low_overlap_focus(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    focus_sets = {
        "tile_lt_0_2": [row for row in rows if float(row["tile_overlap_ratio"]) < 0.2],
        "tile_lt_0_25": [row for row in rows if float(row["tile_overlap_ratio"]) < 0.25],
        "chunk_lt_0_05": [row for row in rows if float(row["chunk_overlap_ratio"]) < 0.05],
        "chunk_lt_0_08": [row for row in rows if float(row["chunk_overlap_ratio"]) < 0.08],
    }
    output: Dict[str, Any] = {}
    for name, focus_rows in focus_sets.items():
        output[name] = {
            "all": summarize_rows(focus_rows),
            "model_only_top1": summarize_rows(
                [row for row in focus_rows if bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
            "baseline_only_top1": summarize_rows(
                [row for row in focus_rows if bool(row["baseline_hit_at_1"]) and not bool(row["model_hit_at_1"])]
            ),
            "both_wrong": summarize_rows(
                [row for row in focus_rows if not bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
        }
    return output


def collect_examples(audits: List[Dict[str, Any]], which: str, limit: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for audit in audits:
        fold_index = int(audit["arguments"]["fold_index"])
        for row in audit[f"{which}_examples"]:
            item = dict(row)
            item["fold_index"] = fold_index
            examples.append(item)

    if which == "baseline_only":
        examples.sort(key=lambda row: (int(row["model_rank"]), -int(row["baseline_rank"])))
    elif which == "model_only":
        examples.sort(key=lambda row: (int(row["baseline_rank"]), -int(row["model_rank"])))
    else:
        examples.sort(key=lambda row: (int(row["model_rank"]) + int(row["baseline_rank"])), reverse=True)
    return examples[:limit]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize multiple dual-view fold error audit JSON reports.")
    parser.add_argument("--reports", type=str, nargs="+", required=True, help="Input audit JSON files.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--example_limit", type=int, default=10, help="How many examples to keep per category.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    audits = [load_json(path) for path in args.reports]
    all_rows = [row for audit in audits for row in audit["all_queries"]]

    summary = {
        "n_reports": len(audits),
        "fold_indices": [int(audit["arguments"]["fold_index"]) for audit in audits],
        "n_queries_total": len(all_rows),
        "agreement_totals": {
            "both_correct": summarize_rows(
                [row for row in all_rows if bool(row["model_hit_at_1"]) and bool(row["baseline_hit_at_1"])]
            ),
            "model_only_top1": summarize_rows(
                [row for row in all_rows if bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
            "baseline_only_top1": summarize_rows(
                [row for row in all_rows if bool(row["baseline_hit_at_1"]) and not bool(row["model_hit_at_1"])]
            ),
            "both_wrong": summarize_rows(
                [row for row in all_rows if not bool(row["model_hit_at_1"]) and not bool(row["baseline_hit_at_1"])]
            ),
        },
        "agreement_per_fold": {
            str(int(audit["arguments"]["fold_index"])): audit["agreement_summary"] for audit in audits
        },
        "subtype_summary": aggregate_subtype_stats(audits),
        "tile_overlap_buckets": aggregate_overlap_stats(audits, "tile_overlap_ratio", DEFAULT_OVERLAP_BINS),
        "chunk_overlap_buckets": aggregate_overlap_stats(audits, "chunk_overlap_ratio", DEFAULT_OVERLAP_BINS),
        "rank_summary": {
            "model_rank": mean_std([int(row["model_rank"]) for row in all_rows]),
            "baseline_rank": mean_std([int(row["baseline_rank"]) for row in all_rows]),
            "rank_delta_model_minus_baseline": mean_std(
                [int(row["model_rank"]) - int(row["baseline_rank"]) for row in all_rows]
            ),
        },
        "rank_bands": {
            "model": summarize_rank_bands(all_rows, "model_rank"),
            "baseline": summarize_rank_bands(all_rows, "baseline_rank"),
        },
        "low_overlap_focus": summarize_low_overlap_focus(all_rows),
        "baseline_only_examples": collect_examples(audits, "baseline_only", args.example_limit),
        "model_only_examples": collect_examples(audits, "model_only", args.example_limit),
        "both_wrong_examples": collect_examples(audits, "both_wrong", args.example_limit),
    }

    print("=" * 90)
    print("Dual-View Fold Error Audit Summary")
    print("=" * 90)
    print(f"Reports: {summary['n_reports']}")
    print(f"Queries total: {summary['n_queries_total']}")
    for key in ("both_correct", "model_only_top1", "baseline_only_top1", "both_wrong"):
        bucket = summary["agreement_totals"][key]
        print(
            f"{key}: count={bucket['count']}, "
            f"model_rank_mean={bucket['model_rank_mean']:.3f}, "
            f"baseline_rank_mean={bucket['baseline_rank_mean']:.3f}, "
            f"tile_overlap_mean={bucket['tile_overlap_mean']:.3f}, "
            f"chunk_overlap_mean={bucket['chunk_overlap_mean']:.3f}"
        )
    print("Subtype deltas (model_top1_rate - baseline_top1_rate):")
    for subtype, subtype_summary in summary["subtype_summary"].items():
        all_stats = subtype_summary["all"]
        delta = all_stats["model_top1_rate"] - all_stats["baseline_top1_rate"]
        print(
            f"  {subtype}: count={all_stats['count']}, "
            f"delta={delta:.3f}, "
            f"model_rank_mean={all_stats['model_rank_mean']:.3f}, "
            f"baseline_rank_mean={all_stats['baseline_rank_mean']:.3f}"
        )
    print("Rank bands:")
    for method_name, band_summary in summary["rank_bands"].items():
        printable = ", ".join(
            f"{label}={band_summary[label]['count']}({band_summary[label]['rate']:.3f})"
            for label, _, _ in DEFAULT_RANK_BANDS
        )
        print(f"  {method_name}: {printable}")
    print("Low-overlap focus:")
    for focus_name, focus_summary in summary["low_overlap_focus"].items():
        all_stats = focus_summary["all"]
        print(
            f"  {focus_name}: count={all_stats['count']}, "
            f"model_top1={all_stats['model_top1_rate']:.3f}, "
            f"baseline_top1={all_stats['baseline_top1_rate']:.3f}, "
            f"both_wrong={focus_summary['both_wrong']['count']}"
        )
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
