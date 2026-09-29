"""Create the final mixed-branch main result table from 3-seed summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_METHOD_ORDER = (
    "lp_only",
    "subtype_aware_masked",
    "generic_only",
    "subtype_plus_0.5_generic",
    "subtype_plus_1.0_generic",
)
METHOD_ALIASES = {
    "subtype_plus_1.0_generic": "subtype_plus_1_generic",
}

METHOD_LABELS = {
    "lp_only": "LP only",
    "subtype_aware_masked": "Subtype-aware masked",
    "generic_only": "Generic-only",
    "subtype_plus_0.5_generic": "Subtype-aware masked + 0.5 Generic",
    "subtype_plus_1.0_generic": "Subtype-aware masked + 1.0 Generic",
}

METHOD_ROLES = {
    "lp_only": "LP baseline",
    "subtype_aware_masked": "Missing-subtype tolerant subtype expert",
    "generic_only": "Generic fallback expert",
    "subtype_plus_0.5_generic": "Frozen main system",
    "subtype_plus_1.0_generic": "Retrieval-biased variant",
}

METRIC_ORDER = ("auc", "eer", "separability", "recall_at_1", "recall_at_5", "mrr")
METRIC_LABELS = {
    "auc": "AUC",
    "eer": "EER",
    "separability": "Sep.",
    "recall_at_1": "R@1",
    "recall_at_5": "R@5",
    "mrr": "MRR",
}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def build_rows(summary: Dict[str, Any], method_order: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    methods = summary["methods"]
    for method in method_order:
        method_key = method if method in methods else METHOD_ALIASES.get(method, method)
        metrics = methods[method_key]
        row = {
            "method_key": method_key,
            "method": METHOD_LABELS.get(method, method),
            "role": METHOD_ROLES.get(method, ""),
        }
        for metric in METRIC_ORDER:
            row[metric] = fmt(float(metrics[metric]["mean"]), float(metrics[metric]["std"]))
        rows.append(row)
    return rows


def rows_to_markdown(rows: List[Dict[str, str]], title: str) -> str:
    headers = ["Method", "Role"] + [METRIC_LABELS[metric] for metric in METRIC_ORDER]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [row["method"], row["role"]] + [row[metric] for metric in METRIC_ORDER]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Frozen main system: Subtype-aware masked + 0.5 Generic.")
    lines.append("All values are mean±std over the same 3 seeds and evaluation protocol.")
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make final mixed main result table.")
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--title", type=str, default="Final Mixed Branch Main Results")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    with Path(args.summary).open("r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = build_rows(summary, list(DEFAULT_METHOD_ORDER))
    output = {
        "title": args.title,
        "source_summary": args.summary,
        "n_runs": summary.get("n_runs"),
        "frozen_main_system": "subtype_plus_0.5_generic",
        "rows": rows,
    }
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON table to: {output_json}")
    markdown = rows_to_markdown(rows, args.title)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        print(f"Saved Markdown table to: {output_md}")
    print(markdown)


if __name__ == "__main__":
    main()
