"""Build budget-controlled SR-MSGS comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stat(summary: Optional[Dict[str, Any]], key: str) -> str:
    if summary is None:
        return "NA"
    value = summary.get("metrics", {}).get(key)
    if value is None:
        return "NA"
    return f"{value['mean']:.4f}±{value['std']:.4f}"


def make_row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def build_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        make_row(headers),
        make_row(["---"] * len(headers)),
    ]
    lines.extend(make_row(row) for row in rows)
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare budget-controlled SR-MSGS variants.")
    parser.add_argument("--reports_dir", type=str, default="paper_results")
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = Path(args.reports_dir)

    variants = {
        "single_local_8/16": {
            "prefix": "sr_msgs_budget_single_local",
            "description": "single scale, local budget",
        },
        "single_global_32/64": {
            "prefix": "sr_msgs_budget_single_global",
            "description": "single scale, large budget",
        },
        "budget_matched_ms": {
            "prefix": "sr_msgs_budget_matched_multiscale",
            "description": "multi-scale, lower total budget",
        },
        "multi_scale_8/16+16/32+32/64": {
            "prefix": "sr_msgs_multiscale_only",
            "description": "main multi-scale setting",
        },
    }

    loaded: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
    for name, spec in variants.items():
        prefix = spec["prefix"]
        loaded[name] = {
            "level0": load_json(
                reports / f"{prefix}_line_polygon_level0_sourceisolated_e3_k5_3seed_summary.json"
            ),
            "auth_far1": load_json(
                reports / f"{prefix}_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p01_beta0p25_3seed_summary.json"
            ),
            "auth_far5": load_json(
                reports / f"{prefix}_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p05_beta0p25_3seed_summary.json"
            ),
            "auth_far10": load_json(
                reports / f"{prefix}_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p10_beta0p25_3seed_summary.json"
            ),
        }

    auth_headers = [
        "Method",
        "AUC",
        "EER",
        "TAR@FAR1",
        "TAR@FAR5",
        "TAR@FAR10",
        "Top1",
        "Top5",
        "MRR",
    ]
    auth_rows: list[list[str]] = []
    for name, summaries in loaded.items():
        far5 = summaries["auth_far5"]
        auth_rows.append(
            [
                name,
                stat(far5, "auc"),
                stat(far5, "eer"),
                stat(summaries["auth_far1"], "tar_at_far"),
                stat(far5, "tar_at_far"),
                stat(summaries["auth_far10"], "tar_at_far"),
                stat(far5, "top1_attribution"),
                stat(far5, "top5_attribution"),
                stat(far5, "attribution_mrr"),
            ]
        )

    level0_headers = [
        "Method",
        "AUC",
        "EER",
        "TAR@FAR1",
        "TAR@FAR5",
        "TAR@FAR10",
        "R@1",
        "R@5",
        "MRR",
    ]
    level0_rows: list[list[str]] = []
    for name, summaries in loaded.items():
        level0 = summaries["level0"]
        level0_rows.append(
            [
                name,
                stat(level0, "auc"),
                stat(level0, "eer"),
                stat(level0, "far_0.01_tar"),
                stat(level0, "far_0.05_tar"),
                stat(level0, "far_0.1_tar"),
                stat(level0, "recall_at_1"),
                stat(level0, "recall_at_5"),
                stat(level0, "mrr"),
            ]
        )

    markdown = (
        "# SR-MSGS Budget-Controlled Multi-Scale Ablation\n\n"
        "## Closed-Loop Authentication (K=5)\n\n"
        + build_table(auth_headers, auth_rows)
        + "\n\n## Level-0 Pairwise / Retrieval\n\n"
        + build_table(level0_headers, level0_rows)
        + "\n"
    )
    print(markdown)

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "auth_headers": auth_headers,
                    "auth_rows": auth_rows,
                    "level0_headers": level0_headers,
                    "level0_rows": level0_rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
