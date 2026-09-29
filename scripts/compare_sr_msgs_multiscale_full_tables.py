"""Build compact comparison tables for SR-MSGS multiscale-only vs full variants."""

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
    parser = argparse.ArgumentParser(
        description="Compare SR-MSGS multiscale-only and full summary metrics."
    )
    parser.add_argument("--reports_dir", type=str, default="paper_results")
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = Path(args.reports_dir)

    variants = {
        "multiscale_only": {
            "level0": reports / "sr_msgs_multiscale_only_line_polygon_level0_sourceisolated_e3_k5_3seed_summary.json",
            "auth_far1": reports / "sr_msgs_multiscale_only_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p01_beta0p25_3seed_summary.json",
            "auth_far5": reports / "sr_msgs_multiscale_only_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p05_beta0p25_3seed_summary.json",
            "auth_far10": reports / "sr_msgs_multiscale_only_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p10_beta0p25_3seed_summary.json",
        },
        "full": {
            "level0": reports / "sr_msgs_full_line_polygon_level0_sourceisolated_e3_k5_3seed_summary.json",
            "auth_far1": reports / "sr_msgs_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p01_beta0p25_3seed_summary.json",
            "auth_far5": reports / "sr_msgs_full_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p05_beta0p25_3seed_summary.json",
            "auth_far10": reports / "sr_msgs_zero_watermark_auth_sourceisolated_e3_k5_k5_far0p10_beta0p25_3seed_summary.json",
        },
    }

    loaded = {
        name: {kind: load_json(path) for kind, path in paths.items()}
        for name, paths in variants.items()
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
    auth_rows = []
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
    level0_rows = []
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
        "# SR-MSGS Variant Comparison\n\n"
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
