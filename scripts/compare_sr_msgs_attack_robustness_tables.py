"""Build SR-MSGS attack robustness tables from strict independent calibration summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


ATTACKS = [
    "clean",
    "random_delete_30",
    "random_delete_50",
    "crop_50",
    "simplify_light",
    "feature_noise_0p02",
    "feature_noise_0p05",
    "minor_missing",
    "point_missing",
]


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
    lines = [make_row(headers), make_row(["---"] * len(headers))]
    lines.extend(make_row(row) for row in rows)
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SR-MSGS attack robustness summaries.")
    parser.add_argument("--reports_dir", type=str, default="paper_results")
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = Path(args.reports_dir)
    rows: list[list[str]] = []
    json_rows: list[dict[str, str]] = []

    for attack in ATTACKS:
        far1 = load_json(reports / f"sr_msgs_attack_{attack}_strictindcal_e3_k5_k5_far0p01_beta0p25_3seed_summary.json")
        far5 = load_json(reports / f"sr_msgs_attack_{attack}_strictindcal_e3_k5_k5_far0p05_beta0p25_3seed_summary.json")
        far10 = load_json(reports / f"sr_msgs_attack_{attack}_strictindcal_e3_k5_k5_far0p10_beta0p25_3seed_summary.json")
        row = [
            attack,
            stat(far5, "auc"),
            stat(far5, "eer"),
            stat(far1, "tar_at_far"),
            stat(far5, "tar_at_far"),
            stat(far10, "tar_at_far"),
            stat(far5, "far_at_accept"),
            stat(far5, "top1_attribution"),
            stat(far5, "top5_attribution"),
            stat(far5, "attribution_mrr"),
        ]
        rows.append(row)
        json_rows.append(
            {
                "attack": row[0],
                "auc": row[1],
                "eer": row[2],
                "tar_at_far1": row[3],
                "tar_at_far5": row[4],
                "tar_at_far10": row[5],
                "actual_far5": row[6],
                "top1": row[7],
                "top5": row[8],
                "mrr": row[9],
            }
        )

    headers = [
        "Attack",
        "AUC",
        "EER",
        "TAR@FAR1",
        "TAR@FAR5",
        "TAR@FAR10",
        "Actual FAR@5",
        "Top1",
        "Top5",
        "MRR",
    ]
    markdown = "# SR-MSGS Attack Robustness (Strict Independent Calibration, K=5)\n\n"
    markdown += build_table(headers, rows) + "\n"
    print(markdown)

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps({"headers": headers, "rows": json_rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
