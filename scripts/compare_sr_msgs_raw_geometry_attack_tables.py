"""Build SR-MSGS raw-geometry attack robustness tables."""

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
    "simplify_strong",
    "coord_noise_0p001",
    "coord_noise_0p003",
    "rotate_10",
    "scale_110",
    "translate_1pct",
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
    return f"{float(value['mean']):.4f}±{float(value['std']):.4f}"


def mean_std(values: list[float]) -> str:
    if not values:
        return "NA"
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return f"{float(np.mean(arr)):.4f}±{float(np.std(arr)):.4f}"


def row_from_seed_reports(reports: Path, attack: str, min_test_packages: int) -> tuple[Optional[list[str]], Optional[str]]:
    seed_reports = [
        load_json(reports / f"sr_msgs_rawgeom_attack_{attack}_strictindcal_seed{seed}_e3_k5_k5_far0p05_beta0p25.json")
        for seed in (7, 17, 27)
    ]
    if any(report is None for report in seed_reports):
        return None, "missing one or more seed reports"

    n_packages = [int(report["test"]["metrics"].get("n_packages", 0)) for report in seed_reports]
    if min(n_packages) < min_test_packages:
        return None, f"invalid sample size: n={','.join(str(value) for value in n_packages)}"

    def values(path: list[str]) -> list[float]:
        out: list[float] = []
        for report in seed_reports:
            current: Any = report
            for key in path:
                current = current[key]
            out.append(float(current))
        return out

    return [
        attack,
        ",".join(str(value) for value in n_packages),
        mean_std(values(["test", "metrics", "pairwise", "auc"])),
        mean_std(values(["test", "metrics", "pairwise", "eer"])),
        mean_std(values(["test", "far_sweep", "0.0100", "tar"])),
        mean_std(values(["test", "far_sweep", "0.0500", "tar"])),
        mean_std(values(["test", "far_sweep", "0.1000", "tar"])),
        mean_std(values(["test", "far_sweep", "0.0500", "far"])),
        mean_std(values(["test", "metrics", "attribution", "top1_attribution_accuracy"])),
        mean_std(values(["test", "metrics", "attribution", "top5_hit_rate"])),
        mean_std(values(["test", "metrics", "attribution", "mrr"])),
    ], None


def make_row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def build_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [make_row(headers), make_row(["---"] * len(headers))]
    lines.extend(make_row(row) for row in rows)
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SR-MSGS raw geometry attack robustness summaries.")
    parser.add_argument("--reports_dir", type=str, default="paper_results")
    parser.add_argument("--min_test_packages", type=int, default=30)
    parser.add_argument("--include_invalid", action="store_true")
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    reports = Path(args.reports_dir)
    rows: list[list[str]] = []
    json_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []

    for attack in ATTACKS:
        far1 = load_json(reports / f"sr_msgs_rawgeom_attack_{attack}_strictindcal_e3_k5_k5_far0p01_beta0p25_3seed_summary.json")
        far5 = load_json(reports / f"sr_msgs_rawgeom_attack_{attack}_strictindcal_e3_k5_k5_far0p05_beta0p25_3seed_summary.json")
        far10 = load_json(reports / f"sr_msgs_rawgeom_attack_{attack}_strictindcal_e3_k5_k5_far0p10_beta0p25_3seed_summary.json")
        row, invalid_reason = row_from_seed_reports(reports, attack, args.min_test_packages)
        if row is None:
            invalid_rows.append({"attack": attack, "reason": invalid_reason or "no valid seed report"})
            if not args.include_invalid:
                continue
            row = [
                attack,
                "invalid",
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
                "n_per_seed": row[1],
                "auc": row[2],
                "eer": row[3],
                "tar_at_far1": row[4],
                "tar_at_far5": row[5],
                "tar_at_far10": row[6],
                "actual_far5": row[7],
                "top1": row[8],
                "top5": row[9],
                "mrr": row[10],
            }
        )

    headers = [
        "Attack",
        "N per seed",
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
    markdown = "# SR-MSGS Raw-Geometry Attack Robustness (Strict Independent Calibration, K=5)\n\n"
    markdown += build_table(headers, rows) + "\n"
    if invalid_rows:
        markdown += "\nExcluded rows (not used in the valid table):\n\n"
        markdown += build_table(["Attack", "Reason"], [[row["attack"], row["reason"]] for row in invalid_rows]) + "\n"
    print(markdown)

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps({"headers": headers, "rows": json_rows, "excluded": invalid_rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
