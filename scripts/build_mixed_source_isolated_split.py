"""Build source-isolated package splits for mixed-branch evaluation.

The split unit is the raw source/package folder under data/raw. For the current
cache layout, this is also the package_id used by the mixed datasets.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonOptionalPointPackageDataset


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    return obj


def split_values(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def select_heldout(
    package_ids: Sequence[str],
    *,
    heldout_sources: Sequence[str],
    heldout_regex: str | None,
    heldout_fraction: float,
    seed: int,
) -> List[str]:
    ids = sorted({str(value).lower() for value in package_ids})
    selected = set()
    exact = {value.lower() for value in heldout_sources}
    for package_id in ids:
        if package_id in exact:
            selected.add(package_id)
    if heldout_regex:
        pattern = re.compile(heldout_regex, flags=re.IGNORECASE)
        for package_id in ids:
            if pattern.search(package_id):
                selected.add(package_id)
    if not selected:
        rng = random.Random(int(seed))
        shuffled = list(ids)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * float(heldout_fraction))))
        selected.update(shuffled[:count])
    return sorted(selected)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build source-isolated split for mixed branch.")
    parser.add_argument("--line_cache_root", type=str, required=True)
    parser.add_argument("--polygon_cache_root", type=str, required=True)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--heldout_sources", type=str, default=None, help="Comma-separated package/source ids.")
    parser.add_argument("--heldout_regex", type=str, default=None)
    parser.add_argument("--heldout_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--line_subtypes", type=str, default="roads,railways,waterways")
    parser.add_argument("--polygon_subtypes", type=str, default="building,landuse,natural")
    parser.add_argument("--point_subtypes", type=str, default="pois,traffic,transport,pofw")
    parser.add_argument("--output", type=str, required=True)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    dataset = MixedLinePolygonOptionalPointPackageDataset(
        args.line_cache_root,
        args.polygon_cache_root,
        args.point_cache_root,
        mode=args.dataset_mode,
        line_subtypes=tuple(split_values(args.line_subtypes)),
        polygon_subtypes=tuple(split_values(args.polygon_subtypes)),
        point_subtypes=tuple(split_values(args.point_subtypes)),
        require_complete_lp=False,
    )
    records: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        records.append(
            {
                "package_id": str(sample["package_id"]).lower(),
                "line_subtypes": list(sample["line_subtypes"]),
                "polygon_subtypes": list(sample["polygon_subtypes"]),
                "point_subtypes": list(sample["point_subtypes"]),
                "has_point": bool(sample["point_subtypes"]),
            }
        )
    package_ids = [record["package_id"] for record in records]
    heldout = select_heldout(
        package_ids,
        heldout_sources=split_values(args.heldout_sources),
        heldout_regex=args.heldout_regex,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
    )
    heldout_set = set(heldout)
    train = sorted(package_id for package_id in package_ids if package_id not in heldout_set)
    report = {
        "split_unit": "raw_source_package",
        "dataset_mode": args.dataset_mode,
        "seed": int(args.seed),
        "line_cache_root": args.line_cache_root,
        "polygon_cache_root": args.polygon_cache_root,
        "point_cache_root": args.point_cache_root,
        "heldout_sources_requested": split_values(args.heldout_sources),
        "heldout_regex": args.heldout_regex,
        "heldout_fraction": float(args.heldout_fraction),
        "n_total": len(package_ids),
        "n_train_sources": len(train),
        "n_heldout_sources": len(heldout),
        "train_packages": train,
        "heldout_packages": heldout,
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)
    print("=" * 90)
    print("Mixed Source-Isolated Split")
    print("=" * 90)
    print(f"Total packages: {report['n_total']}")
    print(f"Train packages: {report['n_train_sources']}")
    print(f"Held-out packages: {report['n_heldout_sources']}")
    print("First held-out:", ", ".join(heldout[:10]))
    print("=" * 90)
    print(f"Saved split to: {output_path}")


if __name__ == "__main__":
    main()
