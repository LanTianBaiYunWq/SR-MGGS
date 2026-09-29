"""Create a model-train/calibration/test split from an existing source-isolated split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split train_packages into model_train and calibration packages.")
    parser.add_argument("--input", type=str, required=True, help="Existing source-isolated split JSON.")
    parser.add_argument("--output", type=str, required=True, help="Output split JSON.")
    parser.add_argument("--calibration_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    with Path(args.input).open("r", encoding="utf-8") as f:
        split: Dict[str, Any] = json.load(f)

    train_packages: List[str] = [str(value) for value in split["train_packages"]]
    heldout_packages: List[str] = [str(value) for value in split["heldout_packages"]]
    rng = random.Random(int(args.seed))
    shuffled = list(train_packages)
    rng.shuffle(shuffled)

    n_calibration = max(1, round(len(shuffled) * float(args.calibration_fraction)))
    calibration_packages = sorted(shuffled[:n_calibration])
    model_train_packages = sorted(shuffled[n_calibration:])

    output: Dict[str, Any] = dict(split)
    output.update(
        {
            "split_unit": split.get("split_unit", "raw_source_package"),
            "derived_from": str(Path(args.input)),
            "calibration_seed": int(args.seed),
            "calibration_fraction": float(args.calibration_fraction),
            "n_model_train_packages": len(model_train_packages),
            "n_calibration_packages": len(calibration_packages),
            "n_heldout_packages": len(heldout_packages),
            "model_train_packages": model_train_packages,
            "calibration_packages": calibration_packages,
            "heldout_packages": sorted(heldout_packages),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("Independent Calibration Split")
    print("=" * 90)
    print(f"Input train packages: {len(train_packages)}")
    print(f"Model-train packages: {len(model_train_packages)}")
    print(f"Calibration packages: {len(calibration_packages)}")
    print(f"Held-out test packages: {len(heldout_packages)}")
    print(f"Saved split to: {output_path}")


if __name__ == "__main__":
    main()
