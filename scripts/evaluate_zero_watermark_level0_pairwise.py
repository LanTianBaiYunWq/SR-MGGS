"""Level-0 dual-view pairwise baseline for zero-watermark diagnosis.

No registration templates, no max-over-K, no attribution database decision.
This measures the raw final-score verification ability under the same split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonOptionalPointPackageDataset
from scripts.evaluate_mixed_generic_score_fusion_ablation import load_package_subset
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_hier_stage1_line_polygon_point_generic import MixedGenericLinePolygonPointEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import compute_separability_from_scores, score_matrix
from scripts.evaluate_mixed_missing_subtype_tolerant_score_fusion import (
    build_tolerant_views,
    mean_point_embedding,
    to_jsonable,
)
from scripts.evaluate_point_hier_stage1_dual_view_hybrid import PointHybridDualViewEvaluator
from train_roads_hier_stage1_minimal import load_config, resolve_path
from utils.metrics import compute_auc, compute_eer
from utils.seed import set_seed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Level-0 dual-view pairwise final-score baseline.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--lp_checkpoint", type=str, required=True)
    parser.add_argument("--point_checkpoint", type=str, required=True)
    parser.add_argument("--generic_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--package_split", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="heldout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default="natural_missing", choices=["natural_missing", "minor_missing", "partial_family_missing", "point_missing"])
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--point_alpha", type=float, default=0.5)
    parser.add_argument("--generic_beta", type=float, default=0.5)
    parser.add_argument("--target_fars", type=str, default="0.01,0.05,0.10")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8))


def threshold_at_far(impostor: np.ndarray, far: float) -> float:
    return float(np.quantile(impostor, 1.0 - far, method="higher")) if impostor.size else 0.0


def rates(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> Dict[str, float]:
    far = float(np.mean(impostor >= threshold)) if impostor.size else 0.0
    frr = float(np.mean(genuine < threshold)) if genuine.size else 0.0
    return {"threshold": float(threshold), "far": far, "frr": frr, "tar": 1.0 - frr}


def main() -> None:
    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    if args.point_cache_root:
        config["data"]["point_cache_root"] = args.point_cache_root
    config["device"]["seed"] = int(args.seed)
    set_seed(config["device"]["seed"])

    dataset = MixedLinePolygonOptionalPointPackageDataset(
        config["data"]["line_cache_root"],
        config["data"]["polygon_cache_root"],
        None if args.scenario == "point_missing" else config["data"].get("point_cache_root"),
        mode=args.dataset_mode,
        line_max_tiles=config["data"].get("line_max_tiles"),
        polygon_max_tiles=config["data"].get("polygon_max_tiles"),
        point_max_tiles=config["data"].get("point_max_tiles"),
        line_tile_selector=config["data"].get("line_tile_selector", "manifest_default"),
        polygon_tile_selector=config["data"].get("polygon_tile_selector", "manifest_default"),
        point_tile_selector=config["data"].get("point_tile_selector", "manifest_default"),
        line_subtypes=config["data"].get("line_subtypes", ("roads", "railways", "waterways")),
        polygon_subtypes=config["data"].get("polygon_subtypes", ("building", "landuse", "natural")),
        point_subtypes=config["data"].get("point_subtypes", ("pois", "traffic", "transport", "pofw")),
        max_packages=config["data"].get("max_packages"),
        require_complete_lp=bool(config["data"].get("require_complete_lp", False)),
    )
    subset = load_package_subset(args.package_split, args.split_name)
    dataset.package_records = [
        record for record in dataset.package_records if str(record["package_id"]).lower() in subset
    ]

    lp_eval = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_eval = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    generic_eval = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)

    lp_q: List[np.ndarray] = []
    lp_g: List[np.ndarray] = []
    generic_q: List[np.ndarray] = []
    generic_g: List[np.ndarray] = []
    point_q: List[np.ndarray | None] = []
    point_g: List[np.ndarray | None] = []
    package_ids: List[str] = []
    point_used: List[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        try:
            views = build_tolerant_views(
                sample,
                config=config,
                scenario=args.scenario,
                rng_a=np.random.default_rng(int(args.seed) * 1000003 + idx * 101),
                rng_b=np.random.default_rng(int(args.seed) * 1000003 + idx * 101 + 1),
                device=lp_eval.device,
            )
        except ValueError:
            continue
        a_lp, _ = lp_eval.encode_package(views["line_view_a"], views["polygon_view_a"], args.prehash_space)
        b_lp, _ = lp_eval.encode_package(views["line_view_b"], views["polygon_view_b"], args.prehash_space)
        a_gen, _ = generic_eval.encode_package(
            views["line_view_a"], views["polygon_view_a"], views["point_view_a"], args.prehash_space
        )
        b_gen, _ = generic_eval.encode_package(
            views["line_view_b"], views["polygon_view_b"], views["point_view_b"], args.prehash_space
        )
        a_point = mean_point_embedding(point_eval, views["point_view_a"], prehash_space=args.prehash_space)
        b_point = mean_point_embedding(point_eval, views["point_view_b"], prehash_space=args.prehash_space)
        lp_q.append(a_lp)
        lp_g.append(b_lp)
        generic_q.append(a_gen)
        generic_g.append(b_gen)
        point_q.append(a_point)
        point_g.append(b_point)
        point_used.append(float(a_point is not None and b_point is not None))
        package_ids.append(str(sample["package_id"]))

    lp_scores = score_matrix(np.asarray(lp_q), np.asarray(lp_g))
    generic_scores = score_matrix(np.asarray(generic_q), np.asarray(generic_g))
    point_scores = np.zeros_like(lp_scores)
    for i, (a_point, _) in enumerate(zip(point_q, point_g)):
        if a_point is None:
            continue
        for j, (_, b_point) in enumerate(zip(point_q, point_g)):
            if b_point is None:
                continue
            point_scores[i, j] = cosine(a_point, b_point)

    final_scores = lp_scores + float(args.point_alpha) * point_scores + float(args.generic_beta) * generic_scores
    genuine = np.asarray([final_scores[i, i] for i in range(final_scores.shape[0])], dtype=np.float64)
    impostor = np.asarray(
        [final_scores[i, j] for i in range(final_scores.shape[0]) for j in range(final_scores.shape[1]) if i != j],
        dtype=np.float64,
    )
    eer, eer_threshold = compute_eer(genuine, impostor)
    operating_points = {
        f"far_{far:g}": rates(genuine, impostor, threshold_at_far(impostor, far))
        for far in [float(value.strip()) for value in args.target_fars.split(",") if value.strip()]
    }
    operating_points["eer"] = rates(genuine, impostor, eer_threshold)

    report = {
        "arguments": vars(args),
        "n_packages": len(package_ids),
        "point_used_rate": float(np.mean(point_used)) if point_used else 0.0,
        "pairwise": {
            "auc": compute_auc(genuine, impostor),
            "eer": eer,
            "eer_threshold": eer_threshold,
            "separability": compute_separability_from_scores(genuine, impostor),
            "genuine_mean": float(np.mean(genuine)),
            "impostor_mean": float(np.mean(impostor)),
        },
        "operating_points": operating_points,
    }
    print("=" * 90)
    print("Level-0 Dual-View Pairwise Final-Score Baseline")
    print("=" * 90)
    print(f"Packages: {report['n_packages']}, point_used_rate={report['point_used_rate']:.4f}")
    print(
        f"AUC={report['pairwise']['auc']:.6f}, EER={report['pairwise']['eer']:.6f}, "
        f"Sep={report['pairwise']['separability']:.6f}"
    )
    for name, result in operating_points.items():
        print(f"{name}: TAR={result['tar']:.6f}, FAR={result['far']:.6f}, FRR={result['frr']:.6f}")
    print("=" * 90)
    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
