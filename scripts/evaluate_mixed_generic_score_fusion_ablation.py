"""Compare subtype-aware, generic-only, and subtype+generic mixed scores."""

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
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_hier_stage1_line_polygon_point_generic import MixedGenericLinePolygonPointEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import metrics_from_score_matrix, score_matrix
from scripts.evaluate_mixed_missing_subtype_tolerant_score_fusion import (
    build_tolerant_views,
    mean_point_embedding,
    to_jsonable,
)
from scripts.evaluate_point_hier_stage1_dual_view_hybrid import PointHybridDualViewEvaluator
from train_roads_hier_stage1_minimal import load_config, resolve_path
from utils.seed import set_seed


def load_package_subset(split_path: str | None, split_name: str) -> set[str] | None:
    if not split_path:
        return None
    with Path(split_path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    key_map = {
        "train": "train_packages",
        "heldout": "heldout_packages",
        "eval": "heldout_packages",
    }
    key = key_map.get(split_name, split_name)
    if key not in split:
        raise KeyError(f"Split file does not contain key: {key}")
    return {str(value).lower() for value in split[key]}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generic/subtype-aware score fusion ablation.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--lp_checkpoint", type=str, required=True)
    parser.add_argument("--point_checkpoint", type=str, required=True)
    parser.add_argument("--generic_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--max_packages", type=int, default=None)
    parser.add_argument("--package_split", type=str, default=None)
    parser.add_argument("--split_name", type=str, default="heldout")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", type=str, default="natural_missing", choices=["natural_missing", "minor_missing", "partial_family_missing", "point_missing"])
    parser.add_argument("--point_alpha", type=float, default=0.5)
    parser.add_argument("--generic_betas", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    if args.point_cache_root:
        config["data"]["point_cache_root"] = args.point_cache_root
    if args.max_packages is not None:
        config["data"]["max_packages"] = int(args.max_packages)
    if args.seed is not None:
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
    package_subset = load_package_subset(args.package_split, args.split_name)
    if package_subset is not None:
        dataset.package_records = [
            record
            for record in dataset.package_records
            if str(record["package_id"]).lower() in package_subset
        ]
    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    generic_evaluator = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)

    lp_query: List[np.ndarray] = []
    lp_gallery: List[np.ndarray] = []
    generic_query: List[np.ndarray] = []
    generic_gallery: List[np.ndarray] = []
    point_query_by_row: List[np.ndarray | None] = []
    point_gallery_by_row: List[np.ndarray | None] = []
    package_ids: List[str] = []
    point_used: List[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        try:
            views = build_tolerant_views(
                sample,
                config=config,
                scenario=args.scenario,
                rng_a=np.random.default_rng(config["device"]["seed"] + idx * 101),
                rng_b=np.random.default_rng(config["device"]["seed"] + idx * 101 + 1),
                device=lp_evaluator.device,
            )
        except ValueError:
            continue
        lp_a, _ = lp_evaluator.encode_package(views["line_view_a"], views["polygon_view_a"], args.prehash_space)
        lp_b, _ = lp_evaluator.encode_package(views["line_view_b"], views["polygon_view_b"], args.prehash_space)
        generic_a, _ = generic_evaluator.encode_package(
            views["line_view_a"], views["polygon_view_a"], views["point_view_a"], args.prehash_space
        )
        generic_b, _ = generic_evaluator.encode_package(
            views["line_view_b"], views["polygon_view_b"], views["point_view_b"], args.prehash_space
        )
        p_a = mean_point_embedding(point_evaluator, views["point_view_a"], prehash_space=args.prehash_space)
        p_b = mean_point_embedding(point_evaluator, views["point_view_b"], prehash_space=args.prehash_space)
        lp_query.append(lp_a)
        lp_gallery.append(lp_b)
        generic_query.append(generic_a)
        generic_gallery.append(generic_b)
        point_query_by_row.append(p_a)
        point_gallery_by_row.append(p_b)
        package_ids.append(str(sample["package_id"]))
        point_used.append(float(p_a is not None and p_b is not None))

    lp_scores = score_matrix(np.asarray(lp_query), np.asarray(lp_gallery))
    subtype_scores = np.array(lp_scores, copy=True)
    point_rows = [
        idx
        for idx, (query, gallery) in enumerate(zip(point_query_by_row, point_gallery_by_row))
        if query is not None and gallery is not None
    ]
    if point_rows:
        point_query = np.stack([point_query_by_row[idx] for idx in point_rows], axis=0)
        point_gallery = np.stack([point_gallery_by_row[idx] for idx in point_rows], axis=0)
        point_scores_sub = score_matrix(point_query, point_gallery)
        for row_i, package_i in enumerate(point_rows):
            for row_j, package_j in enumerate(point_rows):
                subtype_scores[package_i, package_j] = (
                    lp_scores[package_i, package_j] + float(args.point_alpha) * point_scores_sub[row_i, row_j]
                )

    generic_scores = score_matrix(np.asarray(generic_query), np.asarray(generic_gallery))
    beta_reports: Dict[str, Dict[str, Dict[str, float]]] = {}
    for beta in [float(value.strip()) for value in args.generic_betas.split(",") if value.strip()]:
        beta_reports[str(beta)] = metrics_from_score_matrix(subtype_scores + beta * generic_scores, package_ids)

    report = {
        "arguments": {
            "config": args.config,
            "lp_checkpoint": args.lp_checkpoint,
            "point_checkpoint": args.point_checkpoint,
            "generic_checkpoint": args.generic_checkpoint,
            "line_cache_root": config["data"]["line_cache_root"],
            "polygon_cache_root": config["data"]["polygon_cache_root"],
            "point_cache_root": config["data"].get("point_cache_root"),
            "dataset_mode": args.dataset_mode,
            "package_split": args.package_split,
            "split_name": args.split_name,
            "scenario": args.scenario,
            "seed": config["device"]["seed"],
            "point_alpha": float(args.point_alpha),
            "generic_betas": args.generic_betas,
            "prehash_space": args.prehash_space,
        },
        "n_packages": len(package_ids),
        "point_used_rate": float(np.mean(point_used)) if point_used else 0.0,
        "lp_only": metrics_from_score_matrix(lp_scores, package_ids),
        "subtype_aware_masked": metrics_from_score_matrix(subtype_scores, package_ids),
        "generic_only": metrics_from_score_matrix(generic_scores, package_ids),
        "subtype_plus_generic": beta_reports,
    }
    print("=" * 90)
    print("Generic vs Subtype-Aware Score Fusion Ablation")
    print("=" * 90)
    print(f"Scenario: {args.scenario}")
    print(f"Packages: {report['n_packages']}, point_used_rate={report['point_used_rate']:.4f}")
    for name in ("lp_only", "subtype_aware_masked", "generic_only"):
        metrics = report[name]
        print(
            f"{name}: AUC={metrics['pairwise_metrics']['auc']:.6f}, "
            f"EER={metrics['pairwise_metrics']['eer']:.6f}, "
            f"R@1={metrics['retrieval_metrics']['recall_at_1']:.6f}, "
            f"R@5={metrics['retrieval_metrics']['recall_at_5']:.6f}, "
            f"MRR={metrics['retrieval_metrics']['mrr']:.6f}"
        )
    for beta, metrics in report["subtype_plus_generic"].items():
        print(
            f"subtype+{beta}*generic: AUC={metrics['pairwise_metrics']['auc']:.6f}, "
            f"EER={metrics['pairwise_metrics']['eer']:.6f}, "
            f"R@1={metrics['retrieval_metrics']['recall_at_1']:.6f}, "
            f"R@5={metrics['retrieval_metrics']['recall_at_5']:.6f}, "
            f"MRR={metrics['retrieval_metrics']['mrr']:.6f}"
        )
    print("=" * 90)
    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
