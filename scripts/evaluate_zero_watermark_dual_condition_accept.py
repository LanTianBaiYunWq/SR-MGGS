"""Evaluate dual-condition accept rules for zero-watermark authentication.

Decision rules:
  final score >= tau_final AND guard score >= tau_guard
where guard is LP or subtype score calibrated on the same calibration split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonOptionalPointPackageDataset
from scripts.evaluate_mixed_generic_score_fusion_ablation import load_package_subset
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_hier_stage1_line_polygon_point_generic import MixedGenericLinePolygonPointEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import compute_separability_from_scores, cosine_similarity
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
    parser = argparse.ArgumentParser(description="Evaluate dual-condition accept rules.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--lp_checkpoint", type=str, required=True)
    parser.add_argument("--point_checkpoint", type=str, required=True)
    parser.add_argument("--generic_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--package_split", type=str, required=True)
    parser.add_argument("--calibration_split_name", type=str, default="train")
    parser.add_argument("--test_split_name", type=str, default="heldout")
    parser.add_argument("--scenario", type=str, default="natural_missing", choices=["natural_missing", "minor_missing", "partial_family_missing", "point_missing"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--point_alpha", type=float, default=0.5)
    parser.add_argument("--generic_beta", type=float, default=0.25)
    parser.add_argument("--enrollment_templates", type=int, default=5)
    parser.add_argument("--template_aggregation", type=str, default="max", choices=["max", "mean", "top2_mean", "median"])
    parser.add_argument("--target_far", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser


def aggregate(scores: Sequence[float], method: str) -> float:
    arr = np.asarray(list(scores), dtype=np.float64)
    if method == "max":
        return float(np.max(arr))
    if method == "mean":
        return float(np.mean(arr))
    if method == "top2_mean":
        return float(np.mean(np.sort(arr)[-min(2, arr.size) :]))
    if method == "median":
        return float(np.median(arr))
    raise ValueError(f"Unsupported aggregation: {method}")


def build_dataset(config: Dict[str, Any], args: argparse.Namespace, split_name: str) -> MixedLinePolygonOptionalPointPackageDataset:
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
    subset = load_package_subset(args.package_split, split_name)
    dataset.package_records = [
        record for record in dataset.package_records if str(record["package_id"]).lower() in subset
    ]
    return dataset


def encode_views(
    sample: Dict[str, Any],
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    idx_seed: int,
    lp_eval: MixedLinePolygonEvaluator,
    point_eval: PointHybridDualViewEvaluator,
    generic_eval: MixedGenericLinePolygonPointEvaluator,
) -> tuple[Dict[str, np.ndarray | None], List[Dict[str, np.ndarray | None]]]:
    query_views = build_tolerant_views(
        sample,
        config=config,
        scenario=args.scenario,
        rng_a=np.random.default_rng(idx_seed + 1),
        rng_b=np.random.default_rng(idx_seed + 2),
        device=lp_eval.device,
    )

    def encode(side_views: Dict[str, Any], suffix: str) -> Dict[str, np.ndarray | None]:
        lp, _ = lp_eval.encode_package(
            side_views[f"line_view_{suffix}"], side_views[f"polygon_view_{suffix}"], args.prehash_space
        )
        generic, _ = generic_eval.encode_package(
            side_views[f"line_view_{suffix}"],
            side_views[f"polygon_view_{suffix}"],
            side_views[f"point_view_{suffix}"],
            args.prehash_space,
        )
        point = mean_point_embedding(point_eval, side_views[f"point_view_{suffix}"], prehash_space=args.prehash_space)
        return {"lp": lp, "generic": generic, "point": point}

    query = encode(query_views, "a")
    templates: List[Dict[str, np.ndarray | None]] = []
    for template_idx in range(int(args.enrollment_templates)):
        template_views = build_tolerant_views(
            sample,
            config=config,
            scenario=args.scenario,
            rng_a=np.random.default_rng(idx_seed + 101 + template_idx * 17),
            rng_b=np.random.default_rng(idx_seed + 102 + template_idx * 17),
            device=lp_eval.device,
        )
        templates.append(encode(template_views, "b"))
    return query, templates


def component_scores(
    query: Dict[str, np.ndarray | None],
    template: Dict[str, np.ndarray | None],
    *,
    point_alpha: float,
    generic_beta: float,
) -> Dict[str, float]:
    lp = cosine_similarity(query["lp"], template["lp"])
    point = 0.0
    if query["point"] is not None and template["point"] is not None:
        point = cosine_similarity(query["point"], template["point"])
    subtype = lp + point_alpha * point
    generic = cosine_similarity(query["generic"], template["generic"])
    final = subtype + generic_beta * generic
    return {"lp": lp, "point": point, "subtype": subtype, "generic": generic, "final": final}


def build_component_matrices(
    dataset: MixedLinePolygonOptionalPointPackageDataset,
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    lp_eval: MixedLinePolygonEvaluator,
    point_eval: PointHybridDualViewEvaluator,
    generic_eval: MixedGenericLinePolygonPointEvaluator,
) -> tuple[Dict[str, np.ndarray], List[str]]:
    package_ids: List[str] = []
    queries: List[Dict[str, np.ndarray | None]] = []
    template_sets: List[List[Dict[str, np.ndarray | None]]] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        try:
            query, templates = encode_views(
                sample,
                config=config,
                args=args,
                idx_seed=int(args.seed) * 1000003 + idx * 1009,
                lp_eval=lp_eval,
                point_eval=point_eval,
                generic_eval=generic_eval,
            )
        except ValueError:
            continue
        package_ids.append(str(sample["package_id"]))
        queries.append(query)
        template_sets.append(templates)

    matrices = {
        name: np.zeros((len(package_ids), len(package_ids)), dtype=np.float64)
        for name in ("lp", "point", "subtype", "generic", "final")
    }
    for i, query in enumerate(queries):
        for j, templates in enumerate(template_sets):
            per_template = [
                component_scores(
                    query,
                    template,
                    point_alpha=float(args.point_alpha),
                    generic_beta=float(args.generic_beta),
                )
                for template in templates
            ]
            for name in matrices:
                matrices[name][i, j] = aggregate([scores[name] for scores in per_template], args.template_aggregation)
    return matrices, package_ids


def genuine_impostor(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.asarray([scores[i, i] for i in range(scores.shape[0])], dtype=np.float64)
    impostor = np.asarray(
        [scores[i, j] for i in range(scores.shape[0]) for j in range(scores.shape[1]) if i != j],
        dtype=np.float64,
    )
    return genuine, impostor


def threshold_at_far(impostor: np.ndarray, target_far: float) -> float:
    return float(np.quantile(impostor, 1.0 - target_far, method="higher"))


def accept_rates(final_scores: np.ndarray, guard_scores: np.ndarray | None, final_threshold: float, guard_threshold: float | None) -> Dict[str, float]:
    final_g, final_i = genuine_impostor(final_scores)
    if guard_scores is None:
        genuine_accept = final_g >= final_threshold
        impostor_accept = final_i >= final_threshold
    else:
        guard_g, guard_i = genuine_impostor(guard_scores)
        genuine_accept = (final_g >= final_threshold) & (guard_g >= float(guard_threshold))
        impostor_accept = (final_i >= final_threshold) & (guard_i >= float(guard_threshold))
    return {
        "tar": float(np.mean(genuine_accept)),
        "frr": 1.0 - float(np.mean(genuine_accept)),
        "far": float(np.mean(impostor_accept)),
    }


def pairwise_metrics(scores: np.ndarray) -> Dict[str, float]:
    g, i = genuine_impostor(scores)
    eer, threshold = compute_eer(g, i)
    return {
        "auc": compute_auc(g, i),
        "eer": eer,
        "eer_threshold": threshold,
        "separability": compute_separability_from_scores(g, i),
    }


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

    lp_eval = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_eval = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    generic_eval = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)

    cal_matrices, cal_ids = build_component_matrices(
        build_dataset(config, args, args.calibration_split_name),
        config=config,
        args=args,
        lp_eval=lp_eval,
        point_eval=point_eval,
        generic_eval=generic_eval,
    )
    test_matrices, test_ids = build_component_matrices(
        build_dataset(config, args, args.test_split_name),
        config=config,
        args=args,
        lp_eval=lp_eval,
        point_eval=point_eval,
        generic_eval=generic_eval,
    )

    thresholds: Dict[str, float] = {}
    for name, matrix in cal_matrices.items():
        _, impostor = genuine_impostor(matrix)
        thresholds[name] = threshold_at_far(impostor, float(args.target_far))

    results = {
        "final_only": accept_rates(test_matrices["final"], None, thresholds["final"], None),
        "final_and_subtype": accept_rates(
            test_matrices["final"], test_matrices["subtype"], thresholds["final"], thresholds["subtype"]
        ),
        "final_and_lp": accept_rates(
            test_matrices["final"], test_matrices["lp"], thresholds["final"], thresholds["lp"]
        ),
    }
    report = {
        "arguments": vars(args),
        "n_calibration": len(cal_ids),
        "n_test": len(test_ids),
        "thresholds": thresholds,
        "pairwise": {name: pairwise_metrics(matrix) for name, matrix in test_matrices.items()},
        "decision_rules": results,
    }
    print("=" * 90)
    print("Zero-Watermark Dual-Condition Accept")
    print("=" * 90)
    for name, result in results.items():
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
