"""Closed-loop zero-watermark authentication with registered templates.

This script reuses the validated mixed score-fusion system:
LP subtype-aware masked score + optional point score + beta * generic score.
It calibrates thresholds on a calibration split and reports authentication
and attribution metrics on a held-out split.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from scripts.evaluate_mixed_lpp_score_fusion import (
    compute_separability_from_scores,
    cosine_similarity,
)
from scripts.evaluate_mixed_missing_subtype_tolerant_score_fusion import (
    build_tolerant_views,
    mean_point_embedding,
    to_jsonable,
)
from scripts.evaluate_point_hier_stage1_dual_view_hybrid import PointHybridDualViewEvaluator
from train_roads_hier_stage1_minimal import load_config, resolve_path
from utils.metrics import compute_auc, compute_eer
from utils.seed import set_seed


@dataclass
class SignatureBundle:
    lp: np.ndarray
    generic: np.ndarray
    point: np.ndarray | None
    point_available: bool


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate zero-watermark authentication closed loop.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--lp_checkpoint", type=str, required=True)
    parser.add_argument("--point_checkpoint", type=str, required=True)
    parser.add_argument("--generic_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--package_split", type=str, default=None)
    parser.add_argument("--calibration_split_name", type=str, default="train")
    parser.add_argument("--test_split_name", type=str, default="heldout")
    parser.add_argument("--scenario", type=str, default="natural_missing", choices=["natural_missing", "minor_missing", "partial_family_missing", "point_missing"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--point_alpha", type=float, default=0.5)
    parser.add_argument("--generic_beta", type=float, default=0.5)
    parser.add_argument("--enrollment_templates", type=int, default=3)
    parser.add_argument(
        "--template_aggregation",
        type=str,
        default="max",
        choices=["max", "mean", "top2_mean", "median", "consensus"],
    )
    parser.add_argument("--target_far", type=float, default=0.01)
    parser.add_argument("--target_frr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser


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
    package_subset = load_package_subset(args.package_split, split_name)
    if package_subset is not None:
        dataset.package_records = [
            record
            for record in dataset.package_records
            if str(record["package_id"]).lower() in package_subset
        ]
    return dataset


def encode_bundle(
    *,
    sample: Dict[str, Any],
    config: Dict[str, Any],
    scenario: str,
    seed_a: int,
    seed_b: int,
    role: str,
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
    prehash_space: str,
) -> SignatureBundle:
    views = build_tolerant_views(
        sample,
        config=config,
        scenario=scenario,
        rng_a=np.random.default_rng(seed_a),
        rng_b=np.random.default_rng(seed_b),
        device=lp_evaluator.device,
    )
    suffix = "a" if role == "query" else "b"
    lp, _ = lp_evaluator.encode_package(
        views[f"line_view_{suffix}"],
        views[f"polygon_view_{suffix}"],
        prehash_space,
    )
    generic, _ = generic_evaluator.encode_package(
        views[f"line_view_{suffix}"],
        views[f"polygon_view_{suffix}"],
        views[f"point_view_{suffix}"],
        prehash_space,
    )
    point = mean_point_embedding(
        point_evaluator,
        views[f"point_view_{suffix}"],
        prehash_space=prehash_space,
    )
    return SignatureBundle(
        lp=lp,
        generic=generic,
        point=point,
        point_available=point is not None,
    )


def pair_score(
    query: SignatureBundle,
    template: SignatureBundle,
    *,
    point_alpha: float,
    generic_beta: float,
) -> float:
    score = cosine_similarity(query.lp, template.lp)
    if query.point is not None and template.point is not None:
        score += point_alpha * cosine_similarity(query.point, template.point)
    score += generic_beta * cosine_similarity(query.generic, template.generic)
    return float(score)


def aggregate_template_scores(scores: Sequence[float], method: str, threshold: float | None = None) -> float:
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if method == "max":
        return float(np.max(arr))
    if method == "mean":
        return float(np.mean(arr))
    if method == "top2_mean":
        top_k = min(2, arr.size)
        return float(np.mean(np.sort(arr)[-top_k:]))
    if method == "median":
        return float(np.median(arr))
    if method == "consensus":
        if threshold is None:
            raise ValueError("Consensus aggregation requires a template-level threshold.")
        votes = int(np.sum(arr >= threshold))
        return float(np.max(arr) if votes >= 2 else np.min(arr))
    raise ValueError(f"Unsupported template aggregation: {method}")


def build_registered_scores(
    dataset: MixedLinePolygonOptionalPointPackageDataset,
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
    template_aggregation: str,
    template_threshold: float | None = None,
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    package_ids: List[str] = []
    query_bundles: List[SignatureBundle] = []
    template_sets: List[List[SignatureBundle]] = []
    skipped: List[str] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        package_id = str(sample["package_id"])
        base_seed = int(args.seed) * 1000003 + idx * 1009
        try:
            query = encode_bundle(
                sample=sample,
                config=config,
                scenario=args.scenario,
                seed_a=base_seed + 1,
                seed_b=base_seed + 2,
                role="query",
                lp_evaluator=lp_evaluator,
                point_evaluator=point_evaluator,
                generic_evaluator=generic_evaluator,
                prehash_space=args.prehash_space,
            )
            templates: List[SignatureBundle] = []
            for template_idx in range(int(args.enrollment_templates)):
                template = encode_bundle(
                    sample=sample,
                    config=config,
                    scenario=args.scenario,
                    seed_a=base_seed + 101 + template_idx * 17,
                    seed_b=base_seed + 102 + template_idx * 17,
                    role="template",
                    lp_evaluator=lp_evaluator,
                    point_evaluator=point_evaluator,
                    generic_evaluator=generic_evaluator,
                    prehash_space=args.prehash_space,
                )
                templates.append(template)
        except ValueError:
            skipped.append(package_id)
            continue
        package_ids.append(package_id)
        query_bundles.append(query)
        template_sets.append(templates)

    scores = np.zeros((len(package_ids), len(package_ids)), dtype=np.float64)
    for query_idx, query in enumerate(query_bundles):
        for template_owner_idx, templates in enumerate(template_sets):
            template_scores = [
                pair_score(
                    query,
                    template,
                    point_alpha=float(args.point_alpha),
                    generic_beta=float(args.generic_beta),
                )
                for template in templates
            ]
            scores[query_idx, template_owner_idx] = aggregate_template_scores(
                template_scores,
                template_aggregation,
                threshold=template_threshold,
            )

    diagnostics = {
        "skipped_packages": skipped,
        "query_point_available_rate": float(np.mean([bundle.point_available for bundle in query_bundles]))
        if query_bundles
        else 0.0,
        "template_point_available_rate": float(
            np.mean([template.point_available for templates in template_sets for template in templates])
        )
        if template_sets
        else 0.0,
    }
    return scores, package_ids, diagnostics


def score_sets(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.asarray([scores[i, i] for i in range(scores.shape[0])], dtype=np.float64)
    impostor = np.asarray(
        [scores[i, j] for i in range(scores.shape[0]) for j in range(scores.shape[1]) if i != j],
        dtype=np.float64,
    )
    return genuine, impostor


def threshold_at_far(impostor_scores: np.ndarray, target_far: float) -> float:
    if impostor_scores.size == 0:
        return 0.0
    return float(np.quantile(impostor_scores, 1.0 - target_far, method="higher"))


def threshold_at_frr(genuine_scores: np.ndarray, target_frr: float) -> float:
    if genuine_scores.size == 0:
        return 0.0
    return float(np.quantile(genuine_scores, target_frr, method="lower"))


def verification_rates(genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    far = float(np.mean(impostor_scores >= threshold)) if impostor_scores.size else 0.0
    frr = float(np.mean(genuine_scores < threshold)) if genuine_scores.size else 0.0
    return {
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "tar": 1.0 - frr,
    }


def dual_threshold_rates(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    *,
    accept_threshold: float,
    reject_threshold: float,
) -> Dict[str, Dict[str, float]]:
    def rates(values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {"accept": 0.0, "reject": 0.0, "uncertain": 0.0}
        accept = float(np.mean(values >= accept_threshold))
        reject = float(np.mean(values <= reject_threshold))
        uncertain = float(np.mean((values > reject_threshold) & (values < accept_threshold)))
        return {"accept": accept, "reject": reject, "uncertain": uncertain}

    return {
        "accept_threshold": float(accept_threshold),
        "reject_threshold": float(reject_threshold),
        "genuine": rates(genuine_scores),
        "impostor": rates(impostor_scores),
    }


def attribution_metrics(scores: np.ndarray, package_ids: Sequence[str], threshold: float) -> Dict[str, float]:
    top1_hits: List[float] = []
    top5_hits: List[float] = []
    mrrs: List[float] = []
    top1_over_threshold_hits: List[float] = []
    top1_over_threshold_rate: List[float] = []
    for row_idx in range(scores.shape[0]):
        ranked = np.argsort(-scores[row_idx])
        ranked_ids = [package_ids[int(col_idx)] for col_idx in ranked]
        rank = ranked_ids.index(package_ids[row_idx]) + 1
        top1_score = float(scores[row_idx, ranked[0]])
        top1_correct = float(rank == 1)
        over_threshold = float(top1_score >= threshold)
        top1_hits.append(top1_correct)
        top5_hits.append(float(rank <= 5))
        mrrs.append(1.0 / float(rank))
        top1_over_threshold_rate.append(over_threshold)
        top1_over_threshold_hits.append(float(top1_correct and over_threshold))
    return {
        "top1_attribution_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "top5_hit_rate": float(np.mean(top5_hits)) if top5_hits else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "top1_over_accept_threshold_rate": float(np.mean(top1_over_threshold_rate)) if top1_over_threshold_rate else 0.0,
        "top1_correct_over_accept_threshold_rate": float(np.mean(top1_over_threshold_hits))
        if top1_over_threshold_hits
        else 0.0,
    }


def summarize_split(scores: np.ndarray, package_ids: Sequence[str], thresholds: Dict[str, float]) -> Dict[str, Any]:
    genuine, impostor = score_sets(scores)
    eer, eer_threshold = compute_eer(genuine, impostor) if impostor.size else (0.0, 0.0)
    return {
        "n_packages": len(package_ids),
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "pairwise": {
            "auc": compute_auc(genuine, impostor) if impostor.size else 0.0,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "separability": compute_separability_from_scores(genuine, impostor) if impostor.size else 0.0,
            "genuine_mean": float(np.mean(genuine)) if genuine.size else 0.0,
            "impostor_mean": float(np.mean(impostor)) if impostor.size else 0.0,
        },
        "single_threshold_eer_calibrated": verification_rates(genuine, impostor, thresholds["eer"]),
        "single_threshold_far_calibrated": verification_rates(genuine, impostor, thresholds["accept"]),
        "dual_threshold": dual_threshold_rates(
            genuine,
            impostor,
            accept_threshold=thresholds["accept"],
            reject_threshold=thresholds["reject"],
        ),
        "attribution": attribution_metrics(scores, package_ids, thresholds["accept"]),
    }


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
    config["device"]["seed"] = int(args.seed)
    set_seed(config["device"]["seed"])

    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    generic_evaluator = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)

    calibration_dataset = build_dataset(config, args, args.calibration_split_name)
    test_dataset = build_dataset(config, args, args.test_split_name)

    preliminary_aggregation = "max" if args.template_aggregation == "consensus" else args.template_aggregation
    calibration_scores, calibration_ids, calibration_diag = build_registered_scores(
        calibration_dataset,
        config=config,
        args=args,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
        template_aggregation=preliminary_aggregation,
    )
    calibration_genuine, calibration_impostor = score_sets(calibration_scores)
    calibration_eer, calibration_eer_threshold = compute_eer(calibration_genuine, calibration_impostor)
    thresholds = {
        "eer": float(calibration_eer_threshold),
        "accept": threshold_at_far(calibration_impostor, float(args.target_far)),
        "reject": threshold_at_frr(calibration_genuine, float(args.target_frr)),
    }

    if args.template_aggregation == "consensus":
        calibration_scores, calibration_ids, calibration_diag = build_registered_scores(
            calibration_dataset,
            config=config,
            args=args,
            lp_evaluator=lp_evaluator,
            point_evaluator=point_evaluator,
            generic_evaluator=generic_evaluator,
            template_aggregation=args.template_aggregation,
            template_threshold=thresholds["accept"],
        )
        calibration_genuine, calibration_impostor = score_sets(calibration_scores)
        calibration_eer, calibration_eer_threshold = compute_eer(calibration_genuine, calibration_impostor)
        thresholds = {
            "eer": float(calibration_eer_threshold),
            "accept": threshold_at_far(calibration_impostor, float(args.target_far)),
            "reject": threshold_at_frr(calibration_genuine, float(args.target_frr)),
        }

    test_scores, test_ids, test_diag = build_registered_scores(
        test_dataset,
        config=config,
        args=args,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
        template_aggregation=args.template_aggregation,
        template_threshold=thresholds["accept"] if args.template_aggregation == "consensus" else None,
    )

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
            "calibration_split_name": args.calibration_split_name,
            "test_split_name": args.test_split_name,
            "scenario": args.scenario,
            "seed": int(args.seed),
            "prehash_space": args.prehash_space,
            "point_alpha": float(args.point_alpha),
            "generic_beta": float(args.generic_beta),
            "enrollment_templates": int(args.enrollment_templates),
            "template_aggregation": args.template_aggregation,
            "target_far": float(args.target_far),
            "target_frr": float(args.target_frr),
        },
        "calibration": {
            "diagnostics": calibration_diag,
            "metrics": summarize_split(calibration_scores, calibration_ids, thresholds),
            "calibrated_thresholds": {
                **thresholds,
                "calibration_eer": float(calibration_eer),
            },
        },
        "test": {
            "diagnostics": test_diag,
            "metrics": summarize_split(test_scores, test_ids, thresholds),
        },
    }

    test_metrics = report["test"]["metrics"]
    print("=" * 90)
    print("Zero-Watermark Authentication Closed Loop")
    print("=" * 90)
    print(
        f"Calibration packages: {report['calibration']['metrics']['n_packages']}, "
        f"Test packages: {test_metrics['n_packages']}, K={args.enrollment_templates}"
    )
    print(
        f"Thresholds: eer={thresholds['eer']:.6f}, "
        f"accept@FAR{args.target_far:g}={thresholds['accept']:.6f}, "
        f"reject@FRR{args.target_frr:g}={thresholds['reject']:.6f}"
    )
    print(
        f"Test pairwise: AUC={test_metrics['pairwise']['auc']:.6f}, "
        f"EER={test_metrics['pairwise']['eer']:.6f}, "
        f"Sep={test_metrics['pairwise']['separability']:.6f}"
    )
    print(
        f"Test @accept: TAR={test_metrics['single_threshold_far_calibrated']['tar']:.6f}, "
        f"FAR={test_metrics['single_threshold_far_calibrated']['far']:.6f}, "
        f"FRR={test_metrics['single_threshold_far_calibrated']['frr']:.6f}"
    )
    print(
        f"Attribution: Top1={test_metrics['attribution']['top1_attribution_accuracy']:.6f}, "
        f"Top5={test_metrics['attribution']['top5_hit_rate']:.6f}, "
        f"MRR={test_metrics['attribution']['mrr']:.6f}"
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
