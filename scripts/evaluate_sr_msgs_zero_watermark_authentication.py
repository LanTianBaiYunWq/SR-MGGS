"""Closed-loop authentication using SR-MSGS multi-scale LP signatures.

This keeps the existing score-fusion protocol:
SR-MSGS LP score + optional point score + beta * generic score.
Only the LP signature extraction is replaced by multi-scale SR-MSGS pooling.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import copy

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from polygon_hier_stage1.dual_view_protocol import build_fixed_budget_polygon_views
from roads_hier_stage1.dual_view_protocol import build_fixed_budget_dual_views
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_hier_stage1_line_polygon_point_generic import MixedGenericLinePolygonPointEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import cosine_similarity
from scripts.evaluate_mixed_missing_subtype_tolerant_score_fusion import (
    build_tolerant_views,
    mean_point_embedding,
    sample_available_subtypes,
    to_jsonable,
)
from scripts.evaluate_point_hier_stage1_dual_view_hybrid import PointHybridDualViewEvaluator
from scripts.evaluate_zero_watermark_authentication import (
    aggregate_template_scores,
    attribution_metrics,
    build_dataset,
    dual_threshold_rates,
    pair_score,
    score_sets,
    threshold_at_far,
    threshold_at_frr,
    verification_rates,
)
from train_roads_hier_stage1_minimal import load_config, resolve_path
from train_sr_msgs_line_polygon import encode_lp_tensor, mean_signature, parse_scale_specs
from utils.metrics import compute_auc, compute_eer
from utils.seed import set_seed


@dataclass
class SignatureBundle:
    lp: np.ndarray
    generic: np.ndarray
    point: np.ndarray | None
    point_available: bool


@dataclass(frozen=True)
class QueryAttack:
    name: str
    budget_scale: float = 1.0
    feature_noise_std: float = 0.0
    scenario_override: str | None = None


def parse_query_attack(name: str) -> QueryAttack:
    normalized = str(name).strip().lower()
    if normalized in {"", "clean", "none"}:
        return QueryAttack(name="clean")
    if normalized == "random_delete_30":
        return QueryAttack(name=normalized, budget_scale=0.7)
    if normalized == "random_delete_50":
        return QueryAttack(name=normalized, budget_scale=0.5)
    if normalized == "crop_50":
        return QueryAttack(name=normalized, budget_scale=0.5)
    if normalized == "simplify_light":
        return QueryAttack(name=normalized, budget_scale=0.75, feature_noise_std=0.01)
    if normalized == "feature_noise_0p02":
        return QueryAttack(name=normalized, feature_noise_std=0.02)
    if normalized == "feature_noise_0p05":
        return QueryAttack(name=normalized, feature_noise_std=0.05)
    if normalized == "minor_missing":
        return QueryAttack(name=normalized, scenario_override="minor_missing")
    if normalized == "partial_family_missing":
        return QueryAttack(name=normalized, scenario_override="partial_family_missing")
    if normalized == "point_missing":
        return QueryAttack(name=normalized, scenario_override="point_missing")
    raise ValueError(f"Unsupported query attack: {name}")


def scaled_positive_int(value: int, scale: float) -> int:
    return max(1, int(round(int(value) * float(scale))))


def scaled_scale_specs(scale_specs: Sequence[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    return [
        {
            **dict(spec),
            "line_k_chunks": scaled_positive_int(int(spec["line_k_chunks"]), scale),
            "polygon_k_objects": scaled_positive_int(int(spec["polygon_k_objects"]), scale),
        }
        for spec in scale_specs
    ]


def scaled_protocol_config(config: Dict[str, Any], scale: float) -> Dict[str, Any]:
    if float(scale) >= 0.999:
        return config
    cloned = copy.deepcopy(config)
    if "k_chunks_per_tile" in cloned.get("line_protocol", {}):
        cloned["line_protocol"]["k_chunks_per_tile"] = scaled_positive_int(
            int(cloned["line_protocol"]["k_chunks_per_tile"]), scale
        )
    if "k_polygons_per_tile" in cloned.get("polygon_protocol", {}):
        cloned["polygon_protocol"]["k_polygons_per_tile"] = scaled_positive_int(
            int(cloned["polygon_protocol"]["k_polygons_per_tile"]), scale
        )
    if "k_points_per_tile" in cloned.get("point_protocol", {}):
        cloned["point_protocol"]["k_points_per_tile"] = scaled_positive_int(
            int(cloned["point_protocol"]["k_points_per_tile"]), scale
        )
    return cloned


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SR-MSGS zero-watermark authentication.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--lp_checkpoint", type=str, required=True)
    parser.add_argument("--point_checkpoint", type=str, required=True)
    parser.add_argument("--generic_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument(
        "--query_line_cache_root",
        type=str,
        default=None,
        help="Optional attacked query line cache. Calibration and enrollment templates stay on clean caches.",
    )
    parser.add_argument(
        "--query_polygon_cache_root",
        type=str,
        default=None,
        help="Optional attacked query polygon cache. Calibration and enrollment templates stay on clean caches.",
    )
    parser.add_argument(
        "--query_point_cache_root",
        type=str,
        default=None,
        help="Optional attacked query point cache. Calibration and enrollment templates stay on clean caches.",
    )
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--package_split", type=str, default=None)
    parser.add_argument("--calibration_split_name", type=str, default="train_packages")
    parser.add_argument("--test_split_name", type=str, default="heldout_packages")
    parser.add_argument(
        "--scenario",
        type=str,
        default="natural_missing",
        choices=["natural_missing", "minor_missing", "partial_family_missing", "point_missing"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--scale_specs", type=str, default="local:8:16,regional:16:32,global:32:64")
    parser.add_argument("--point_alpha", type=float, default=0.5)
    parser.add_argument("--generic_beta", type=float, default=0.25)
    parser.add_argument("--enrollment_templates", type=int, default=5)
    parser.add_argument(
        "--template_aggregation",
        type=str,
        default="max",
        choices=["max", "mean", "top2_mean", "median", "consensus"],
    )
    parser.add_argument(
        "--query_attack",
        type=str,
        default="clean",
        choices=[
            "clean",
            "random_delete_30",
            "random_delete_50",
            "crop_50",
            "simplify_light",
            "feature_noise_0p02",
            "feature_noise_0p05",
            "minor_missing",
            "partial_family_missing",
            "point_missing",
        ],
        help="Attack applied to query signatures only. Calibration and enrollment templates remain clean.",
    )
    parser.add_argument("--target_far", type=float, default=0.05)
    parser.add_argument("--target_frr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser


def encode_sr_lp_signature(
    *,
    sample: Dict[str, Any],
    config: Dict[str, Any],
    scale_specs: Sequence[Dict[str, Any]],
    seed_a: int,
    seed_b: int,
    role: str,
    lp_evaluator: MixedLinePolygonEvaluator,
    prehash_space: str,
    attack: QueryAttack | None = None,
) -> np.ndarray:
    attack = attack or QueryAttack(name="clean")
    line_protocol = config["line_protocol"]
    polygon_protocol = config["polygon_protocol"]
    line_depth_template = {int(key): int(value) for key, value in line_protocol["depth_template"].items()}
    polygon_depth_template = {int(key): int(value) for key, value in polygon_protocol["depth_template"].items()}
    suffix = "a" if role == "query" else "b"
    features: List[torch.Tensor] = []
    available_line = [
        subtype for subtype in config["data"].get("line_subtypes", ()) if subtype in sample["line_samples"]
    ]
    available_polygon = [
        subtype for subtype in config["data"].get("polygon_subtypes", ()) if subtype in sample["polygon_samples"]
    ]
    if not available_line or not available_polygon:
        raise ValueError(f"Package {sample['package_id']} has no usable LP subtype.")
    effective_scale_specs = scaled_scale_specs(scale_specs, attack.budget_scale)
    for scale_idx, spec in enumerate(effective_scale_specs):
        rng_a = np.random.default_rng(int(seed_a) + scale_idx * 10007)
        rng_b = np.random.default_rng(int(seed_b) + scale_idx * 10007)
        line_a = sample_available_subtypes(
            available_line,
            per_view=config["data"].get("line_subtypes_per_view"),
            rng=rng_a,
        )
        line_b = sample_available_subtypes(
            available_line,
            per_view=config["data"].get("line_subtypes_per_view"),
            rng=rng_b,
        )
        polygon_a = sample_available_subtypes(
            available_polygon,
            per_view=config["data"].get("polygon_subtypes_per_view"),
            rng=rng_a,
        )
        polygon_b = sample_available_subtypes(
            available_polygon,
            per_view=config["data"].get("polygon_subtypes_per_view"),
            rng=rng_b,
        )
        selected_line = sorted(set(line_a) | set(line_b))
        selected_polygon = sorted(set(polygon_a) | set(polygon_b))
        if not selected_line or not selected_polygon:
            raise ValueError(f"Package {sample['package_id']} has no usable LP view.")
        line_view_a: Dict[str, object] = {}
        line_view_b: Dict[str, object] = {}
        polygon_view_a: Dict[str, object] = {}
        polygon_view_b: Dict[str, object] = {}
        for subtype in selected_line:
            dual = build_fixed_budget_dual_views(
                sample["line_samples"][subtype],
                n_tiles=int(line_protocol["n_tiles"]),
                k_chunks=int(spec["line_k_chunks"]),
                depth_template=line_depth_template,
                rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
                rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
                feature_noise_std=float(attack.feature_noise_std),
                device=lp_evaluator.device,
            )
            if subtype in line_a:
                line_view_a[subtype] = dual["view_a"]
            if subtype in line_b:
                line_view_b[subtype] = dual["view_b"]
        for subtype in selected_polygon:
            dual = build_fixed_budget_polygon_views(
                sample["polygon_samples"][subtype],
                n_tiles=int(polygon_protocol["n_tiles"]),
                k_polygons=int(spec["polygon_k_objects"]),
                depth_template=polygon_depth_template,
                rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
                rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
                feature_noise_std=float(attack.feature_noise_std),
                device=lp_evaluator.device,
            )
            if subtype in polygon_a:
                polygon_view_a[subtype] = dual["view_a"]
            if subtype in polygon_b:
                polygon_view_b[subtype] = dual["view_b"]
        view = {
            "line_view_a": line_view_a,
            "line_view_b": line_view_b,
            "polygon_view_a": polygon_view_a,
            "polygon_view_b": polygon_view_b,
        }
        features.append(encode_lp_tensor(lp_evaluator, view, suffix, prehash_space))
    return mean_signature(torch.stack(features, dim=0)).detach().cpu().numpy()


def encode_bundle(
    *,
    sample: Dict[str, Any],
    config: Dict[str, Any],
    scenario: str,
    seed_a: int,
    seed_b: int,
    role: str,
    scale_specs: Sequence[Dict[str, Any]],
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
    prehash_space: str,
    attack: QueryAttack | None = None,
) -> SignatureBundle:
    attack = attack or QueryAttack(name="clean")
    lp = encode_sr_lp_signature(
        sample=sample,
        config=config,
        scale_specs=scale_specs,
        seed_a=seed_a,
        seed_b=seed_b,
        role=role,
        lp_evaluator=lp_evaluator,
        prehash_space=prehash_space,
        attack=attack,
    )
    effective_config = scaled_protocol_config(config, attack.budget_scale)
    effective_scenario = attack.scenario_override or scenario
    views = build_tolerant_views(
        sample,
        config=effective_config,
        scenario=effective_scenario,
        rng_a=np.random.default_rng(seed_a),
        rng_b=np.random.default_rng(seed_b),
        device=lp_evaluator.device,
        feature_noise_std=float(attack.feature_noise_std),
    )
    suffix = "a" if role == "query" else "b"
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


def build_registered_scores(
    dataset: Any,
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    scale_specs: Sequence[Dict[str, Any]],
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
    template_aggregation: str,
    template_threshold: float | None = None,
    query_attack: QueryAttack | None = None,
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    query_attack = query_attack or QueryAttack(name="clean")
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
                scale_specs=scale_specs,
                lp_evaluator=lp_evaluator,
                point_evaluator=point_evaluator,
                generic_evaluator=generic_evaluator,
                prehash_space=args.prehash_space,
                attack=query_attack,
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
                    scale_specs=scale_specs,
                    lp_evaluator=lp_evaluator,
                    point_evaluator=point_evaluator,
                    generic_evaluator=generic_evaluator,
                    prehash_space=args.prehash_space,
                    attack=QueryAttack(name="clean"),
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
        "query_attack": query_attack.name,
        "query_attack_budget_scale": float(query_attack.budget_scale),
        "query_attack_feature_noise_std": float(query_attack.feature_noise_std),
        "query_attack_scenario_override": query_attack.scenario_override,
    }
    return scores, package_ids, diagnostics


def build_cross_registered_scores(
    query_dataset: Any,
    template_dataset: Any,
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    scale_specs: Sequence[Dict[str, Any]],
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
    template_aggregation: str,
    template_threshold: float | None = None,
    query_attack: QueryAttack | None = None,
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    query_attack = query_attack or QueryAttack(name="clean")
    template_by_id = {str(record["package_id"]).lower(): idx for idx, record in enumerate(template_dataset.package_records)}
    query_records = [
        (idx, str(record["package_id"]).lower())
        for idx, record in enumerate(query_dataset.package_records)
        if str(record["package_id"]).lower() in template_by_id
    ]

    package_ids: List[str] = []
    query_bundles: List[SignatureBundle] = []
    template_sets: List[List[SignatureBundle]] = []
    skipped: List[str] = []

    for ordinal, (query_idx, package_id) in enumerate(query_records):
        query_sample = query_dataset[query_idx]
        template_sample = template_dataset[template_by_id[package_id]]
        base_seed = int(args.seed) * 1000003 + ordinal * 1009
        try:
            query = encode_bundle(
                sample=query_sample,
                config=config,
                scenario=args.scenario,
                seed_a=base_seed + 1,
                seed_b=base_seed + 2,
                role="query",
                scale_specs=scale_specs,
                lp_evaluator=lp_evaluator,
                point_evaluator=point_evaluator,
                generic_evaluator=generic_evaluator,
                prehash_space=args.prehash_space,
                attack=query_attack,
            )
            templates: List[SignatureBundle] = []
            for template_idx in range(int(args.enrollment_templates)):
                template = encode_bundle(
                    sample=template_sample,
                    config=config,
                    scenario=args.scenario,
                    seed_a=base_seed + 101 + template_idx * 17,
                    seed_b=base_seed + 102 + template_idx * 17,
                    role="template",
                    scale_specs=scale_specs,
                    lp_evaluator=lp_evaluator,
                    point_evaluator=point_evaluator,
                    generic_evaluator=generic_evaluator,
                    prehash_space=args.prehash_space,
                    attack=QueryAttack(name="clean"),
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
        "query_attack": query_attack.name,
        "query_attack_budget_scale": float(query_attack.budget_scale),
        "query_attack_feature_noise_std": float(query_attack.feature_noise_std),
        "query_attack_scenario_override": query_attack.scenario_override,
        "query_cache_roots": {
            "line": args.query_line_cache_root,
            "polygon": args.query_polygon_cache_root,
            "point": args.query_point_cache_root,
        },
    }
    return scores, package_ids, diagnostics


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
            "separability": float((np.mean(genuine) - np.mean(impostor)) / (np.std(genuine) + np.std(impostor) + 1e-8))
            if impostor.size
            else 0.0,
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


def far_sweep_rates(
    calibration_scores: np.ndarray,
    scores: np.ndarray,
    fars: Sequence[float],
) -> Dict[str, Any]:
    calibration_genuine, calibration_impostor = score_sets(calibration_scores)
    genuine, impostor = score_sets(scores)
    result: Dict[str, Any] = {}
    for far in fars:
        threshold = threshold_at_far(calibration_impostor, float(far))
        result[f"{float(far):.4f}"] = verification_rates(genuine, impostor, threshold)
    return result


def main() -> None:
    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    if args.point_cache_root:
        config["data"]["point_cache_root"] = args.point_cache_root
    clean_cache_roots = {
        "line": config["data"]["line_cache_root"],
        "polygon": config["data"]["polygon_cache_root"],
        "point": config["data"].get("point_cache_root"),
    }
    config["device"]["seed"] = int(args.seed)
    set_seed(config["device"]["seed"])
    scale_specs = parse_scale_specs(args.scale_specs)
    query_attack = parse_query_attack(args.query_attack)

    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    generic_evaluator = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)

    calibration_dataset = build_dataset(config, args, args.calibration_split_name)
    test_dataset = build_dataset(config, args, args.test_split_name)
    query_dataset = None
    if args.query_line_cache_root or args.query_polygon_cache_root or args.query_point_cache_root:
        query_config = copy.deepcopy(config)
        if args.query_line_cache_root:
            query_config["data"]["line_cache_root"] = args.query_line_cache_root
        if args.query_polygon_cache_root:
            query_config["data"]["polygon_cache_root"] = args.query_polygon_cache_root
        if args.query_point_cache_root:
            query_config["data"]["point_cache_root"] = args.query_point_cache_root
        query_dataset = build_dataset(query_config, args, args.test_split_name)

    preliminary_aggregation = "max" if args.template_aggregation == "consensus" else args.template_aggregation
    calibration_scores, calibration_ids, calibration_diag = build_registered_scores(
        calibration_dataset,
        config=config,
        args=args,
        scale_specs=scale_specs,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
        template_aggregation=preliminary_aggregation,
        query_attack=QueryAttack(name="clean"),
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
            scale_specs=scale_specs,
            lp_evaluator=lp_evaluator,
            point_evaluator=point_evaluator,
            generic_evaluator=generic_evaluator,
            template_aggregation=args.template_aggregation,
            template_threshold=thresholds["accept"],
            query_attack=QueryAttack(name="clean"),
        )
        calibration_genuine, calibration_impostor = score_sets(calibration_scores)
        calibration_eer, calibration_eer_threshold = compute_eer(calibration_genuine, calibration_impostor)
        thresholds = {
            "eer": float(calibration_eer_threshold),
            "accept": threshold_at_far(calibration_impostor, float(args.target_far)),
            "reject": threshold_at_frr(calibration_genuine, float(args.target_frr)),
        }

    if query_dataset is None:
        test_scores, test_ids, test_diag = build_registered_scores(
            test_dataset,
            config=config,
            args=args,
            scale_specs=scale_specs,
            lp_evaluator=lp_evaluator,
            point_evaluator=point_evaluator,
            generic_evaluator=generic_evaluator,
            template_aggregation=args.template_aggregation,
            template_threshold=thresholds["accept"] if args.template_aggregation == "consensus" else None,
            query_attack=query_attack,
        )
    else:
        test_scores, test_ids, test_diag = build_cross_registered_scores(
            query_dataset,
            test_dataset,
            config=config,
            args=args,
            scale_specs=scale_specs,
            lp_evaluator=lp_evaluator,
            point_evaluator=point_evaluator,
            generic_evaluator=generic_evaluator,
            template_aggregation=args.template_aggregation,
            template_threshold=thresholds["accept"] if args.template_aggregation == "consensus" else None,
            query_attack=query_attack,
        )

    report = {
        "arguments": {
            **vars(args),
            "line_cache_root": config["data"]["line_cache_root"],
            "polygon_cache_root": config["data"]["polygon_cache_root"],
            "point_cache_root": config["data"].get("point_cache_root"),
            "clean_cache_roots": clean_cache_roots,
            "query_line_cache_root": args.query_line_cache_root,
            "query_polygon_cache_root": args.query_polygon_cache_root,
            "query_point_cache_root": args.query_point_cache_root,
            "scale_specs_parsed": scale_specs,
        },
        "calibration": {
            "diagnostics": calibration_diag,
            "metrics": summarize_split(calibration_scores, calibration_ids, thresholds),
            "far_sweep": far_sweep_rates(calibration_scores, calibration_scores, [0.01, 0.05, 0.10]),
            "calibrated_thresholds": {
                **thresholds,
                "calibration_eer": float(calibration_eer),
            },
        },
        "test": {
            "diagnostics": test_diag,
            "metrics": summarize_split(test_scores, test_ids, thresholds),
            "far_sweep": far_sweep_rates(calibration_scores, test_scores, [0.01, 0.05, 0.10]),
        },
    }

    test_metrics = report["test"]["metrics"]
    print("=" * 90)
    print("SR-MSGS Zero-Watermark Authentication")
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
