"""Evaluate pairwise verification head for zero-watermark authentication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.verification_head import PairwiseVerificationHead
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_hier_stage1_line_polygon_point_generic import MixedGenericLinePolygonPointEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import compute_separability_from_scores, cosine_similarity
from scripts.evaluate_mixed_missing_subtype_tolerant_score_fusion import build_tolerant_views, mean_point_embedding, to_jsonable
from scripts.evaluate_point_hier_stage1_dual_view_hybrid import PointHybridDualViewEvaluator
from scripts.evaluate_zero_watermark_authentication import (
    attribution_metrics,
    build_dataset,
    dual_threshold_rates,
    threshold_at_far,
    threshold_at_frr,
    verification_rates,
)
from train_roads_hier_stage1_minimal import load_config, resolve_path
from train_zero_watermark_verification_head import l2_normalize, modality_pair_features
from utils.metrics import compute_auc, compute_eer
from utils.seed import set_seed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate zero-watermark pairwise verification head.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--head_checkpoint", type=str, required=True)
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
    parser.add_argument("--target_frr", type=float, default=0.01)
    parser.add_argument("--invert_score", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser


def load_head(path: str, device: torch.device) -> PairwiseVerificationHead:
    checkpoint = torch.load(resolve_path(path), map_location=device)
    model = PairwiseVerificationHead(
        int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def zero_point_like(lp: np.ndarray) -> np.ndarray:
    return np.zeros_like(lp, dtype=np.float32)


def encode_bundle(
    *,
    sample: Dict[str, Any],
    config: Dict[str, Any],
    args: argparse.Namespace,
    seed_a: int,
    seed_b: int,
    suffix: str,
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
) -> Dict[str, Any]:
    views = build_tolerant_views(
        sample,
        config=config,
        scenario=args.scenario,
        rng_a=np.random.default_rng(seed_a),
        rng_b=np.random.default_rng(seed_b),
        device=lp_evaluator.device,
    )
    lp, _ = lp_evaluator.encode_package(views[f"line_view_{suffix}"], views[f"polygon_view_{suffix}"], args.prehash_space)
    generic, _ = generic_evaluator.encode_package(
        views[f"line_view_{suffix}"],
        views[f"polygon_view_{suffix}"],
        views[f"point_view_{suffix}"],
        args.prehash_space,
    )
    point = mean_point_embedding(point_evaluator, views[f"point_view_{suffix}"], prehash_space=args.prehash_space)
    return {
        "lp": l2_normalize(np.asarray(lp, dtype=np.float32)),
        "generic": l2_normalize(np.asarray(generic, dtype=np.float32)),
        "point": l2_normalize(np.asarray(point, dtype=np.float32)) if point is not None else zero_point_like(lp),
        "point_available": float(point is not None),
    }


def pair_feature_bundle(query: Dict[str, Any], gallery: Dict[str, Any], point_alpha: float, generic_beta: float) -> np.ndarray:
    lp_score = cosine_similarity(query["lp"], gallery["lp"])
    generic_score = cosine_similarity(query["generic"], gallery["generic"])
    point_available = float(query["point_available"] > 0.5 and gallery["point_available"] > 0.5)
    point_score = cosine_similarity(query["point"], gallery["point"]) if point_available else 0.0
    final_score = lp_score + float(point_alpha) * point_score + float(generic_beta) * generic_score
    return np.concatenate(
        [
            modality_pair_features(query["lp"], gallery["lp"]),
            modality_pair_features(query["generic"], gallery["generic"]),
            modality_pair_features(query["point"], gallery["point"]) * point_available,
            np.asarray([lp_score, generic_score, point_score, final_score, point_available], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


@torch.no_grad()
def head_score(
    head: PairwiseVerificationHead,
    device: torch.device,
    query: Dict[str, Any],
    gallery: Dict[str, Any],
    *,
    point_alpha: float,
    generic_beta: float,
) -> float:
    x = torch.as_tensor(
        pair_feature_bundle(query, gallery, point_alpha, generic_beta),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    value = float(head(x)[0].detach().cpu().item())
    return -value if getattr(head, "invert_score", False) else value


def aggregate(scores: Sequence[float], method: str) -> float:
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if method == "max":
        return float(np.max(arr))
    if method == "mean":
        return float(np.mean(arr))
    if method == "top2_mean":
        return float(np.mean(np.sort(arr)[-min(2, arr.size):]))
    if method == "median":
        return float(np.median(arr))
    raise ValueError(f"Unsupported aggregation: {method}")


def score_sets(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.asarray([scores[i, i] for i in range(scores.shape[0])], dtype=np.float64)
    impostor = np.asarray(
        [scores[i, j] for i in range(scores.shape[0]) for j in range(scores.shape[1]) if i != j],
        dtype=np.float64,
    )
    return genuine, impostor


def summarize(scores: np.ndarray, package_ids: Sequence[str], thresholds: Dict[str, float]) -> Dict[str, Any]:
    genuine, impostor = score_sets(scores)
    eer, eer_threshold = compute_eer(genuine, impostor)
    return {
        "n_packages": len(package_ids),
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "pairwise": {
            "auc": compute_auc(genuine, impostor),
            "eer": eer,
            "eer_threshold": eer_threshold,
            "separability": compute_separability_from_scores(genuine, impostor),
            "genuine_mean": float(np.mean(genuine)) if genuine.size else 0.0,
            "impostor_mean": float(np.mean(impostor)) if impostor.size else 0.0,
        },
        "single_threshold_far_calibrated": verification_rates(genuine, impostor, thresholds["accept"]),
        "single_threshold_eer_calibrated": verification_rates(genuine, impostor, thresholds["eer"]),
        "dual_threshold": dual_threshold_rates(
            genuine,
            impostor,
            accept_threshold=thresholds["accept"],
            reject_threshold=thresholds["reject"],
        ),
        "attribution": attribution_metrics(scores, package_ids, thresholds["accept"]),
    }


def build_level0_scores(
    dataset: Any,
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    head: PairwiseVerificationHead,
    device: torch.device,
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
) -> tuple[np.ndarray, List[str]]:
    package_ids: List[str] = []
    queries: List[Dict[str, Any]] = []
    galleries: List[Dict[str, Any]] = []
    for idx in tqdm(range(len(dataset)), desc="Build Level-0 head bundles"):
        sample = dataset[idx]
        base_seed = int(args.seed) * 1000003 + idx * 1009
        try:
            queries.append(
                encode_bundle(
                    sample=sample,
                    config=config,
                    args=args,
                    seed_a=base_seed + 1,
                    seed_b=base_seed + 2,
                    suffix="a",
                    lp_evaluator=lp_evaluator,
                    point_evaluator=point_evaluator,
                    generic_evaluator=generic_evaluator,
                )
            )
            galleries.append(
                encode_bundle(
                    sample=sample,
                    config=config,
                    args=args,
                    seed_a=base_seed + 1,
                    seed_b=base_seed + 2,
                    suffix="b",
                    lp_evaluator=lp_evaluator,
                    point_evaluator=point_evaluator,
                    generic_evaluator=generic_evaluator,
                )
            )
        except ValueError:
            continue
        package_ids.append(str(sample["package_id"]))
    scores = np.zeros((len(package_ids), len(package_ids)), dtype=np.float64)
    for i, query in enumerate(tqdm(queries, desc="Score Level-0 head matrix")):
        for j, gallery in enumerate(galleries):
            scores[i, j] = head_score(
                head,
                device,
                query,
                gallery,
                point_alpha=float(args.point_alpha),
                generic_beta=float(args.generic_beta),
            )
    return scores, package_ids


def build_registered_scores(
    dataset: Any,
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    head: PairwiseVerificationHead,
    device: torch.device,
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
) -> tuple[np.ndarray, List[str]]:
    package_ids: List[str] = []
    queries: List[Dict[str, Any]] = []
    template_sets: List[List[Dict[str, Any]]] = []
    for idx in tqdm(range(len(dataset)), desc="Build registered head bundles"):
        sample = dataset[idx]
        base_seed = int(args.seed) * 1000003 + idx * 1009
        try:
            query = encode_bundle(
                sample=sample,
                config=config,
                args=args,
                seed_a=base_seed + 1,
                seed_b=base_seed + 2,
                suffix="a",
                lp_evaluator=lp_evaluator,
                point_evaluator=point_evaluator,
                generic_evaluator=generic_evaluator,
            )
            templates: List[Dict[str, Any]] = []
            for template_idx in range(int(args.enrollment_templates)):
                templates.append(
                    encode_bundle(
                        sample=sample,
                        config=config,
                        args=args,
                        seed_a=base_seed + 101 + template_idx * 17,
                        seed_b=base_seed + 102 + template_idx * 17,
                        suffix="b",
                        lp_evaluator=lp_evaluator,
                        point_evaluator=point_evaluator,
                        generic_evaluator=generic_evaluator,
                    )
                )
        except ValueError:
            continue
        package_ids.append(str(sample["package_id"]))
        queries.append(query)
        template_sets.append(templates)
    scores = np.zeros((len(package_ids), len(package_ids)), dtype=np.float64)
    for i, query in enumerate(tqdm(queries, desc="Score registered head matrix")):
        for j, templates in enumerate(template_sets):
            template_scores = [
                head_score(
                    head,
                    device,
                    query,
                    template,
                    point_alpha=float(args.point_alpha),
                    generic_beta=float(args.generic_beta),
                )
                for template in templates
            ]
            scores[i, j] = aggregate(template_scores, args.template_aggregation)
    return scores, package_ids


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
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    head = load_head(args.head_checkpoint, device)
    setattr(head, "invert_score", bool(args.invert_score))
    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    generic_evaluator = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)

    calibration_dataset = build_dataset(config, args, args.calibration_split_name)
    test_dataset = build_dataset(config, args, args.test_split_name)

    calibration_level0, calibration_ids = build_level0_scores(
        calibration_dataset,
        config=config,
        args=args,
        head=head,
        device=device,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
    )
    calibration_genuine, calibration_impostor = score_sets(calibration_level0)
    calibration_eer, calibration_eer_threshold = compute_eer(calibration_genuine, calibration_impostor)
    thresholds = {
        "eer": float(calibration_eer_threshold),
        "accept": threshold_at_far(calibration_impostor, float(args.target_far)),
        "reject": threshold_at_frr(calibration_genuine, float(args.target_frr)),
    }

    test_level0, test_level0_ids = build_level0_scores(
        test_dataset,
        config=config,
        args=args,
        head=head,
        device=device,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
    )
    calibration_registered, calibration_reg_ids = build_registered_scores(
        calibration_dataset,
        config=config,
        args=args,
        head=head,
        device=device,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
    )
    calibration_registered_genuine, calibration_registered_impostor = score_sets(calibration_registered)
    registered_eer, registered_eer_threshold = compute_eer(calibration_registered_genuine, calibration_registered_impostor)
    registered_thresholds = {
        "eer": float(registered_eer_threshold),
        "accept": threshold_at_far(calibration_registered_impostor, float(args.target_far)),
        "reject": threshold_at_frr(calibration_registered_genuine, float(args.target_frr)),
    }
    test_registered, test_reg_ids = build_registered_scores(
        test_dataset,
        config=config,
        args=args,
        head=head,
        device=device,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
    )

    report = {
        "arguments": vars(args),
        "level0": {
            "calibration": summarize(calibration_level0, calibration_ids, thresholds),
            "test": summarize(test_level0, test_level0_ids, thresholds),
            "thresholds": {
                **thresholds,
                "calibration_eer": float(calibration_eer),
            },
        },
        "registered": {
            "calibration": summarize(calibration_registered, calibration_reg_ids, registered_thresholds),
            "test": summarize(test_registered, test_reg_ids, registered_thresholds),
            "thresholds": {
                **registered_thresholds,
                "calibration_eer": float(registered_eer),
            },
        },
    }

    level0_test = report["level0"]["test"]
    reg_test = report["registered"]["test"]
    print("=" * 90)
    print("Zero-Watermark Verification Head Evaluation")
    print("=" * 90)
    print(
        f"Level-0: AUC={level0_test['pairwise']['auc']:.6f}, "
        f"EER={level0_test['pairwise']['eer']:.6f}, "
        f"TAR@FAR{args.target_far:g}={level0_test['single_threshold_far_calibrated']['tar']:.6f}, "
        f"FAR={level0_test['single_threshold_far_calibrated']['far']:.6f}"
    )
    print(
        f"Registered K={args.enrollment_templates}: AUC={reg_test['pairwise']['auc']:.6f}, "
        f"EER={reg_test['pairwise']['eer']:.6f}, "
        f"TAR@FAR{args.target_far:g}={reg_test['single_threshold_far_calibrated']['tar']:.6f}, "
        f"FAR={reg_test['single_threshold_far_calibrated']['far']:.6f}, "
        f"Top1={reg_test['attribution']['top1_attribution_accuracy']:.6f}, "
        f"MRR={reg_test['attribution']['mrr']:.6f}"
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
