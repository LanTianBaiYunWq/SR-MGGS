"""Score normalization/calibration diagnostics for zero-watermark authentication."""

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
from scripts.evaluate_mixed_lpp_score_fusion import (
    compute_separability_from_scores,
    cosine_similarity,
    score_matrix,
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate score normalization/calibration for authentication.")
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
    parser.add_argument("--generic_beta", type=float, default=0.5)
    parser.add_argument("--target_far", type=float, default=0.05)
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
    dataset.package_records = [
        record
        for record in dataset.package_records
        if str(record["package_id"]).lower() in package_subset
    ]
    return dataset


def encode_split_components(
    dataset: MixedLinePolygonOptionalPointPackageDataset,
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    lp_evaluator: MixedLinePolygonEvaluator,
    point_evaluator: PointHybridDualViewEvaluator,
    generic_evaluator: MixedGenericLinePolygonPointEvaluator,
) -> Dict[str, Any]:
    package_ids: List[str] = []
    lp_query: List[np.ndarray] = []
    lp_gallery: List[np.ndarray] = []
    generic_query: List[np.ndarray] = []
    generic_gallery: List[np.ndarray] = []
    point_query_by_row: List[np.ndarray | None] = []
    point_gallery_by_row: List[np.ndarray | None] = []
    point_available: List[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        try:
            views = build_tolerant_views(
                sample,
                config=config,
                scenario=args.scenario,
                rng_a=np.random.default_rng(int(args.seed) * 1000003 + idx * 101),
                rng_b=np.random.default_rng(int(args.seed) * 1000003 + idx * 101 + 1),
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
        package_ids.append(str(sample["package_id"]))
        lp_query.append(lp_a)
        lp_gallery.append(lp_b)
        generic_query.append(generic_a)
        generic_gallery.append(generic_b)
        point_query_by_row.append(p_a)
        point_gallery_by_row.append(p_b)
        point_available.append(float(p_a is not None and p_b is not None))

    lp_scores = score_matrix(np.asarray(lp_query), np.asarray(lp_gallery))
    generic_scores = score_matrix(np.asarray(generic_query), np.asarray(generic_gallery))
    point_scores_full = np.zeros_like(lp_scores)
    point_mask = np.zeros_like(lp_scores)
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
                point_scores_full[package_i, package_j] = point_scores_sub[row_i, row_j]
                point_mask[package_i, package_j] = 1.0
    subtype_scores = lp_scores + float(args.point_alpha) * point_scores_full * point_mask
    final_scores = subtype_scores + float(args.generic_beta) * generic_scores
    return {
        "package_ids": package_ids,
        "lp": lp_scores,
        "point": point_scores_full,
        "point_mask": point_mask,
        "subtype": subtype_scores,
        "generic": generic_scores,
        "final": final_scores,
        "point_used_rate": float(np.mean(point_available)) if point_available else 0.0,
    }


def score_sets(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.asarray([scores[i, i] for i in range(scores.shape[0])], dtype=np.float64)
    impostor = np.asarray(
        [scores[i, j] for i in range(scores.shape[0]) for j in range(scores.shape[1]) if i != j],
        dtype=np.float64,
    )
    return genuine, impostor


def metrics(scores: np.ndarray) -> Dict[str, float]:
    genuine, impostor = score_sets(scores)
    eer, eer_threshold = compute_eer(genuine, impostor) if impostor.size else (0.0, 0.0)
    return {
        "auc": compute_auc(genuine, impostor) if impostor.size else 0.0,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "separability": compute_separability_from_scores(genuine, impostor) if impostor.size else 0.0,
        "genuine_mean": float(np.mean(genuine)) if genuine.size else 0.0,
        "impostor_mean": float(np.mean(impostor)) if impostor.size else 0.0,
        "genuine_std": float(np.std(genuine)) if genuine.size else 0.0,
        "impostor_std": float(np.std(impostor)) if impostor.size else 0.0,
    }


def threshold_at_far(impostor_scores: np.ndarray, target_far: float) -> float:
    return float(np.quantile(impostor_scores, 1.0 - target_far, method="higher")) if impostor_scores.size else 0.0


def threshold_result(scores: np.ndarray, threshold: float) -> Dict[str, float]:
    genuine, impostor = score_sets(scores)
    far = float(np.mean(impostor >= threshold)) if impostor.size else 0.0
    frr = float(np.mean(genuine < threshold)) if genuine.size else 0.0
    return {"threshold": float(threshold), "far": far, "frr": frr, "tar": 1.0 - frr}


def z_norm(scores: np.ndarray, impostor_mean: float, impostor_std: float) -> np.ndarray:
    return (scores - impostor_mean) / (impostor_std + 1e-8)


def row_cohort_norm(scores: np.ndarray, top_k: int = 20) -> np.ndarray:
    normalized = np.zeros_like(scores, dtype=np.float64)
    for row_idx in range(scores.shape[0]):
        row = scores[row_idx]
        impostors = np.delete(row, row_idx) if scores.shape[0] > 1 else row
        k = min(top_k, impostors.shape[0])
        cohort = np.sort(impostors)[-k:] if k > 0 else impostors
        mean = float(np.mean(cohort)) if cohort.size else 0.0
        std = float(np.std(cohort)) if cohort.size else 1.0
        normalized[row_idx] = (row - mean) / (std + 1e-8)
    return normalized


def affine_calibrate(scores: np.ndarray, cal_genuine_mean: float, cal_impostor_mean: float) -> np.ndarray:
    center = 0.5 * (cal_genuine_mean + cal_impostor_mean)
    scale = abs(cal_genuine_mean - cal_impostor_mean) + 1e-8
    return (scores - center) / scale


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

    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    generic_evaluator = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)

    calibration = encode_split_components(
        build_dataset(config, args, args.calibration_split_name),
        config=config,
        args=args,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
    )
    test = encode_split_components(
        build_dataset(config, args, args.test_split_name),
        config=config,
        args=args,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
    )

    components: Dict[str, Dict[str, Any]] = {}
    for name in ("lp", "point", "subtype", "generic", "final"):
        cal_scores = calibration[name]
        test_scores = test[name]
        if name == "point":
            cal_scores = np.where(calibration["point_mask"] > 0, cal_scores, 0.0)
            test_scores = np.where(test["point_mask"] > 0, test_scores, 0.0)
        cal_metrics = metrics(cal_scores)
        test_metrics = metrics(test_scores)
        components[name] = {
            "calibration": cal_metrics,
            "test": test_metrics,
            "shift": {
                "genuine_mean_test_minus_calibration": test_metrics["genuine_mean"] - cal_metrics["genuine_mean"],
                "impostor_mean_test_minus_calibration": test_metrics["impostor_mean"] - cal_metrics["impostor_mean"],
                "separability_test_minus_calibration": test_metrics["separability"] - cal_metrics["separability"],
                "auc_test_minus_calibration": test_metrics["auc"] - cal_metrics["auc"],
                "eer_test_minus_calibration": test_metrics["eer"] - cal_metrics["eer"],
            },
        }

    final_cal = calibration["final"]
    final_test = test["final"]
    final_cal_genuine, final_cal_impostor = score_sets(final_cal)
    cal_final_metrics = metrics(final_cal)
    operating_points: Dict[str, Dict[str, Any]] = {}

    raw_threshold = threshold_at_far(final_cal_impostor, float(args.target_far))
    operating_points["raw"] = threshold_result(final_test, raw_threshold)

    z_cal = z_norm(final_cal, float(np.mean(final_cal_impostor)), float(np.std(final_cal_impostor)))
    z_test = z_norm(final_test, float(np.mean(final_cal_impostor)), float(np.std(final_cal_impostor)))
    _, z_cal_impostor = score_sets(z_cal)
    operating_points["z_norm"] = threshold_result(z_test, threshold_at_far(z_cal_impostor, float(args.target_far)))

    cohort_cal = row_cohort_norm(final_cal)
    cohort_test = row_cohort_norm(final_test)
    _, cohort_cal_impostor = score_sets(cohort_cal)
    operating_points["cohort_row_norm"] = threshold_result(
        cohort_test, threshold_at_far(cohort_cal_impostor, float(args.target_far))
    )

    affine_cal = affine_calibrate(
        final_cal,
        cal_genuine_mean=float(np.mean(final_cal_genuine)),
        cal_impostor_mean=float(np.mean(final_cal_impostor)),
    )
    affine_test = affine_calibrate(
        final_test,
        cal_genuine_mean=float(np.mean(final_cal_genuine)),
        cal_impostor_mean=float(np.mean(final_cal_impostor)),
    )
    _, affine_cal_impostor = score_sets(affine_cal)
    operating_points["affine"] = threshold_result(
        affine_test, threshold_at_far(affine_cal_impostor, float(args.target_far))
    )

    eer_threshold = float(cal_final_metrics["eer_threshold"])
    operating_points["raw_eer_threshold"] = threshold_result(final_test, eer_threshold)

    report = {
        "arguments": vars(args),
        "calibration_n": len(calibration["package_ids"]),
        "test_n": len(test["package_ids"]),
        "calibration_point_used_rate": calibration["point_used_rate"],
        "test_point_used_rate": test["point_used_rate"],
        "components": components,
        "operating_points": operating_points,
    }

    print("=" * 90)
    print("Zero-Watermark Score Calibration Diagnostics")
    print("=" * 90)
    for name, result in operating_points.items():
        print(
            f"{name}: TAR={result['tar']:.6f}, FAR={result['far']:.6f}, "
            f"FRR={result['frr']:.6f}, threshold={result['threshold']:.6f}"
        )
    print("-" * 90)
    for name in ("lp", "point", "subtype", "generic", "final"):
        shift = components[name]["shift"]
        print(
            f"{name}: genuine_shift={shift['genuine_mean_test_minus_calibration']:.6f}, "
            f"impostor_shift={shift['impostor_mean_test_minus_calibration']:.6f}, "
            f"sep_shift={shift['separability_test_minus_calibration']:.6f}"
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
