"""Diagnose point contribution with point-only and LP + alpha * P score fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonPointPackageDataset
from mixed_hier_stage1.line_polygon_point_protocol import build_partial_line_polygon_point_views
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_point_hier_stage1_dual_view_hybrid import PointHybridDualViewEvaluator
from train_roads_hier_stage1_minimal import load_config, resolve_path
from utils.metrics import compute_auc, compute_eer
from utils.seed import set_seed


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
    return float(np.dot(x, y) / denom)


def cosine_to_ber(score: float) -> float:
    return float((1.0 - score) / 2.0)


def compute_separability_from_scores(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    genuine_bers = np.asarray([cosine_to_ber(score) for score in genuine_scores], dtype=np.float64)
    impostor_bers = np.asarray([cosine_to_ber(score) for score in impostor_scores], dtype=np.float64)
    return float(
        (np.mean(impostor_bers) - np.mean(genuine_bers))
        / (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8)
    )


def score_matrix(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros((query_embeddings.shape[0], gallery_embeddings.shape[0]), dtype=np.float64)
    for i in range(query_embeddings.shape[0]):
        for j in range(gallery_embeddings.shape[0]):
            scores[i, j] = cosine_similarity(query_embeddings[i], gallery_embeddings[j])
    return scores


def metrics_from_score_matrix(scores: np.ndarray, package_ids: Sequence[str]) -> Dict[str, Dict[str, float]]:
    genuine_scores = np.asarray([scores[i, i] for i in range(scores.shape[0])], dtype=np.float64)
    impostor_scores = np.asarray(
        [scores[i, j] for i in range(scores.shape[0]) for j in range(scores.shape[1]) if i != j],
        dtype=np.float64,
    )
    recalls_1: List[float] = []
    recalls_5: List[float] = []
    reciprocal_ranks: List[float] = []
    for i in range(scores.shape[0]):
        ranked = np.argsort(-scores[i])
        ranked_ids = [package_ids[int(j)] for j in ranked]
        rank = ranked_ids.index(package_ids[i]) + 1
        recalls_1.append(float(rank <= 1))
        recalls_5.append(float(rank <= 5))
        reciprocal_ranks.append(1.0 / float(rank))
    return {
        "pairwise_metrics": {
            "auc": compute_auc(genuine_scores, impostor_scores) if len(impostor_scores) else 0.0,
            "eer": compute_eer(genuine_scores, impostor_scores)[0] if len(impostor_scores) else 0.0,
            "separability": compute_separability_from_scores(genuine_scores, impostor_scores)
            if len(impostor_scores)
            else 0.0,
            "genuine_mean_score": float(np.mean(genuine_scores)) if len(genuine_scores) else 0.0,
            "impostor_mean_score": float(np.mean(impostor_scores)) if len(impostor_scores) else 0.0,
        },
        "retrieval_metrics": {
            "recall_at_1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
            "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        },
    }


def ranks_from_score_matrix(scores: np.ndarray, package_ids: Sequence[str]) -> List[int]:
    ranks: List[int] = []
    for i in range(scores.shape[0]):
        ranked = np.argsort(-scores[i])
        ranked_ids = [package_ids[int(j)] for j in ranked]
        ranks.append(int(ranked_ids.index(package_ids[i]) + 1))
    return ranks


def positive_margin_from_score_matrix(scores: np.ndarray) -> List[float]:
    margins: List[float] = []
    for i in range(scores.shape[0]):
        impostor = np.delete(scores[i], i)
        hardest = float(np.max(impostor)) if impostor.size else 0.0
        margins.append(float(scores[i, i] - hardest))
    return margins


def stratified_rank_summary(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if values.size == 0:
        return {}
    quantiles = np.quantile(values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    buckets: Dict[str, List[Dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    for row, value in zip(rows, values):
        if value <= quantiles[1]:
            buckets["low"].append(row)
        elif value <= quantiles[2]:
            buckets["mid"].append(row)
        else:
            buckets["high"].append(row)
    summary: Dict[str, Dict[str, float]] = {}
    for name, bucket_rows in buckets.items():
        if not bucket_rows:
            continue
        summary[name] = {
            "n": float(len(bucket_rows)),
            "field_min": float(min(float(row[field]) for row in bucket_rows)),
            "field_max": float(max(float(row[field]) for row in bucket_rows)),
            "mean_rank_delta_lp_minus_fusion": float(
                np.mean([float(row["rank_delta_lp_minus_fusion"]) for row in bucket_rows])
            ),
            "mean_mrr_delta_fusion_minus_lp": float(
                np.mean([float(row["mrr_delta_fusion_minus_lp"]) for row in bucket_rows])
            ),
            "fusion_improved_rate": float(np.mean([float(row["fusion_improved"]) for row in bucket_rows])),
            "fusion_hurt_rate": float(np.mean([float(row["fusion_hurt"]) for row in bucket_rows])),
        }
    return summary


def row_top2_margin(scores: np.ndarray) -> np.ndarray:
    margins = np.zeros(scores.shape[0], dtype=np.float64)
    for i in range(scores.shape[0]):
        row = np.sort(scores[i])[::-1]
        if row.shape[0] < 2:
            margins[i] = 0.0
        else:
            margins[i] = float(row[0] - row[1])
    return margins


def normalize_confidence(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values)) if values.size else 0.0
    hi = float(np.max(values)) if values.size else 0.0
    if hi <= lo + 1e-12:
        return np.ones_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def confidence_gated_scores(
    lp_scores: np.ndarray,
    point_scores: np.ndarray,
    *,
    alpha: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    lp_conf = normalize_confidence(row_top2_margin(lp_scores))
    point_conf = normalize_confidence(row_top2_margin(point_scores))
    if mode == "point":
        gate = point_conf
    elif mode == "point_over_lp":
        gate = point_conf / (point_conf + lp_conf + 1e-8)
    elif mode == "point_minus_lp":
        gate = 1.0 / (1.0 + np.exp(-(point_conf - lp_conf) * 6.0))
    else:
        raise ValueError(f"Unsupported confidence gate mode: {mode}")
    gated = lp_scores + alpha * gate[:, None] * point_scores
    return gated, gate


def point_margin_selective_scores(
    lp_scores: np.ndarray,
    point_scores: np.ndarray,
    *,
    alpha: float,
    quantile: float,
    soft: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    margins = np.asarray(positive_margin_from_score_matrix(point_scores), dtype=np.float64)
    threshold = float(np.quantile(margins, quantile)) if margins.size else 0.0
    if soft:
        scale = float(np.std(margins)) + 1e-8
        gate = 1.0 / (1.0 + np.exp(-(margins - threshold) / scale))
    else:
        gate = (margins >= threshold).astype(np.float64)
    fused = lp_scores + alpha * gate[:, None] * point_scores
    return fused, gate, threshold


def mean_point_embedding(
    point_evaluator: PointHybridDualViewEvaluator,
    point_view_map: Dict[str, object],
    *,
    prehash_space: str,
) -> np.ndarray:
    embeddings = [
        point_evaluator.encode_view(point_view_map[subtype], prehash_space=prehash_space)
        for subtype in sorted(point_view_map.keys())
    ]
    if not embeddings:
        raise ValueError("point_view_map is empty")
    return np.mean(np.stack(embeddings, axis=0), axis=0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate point-only and LP+alpha*P score fusion on mixed subset.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_prototype.yaml")
    parser.add_argument("--lp_checkpoint", type=str, required=True)
    parser.add_argument("--point_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--max_packages", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--feature_noise_std", type=float, default=0.0)
    parser.add_argument("--alphas", type=str, default="0,0.1,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--gate_alphas", type=str, default="0.25,0.5,0.75,1")
    parser.add_argument("--gate_modes", type=str, default="point,point_over_lp,point_minus_lp")
    parser.add_argument("--selective_alphas", type=str, default="0.5,0.75,1")
    parser.add_argument("--selective_quantiles", type=str, default="0.25,0.5,0.67")
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

    dataset = MixedLinePolygonPointPackageDataset(
        config["data"]["line_cache_root"],
        config["data"]["polygon_cache_root"],
        config["data"]["point_cache_root"],
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
        require_complete_packages=bool(config["data"].get("require_complete_packages", True)),
    )
    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)

    line_protocol = config["line_protocol"]
    polygon_protocol = config["polygon_protocol"]
    point_protocol = config["point_protocol"]
    line_depth_template = {int(key): int(value) for key, value in line_protocol["depth_template"].items()}
    polygon_depth_template = {int(key): int(value) for key, value in polygon_protocol["depth_template"].items()}
    point_depth_template = {int(key): int(value) for key, value in point_protocol["depth_template"].items()}
    line_subtypes = tuple(config["data"].get("line_subtypes", ("roads", "railways", "waterways")))
    polygon_subtypes = tuple(config["data"].get("polygon_subtypes", ("building", "landuse", "natural")))
    point_subtypes = tuple(config["data"].get("point_subtypes", ("pois", "traffic", "transport", "pofw")))

    lp_query: List[np.ndarray] = []
    lp_gallery: List[np.ndarray] = []
    point_query: List[np.ndarray] = []
    point_gallery: List[np.ndarray] = []
    package_ids: List[str] = []
    overlaps: Dict[str, List[float]] = {
        "line_subtype": [],
        "line_tile": [],
        "line_chunk": [],
        "polygon_subtype": [],
        "polygon_tile": [],
        "polygon": [],
        "point_subtype": [],
        "point_tile": [],
        "point": [],
    }

    for idx in range(len(dataset)):
        package_sample = dataset[idx]
        views = build_partial_line_polygon_point_views(
            package_sample,
            line_subtypes=line_subtypes,
            polygon_subtypes=polygon_subtypes,
            point_subtypes=point_subtypes,
            line_subtypes_per_view=config["data"].get("line_subtypes_per_view"),
            polygon_subtypes_per_view=config["data"].get("polygon_subtypes_per_view"),
            point_subtypes_per_view=config["data"].get("point_subtypes_per_view"),
            line_n_tiles=int(line_protocol["n_tiles"]),
            line_k_chunks=int(line_protocol["k_chunks_per_tile"]),
            line_depth_template=line_depth_template,
            polygon_n_tiles=int(polygon_protocol["n_tiles"]),
            polygon_k_polygons=int(polygon_protocol["k_polygons_per_tile"]),
            polygon_depth_template=polygon_depth_template,
            point_n_tiles=int(point_protocol["n_tiles"]),
            point_k_points=int(point_protocol["k_points_per_tile"]),
            point_depth_template=point_depth_template,
            rng_a=np.random.default_rng(config["device"]["seed"] + idx * 101),
            rng_b=np.random.default_rng(config["device"]["seed"] + idx * 101 + 1),
            feature_noise_std=float(args.feature_noise_std),
            device=lp_evaluator.device,
        )
        lp_a, _ = lp_evaluator.encode_package(
            views["line_view_a"],
            views["polygon_view_a"],
            prehash_space=args.prehash_space,
        )
        lp_b, _ = lp_evaluator.encode_package(
            views["line_view_b"],
            views["polygon_view_b"],
            prehash_space=args.prehash_space,
        )
        p_a = mean_point_embedding(point_evaluator, views["point_view_a"], prehash_space=args.prehash_space)
        p_b = mean_point_embedding(point_evaluator, views["point_view_b"], prehash_space=args.prehash_space)

        lp_query.append(lp_a)
        lp_gallery.append(lp_b)
        point_query.append(p_a)
        point_gallery.append(p_b)
        package_ids.append(str(package_sample["package_id"]))
        overlaps["line_subtype"].append(float(views["line_subtype_overlap"]))
        overlaps["line_tile"].append(float(views["line_tile_overlap"]))
        overlaps["line_chunk"].append(float(views["line_chunk_overlap"]))
        overlaps["polygon_subtype"].append(float(views["polygon_subtype_overlap"]))
        overlaps["polygon_tile"].append(float(views["polygon_tile_overlap"]))
        overlaps["polygon"].append(float(views["polygon_overlap"]))
        overlaps["point_subtype"].append(float(views["point_subtype_overlap"]))
        overlaps["point_tile"].append(float(views["point_tile_overlap"]))
        overlaps["point"].append(float(views["point_overlap"]))

    lp_scores = score_matrix(np.asarray(lp_query), np.asarray(lp_gallery))
    point_scores = score_matrix(np.asarray(point_query), np.asarray(point_gallery))
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    gate_alphas = [float(value.strip()) for value in args.gate_alphas.split(",") if value.strip()]
    gate_modes = [value.strip() for value in args.gate_modes.split(",") if value.strip()]
    selective_alphas = [float(value.strip()) for value in args.selective_alphas.split(",") if value.strip()]
    selective_quantiles = [float(value.strip()) for value in args.selective_quantiles.split(",") if value.strip()]
    alpha_reports: Dict[str, Dict[str, Dict[str, float]]] = {}
    for alpha in alphas:
        fused_scores = lp_scores + alpha * point_scores
        alpha_reports[str(alpha)] = metrics_from_score_matrix(fused_scores, package_ids)
    gated_reports: Dict[str, Dict[str, object]] = {}
    for mode in gate_modes:
        for alpha in gate_alphas:
            fused_scores, gate = confidence_gated_scores(lp_scores, point_scores, alpha=alpha, mode=mode)
            key = f"{mode}:alpha={alpha}"
            gated_reports[key] = {
                "mode": mode,
                "alpha": alpha,
                "gate_mean": float(np.mean(gate)) if len(gate) else 0.0,
                "gate_std": float(np.std(gate)) if len(gate) else 0.0,
                "metrics": metrics_from_score_matrix(fused_scores, package_ids),
            }
    selective_reports: Dict[str, Dict[str, object]] = {}
    for alpha in selective_alphas:
        for quantile in selective_quantiles:
            for soft in (False, True):
                fused_scores, gate, threshold = point_margin_selective_scores(
                    lp_scores,
                    point_scores,
                    alpha=alpha,
                    quantile=quantile,
                    soft=soft,
                )
                key = f"{'soft' if soft else 'hard'}:alpha={alpha}:q={quantile}"
                selective_reports[key] = {
                    "alpha": alpha,
                    "quantile": quantile,
                    "soft": soft,
                    "threshold": threshold,
                    "gate_mean": float(np.mean(gate)) if len(gate) else 0.0,
                    "gate_std": float(np.std(gate)) if len(gate) else 0.0,
                    "metrics": metrics_from_score_matrix(fused_scores, package_ids),
                }

    point_only = metrics_from_score_matrix(point_scores, package_ids)
    line_polygon_only = metrics_from_score_matrix(lp_scores, package_ids)
    best_alpha = max(
        alphas,
        key=lambda alpha: (
            alpha_reports[str(alpha)]["retrieval_metrics"]["mrr"],
            alpha_reports[str(alpha)]["retrieval_metrics"]["recall_at_1"],
            alpha_reports[str(alpha)]["pairwise_metrics"]["auc"],
        ),
    )
    best_gate_key = max(
        gated_reports.keys(),
        key=lambda key: (
            gated_reports[key]["metrics"]["retrieval_metrics"]["mrr"],
            gated_reports[key]["metrics"]["retrieval_metrics"]["recall_at_1"],
            gated_reports[key]["metrics"]["pairwise_metrics"]["auc"],
        ),
    ) if gated_reports else None
    best_selective_key = max(
        selective_reports.keys(),
        key=lambda key: (
            selective_reports[key]["metrics"]["retrieval_metrics"]["mrr"],
            selective_reports[key]["metrics"]["retrieval_metrics"]["recall_at_1"],
            selective_reports[key]["metrics"]["pairwise_metrics"]["auc"],
        ),
    ) if selective_reports else None
    analysis_alpha = 0.5
    analysis_scores = lp_scores + analysis_alpha * point_scores
    lp_ranks = ranks_from_score_matrix(lp_scores, package_ids)
    point_ranks = ranks_from_score_matrix(point_scores, package_ids)
    fusion_ranks = ranks_from_score_matrix(analysis_scores, package_ids)
    lp_margins = positive_margin_from_score_matrix(lp_scores)
    point_margins = positive_margin_from_score_matrix(point_scores)
    fusion_margins = positive_margin_from_score_matrix(analysis_scores)
    package_rows: List[Dict[str, Any]] = []
    for i, package_id in enumerate(package_ids):
        lp_rank = int(lp_ranks[i])
        point_rank = int(point_ranks[i])
        fusion_rank = int(fusion_ranks[i])
        package_rows.append(
            {
                "package_id": package_id,
                "lp_rank": lp_rank,
                "point_rank": point_rank,
                "fusion_alpha_0.5_rank": fusion_rank,
                "rank_delta_lp_minus_fusion": int(lp_rank - fusion_rank),
                "mrr_delta_fusion_minus_lp": float((1.0 / fusion_rank) - (1.0 / lp_rank)),
                "fusion_improved": bool(fusion_rank < lp_rank),
                "fusion_hurt": bool(fusion_rank > lp_rank),
                "lp_positive_margin": float(lp_margins[i]),
                "point_positive_margin": float(point_margins[i]),
                "fusion_alpha_0.5_positive_margin": float(fusion_margins[i]),
                "line_subtype_overlap": float(overlaps["line_subtype"][i]),
                "line_tile_overlap": float(overlaps["line_tile"][i]),
                "line_chunk_overlap": float(overlaps["line_chunk"][i]),
                "polygon_subtype_overlap": float(overlaps["polygon_subtype"][i]),
                "polygon_tile_overlap": float(overlaps["polygon_tile"][i]),
                "polygon_overlap": float(overlaps["polygon"][i]),
                "point_subtype_overlap": float(overlaps["point_subtype"][i]),
                "point_tile_overlap": float(overlaps["point_tile"][i]),
                "point_overlap": float(overlaps["point"][i]),
            }
        )
    improved_rows = [row for row in package_rows if row["fusion_improved"]]
    hurt_rows = [row for row in package_rows if row["fusion_hurt"]]
    unchanged_rows = [row for row in package_rows if row["fusion_alpha_0.5_rank"] == row["lp_rank"]]
    package_analysis = {
        "analysis_alpha": analysis_alpha,
        "n_improved": len(improved_rows),
        "n_hurt": len(hurt_rows),
        "n_unchanged": len(unchanged_rows),
        "mean_rank_delta_lp_minus_fusion": float(np.mean([row["rank_delta_lp_minus_fusion"] for row in package_rows]))
        if package_rows
        else 0.0,
        "mean_mrr_delta_fusion_minus_lp": float(np.mean([row["mrr_delta_fusion_minus_lp"] for row in package_rows]))
        if package_rows
        else 0.0,
        "top_improved": sorted(package_rows, key=lambda row: row["rank_delta_lp_minus_fusion"], reverse=True)[:10],
        "top_hurt": sorted(package_rows, key=lambda row: row["rank_delta_lp_minus_fusion"])[:10],
        "stratified_by_point_subtype_overlap": stratified_rank_summary(package_rows, "point_subtype_overlap"),
        "stratified_by_point_tile_overlap": stratified_rank_summary(package_rows, "point_tile_overlap"),
        "stratified_by_point_overlap": stratified_rank_summary(package_rows, "point_overlap"),
        "stratified_by_point_positive_margin": stratified_rank_summary(package_rows, "point_positive_margin"),
        "package_rows": package_rows,
    }
    report = {
        "arguments": {
            "config": args.config,
            "lp_checkpoint": args.lp_checkpoint,
            "point_checkpoint": args.point_checkpoint,
            "line_cache_root": config["data"]["line_cache_root"],
            "polygon_cache_root": config["data"]["polygon_cache_root"],
            "point_cache_root": config["data"]["point_cache_root"],
            "dataset_mode": args.dataset_mode,
            "max_packages": config["data"].get("max_packages"),
            "prehash_space": args.prehash_space,
            "feature_noise_std": args.feature_noise_std,
            "seed": config["device"]["seed"],
            "alphas": alphas,
            "gate_alphas": gate_alphas,
            "gate_modes": gate_modes,
            "selective_alphas": selective_alphas,
            "selective_quantiles": selective_quantiles,
        },
        "n_packages": len(package_ids),
        "n_genuine_pairs": len(package_ids),
        "n_impostor_pairs": len(package_ids) * max(len(package_ids) - 1, 0),
        "line_polygon_only": line_polygon_only,
        "point_only": point_only,
        "score_fusion": alpha_reports,
        "confidence_gated_score_fusion": gated_reports,
        "point_margin_selective_score_fusion": selective_reports,
        "best_alpha_by_mrr": {
            "alpha": best_alpha,
            "metrics": alpha_reports[str(best_alpha)],
        },
        "best_gate_by_mrr": {
            "key": best_gate_key,
            "metrics": gated_reports[best_gate_key] if best_gate_key else None,
        },
        "best_selective_by_mrr": {
            "key": best_selective_key,
            "metrics": selective_reports[best_selective_key] if best_selective_key else None,
        },
        "package_analysis_alpha_0.5": package_analysis,
        "overlap_metrics": {f"{key}_mean": float(np.mean(value)) if value else 0.0 for key, value in overlaps.items()},
    }

    print("=" * 90)
    print("Mixed L+P/P Point-Only and Score Fusion Diagnostic")
    print("=" * 90)
    print(f"Packages: {report['n_packages']}")
    lp = line_polygon_only
    po = point_only
    print(
        "Line+Polygon: "
        f"AUC={lp['pairwise_metrics']['auc']:.6f}, EER={lp['pairwise_metrics']['eer']:.6f}, "
        f"R@1={lp['retrieval_metrics']['recall_at_1']:.6f}, R@5={lp['retrieval_metrics']['recall_at_5']:.6f}, "
        f"MRR={lp['retrieval_metrics']['mrr']:.6f}"
    )
    print(
        "Point-only: "
        f"AUC={po['pairwise_metrics']['auc']:.6f}, EER={po['pairwise_metrics']['eer']:.6f}, "
        f"R@1={po['retrieval_metrics']['recall_at_1']:.6f}, R@5={po['retrieval_metrics']['recall_at_5']:.6f}, "
        f"MRR={po['retrieval_metrics']['mrr']:.6f}"
    )
    for alpha in alphas:
        metrics = alpha_reports[str(alpha)]
        print(
            f"LP + {alpha:g}*P: "
            f"AUC={metrics['pairwise_metrics']['auc']:.6f}, EER={metrics['pairwise_metrics']['eer']:.6f}, "
            f"R@1={metrics['retrieval_metrics']['recall_at_1']:.6f}, "
            f"R@5={metrics['retrieval_metrics']['recall_at_5']:.6f}, "
            f"MRR={metrics['retrieval_metrics']['mrr']:.6f}"
        )
    if best_gate_key:
        best_gate = gated_reports[best_gate_key]
        metrics = best_gate["metrics"]
        print(
            f"Best confidence gate: {best_gate_key}, "
            f"gate_mean={best_gate['gate_mean']:.6f}, gate_std={best_gate['gate_std']:.6f}, "
            f"AUC={metrics['pairwise_metrics']['auc']:.6f}, EER={metrics['pairwise_metrics']['eer']:.6f}, "
            f"R@1={metrics['retrieval_metrics']['recall_at_1']:.6f}, "
            f"R@5={metrics['retrieval_metrics']['recall_at_5']:.6f}, "
            f"MRR={metrics['retrieval_metrics']['mrr']:.6f}"
        )
    if best_selective_key:
        best_selective = selective_reports[best_selective_key]
        metrics = best_selective["metrics"]
        print(
            f"Best point-margin selective: {best_selective_key}, "
            f"gate_mean={best_selective['gate_mean']:.6f}, threshold={best_selective['threshold']:.6f}, "
            f"AUC={metrics['pairwise_metrics']['auc']:.6f}, EER={metrics['pairwise_metrics']['eer']:.6f}, "
            f"R@1={metrics['retrieval_metrics']['recall_at_1']:.6f}, "
            f"R@5={metrics['retrieval_metrics']['recall_at_5']:.6f}, "
            f"MRR={metrics['retrieval_metrics']['mrr']:.6f}"
        )
    print(f"Best alpha by MRR: {best_alpha:g}")
    print(
        f"Package analysis alpha=0.5: improved={package_analysis['n_improved']}, "
        f"hurt={package_analysis['n_hurt']}, unchanged={package_analysis['n_unchanged']}, "
        f"mean_rank_delta={package_analysis['mean_rank_delta_lp_minus_fusion']:.3f}, "
        f"mean_mrr_delta={package_analysis['mean_mrr_delta_fusion_minus_lp']:.6f}"
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
