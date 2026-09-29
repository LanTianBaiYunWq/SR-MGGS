"""Evaluate a view-visible coarse baseline for the point branch."""

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

from point_hier_stage1.dual_view_protocol import (
    build_fixed_budget_point_views,
    infer_point_subtype,
    visible_point_stats_from_tile_features,
)
from point_hier_stage1.point_hier_dataset import PointHierDataset
from roads_hier_stage1.fold_utils import split_indices_kfold
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


def compute_separability(genuine_bers: np.ndarray, impostor_bers: np.ndarray) -> float:
    return float(
        (np.mean(impostor_bers) - np.mean(genuine_bers))
        / (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8)
    )


def compute_retrieval_metrics(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    file_ids: List[str],
    subtypes: List[str],
) -> Dict[str, float]:
    recalls_1: List[float] = []
    recalls_5: List[float] = []
    reciprocal_ranks: List[float] = []

    for idx, query in enumerate(query_features):
        candidate_indices = [j for j, subtype in enumerate(subtypes) if subtype == subtypes[idx]]
        scores = [(j, cosine_similarity(query, gallery_features[j])) for j in candidate_indices]
        scores.sort(key=lambda item: item[1], reverse=True)
        ranked_ids = [file_ids[j] for j, _ in scores]
        rank = ranked_ids.index(file_ids[idx]) + 1
        recalls_1.append(float(rank <= 1))
        recalls_5.append(float(rank <= 5))
        reciprocal_ranks.append(1.0 / float(rank))

    return {
        "recall_at_1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
        "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate point visible coarse baseline under fixed-budget dual-view.")
    parser.add_argument("--config", type=str, default="configs/point_hier_stage1_dual_view.yaml")
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cache_root:
        config["data"]["cache_root"] = args.cache_root
    if args.max_tiles is not None:
        config["data"]["max_tiles"] = args.max_tiles
    if args.max_files is not None:
        config["data"]["max_files"] = args.max_files
    if args.seed is not None:
        config["device"]["seed"] = args.seed
    if args.num_folds is not None or args.fold_index is not None:
        if args.num_folds is None or args.fold_index is None:
            raise ValueError("Both --num_folds and --fold_index must be provided together.")

    set_seed(config["device"]["seed"])
    protocol_cfg = config["protocol"]
    n_tiles = int(protocol_cfg["n_tiles"])
    k_points = int(protocol_cfg["k_points_per_tile"])
    depth_template = {int(key): int(value) for key, value in protocol_cfg["depth_template"].items()}

    dataset = PointHierDataset(
        config["data"]["cache_root"],
        mode=args.dataset_mode,
        max_tiles=config["data"].get("max_tiles"),
        tile_selector=config["data"].get("tile_selector", "manifest_default"),
    )
    total_files = len(dataset) if args.max_files is None else min(len(dataset), args.max_files)
    if args.num_folds is not None:
        _, eval_indices = split_indices_kfold(
            total_files,
            num_folds=args.num_folds,
            fold_index=args.fold_index,
            seed=config["device"]["seed"],
        )
    else:
        eval_indices = list(range(total_files))
    num_files = len(eval_indices)

    query_rows: List[np.ndarray] = []
    gallery_rows: List[np.ndarray] = []
    file_ids: List[str] = []
    file_paths: List[str] = []
    subtypes: List[str] = []
    tile_overlaps: List[float] = []
    point_overlaps: List[float] = []
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []
    genuine_examples: List[Dict[str, Any]] = []
    feature_names: List[str] | None = None

    for idx, dataset_idx in enumerate(eval_indices):
        sample = dataset[dataset_idx]
        dual_views = build_fixed_budget_point_views(
            sample,
            n_tiles=n_tiles,
            k_points=k_points,
            depth_template=depth_template,
            rng_a=np.random.default_rng(config["device"]["seed"] + idx * 2),
            rng_b=np.random.default_rng(config["device"]["seed"] + idx * 2 + 1),
            feature_noise_std=0.0,
            device=None,
        )
        names_a, feat_a = visible_point_stats_from_tile_features(dual_views["view_a"].tile_feature_list)
        names_b, feat_b = visible_point_stats_from_tile_features(dual_views["view_b"].tile_feature_list)
        if feature_names is None:
            feature_names = names_a
        elif feature_names != names_a or feature_names != names_b:
            raise ValueError("Inconsistent visible coarse feature names across views.")

        query_rows.append(feat_a)
        gallery_rows.append(feat_b)
        file_ids.append(str(sample["file_id"]))
        file_paths.append(str(sample["file_path"]))
        subtype = infer_point_subtype(sample["file_path"])
        subtypes.append(subtype)
        tile_overlaps.append(float(dual_views["tile_overlap_ratio"]))
        point_overlaps.append(float(dual_views["point_overlap_ratio"]))

    query_matrix = np.vstack(query_rows)
    gallery_matrix = np.vstack(gallery_rows)
    stacked = np.vstack([query_matrix, gallery_matrix])
    mean = np.mean(stacked, axis=0, keepdims=True)
    std = np.std(stacked, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    query_matrix = (query_matrix - mean) / std
    gallery_matrix = (gallery_matrix - mean) / std

    for idx in range(num_files):
        score = cosine_similarity(query_matrix[idx], gallery_matrix[idx])
        ber = cosine_to_ber(score)
        genuine_scores.append(score)
        genuine_bers.append(ber)
        genuine_examples.append(
            {
                "file_id": file_ids[idx],
                "file_path": file_paths[idx],
                "subtype": subtypes[idx],
                "prehash_score": score,
                "ber": ber,
                "tile_overlap_ratio": tile_overlaps[idx],
                "point_overlap_ratio": point_overlaps[idx],
            }
        )

    impostor_scores: List[float] = []
    impostor_bers: List[float] = []
    impostor_rows: List[Dict[str, Any]] = []
    for i in range(num_files):
        for j in range(num_files):
            if i == j or subtypes[i] != subtypes[j]:
                continue
            score = cosine_similarity(query_matrix[i], gallery_matrix[j])
            ber = cosine_to_ber(score)
            impostor_scores.append(score)
            impostor_bers.append(ber)
            impostor_rows.append(
                {
                    "file_id_i": file_ids[i],
                    "file_id_j": file_ids[j],
                    "file_path_i": file_paths[i],
                    "file_path_j": file_paths[j],
                    "subtype": subtypes[i],
                    "prehash_score": score,
                    "ber": ber,
                }
            )

    genuine_scores_np = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores_np = np.asarray(impostor_scores, dtype=np.float64)
    genuine_bers_np = np.asarray(genuine_bers, dtype=np.float64)
    impostor_bers_np = np.asarray(impostor_bers, dtype=np.float64)
    retrieval_metrics = compute_retrieval_metrics(query_matrix, gallery_matrix, file_ids, subtypes)
    pairwise_metrics = {
        "genuine_mean": float(np.mean(genuine_bers_np)) if len(genuine_bers_np) else 0.0,
        "impostor_mean": float(np.mean(impostor_bers_np)) if len(impostor_bers_np) else 0.0,
        "auc": compute_auc(genuine_scores_np, impostor_scores_np) if len(impostor_scores_np) else 0.0,
        "eer": compute_eer(genuine_scores_np, impostor_scores_np)[0] if len(impostor_scores_np) else 0.0,
        "separability": compute_separability(genuine_bers_np, impostor_bers_np) if len(impostor_bers_np) else 0.0,
    }

    report = {
        "arguments": {
            "config": args.config,
            "cache_root": config["data"]["cache_root"],
            "dataset_mode": args.dataset_mode,
            "max_files": args.max_files,
            "max_tiles": config["data"].get("max_tiles"),
            "seed": config["device"]["seed"],
            "num_folds": args.num_folds,
            "fold_index": args.fold_index,
            "protocol": {
                "n_tiles": n_tiles,
                "k_points_per_tile": k_points,
                "depth_template": depth_template,
            },
        },
        "n_files": num_files,
        "n_features": len(feature_names or []),
        "feature_names": feature_names or [],
        "subtype_counts": {
            subtype: subtypes.count(subtype)
            for subtype in sorted(set(subtypes))
        },
        "n_genuine_pairs": len(genuine_examples),
        "n_impostor_pairs": len(impostor_rows),
        "pairwise_metrics": pairwise_metrics,
        "retrieval_metrics": retrieval_metrics,
        "overlap_metrics": {
            "tile_overlap_mean": float(np.mean(tile_overlaps)) if tile_overlaps else 0.0,
            "tile_overlap_p95": float(np.percentile(tile_overlaps, 95)) if tile_overlaps else 0.0,
            "point_overlap_mean": float(np.mean(point_overlaps)) if point_overlaps else 0.0,
            "point_overlap_p95": float(np.percentile(point_overlaps, 95)) if point_overlaps else 0.0,
        },
        "genuine_examples": genuine_examples[:10],
        "hard_impostors": sorted(impostor_rows, key=lambda row: row["ber"])[:10],
    }

    print("=" * 90)
    print("Point Hier Dual-View Visible Coarse Baseline")
    print("=" * 90)
    print(f"Files: {num_files}")
    print(f"Genuine pairs: {len(genuine_examples)}")
    print(f"Impostor pairs: {len(impostor_rows)}")
    print(
        f"Pairwise: AUC={pairwise_metrics['auc']:.6f}, "
        f"EER={pairwise_metrics['eer']:.6f}, "
        f"Separability={pairwise_metrics['separability']:.6f}"
    )
    print(
        f"Retrieval: R@1={retrieval_metrics['recall_at_1']:.6f}, "
        f"R@5={retrieval_metrics['recall_at_5']:.6f}, "
        f"MRR={retrieval_metrics['mrr']:.6f}"
    )
    print(
        f"Overlap: tile_mean={report['overlap_metrics']['tile_overlap_mean']:.6f}, "
        f"point_mean={report['overlap_metrics']['point_overlap_mean']:.6f}"
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
