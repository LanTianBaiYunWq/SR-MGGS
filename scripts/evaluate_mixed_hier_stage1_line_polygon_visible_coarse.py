"""Evaluate mixed V1 line+polygon visible coarse baseline."""

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

from mixed_hier_stage1.line_polygon_dataset import MixedLinePolygonPackageDataset
from mixed_hier_stage1.line_polygon_protocol import build_partial_package_views
from polygon_hier_stage1.dual_view_protocol import visible_polygon_stats_from_tile_features
from roads_hier_stage1.dual_view_protocol import visible_chunk_stats_from_tile_features
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
    return float((np.mean(impostor_bers) - np.mean(genuine_bers)) / (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8))


def compute_retrieval_metrics(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, package_ids: List[str]) -> Dict[str, float]:
    recalls_1: List[float] = []
    recalls_5: List[float] = []
    reciprocal_ranks: List[float] = []
    for idx, query in enumerate(query_embeddings):
        scores = [(j, cosine_similarity(query, gallery_embeddings[j])) for j in range(len(gallery_embeddings))]
        scores.sort(key=lambda item: item[1], reverse=True)
        ranked_ids = [package_ids[j] for j, _ in scores]
        rank = ranked_ids.index(package_ids[idx]) + 1
        recalls_1.append(float(rank <= 1))
        recalls_5.append(float(rank <= 5))
        reciprocal_ranks.append(1.0 / float(rank))
    return {
        "recall_at_1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
        "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mixed V1 line+polygon visible coarse baseline.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon.yaml")
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--max_packages", type=int, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser


def _package_visible_vector(
    views: Dict[str, object],
    *,
    line_subtypes: List[str],
    polygon_subtypes: List[str],
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    line_query_parts: List[np.ndarray] = []
    line_gallery_parts: List[np.ndarray] = []
    polygon_query_parts: List[np.ndarray] = []
    polygon_gallery_parts: List[np.ndarray] = []
    line_dim: int | None = None
    polygon_dim: int | None = None

    for subtype in line_subtypes:
        if subtype in views["line_view_a"]:
            _, feat_a = visible_chunk_stats_from_tile_features(views["line_view_a"][subtype].tile_feature_list)
            line_dim = int(feat_a.shape[0])
            line_query_parts.append(feat_a.astype(np.float64, copy=False))
        else:
            if line_dim is None:
                for candidate in views["line_view_a"].values():
                    _, probe = visible_chunk_stats_from_tile_features(candidate.tile_feature_list)
                    line_dim = int(probe.shape[0])
                    break
                if line_dim is None:
                    raise ValueError("Unable to infer line visible feature dimension.")
            line_query_parts.append(np.zeros(line_dim, dtype=np.float64))
        if subtype in views["line_view_b"]:
            _, feat_b = visible_chunk_stats_from_tile_features(views["line_view_b"][subtype].tile_feature_list)
            line_dim = int(feat_b.shape[0])
            line_gallery_parts.append(feat_b.astype(np.float64, copy=False))
        else:
            if line_dim is None:
                for candidate in views["line_view_b"].values():
                    _, probe = visible_chunk_stats_from_tile_features(candidate.tile_feature_list)
                    line_dim = int(probe.shape[0])
                    break
                if line_dim is None:
                    raise ValueError("Unable to infer line visible feature dimension.")
            line_gallery_parts.append(np.zeros(line_dim, dtype=np.float64))

    for subtype in polygon_subtypes:
        if subtype in views["polygon_view_a"]:
            _, feat_a = visible_polygon_stats_from_tile_features(views["polygon_view_a"][subtype].tile_feature_list)
            polygon_dim = int(feat_a.shape[0])
            polygon_query_parts.append(feat_a.astype(np.float64, copy=False))
        else:
            if polygon_dim is None:
                for candidate in views["polygon_view_a"].values():
                    _, probe = visible_polygon_stats_from_tile_features(candidate.tile_feature_list)
                    polygon_dim = int(probe.shape[0])
                    break
                if polygon_dim is None:
                    raise ValueError("Unable to infer polygon visible feature dimension.")
            polygon_query_parts.append(np.zeros(polygon_dim, dtype=np.float64))
        if subtype in views["polygon_view_b"]:
            _, feat_b = visible_polygon_stats_from_tile_features(views["polygon_view_b"][subtype].tile_feature_list)
            polygon_dim = int(feat_b.shape[0])
            polygon_gallery_parts.append(feat_b.astype(np.float64, copy=False))
        else:
            if polygon_dim is None:
                for candidate in views["polygon_view_b"].values():
                    _, probe = visible_polygon_stats_from_tile_features(candidate.tile_feature_list)
                    polygon_dim = int(probe.shape[0])
                    break
                if polygon_dim is None:
                    raise ValueError("Unable to infer polygon visible feature dimension.")
            polygon_gallery_parts.append(np.zeros(polygon_dim, dtype=np.float64))

    overlap = {
        "line_subtype": float(views["line_subtype_overlap"]),
        "line_tile": float(views["line_tile_overlap"]),
        "line_chunk": float(views["line_chunk_overlap"]),
        "polygon_subtype": float(views["polygon_subtype_overlap"]),
        "polygon_tile": float(views["polygon_tile_overlap"]),
        "polygon": float(views["polygon_overlap"]),
    }
    return (
        np.concatenate(line_query_parts + polygon_query_parts, axis=0),
        np.concatenate(line_gallery_parts + polygon_gallery_parts, axis=0),
        overlap,
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    if args.max_packages is not None:
        config["data"]["max_packages"] = int(args.max_packages)
    if args.seed is not None:
        config["device"]["seed"] = int(args.seed)

    set_seed(config["device"]["seed"])
    dataset = MixedLinePolygonPackageDataset(
        config["data"]["line_cache_root"],
        config["data"]["polygon_cache_root"],
        mode=args.dataset_mode,
        line_max_tiles=config["data"].get("line_max_tiles"),
        polygon_max_tiles=config["data"].get("polygon_max_tiles"),
        line_tile_selector=config["data"].get("line_tile_selector", "manifest_default"),
        polygon_tile_selector=config["data"].get("polygon_tile_selector", "manifest_default"),
        line_subtypes=config["data"].get("line_subtypes", ("roads", "railways", "waterways")),
        polygon_subtypes=config["data"].get("polygon_subtypes", ("building", "landuse", "natural", "water")),
        max_packages=config["data"].get("max_packages"),
        require_complete_packages=bool(config["data"].get("require_complete_packages", True)),
    )

    line_subtypes = list(config["data"].get("line_subtypes", ("roads", "railways", "waterways")))
    polygon_subtypes = list(config["data"].get("polygon_subtypes", ("building", "landuse", "natural", "water")))
    line_subtypes_per_view = config["data"].get("line_subtypes_per_view")
    polygon_subtypes_per_view = config["data"].get("polygon_subtypes_per_view")
    line_protocol = config["line_protocol"]
    polygon_protocol = config["polygon_protocol"]

    query_rows: List[np.ndarray] = []
    gallery_rows: List[np.ndarray] = []
    package_ids: List[str] = []
    line_subtype_overlaps: List[float] = []
    line_tile_overlaps: List[float] = []
    line_chunk_overlaps: List[float] = []
    polygon_subtype_overlaps: List[float] = []
    polygon_tile_overlaps: List[float] = []
    polygon_overlaps: List[float] = []
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []

    for idx in range(len(dataset)):
        package_sample = dataset[idx]
        views = build_partial_package_views(
            package_sample,
            line_subtypes=line_subtypes,
            polygon_subtypes=polygon_subtypes,
            line_subtypes_per_view=line_subtypes_per_view,
            polygon_subtypes_per_view=polygon_subtypes_per_view,
            line_n_tiles=int(line_protocol["n_tiles"]),
            line_k_chunks=int(line_protocol["k_chunks_per_tile"]),
            line_depth_template={int(key): int(value) for key, value in line_protocol["depth_template"].items()},
            polygon_n_tiles=int(polygon_protocol["n_tiles"]),
            polygon_k_polygons=int(polygon_protocol["k_polygons_per_tile"]),
            polygon_depth_template={int(key): int(value) for key, value in polygon_protocol["depth_template"].items()},
            rng_a=np.random.default_rng(int(config["device"]["seed"]) + idx * 101),
            rng_b=np.random.default_rng(int(config["device"]["seed"]) + idx * 101 + 1),
            feature_noise_std=0.0,
            device=None,
        )
        query_vec, gallery_vec, overlap = _package_visible_vector(
            views,
            line_subtypes=line_subtypes,
            polygon_subtypes=polygon_subtypes,
        )
        query_rows.append(query_vec)
        gallery_rows.append(gallery_vec)
        package_ids.append(str(package_sample["package_id"]))
        line_subtype_overlaps.append(overlap["line_subtype"])
        line_tile_overlaps.append(overlap["line_tile"])
        line_chunk_overlaps.append(overlap["line_chunk"])
        polygon_subtype_overlaps.append(overlap["polygon_subtype"])
        polygon_tile_overlaps.append(overlap["polygon_tile"])
        polygon_overlaps.append(overlap["polygon"])

    query_matrix = np.vstack(query_rows)
    gallery_matrix = np.vstack(gallery_rows)
    stacked = np.vstack([query_matrix, gallery_matrix])
    mean = np.mean(stacked, axis=0, keepdims=True)
    std = np.std(stacked, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    query_matrix = (query_matrix - mean) / std
    gallery_matrix = (gallery_matrix - mean) / std

    for idx in range(len(package_ids)):
        score = cosine_similarity(query_matrix[idx], gallery_matrix[idx])
        genuine_scores.append(score)
        genuine_bers.append(cosine_to_ber(score))

    impostor_scores: List[float] = []
    impostor_bers: List[float] = []
    for i in range(len(package_ids)):
        for j in range(len(package_ids)):
            if i == j:
                continue
            score = cosine_similarity(query_matrix[i], gallery_matrix[j])
            impostor_scores.append(score)
            impostor_bers.append(cosine_to_ber(score))

    genuine_scores_np = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores_np = np.asarray(impostor_scores, dtype=np.float64)
    genuine_bers_np = np.asarray(genuine_bers, dtype=np.float64)
    impostor_bers_np = np.asarray(impostor_bers, dtype=np.float64)
    retrieval_metrics = compute_retrieval_metrics(query_matrix, gallery_matrix, package_ids)
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
            "line_cache_root": config["data"]["line_cache_root"],
            "polygon_cache_root": config["data"]["polygon_cache_root"],
            "dataset_mode": args.dataset_mode,
            "max_packages": config["data"].get("max_packages"),
            "seed": config["device"]["seed"],
        },
        "n_packages": len(package_ids),
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
        "n_features": int(query_matrix.shape[1]) if len(query_matrix) else 0,
        "pairwise_metrics": pairwise_metrics,
        "retrieval_metrics": retrieval_metrics,
        "overlap_metrics": {
            "line_subtype_mean": float(np.mean(line_subtype_overlaps)) if line_subtype_overlaps else 0.0,
            "line_tile_mean": float(np.mean(line_tile_overlaps)) if line_tile_overlaps else 0.0,
            "line_chunk_mean": float(np.mean(line_chunk_overlaps)) if line_chunk_overlaps else 0.0,
            "polygon_subtype_mean": float(np.mean(polygon_subtype_overlaps)) if polygon_subtype_overlaps else 0.0,
            "polygon_tile_mean": float(np.mean(polygon_tile_overlaps)) if polygon_tile_overlaps else 0.0,
            "polygon_mean": float(np.mean(polygon_overlaps)) if polygon_overlaps else 0.0,
        },
    }

    print("=" * 90)
    print("Mixed Hier Stage1 Line+Polygon Visible Coarse Baseline")
    print("=" * 90)
    print(f"Packages: {report['n_packages']}")
    print(f"Genuine pairs: {report['n_genuine_pairs']}")
    print(f"Impostor pairs: {report['n_impostor_pairs']}")
    print(f"Pairwise: AUC={pairwise_metrics['auc']:.6f}, EER={pairwise_metrics['eer']:.6f}, Separability={pairwise_metrics['separability']:.6f}")
    print(f"Retrieval: R@1={retrieval_metrics['recall_at_1']:.6f}, R@5={retrieval_metrics['recall_at_5']:.6f}, MRR={retrieval_metrics['mrr']:.6f}")
    print(
        f"Overlap: line_subtype_mean={report['overlap_metrics']['line_subtype_mean']:.6f}, "
        f"line_tile_mean={report['overlap_metrics']['line_tile_mean']:.6f}, "
        f"line_chunk_mean={report['overlap_metrics']['line_chunk_mean']:.6f}, "
        f"polygon_subtype_mean={report['overlap_metrics']['polygon_subtype_mean']:.6f}, "
        f"polygon_tile_mean={report['overlap_metrics']['polygon_tile_mean']:.6f}, "
        f"polygon_mean={report['overlap_metrics']['polygon_mean']:.6f}"
    )
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
