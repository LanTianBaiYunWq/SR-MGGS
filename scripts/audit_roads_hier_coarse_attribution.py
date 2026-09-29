"""
Audit which coarse-stat feature groups dominate the roads_hier_stage1 baseline.

Outputs:
1. Full-feature baseline metrics.
2. Single-feature metrics ranked by AUC.
3. Feature-group metrics.
4. Leave-one-group-out metrics.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from roads_hier_stage1.hard_negative_mining import (
    build_coarse_matched_pairs,
    build_size_matched_pairs,
    build_subtype_matched_pairs,
    build_subtype_size_matched_pairs,
    collect_impostor_pairs,
    load_cache_feature_rows,
    resolve_path,
)
from utils.metrics import compute_auc, compute_eer


def to_jsonable(obj: Any) -> Any:
    """Convert numpy-heavy structures into JSON-safe objects."""
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Define CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit coarse-stat attribution on roads hierarchical cache.")
    parser.add_argument("--cache_root", type=str, required=True, help="Gate 1 cache root.")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files.")
    parser.add_argument(
        "--genuine_jitter_std",
        type=float,
        default=0.01,
        help="Noise std used to form same-file genuine pairs.",
    )
    parser.add_argument("--max_impostor_pairs", type=int, default=None, help="Maximum impostor pairs.")
    parser.add_argument(
        "--negative_mode",
        type=str,
        default="subtype_size_matched",
        choices=["all", "size_matched", "coarse_matched", "subtype_matched", "subtype_size_matched"],
        help="How to construct impostor pairs.",
    )
    parser.add_argument("--hard_negative_topk", type=int, default=2, help="Top-k impostors per file for matched modes.")
    parser.add_argument("--match_ratio_tol", type=float, default=0.25, help="Coarse-matched ratio tolerance.")
    parser.add_argument("--match_budget_tol", type=float, default=0.10, help="Coarse-matched budget tolerance.")
    parser.add_argument("--match_diag_tol", type=float, default=0.25, help="Coarse-matched bbox diagonal tolerance.")
    parser.add_argument("--top_single", type=int, default=10, help="How many top single features to print/save.")
    parser.add_argument("--top_groups", type=int, default=10, help="How many top groups to print/save.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity."""
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
    return float(np.dot(x, y) / denom)


def cosine_to_ber(score: float) -> float:
    """Map cosine score to BER-like distance."""
    return float((1.0 - score) / 2.0)


def compute_separability(genuine_bers: np.ndarray, impostor_bers: np.ndarray) -> float:
    """Same separability definition used elsewhere in this repo."""
    return float(
        (np.mean(impostor_bers) - np.mean(genuine_bers))
        / (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8)
    )


def select_impostor_pairs(
    cache_root: str,
    *,
    num_files: int,
    negative_mode: str,
    hard_negative_topk: int,
    max_impostor_pairs: int | None,
    match_ratio_tol: float,
    match_budget_tol: float,
    match_diag_tol: float,
) -> List[Tuple[int, int]]:
    """Choose impostor pairs using the same logic as evaluation."""
    if negative_mode == "size_matched":
        return build_size_matched_pairs(
            cache_root,
            max_files=num_files,
            topk_per_file=hard_negative_topk,
            max_pairs=max_impostor_pairs,
        )
    if negative_mode == "subtype_matched":
        return build_subtype_matched_pairs(
            cache_root,
            max_files=num_files,
            max_pairs=max_impostor_pairs,
        )
    if negative_mode == "subtype_size_matched":
        return build_subtype_size_matched_pairs(
            cache_root,
            max_files=num_files,
            topk_per_file=hard_negative_topk,
            max_pairs=max_impostor_pairs,
        )
    if negative_mode == "coarse_matched":
        return build_coarse_matched_pairs(
            cache_root,
            max_files=num_files,
            topk_per_file=hard_negative_topk,
            max_pairs=max_impostor_pairs,
            ratio_tolerance=match_ratio_tol,
            budget_tolerance=match_budget_tol,
            diag_tolerance=match_diag_tol,
        )
    return collect_impostor_pairs(num_files, max_impostor_pairs)


def evaluate_feature_subspace(
    feature_matrix: np.ndarray,
    impostor_pairs: Sequence[Tuple[int, int]],
    *,
    genuine_jitter_std: float,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate coarse baseline using only a chosen feature subspace."""
    rng = np.random.default_rng(seed)
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []

    for feature in feature_matrix:
        jitter_a = rng.normal(0.0, genuine_jitter_std, size=feature.shape)
        jitter_b = rng.normal(0.0, genuine_jitter_std, size=feature.shape)
        feat_a = feature + jitter_a
        feat_b = feature + jitter_b
        score = cosine_similarity(feat_a, feat_b)
        genuine_scores.append(score)
        genuine_bers.append(cosine_to_ber(score))

    impostor_scores: List[float] = []
    impostor_bers: List[float] = []
    for i, j in impostor_pairs:
        score = cosine_similarity(feature_matrix[i], feature_matrix[j])
        impostor_scores.append(score)
        impostor_bers.append(cosine_to_ber(score))

    genuine_scores_np = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores_np = np.asarray(impostor_scores, dtype=np.float64)
    genuine_bers_np = np.asarray(genuine_bers, dtype=np.float64)
    impostor_bers_np = np.asarray(impostor_bers, dtype=np.float64)

    return {
        "genuine_mean": float(np.mean(genuine_bers_np)) if len(genuine_bers_np) else 0.0,
        "impostor_mean": float(np.mean(impostor_bers_np)) if len(impostor_bers_np) else 0.0,
        "auc": compute_auc(genuine_scores_np, impostor_scores_np) if len(impostor_scores_np) else 0.0,
        "eer": compute_eer(genuine_scores_np, impostor_scores_np)[0] if len(impostor_scores_np) else 0.0,
        "separability": compute_separability(genuine_bers_np, impostor_bers_np) if len(impostor_bers_np) else 0.0,
    }


def build_feature_groups(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    """Group coarse features into interpretable blocks."""
    groups: Dict[str, List[int]] = {
        "manifest_scale": [],
        "manifest_chunk_stats": [],
        "manifest_budget_compression": [],
        "tile_counts": [],
        "tile_length_density": [],
        "tile_area_depth": [],
    }
    for idx, name in enumerate(feature_names):
        if name in {"num_tiles_manifest", "num_chunks_manifest", "file_bbox_diagonal"}:
            groups["manifest_scale"].append(idx)
        elif name in {"chunk_p50_manifest", "chunk_p95_manifest", "chunk_max_manifest"}:
            groups["manifest_chunk_stats"].append(idx)
        elif name in {"compression_p50_manifest", "compression_p95_manifest", "budget_hit_ratio_manifest"}:
            groups["manifest_budget_compression"].append(idx)
        elif name in {
            "tile_count",
            "raw_line_sum",
            "raw_line_mean",
            "raw_line_p95",
            "raw_chunk_sum",
            "raw_chunk_mean",
            "raw_chunk_p95",
            "budgeted_chunk_sum",
            "budgeted_chunk_mean",
            "budgeted_chunk_p95",
        }:
            groups["tile_counts"].append(idx)
        elif name in {
            "chunk_length_sum_total",
            "chunk_length_sum_mean",
            "chunk_length_sum_p95",
            "chunk_density_mean",
            "chunk_density_p95",
            "compression_ratio_mean",
            "compression_ratio_p95",
        }:
            groups["tile_length_density"].append(idx)
        elif name in {"tile_area_sum", "tile_area_mean", "depth_mean", "depth_max"}:
            groups["tile_area_depth"].append(idx)
        else:
            raise ValueError(f"Unassigned feature name: {name}")
    return {name: idxs for name, idxs in groups.items() if idxs}


def evaluate_named_subspaces(
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    named_indices: Dict[str, List[int]],
    *,
    impostor_pairs: Sequence[Tuple[int, int]],
    genuine_jitter_std: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Evaluate many named subspaces with consistent output shape."""
    results: List[Dict[str, Any]] = []
    full_metrics = None
    for name, indices in named_indices.items():
        subspace = feature_matrix[:, indices]
        metrics = evaluate_feature_subspace(
            subspace,
            impostor_pairs,
            genuine_jitter_std=genuine_jitter_std,
            seed=seed,
        )
        results.append(
            {
                "name": name,
                "feature_names": [feature_names[i] for i in indices],
                "n_dims": len(indices),
                "metrics": metrics,
            }
        )
        if name == "__full__":
            full_metrics = metrics

    if full_metrics is not None:
        for row in results:
            row["delta_auc_vs_full"] = float(row["metrics"]["auc"] - full_metrics["auc"])
            row["delta_eer_vs_full"] = float(row["metrics"]["eer"] - full_metrics["eer"])
            row["delta_sep_vs_full"] = float(row["metrics"]["separability"] - full_metrics["separability"])
    return results


def main() -> None:
    """Script entry."""
    parser = build_argument_parser()
    args = parser.parse_args()

    cache_root = resolve_path(args.cache_root)
    feature_rows, feature_names, feature_matrix = load_cache_feature_rows(str(cache_root), args.max_files)
    num_files = len(feature_rows)
    impostor_pairs = select_impostor_pairs(
        str(cache_root),
        num_files=num_files,
        negative_mode=args.negative_mode,
        hard_negative_topk=args.hard_negative_topk,
        max_impostor_pairs=args.max_impostor_pairs,
        match_ratio_tol=args.match_ratio_tol,
        match_budget_tol=args.match_budget_tol,
        match_diag_tol=args.match_diag_tol,
    )

    feature_groups = build_feature_groups(feature_names)

    full_named = {"__full__": list(range(len(feature_names)))}
    full_metrics = evaluate_named_subspaces(
        feature_matrix,
        feature_names,
        full_named,
        impostor_pairs=impostor_pairs,
        genuine_jitter_std=args.genuine_jitter_std,
    )[0]["metrics"]

    single_feature_indices = {
        feature_names[i]: [i]
        for i in range(len(feature_names))
    }
    single_feature_results = evaluate_named_subspaces(
        feature_matrix,
        feature_names,
        {"__full__": list(range(len(feature_names))), **single_feature_indices},
        impostor_pairs=impostor_pairs,
        genuine_jitter_std=args.genuine_jitter_std,
    )
    single_feature_results = [row for row in single_feature_results if row["name"] != "__full__"]
    single_feature_results.sort(key=lambda row: row["metrics"]["auc"], reverse=True)

    group_results = evaluate_named_subspaces(
        feature_matrix,
        feature_names,
        {"__full__": list(range(len(feature_names))), **feature_groups},
        impostor_pairs=impostor_pairs,
        genuine_jitter_std=args.genuine_jitter_std,
    )
    group_results = [row for row in group_results if row["name"] != "__full__"]
    group_results.sort(key=lambda row: row["metrics"]["auc"], reverse=True)

    leave_one_group_out_indices = {
        f"without_{group_name}": [idx for idx in range(len(feature_names)) if idx not in indices]
        for group_name, indices in feature_groups.items()
    }
    leave_one_group_out_results = evaluate_named_subspaces(
        feature_matrix,
        feature_names,
        {"__full__": list(range(len(feature_names))), **leave_one_group_out_indices},
        impostor_pairs=impostor_pairs,
        genuine_jitter_std=args.genuine_jitter_std,
    )
    leave_one_group_out_results = [row for row in leave_one_group_out_results if row["name"] != "__full__"]
    leave_one_group_out_results.sort(key=lambda row: row["delta_auc_vs_full"])

    report = {
        "arguments": {
            "cache_root": str(cache_root),
            "max_files": args.max_files,
            "genuine_jitter_std": args.genuine_jitter_std,
            "max_impostor_pairs": args.max_impostor_pairs,
            "negative_mode": args.negative_mode,
            "hard_negative_topk": args.hard_negative_topk,
            "match_ratio_tol": args.match_ratio_tol,
            "match_budget_tol": args.match_budget_tol,
            "match_diag_tol": args.match_diag_tol,
        },
        "n_files": num_files,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "feature_groups": {
            name: [feature_names[i] for i in indices]
            for name, indices in feature_groups.items()
        },
        "n_impostor_pairs": len(impostor_pairs),
        "full_metrics": full_metrics,
        "top_single_features": single_feature_results[: args.top_single],
        "all_single_features": single_feature_results,
        "top_feature_groups": group_results[: args.top_groups],
        "all_feature_groups": group_results,
        "leave_one_group_out": leave_one_group_out_results,
    }

    print("=" * 90)
    print("Roads Hier Coarse Attribution Audit")
    print("=" * 90)
    print(f"Files: {num_files}")
    print(f"Impostor pairs: {len(impostor_pairs)}")
    print(
        f"Full baseline: AUC={full_metrics['auc']:.6f}, "
        f"EER={full_metrics['eer']:.6f}, "
        f"Separability={full_metrics['separability']:.6f}"
    )
    print("Top single features:")
    for row in single_feature_results[: args.top_single]:
        print(
            f"  {row['name']}: "
            f"AUC={row['metrics']['auc']:.6f}, "
            f"EER={row['metrics']['eer']:.6f}, "
            f"dAUC={row['delta_auc_vs_full']:.6f}"
        )
    print("Top feature groups:")
    for row in group_results[: args.top_groups]:
        print(
            f"  {row['name']}: "
            f"AUC={row['metrics']['auc']:.6f}, "
            f"EER={row['metrics']['eer']:.6f}, "
            f"dAUC={row['delta_auc_vs_full']:.6f}"
        )
    print("Leave-one-group-out:")
    for row in leave_one_group_out_results:
        print(
            f"  {row['name']}: "
            f"AUC={row['metrics']['auc']:.6f}, "
            f"EER={row['metrics']['eer']:.6f}, "
            f"dAUC={row['delta_auc_vs_full']:.6f}"
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
