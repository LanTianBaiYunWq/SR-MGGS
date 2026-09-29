"""
Audit per-query retrieval errors for one dual-view fold.
Compare the enhanced model against the visible coarse baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roads_hier_stage1.dual_view_protocol import (
    build_fixed_budget_dual_views,
    infer_line_subtype,
    visible_chunk_stats_from_tile_features,
)
from roads_hier_stage1.enhanced_signature_model import EnhancedRoadsHierarchicalEncoder
from roads_hier_stage1.fold_utils import split_indices_kfold
from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset
from train_roads_hier_stage1_minimal import load_config, resolve_path
from models.projector import SimCLRProjector
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


class EnhancedFoldEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location=self.device)
        model_cfg = checkpoint["config"]["model"]
        protocol_cfg = checkpoint["config"]["protocol"]
        max_depth = max(int(key) for key in protocol_cfg["depth_template"].keys())

        self.encoder = EnhancedRoadsHierarchicalEncoder(
            chunk_feature_dim=int(model_cfg.get("chunk_feature_dim", 10)),
            hidden_dim=int(model_cfg.get("hidden_dim", 192)),
            file_dim=int(model_cfg.get("file_dim", 192)),
            n_tiles=int(protocol_cfg["n_tiles"]),
            max_depth=max_depth,
            chunk_num_heads=int(model_cfg.get("chunk_num_heads", 4)),
            file_num_heads=int(model_cfg.get("file_num_heads", 4)),
            num_chunk_layers=int(model_cfg.get("num_chunk_layers", 2)),
            num_file_layers=int(model_cfg.get("num_file_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        ).to(self.device)
        self.projector = SimCLRProjector(
            input_dim=int(model_cfg.get("file_dim", 192)),
            hidden_dim=int(model_cfg.get("file_dim", 192)),
            output_dim=int(model_cfg.get("projector_dim", 128)),
        ).to(self.device)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.projector.load_state_dict(checkpoint["projector"])
        self.encoder.eval()
        self.projector.eval()

    @torch.no_grad()
    def encode_view(self, dual_view: Any, prehash_space: str) -> np.ndarray:
        file_embedding = self.encoder(dual_view.tile_feature_list, dual_view.tile_depths)
        if prehash_space == "projector":
            file_embedding = self.projector(file_embedding.unsqueeze(0))[0]
        return file_embedding.detach().cpu().numpy()


def rank_target(
    query: np.ndarray,
    gallery_matrix: np.ndarray,
    candidate_indices: List[int],
    file_ids: List[str],
    target_id: str,
    top_k: int,
) -> Dict[str, Any]:
    scored = [(j, cosine_similarity(query, gallery_matrix[j])) for j in candidate_indices]
    scored.sort(key=lambda item: item[1], reverse=True)
    ranked_ids = [file_ids[j] for j, _ in scored]
    rank = ranked_ids.index(target_id) + 1
    return {
        "rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_5": rank <= 5,
        "top_candidates": [
            {"file_id": file_ids[j], "score": float(score)}
            for j, score in scored[:top_k]
        ],
    }


def summarize_bucket(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    model_ranks = [int(row["model_rank"]) for row in rows]
    baseline_ranks = [int(row["baseline_rank"]) for row in rows]
    return {
        "count": len(rows),
        "model_rank_mean": float(np.mean(model_ranks)),
        "baseline_rank_mean": float(np.mean(baseline_ranks)),
        "tile_overlap_mean": float(np.mean([float(row["tile_overlap_ratio"]) for row in rows])),
        "chunk_overlap_mean": float(np.mean([float(row["chunk_overlap_ratio"]) for row in rows])),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit fold-level retrieval errors for enhanced model vs visible coarse baseline.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Enhanced model checkpoint path.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/roads_hier_stage1_dual_view_enhanced.yaml",
        help="Enhanced model config path.",
    )
    parser.add_argument("--cache_root", type=str, default=None, help="Override cache root.")
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--max_files", type=int, default=None, help="Maximum files to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_folds", type=int, required=True, help="Number of folds.")
    parser.add_argument("--fold_index", type=int, required=True, help="Validation fold index.")
    parser.add_argument(
        "--prehash_space",
        type=str,
        default="encoder",
        choices=["encoder", "projector"],
        help="Enhanced model space to audit.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="How many candidates to store per query.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cache_root:
        config["data"]["cache_root"] = args.cache_root
    if args.max_files is not None:
        config["data"]["max_files"] = args.max_files
    config["device"]["seed"] = args.seed

    set_seed(config["device"]["seed"])
    protocol_cfg = config["protocol"]
    n_tiles = int(protocol_cfg["n_tiles"])
    k_chunks = int(protocol_cfg["k_chunks_per_tile"])
    depth_template = {int(key): int(value) for key, value in protocol_cfg["depth_template"].items()}

    dataset = GeometryCoreHierDataset(
        config["data"]["cache_root"],
        mode=args.dataset_mode,
        max_tiles=config["data"].get("max_tiles"),
        tile_selector=config["data"].get("tile_selector", "manifest_default"),
    )
    total_files = len(dataset) if args.max_files is None else min(len(dataset), args.max_files)
    _, eval_indices = split_indices_kfold(
        total_files,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        seed=config["device"]["seed"],
    )

    model = EnhancedFoldEvaluator(args.checkpoint)

    model_query_rows: List[np.ndarray] = []
    model_gallery_rows: List[np.ndarray] = []
    coarse_query_rows: List[np.ndarray] = []
    coarse_gallery_rows: List[np.ndarray] = []
    file_ids: List[str] = []
    file_paths: List[str] = []
    subtypes: List[str] = []
    tile_overlaps: List[float] = []
    chunk_overlaps: List[float] = []

    for idx, dataset_idx in enumerate(eval_indices):
        sample = dataset[dataset_idx]
        dual_views = build_fixed_budget_dual_views(
            sample,
            n_tiles=n_tiles,
            k_chunks=k_chunks,
            depth_template=depth_template,
            rng_a=np.random.default_rng(config["device"]["seed"] + idx * 2),
            rng_b=np.random.default_rng(config["device"]["seed"] + idx * 2 + 1),
            feature_noise_std=0.0,
            device=model.device,
        )

        model_query_rows.append(model.encode_view(dual_views["view_a"], prehash_space=args.prehash_space))
        model_gallery_rows.append(model.encode_view(dual_views["view_b"], prehash_space=args.prehash_space))

        _, coarse_a = visible_chunk_stats_from_tile_features(dual_views["view_a"].tile_feature_list)
        _, coarse_b = visible_chunk_stats_from_tile_features(dual_views["view_b"].tile_feature_list)
        coarse_query_rows.append(np.asarray(coarse_a, dtype=np.float64))
        coarse_gallery_rows.append(np.asarray(coarse_b, dtype=np.float64))

        file_ids.append(str(sample["file_id"]))
        file_paths.append(str(sample["file_path"]))
        subtypes.append(infer_line_subtype(sample["file_path"]))
        tile_overlaps.append(float(dual_views["tile_overlap_ratio"]))
        chunk_overlaps.append(float(dual_views["chunk_overlap_ratio"]))

    model_query = np.asarray(model_query_rows, dtype=np.float64)
    model_gallery = np.asarray(model_gallery_rows, dtype=np.float64)
    coarse_query = np.asarray(coarse_query_rows, dtype=np.float64)
    coarse_gallery = np.asarray(coarse_gallery_rows, dtype=np.float64)

    stacked = np.vstack([coarse_query, coarse_gallery])
    mean = np.mean(stacked, axis=0, keepdims=True)
    std = np.std(stacked, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    coarse_query = (coarse_query - mean) / std
    coarse_gallery = (coarse_gallery - mean) / std

    rows: List[Dict[str, Any]] = []
    model_only: List[Dict[str, Any]] = []
    baseline_only: List[Dict[str, Any]] = []
    both_correct: List[Dict[str, Any]] = []
    both_wrong: List[Dict[str, Any]] = []

    for idx in range(len(file_ids)):
        candidate_indices = [j for j, subtype in enumerate(subtypes) if subtype == subtypes[idx]]
        target_id = file_ids[idx]
        model_rank_info = rank_target(
            model_query[idx], model_gallery, candidate_indices, file_ids, target_id, args.top_k
        )
        baseline_rank_info = rank_target(
            coarse_query[idx], coarse_gallery, candidate_indices, file_ids, target_id, args.top_k
        )

        row = {
            "file_id": file_ids[idx],
            "file_path": file_paths[idx],
            "subtype": subtypes[idx],
            "model_rank": model_rank_info["rank"],
            "baseline_rank": baseline_rank_info["rank"],
            "model_hit_at_1": model_rank_info["hit_at_1"],
            "baseline_hit_at_1": baseline_rank_info["hit_at_1"],
            "model_hit_at_5": model_rank_info["hit_at_5"],
            "baseline_hit_at_5": baseline_rank_info["hit_at_5"],
            "tile_overlap_ratio": tile_overlaps[idx],
            "chunk_overlap_ratio": chunk_overlaps[idx],
            "model_top_candidates": model_rank_info["top_candidates"],
            "baseline_top_candidates": baseline_rank_info["top_candidates"],
        }
        rows.append(row)

        if row["model_hit_at_1"] and not row["baseline_hit_at_1"]:
            model_only.append(row)
        elif row["baseline_hit_at_1"] and not row["model_hit_at_1"]:
            baseline_only.append(row)
        elif row["model_hit_at_1"] and row["baseline_hit_at_1"]:
            both_correct.append(row)
        else:
            both_wrong.append(row)

    by_subtype: Dict[str, Dict[str, Any]] = {}
    for subtype in sorted(set(subtypes)):
        subtype_rows = [row for row in rows if row["subtype"] == subtype]
        by_subtype[subtype] = {
            "all": summarize_bucket(subtype_rows),
            "model_only_top1": summarize_bucket([row for row in model_only if row["subtype"] == subtype]),
            "baseline_only_top1": summarize_bucket([row for row in baseline_only if row["subtype"] == subtype]),
            "both_correct": summarize_bucket([row for row in both_correct if row["subtype"] == subtype]),
            "both_wrong": summarize_bucket([row for row in both_wrong if row["subtype"] == subtype]),
        }

    report = {
        "arguments": {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "cache_root": config["data"]["cache_root"],
            "dataset_mode": args.dataset_mode,
            "max_files": args.max_files,
            "seed": args.seed,
            "num_folds": args.num_folds,
            "fold_index": args.fold_index,
            "prehash_space": args.prehash_space,
            "protocol": {
                "n_tiles": n_tiles,
                "k_chunks_per_tile": k_chunks,
                "depth_template": depth_template,
            },
        },
        "n_files": len(rows),
        "agreement_summary": {
            "both_correct": summarize_bucket(both_correct),
            "both_wrong": summarize_bucket(both_wrong),
            "model_only_top1": summarize_bucket(model_only),
            "baseline_only_top1": summarize_bucket(baseline_only),
        },
        "by_subtype": by_subtype,
        "model_only_examples": sorted(model_only, key=lambda row: (row["baseline_rank"], -row["model_rank"]))[:10],
        "baseline_only_examples": sorted(baseline_only, key=lambda row: (row["model_rank"], -row["baseline_rank"]))[:10],
        "both_wrong_examples": sorted(both_wrong, key=lambda row: (row["model_rank"] + row["baseline_rank"]))[:10],
        "all_queries": rows,
    }

    print("=" * 90)
    print("Dual-View Fold Error Audit")
    print("=" * 90)
    print(f"Files: {len(rows)}")
    print(f"Both correct: {len(both_correct)}")
    print(f"Model-only top1: {len(model_only)}")
    print(f"Baseline-only top1: {len(baseline_only)}")
    print(f"Both wrong: {len(both_wrong)}")
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)
        print(f"Saved audit to: {output_path}")


if __name__ == "__main__":
    main()
