"""
Evaluate the stronger dual-view encoder.
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

from models.projector import SimCLRProjector
from roads_hier_stage1.dual_view_protocol import build_fixed_budget_dual_views, infer_line_subtype
from roads_hier_stage1.enhanced_signature_model import EnhancedRoadsHierarchicalEncoder
from roads_hier_stage1.fold_utils import split_indices_kfold
from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset
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
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    file_ids: List[str],
    subtypes: List[str],
) -> Dict[str, float]:
    recalls_1: List[float] = []
    recalls_5: List[float] = []
    reciprocal_ranks: List[float] = []

    for idx, query in enumerate(query_embeddings):
        candidate_indices = [j for j, subtype in enumerate(subtypes) if subtype == subtypes[idx]]
        scores = [(j, cosine_similarity(query, gallery_embeddings[j])) for j in candidate_indices]
        scores.sort(key=lambda item: item[1], reverse=True)
        ranked_ids = [file_ids[j] for j, _ in scores]
        target_id = file_ids[idx]
        rank = ranked_ids.index(target_id) + 1
        recalls_1.append(float(rank <= 1))
        recalls_5.append(float(rank <= 5))
        reciprocal_ranks.append(1.0 / float(rank))

    return {
        "recall_at_1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
        "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


class EnhancedDualViewEvaluator:
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate stronger dual-view encoder.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/roads_hier_stage1_dual_view_enhanced.yaml",
        help="Config path.",
    )
    parser.add_argument("--cache_root", type=str, default=None, help="Override cache root.")
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="eval",
        choices=["train", "eval", "all"],
        help="Dataset mode used for tile selection.",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Maximum files to evaluate.")
    parser.add_argument("--max_tiles", type=int, default=None, help="Override cache read max tiles.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--num_folds", type=int, default=None, help="Use k-fold split with this many folds.")
    parser.add_argument("--fold_index", type=int, default=None, help="Evaluate this validation fold only.")
    parser.add_argument(
        "--prehash_space",
        type=str,
        default="encoder",
        choices=["encoder", "projector"],
        help="Whether to evaluate encoder or projector space.",
    )
    parser.add_argument("--feature_noise_std", type=float, default=0.0, help="Optional eval-time feature noise.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
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
    k_chunks = int(protocol_cfg["k_chunks_per_tile"])
    depth_template = {int(key): int(value) for key, value in protocol_cfg["depth_template"].items()}

    dataset = GeometryCoreHierDataset(
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
    evaluator = EnhancedDualViewEvaluator(args.checkpoint, device=args.device)

    query_embeddings: List[np.ndarray] = []
    gallery_embeddings: List[np.ndarray] = []
    file_ids: List[str] = []
    file_paths: List[str] = []
    subtypes: List[str] = []
    tile_overlaps: List[float] = []
    chunk_overlaps: List[float] = []
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []
    genuine_rows: List[Dict[str, Any]] = []

    for idx, dataset_idx in enumerate(eval_indices):
        sample = dataset[dataset_idx]
        dual_views = build_fixed_budget_dual_views(
            sample,
            n_tiles=n_tiles,
            k_chunks=k_chunks,
            depth_template=depth_template,
            rng_a=np.random.default_rng(config["device"]["seed"] + idx * 2),
            rng_b=np.random.default_rng(config["device"]["seed"] + idx * 2 + 1),
            feature_noise_std=float(args.feature_noise_std),
            device=evaluator.device,
        )
        query = evaluator.encode_view(dual_views["view_a"], prehash_space=args.prehash_space)
        gallery = evaluator.encode_view(dual_views["view_b"], prehash_space=args.prehash_space)
        score = cosine_similarity(query, gallery)
        ber = cosine_to_ber(score)

        query_embeddings.append(query)
        gallery_embeddings.append(gallery)
        file_ids.append(str(sample["file_id"]))
        file_paths.append(str(sample["file_path"]))
        subtype = infer_line_subtype(sample["file_path"])
        subtypes.append(subtype)
        tile_overlaps.append(float(dual_views["tile_overlap_ratio"]))
        chunk_overlaps.append(float(dual_views["chunk_overlap_ratio"]))
        genuine_scores.append(score)
        genuine_bers.append(ber)
        genuine_rows.append(
            {
                "file_id": sample["file_id"],
                "file_path": sample["file_path"],
                "subtype": subtype,
                "prehash_score": score,
                "ber": ber,
                "tile_overlap_ratio": float(dual_views["tile_overlap_ratio"]),
                "chunk_overlap_ratio": float(dual_views["chunk_overlap_ratio"]),
            }
        )

    query_matrix = np.asarray(query_embeddings, dtype=np.float64)
    gallery_matrix = np.asarray(gallery_embeddings, dtype=np.float64)

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
            "checkpoint": args.checkpoint,
            "config": args.config,
            "cache_root": config["data"]["cache_root"],
            "dataset_mode": args.dataset_mode,
            "max_files": args.max_files,
            "max_tiles": config["data"].get("max_tiles"),
            "prehash_space": args.prehash_space,
            "feature_noise_std": args.feature_noise_std,
            "device": args.device,
            "seed": config["device"]["seed"],
            "num_folds": args.num_folds,
            "fold_index": args.fold_index,
            "protocol": {
                "n_tiles": n_tiles,
                "k_chunks_per_tile": k_chunks,
                "depth_template": depth_template,
            },
        },
        "n_files": num_files,
        "subtype_counts": {
            subtype: subtypes.count(subtype)
            for subtype in sorted(set(subtypes))
        },
        "n_genuine_pairs": len(genuine_rows),
        "n_impostor_pairs": len(impostor_rows),
        "pairwise_metrics": pairwise_metrics,
        "retrieval_metrics": retrieval_metrics,
        "overlap_metrics": {
            "tile_overlap_mean": float(np.mean(tile_overlaps)) if tile_overlaps else 0.0,
            "tile_overlap_p95": float(np.percentile(tile_overlaps, 95)) if tile_overlaps else 0.0,
            "chunk_overlap_mean": float(np.mean(chunk_overlaps)) if chunk_overlaps else 0.0,
            "chunk_overlap_p95": float(np.percentile(chunk_overlaps, 95)) if chunk_overlaps else 0.0,
        },
        "genuine_examples": genuine_rows[:10],
        "hard_impostors": sorted(impostor_rows, key=lambda row: row["ber"])[:10],
    }

    print("=" * 90)
    print("Roads Hier Stage1 Dual-View Enhanced Evaluation")
    print("=" * 90)
    print(f"Files: {num_files}")
    print(f"Genuine pairs: {len(genuine_rows)}")
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
        f"chunk_mean={report['overlap_metrics']['chunk_overlap_mean']:.6f}"
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
