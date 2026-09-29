"""Evaluate polygon branch V1 dual-view encoder."""

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
from polygon_hier_stage1.dual_view_protocol import build_fixed_budget_polygon_views, infer_polygon_subtype
from polygon_hier_stage1.polygon_hier_dataset import PolygonHierDataset
from polygon_hier_stage1.polygon_signature_model import PolygonHierarchicalEncoder
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
    return float((np.mean(impostor_bers) - np.mean(genuine_bers)) / (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8))


def compute_retrieval_metrics(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, file_ids: List[str], subtypes: List[str]) -> Dict[str, float]:
    recalls_1: List[float] = []
    recalls_5: List[float] = []
    reciprocal_ranks: List[float] = []
    for idx, query in enumerate(query_embeddings):
        candidate_indices = [j for j, subtype in enumerate(subtypes) if subtype == subtypes[idx]]
        scores = [(j, cosine_similarity(query, gallery_embeddings[j])) for j in candidate_indices]
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


class PolygonDualViewEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location=self.device)
        model_cfg = checkpoint["config"]["model"]
        protocol_cfg = checkpoint["config"]["protocol"]
        max_depth = max(int(key) for key in protocol_cfg["depth_template"].keys())
        self.encoder = PolygonHierarchicalEncoder(
            polygon_feature_dim=int(model_cfg.get("polygon_feature_dim", 10)),
            hidden_dim=int(model_cfg.get("hidden_dim", 192)),
            file_dim=int(model_cfg.get("file_dim", 192)),
            n_tiles=int(protocol_cfg["n_tiles"]),
            max_depth=max_depth,
            tile_encoder_type=str(model_cfg.get("tile_encoder_type", "set_transformer")),
            tile_num_heads=int(model_cfg.get("tile_num_heads", 4)),
            num_tile_layers=int(model_cfg.get("num_tile_layers", 2)),
            file_num_heads=int(model_cfg.get("file_num_heads", 4)),
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
    parser = argparse.ArgumentParser(description="Evaluate polygon branch V1.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/polygon_hier_stage1_dual_view.yaml")
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--feature_noise_std", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
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
    k_polygons = int(protocol_cfg["k_polygons_per_tile"])
    depth_template = {int(key): int(value) for key, value in protocol_cfg["depth_template"].items()}

    dataset = PolygonHierDataset(config["data"]["cache_root"], mode=args.dataset_mode, max_tiles=config["data"].get("max_tiles"), tile_selector=config["data"].get("tile_selector", "manifest_default"))
    total_files = len(dataset) if args.max_files is None else min(len(dataset), args.max_files)
    if args.num_folds is not None:
        _, eval_indices = split_indices_kfold(total_files, num_folds=args.num_folds, fold_index=args.fold_index, seed=config["device"]["seed"])
    else:
        eval_indices = list(range(total_files))

    evaluator = PolygonDualViewEvaluator(args.checkpoint, device=args.device)
    query_embeddings: List[np.ndarray] = []
    gallery_embeddings: List[np.ndarray] = []
    file_ids: List[str] = []
    subtypes: List[str] = []
    tile_overlaps: List[float] = []
    polygon_overlaps: List[float] = []
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []

    for idx, dataset_idx in enumerate(eval_indices):
        sample = dataset[dataset_idx]
        dual_views = build_fixed_budget_polygon_views(
            sample,
            n_tiles=n_tiles,
            k_polygons=k_polygons,
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
        subtypes.append(infer_polygon_subtype(sample["file_path"]))
        tile_overlaps.append(float(dual_views["tile_overlap_ratio"]))
        polygon_overlaps.append(float(dual_views["polygon_overlap_ratio"]))
        genuine_scores.append(score)
        genuine_bers.append(ber)

    query_matrix = np.asarray(query_embeddings, dtype=np.float64)
    gallery_matrix = np.asarray(gallery_embeddings, dtype=np.float64)
    impostor_scores: List[float] = []
    impostor_bers: List[float] = []
    for i in range(len(eval_indices)):
        for j in range(len(eval_indices)):
            if i == j or subtypes[i] != subtypes[j]:
                continue
            score = cosine_similarity(query_matrix[i], gallery_matrix[j])
            impostor_scores.append(score)
            impostor_bers.append(cosine_to_ber(score))

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
            "protocol": {"n_tiles": n_tiles, "k_polygons_per_tile": k_polygons, "depth_template": depth_template},
        },
        "n_files": len(eval_indices),
        "subtype_counts": {subtype: subtypes.count(subtype) for subtype in sorted(set(subtypes))},
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
        "pairwise_metrics": pairwise_metrics,
        "retrieval_metrics": retrieval_metrics,
        "overlap_metrics": {
            "tile_mean": float(np.mean(tile_overlaps)) if tile_overlaps else 0.0,
            "polygon_mean": float(np.mean(polygon_overlaps)) if polygon_overlaps else 0.0,
        },
    }

    print("=" * 90)
    print("Polygon Hier Stage1 Dual-View Evaluation")
    print("=" * 90)
    print(f"Files: {report['n_files']}")
    print(f"Genuine pairs: {report['n_genuine_pairs']}")
    print(f"Impostor pairs: {report['n_impostor_pairs']}")
    print(f"Pairwise: AUC={pairwise_metrics['auc']:.6f}, EER={pairwise_metrics['eer']:.6f}, Separability={pairwise_metrics['separability']:.6f}")
    print(f"Retrieval: R@1={retrieval_metrics['recall_at_1']:.6f}, R@5={retrieval_metrics['recall_at_5']:.6f}, MRR={retrieval_metrics['mrr']:.6f}")
    print(f"Overlap: tile_mean={report['overlap_metrics']['tile_mean']:.6f}, polygon_mean={report['overlap_metrics']['polygon_mean']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
