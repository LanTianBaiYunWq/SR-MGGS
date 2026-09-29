"""Evaluate mixed prototype line + polygon + point fusion model."""

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

from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonPointPackageDataset
from mixed_hier_stage1.line_polygon_point_fusion_model import MixedLinePolygonPointFusionEncoder
from mixed_hier_stage1.line_polygon_point_protocol import build_partial_line_polygon_point_views
from models.projector import SimCLRProjector
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


class MixedLinePolygonPointEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location=self.device)
        cfg = checkpoint["config"]
        line_protocol = cfg["line_protocol"]
        polygon_protocol = cfg["polygon_protocol"]
        point_protocol = cfg["point_protocol"]
        line_model = cfg["line_model"]
        polygon_model = cfg["polygon_model"]
        point_model = cfg["point_model"]
        fusion_model = cfg["fusion_model"]
        data_cfg = cfg["data"]

        self.line_subtypes = tuple(data_cfg.get("line_subtypes", ("roads", "railways", "waterways")))
        self.polygon_subtypes = tuple(data_cfg.get("polygon_subtypes", ("building", "landuse", "natural")))
        self.point_subtypes = tuple(data_cfg.get("point_subtypes", ("pois", "traffic", "transport", "pofw")))
        self.encoder = MixedLinePolygonPointFusionEncoder(
            line_chunk_feature_dim=int(line_model.get("chunk_feature_dim", 10)),
            polygon_feature_dim=int(polygon_model.get("polygon_feature_dim", 10)),
            point_feature_dim=int(point_model.get("point_feature_dim", 8)),
            point_visible_feature_dim=int(point_model.get("visible_feature_dim", 103)),
            line_hidden_dim=int(line_model.get("hidden_dim", 192)),
            line_file_dim=int(line_model.get("file_dim", 192)),
            polygon_hidden_dim=int(polygon_model.get("hidden_dim", 192)),
            polygon_file_dim=int(polygon_model.get("file_dim", 192)),
            point_hidden_dim=int(point_model.get("hidden_dim", 192)),
            point_file_dim=int(point_model.get("file_dim", 192)),
            point_visible_hidden_dim=int(point_model.get("visible_hidden_dim", 160)),
            point_visible_dim=int(point_model.get("visible_dim", 128)),
            point_fusion_hidden_dim=int(point_model.get("fusion_hidden_dim", 256)),
            point_fusion_dim=int(point_model.get("fusion_dim", 192)),
            line_n_tiles=int(line_protocol["n_tiles"]),
            polygon_n_tiles=int(polygon_protocol["n_tiles"]),
            point_n_tiles=int(point_protocol["n_tiles"]),
            line_max_depth=max(int(key) for key in line_protocol["depth_template"].keys()),
            polygon_max_depth=max(int(key) for key in polygon_protocol["depth_template"].keys()),
            point_max_depth=max(int(key) for key in point_protocol["depth_template"].keys()),
            polygon_tile_encoder_type=str(polygon_model.get("tile_encoder_type", "set_transformer")),
            point_tile_encoder_type=str(point_model.get("tile_encoder_type", "set_transformer")),
            line_chunk_num_heads=int(line_model.get("chunk_num_heads", 4)),
            line_file_num_heads=int(line_model.get("file_num_heads", 4)),
            line_num_chunk_layers=int(line_model.get("num_chunk_layers", 2)),
            line_num_file_layers=int(line_model.get("num_file_layers", 3)),
            polygon_tile_num_heads=int(polygon_model.get("tile_num_heads", 4)),
            polygon_num_tile_layers=int(polygon_model.get("num_tile_layers", 2)),
            polygon_file_num_heads=int(polygon_model.get("file_num_heads", 4)),
            polygon_num_file_layers=int(polygon_model.get("num_file_layers", 3)),
            point_tile_num_heads=int(point_model.get("tile_num_heads", 4)),
            point_num_tile_layers=int(point_model.get("num_tile_layers", 2)),
            point_file_num_heads=int(point_model.get("file_num_heads", 4)),
            point_num_file_layers=int(point_model.get("num_file_layers", 3)),
            branch_num_heads=int(fusion_model.get("branch_num_heads", 4)),
            branch_num_layers=int(fusion_model.get("branch_num_layers", 2)),
            modality_num_heads=int(fusion_model.get("modality_num_heads", 4)),
            modality_num_layers=int(fusion_model.get("modality_num_layers", 2)),
            fusion_dim=int(fusion_model.get("fusion_dim", 192)),
            dropout=float(fusion_model.get("dropout", 0.1)),
            line_subtypes=self.line_subtypes,
            polygon_subtypes=self.polygon_subtypes,
            point_subtypes=self.point_subtypes,
            point_include_local_structure=bool(point_model.get("include_local_structure", True)),
            point_local_grid_size=int(point_model.get("local_grid_size", 4)),
            point_local_radial_bins=int(point_model.get("local_radial_bins", 4)),
        ).to(self.device)
        self.projector = SimCLRProjector(
            input_dim=int(fusion_model.get("fusion_dim", 192)),
            hidden_dim=int(fusion_model.get("fusion_dim", 192)),
            output_dim=int(fusion_model.get("projector_dim", 128)),
        ).to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.projector.load_state_dict(checkpoint["projector"])
        self.encoder.eval()
        self.projector.eval()

    @torch.no_grad()
    def encode_package(
        self,
        line_view_map: Dict[str, object],
        polygon_view_map: Dict[str, object],
        point_view_map: Dict[str, object],
        prehash_space: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        fused, modalities = self.encoder(line_view_map, polygon_view_map, point_view_map)
        if prehash_space == "projector":
            fused = self.projector(fused.unsqueeze(0))[0]
        return fused.detach().cpu().numpy(), modalities.detach().cpu().numpy()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mixed prototype line+polygon+point fusion model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_prototype.yaml")
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--max_packages", type=int, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--feature_noise_std", type=float, default=0.0)
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
    evaluator = MixedLinePolygonPointEvaluator(args.checkpoint, device=args.device)
    line_protocol = config["line_protocol"]
    polygon_protocol = config["polygon_protocol"]
    point_protocol = config["point_protocol"]
    line_depth_template = {int(key): int(value) for key, value in line_protocol["depth_template"].items()}
    polygon_depth_template = {int(key): int(value) for key, value in polygon_protocol["depth_template"].items()}
    point_depth_template = {int(key): int(value) for key, value in point_protocol["depth_template"].items()}
    line_subtypes = tuple(config["data"].get("line_subtypes", ("roads", "railways", "waterways")))
    polygon_subtypes = tuple(config["data"].get("polygon_subtypes", ("building", "landuse", "natural")))
    point_subtypes = tuple(config["data"].get("point_subtypes", ("pois", "traffic", "transport", "pofw")))

    query_embeddings: List[np.ndarray] = []
    gallery_embeddings: List[np.ndarray] = []
    package_ids: List[str] = []
    line_subtype_overlaps: List[float] = []
    line_tile_overlaps: List[float] = []
    line_chunk_overlaps: List[float] = []
    polygon_subtype_overlaps: List[float] = []
    polygon_tile_overlaps: List[float] = []
    polygon_overlaps: List[float] = []
    point_subtype_overlaps: List[float] = []
    point_tile_overlaps: List[float] = []
    point_overlaps: List[float] = []
    modality_norms: List[float] = []
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []

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
            device=evaluator.device,
        )

        query, modalities_a = evaluator.encode_package(
            views["line_view_a"],
            views["polygon_view_a"],
            views["point_view_a"],
            prehash_space=args.prehash_space,
        )
        gallery, modalities_b = evaluator.encode_package(
            views["line_view_b"],
            views["polygon_view_b"],
            views["point_view_b"],
            prehash_space=args.prehash_space,
        )
        score = cosine_similarity(query, gallery)
        ber = cosine_to_ber(score)
        query_embeddings.append(query)
        gallery_embeddings.append(gallery)
        package_ids.append(str(package_sample["package_id"]))
        modality_norms.append(float((np.linalg.norm(modalities_a, axis=1).mean() + np.linalg.norm(modalities_b, axis=1).mean()) * 0.5))
        line_subtype_overlaps.append(float(views["line_subtype_overlap"]))
        line_tile_overlaps.append(float(views["line_tile_overlap"]))
        line_chunk_overlaps.append(float(views["line_chunk_overlap"]))
        polygon_subtype_overlaps.append(float(views["polygon_subtype_overlap"]))
        polygon_tile_overlaps.append(float(views["polygon_tile_overlap"]))
        polygon_overlaps.append(float(views["polygon_overlap"]))
        point_subtype_overlaps.append(float(views["point_subtype_overlap"]))
        point_tile_overlaps.append(float(views["point_tile_overlap"]))
        point_overlaps.append(float(views["point_overlap"]))
        genuine_scores.append(score)
        genuine_bers.append(ber)

    query_matrix = np.asarray(query_embeddings, dtype=np.float64)
    gallery_matrix = np.asarray(gallery_embeddings, dtype=np.float64)
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
            "checkpoint": args.checkpoint,
            "config": args.config,
            "line_cache_root": config["data"]["line_cache_root"],
            "polygon_cache_root": config["data"]["polygon_cache_root"],
            "point_cache_root": config["data"]["point_cache_root"],
            "dataset_mode": args.dataset_mode,
            "max_packages": config["data"].get("max_packages"),
            "prehash_space": args.prehash_space,
            "feature_noise_std": args.feature_noise_std,
            "device": args.device,
            "seed": config["device"]["seed"],
        },
        "n_packages": len(package_ids),
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
        "pairwise_metrics": pairwise_metrics,
        "retrieval_metrics": retrieval_metrics,
        "overlap_metrics": {
            "line_subtype_mean": float(np.mean(line_subtype_overlaps)) if line_subtype_overlaps else 0.0,
            "line_tile_mean": float(np.mean(line_tile_overlaps)) if line_tile_overlaps else 0.0,
            "line_chunk_mean": float(np.mean(line_chunk_overlaps)) if line_chunk_overlaps else 0.0,
            "polygon_subtype_mean": float(np.mean(polygon_subtype_overlaps)) if polygon_subtype_overlaps else 0.0,
            "polygon_tile_mean": float(np.mean(polygon_tile_overlaps)) if polygon_tile_overlaps else 0.0,
            "polygon_mean": float(np.mean(polygon_overlaps)) if polygon_overlaps else 0.0,
            "point_subtype_mean": float(np.mean(point_subtype_overlaps)) if point_subtype_overlaps else 0.0,
            "point_tile_mean": float(np.mean(point_tile_overlaps)) if point_tile_overlaps else 0.0,
            "point_mean": float(np.mean(point_overlaps)) if point_overlaps else 0.0,
        },
        "fusion_metrics": {
            "modality_norm_mean": float(np.mean(modality_norms)) if modality_norms else 0.0,
            "modality_norm_std": float(np.std(modality_norms)) if modality_norms else 0.0,
        },
    }

    print("=" * 90)
    print("Mixed Hier Stage1 Line+Polygon+Point Prototype Evaluation")
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
        f"polygon_mean={report['overlap_metrics']['polygon_mean']:.6f}, "
        f"point_subtype_mean={report['overlap_metrics']['point_subtype_mean']:.6f}, "
        f"point_tile_mean={report['overlap_metrics']['point_tile_mean']:.6f}, "
        f"point_mean={report['overlap_metrics']['point_mean']:.6f}"
    )
    print(
        f"Fusion modality norm: mean={report['fusion_metrics']['modality_norm_mean']:.6f}, "
        f"std={report['fusion_metrics']['modality_norm_std']:.6f}"
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
