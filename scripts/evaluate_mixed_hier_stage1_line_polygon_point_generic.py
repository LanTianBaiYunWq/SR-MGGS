"""Evaluate generic mixed line+polygon+optional-point backbone."""

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

from mixed_hier_stage1.generic_line_polygon_point_fusion_model import MixedGenericLinePolygonPointFusionEncoder
from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonOptionalPointPackageDataset
from models.projector import SimCLRProjector
from scripts.evaluate_mixed_lpp_score_fusion import cosine_similarity, metrics_from_score_matrix, score_matrix
from scripts.evaluate_mixed_missing_subtype_tolerant_score_fusion import build_tolerant_views
from train_roads_hier_stage1_minimal import load_config, resolve_path
from utils.seed import set_seed


def load_package_subset(split_path: str | None, split_name: str) -> set[str] | None:
    if not split_path:
        return None
    with Path(split_path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    key_map = {
        "train": "train_packages",
        "heldout": "heldout_packages",
        "eval": "heldout_packages",
    }
    key = key_map.get(split_name, split_name)
    if key not in split:
        raise KeyError(f"Split file does not contain key: {key}")
    return {str(value).lower() for value in split[key]}


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


class MixedGenericLinePolygonPointEvaluator:
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
        self.encoder = MixedGenericLinePolygonPointFusionEncoder(
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
    ) -> tuple[np.ndarray, int]:
        fused, modalities = self.encoder(line_view_map, polygon_view_map, point_view_map)
        if prehash_space == "projector":
            fused = self.projector(fused.unsqueeze(0))[0]
        return fused.detach().cpu().numpy(), int(modalities.shape[0])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generic mixed line+polygon+optional-point model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--max_packages", type=int, default=None)
    parser.add_argument("--package_split", type=str, default=None)
    parser.add_argument("--split_name", type=str, default="heldout")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", type=str, default="natural_missing", choices=["natural_missing", "minor_missing", "partial_family_missing", "point_missing"])
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
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
    package_subset = load_package_subset(args.package_split, args.split_name)
    if package_subset is not None:
        dataset.package_records = [
            record
            for record in dataset.package_records
            if str(record["package_id"]).lower() in package_subset
        ]
    evaluator = MixedGenericLinePolygonPointEvaluator(args.checkpoint, device=args.device)
    query_embeddings: List[np.ndarray] = []
    gallery_embeddings: List[np.ndarray] = []
    package_ids: List[str] = []
    point_used: List[float] = []
    modality_counts: List[int] = []
    genuine_scores: List[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        try:
            views = build_tolerant_views(
                sample,
                config=config,
                scenario=args.scenario,
                rng_a=np.random.default_rng(config["device"]["seed"] + idx * 101),
                rng_b=np.random.default_rng(config["device"]["seed"] + idx * 101 + 1),
                device=evaluator.device,
            )
        except ValueError:
            continue
        query, n_mod_a = evaluator.encode_package(
            views["line_view_a"], views["polygon_view_a"], views["point_view_a"], args.prehash_space
        )
        gallery, n_mod_b = evaluator.encode_package(
            views["line_view_b"], views["polygon_view_b"], views["point_view_b"], args.prehash_space
        )
        query_embeddings.append(query)
        gallery_embeddings.append(gallery)
        package_ids.append(str(sample["package_id"]))
        point_used.append(float(bool(views["point_view_a"]) and bool(views["point_view_b"])))
        modality_counts.append(int(round((n_mod_a + n_mod_b) * 0.5)))
        genuine_scores.append(cosine_similarity(query, gallery))

    scores = score_matrix(np.asarray(query_embeddings), np.asarray(gallery_embeddings))
    metrics = metrics_from_score_matrix(scores, package_ids)
    report = {
        "arguments": {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "line_cache_root": config["data"]["line_cache_root"],
            "polygon_cache_root": config["data"]["polygon_cache_root"],
            "point_cache_root": config["data"].get("point_cache_root"),
            "dataset_mode": args.dataset_mode,
            "package_split": args.package_split,
            "split_name": args.split_name,
            "scenario": args.scenario,
            "seed": config["device"]["seed"],
            "prehash_space": args.prehash_space,
        },
        "n_packages": len(package_ids),
        "point_used_rate": float(np.mean(point_used)) if point_used else 0.0,
        "modality_count_mean": float(np.mean(modality_counts)) if modality_counts else 0.0,
        "metrics": metrics,
    }
    print("=" * 90)
    print("Generic Mixed Line+Polygon+Optional-Point Evaluation")
    print("=" * 90)
    print(f"Scenario: {args.scenario}")
    print(f"Packages: {report['n_packages']}, point_used_rate={report['point_used_rate']:.4f}")
    print(
        f"Pairwise: AUC={metrics['pairwise_metrics']['auc']:.6f}, "
        f"EER={metrics['pairwise_metrics']['eer']:.6f}, "
        f"Separability={metrics['pairwise_metrics']['separability']:.6f}"
    )
    print(
        f"Retrieval: R@1={metrics['retrieval_metrics']['recall_at_1']:.6f}, "
        f"R@5={metrics['retrieval_metrics']['recall_at_5']:.6f}, "
        f"MRR={metrics['retrieval_metrics']['mrr']:.6f}"
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
