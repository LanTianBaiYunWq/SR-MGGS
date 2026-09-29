"""Evaluate SR-MSGS LP signatures with multi-scale Level-0 pairwise/retrieval metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mixed_hier_stage1.line_polygon_dataset import MixedLinePolygonPackageDataset
from mixed_hier_stage1.line_polygon_protocol import build_partial_package_views
from scripts.evaluate_mixed_generic_score_fusion_ablation import load_package_subset
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import compute_separability_from_scores, score_matrix
from train_roads_hier_stage1_minimal import load_config, resolve_path
from train_sr_msgs_line_polygon import encode_multiscale_features, mean_signature, parse_scale_specs
from utils.metrics import compute_auc, compute_eer
from utils.seed import set_seed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SR-MSGS LP Level-0 multi-scale signatures.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--package_split", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="heldout_packages")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument("--scale_specs", type=str, default="local:8:16,regional:16:32,global:32:64")
    parser.add_argument("--target_fars", type=str, default="0.01,0.05,0.1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser


def build_dataset(config: Dict[str, Any], args: argparse.Namespace) -> MixedLinePolygonPackageDataset:
    data_cfg = config["data"]
    dataset = MixedLinePolygonPackageDataset(
        data_cfg["line_cache_root"],
        data_cfg["polygon_cache_root"],
        mode=args.dataset_mode,
        line_max_tiles=data_cfg.get("line_max_tiles"),
        polygon_max_tiles=data_cfg.get("polygon_max_tiles"),
        line_tile_selector=data_cfg.get("line_tile_selector", "manifest_default"),
        polygon_tile_selector=data_cfg.get("polygon_tile_selector", "manifest_default"),
        line_subtypes=data_cfg.get("line_subtypes", ("roads", "railways", "waterways")),
        polygon_subtypes=data_cfg.get("polygon_subtypes", ("building", "landuse", "natural", "water")),
        max_packages=data_cfg.get("max_packages"),
        require_complete_packages=bool(data_cfg.get("require_complete_packages", False)),
    )
    subset = load_package_subset(args.package_split, args.split_name)
    if subset is not None:
        dataset.package_records = [
            record for record in dataset.package_records if str(record["package_id"]).lower() in subset
        ]
    return dataset


def build_scale_views(
    sample: Dict[str, Any],
    config: Dict[str, Any],
    scale_specs: Sequence[Dict[str, Any]],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    line_protocol = config["line_protocol"]
    polygon_protocol = config["polygon_protocol"]
    line_depth_template = {int(key): int(value) for key, value in line_protocol["depth_template"].items()}
    polygon_depth_template = {int(key): int(value) for key, value in polygon_protocol["depth_template"].items()}
    scale_views: Dict[str, Any] = {}
    for scale_idx, spec in enumerate(scale_specs):
        scale_seed = int(seed) + scale_idx * 10007
        scale_views[str(spec["name"])] = build_partial_package_views(
            sample,
            line_subtypes=tuple(config["data"].get("line_subtypes", ("roads", "railways", "waterways"))),
            polygon_subtypes=tuple(config["data"].get("polygon_subtypes", ("building", "landuse", "natural", "water"))),
            line_subtypes_per_view=config["data"].get("line_subtypes_per_view"),
            polygon_subtypes_per_view=config["data"].get("polygon_subtypes_per_view"),
            line_n_tiles=int(line_protocol["n_tiles"]),
            line_k_chunks=int(spec["line_k_chunks"]),
            line_depth_template=line_depth_template,
            polygon_n_tiles=int(polygon_protocol["n_tiles"]),
            polygon_k_polygons=int(spec["polygon_k_objects"]),
            polygon_depth_template=polygon_depth_template,
            rng_a=np.random.default_rng(scale_seed + 1),
            rng_b=np.random.default_rng(scale_seed + 2),
            feature_noise_std=0.0,
            device=device,
        )
    return scale_views


@torch.no_grad()
def encode_split(
    dataset: MixedLinePolygonPackageDataset,
    config: Dict[str, Any],
    args: argparse.Namespace,
    evaluator: MixedLinePolygonEvaluator,
) -> tuple[List[str], np.ndarray, np.ndarray]:
    scale_specs = parse_scale_specs(args.scale_specs)
    query_embeddings: List[np.ndarray] = []
    gallery_embeddings: List[np.ndarray] = []
    package_ids: List[str] = []
    evaluator.encoder.eval()
    evaluator.projector.eval()
    for idx in tqdm(range(len(dataset)), desc="Encode SR-MSGS Level-0"):
        sample = dataset[idx]
        base_seed = int(args.seed) * 1000003 + idx * 1009
        scale_views = build_scale_views(sample, config, scale_specs, base_seed, evaluator.device)
        features = encode_multiscale_features(evaluator, scale_views, scale_specs, args.prehash_space)
        n_scales = len(scale_specs)
        query_features = features[0 : 2 * n_scales : 2]
        gallery_features = features[1 : 2 * n_scales : 2]
        query_embeddings.append(mean_signature(query_features).detach().cpu().numpy())
        gallery_embeddings.append(mean_signature(gallery_features).detach().cpu().numpy())
        package_ids.append(str(sample["package_id"]))
    return package_ids, np.asarray(query_embeddings), np.asarray(gallery_embeddings)


def score_sets(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.asarray([scores[i, i] for i in range(scores.shape[0])], dtype=np.float64)
    impostor = np.asarray(
        [scores[i, j] for i in range(scores.shape[0]) for j in range(scores.shape[1]) if i != j],
        dtype=np.float64,
    )
    return genuine, impostor


def rates_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> Dict[str, float]:
    threshold = float(np.quantile(impostor, 1.0 - float(target_far), method="higher"))
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))
    return {"threshold": threshold, "tar": 1.0 - frr, "far": far, "frr": frr}


def retrieval_metrics(scores: np.ndarray) -> Dict[str, float]:
    r1: List[float] = []
    r5: List[float] = []
    mrr: List[float] = []
    for i in range(scores.shape[0]):
        ranked = np.argsort(-scores[i])
        rank = int(np.where(ranked == i)[0][0]) + 1
        r1.append(float(rank <= 1))
        r5.append(float(rank <= 5))
        mrr.append(1.0 / float(rank))
    return {
        "recall_at_1": float(np.mean(r1)) if r1 else 0.0,
        "recall_at_5": float(np.mean(r5)) if r5 else 0.0,
        "mrr": float(np.mean(mrr)) if mrr else 0.0,
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    config["device"]["seed"] = int(args.seed)
    set_seed(int(args.seed))

    dataset = build_dataset(config, args)
    evaluator = MixedLinePolygonEvaluator(args.checkpoint, device=args.device)
    package_ids, queries, galleries = encode_split(dataset, config, args, evaluator)
    scores = score_matrix(queries, galleries)
    genuine, impostor = score_sets(scores)
    eer, eer_threshold = compute_eer(genuine, impostor)
    target_fars = [float(value) for value in str(args.target_fars).split(",") if value.strip()]
    report = {
        "arguments": vars(args),
        "n_packages": len(package_ids),
        "package_ids": package_ids,
        "pairwise": {
            "auc": compute_auc(genuine, impostor),
            "eer": eer,
            "eer_threshold": eer_threshold,
            "separability": compute_separability_from_scores(genuine, impostor),
            "genuine_mean": float(np.mean(genuine)),
            "impostor_mean": float(np.mean(impostor)),
        },
        "operating_points": {f"far_{far:g}": rates_at_far(genuine, impostor, far) for far in target_fars},
        "retrieval": retrieval_metrics(scores),
    }

    print("=" * 90)
    print("SR-MSGS LP Level-0 Evaluation")
    print("=" * 90)
    print(f"Packages: {len(package_ids)}")
    print(
        f"Pairwise: AUC={report['pairwise']['auc']:.6f}, "
        f"EER={report['pairwise']['eer']:.6f}, Sep={report['pairwise']['separability']:.6f}"
    )
    for key, value in report["operating_points"].items():
        print(f"{key}: TAR={value['tar']:.6f}, FAR={value['far']:.6f}, FRR={value['frr']:.6f}")
    print(
        f"Retrieval: R@1={report['retrieval']['recall_at_1']:.6f}, "
        f"R@5={report['retrieval']['recall_at_5']:.6f}, MRR={report['retrieval']['mrr']:.6f}"
    )
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
