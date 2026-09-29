"""Evaluate missing-subtype tolerant late score fusion.

This script is a low-cost diagnostic for the fallback path:
- line/polygon branches use any available subtypes instead of requiring complete subtype coverage;
- point is optional and averaged over available point subtypes;
- if point is unavailable or masked out, the fused score falls back to LP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mixed_hier_stage1.line_polygon_dataset import MixedLinePolygonPackageDataset
from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonPointPackageDataset
from point_hier_stage1.dual_view_protocol import build_fixed_budget_point_views
from polygon_hier_stage1.dual_view_protocol import build_fixed_budget_polygon_views
from roads_hier_stage1.dual_view_protocol import build_fixed_budget_dual_views
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import cosine_similarity, metrics_from_score_matrix
from scripts.evaluate_point_hier_stage1_dual_view_hybrid import PointHybridDualViewEvaluator
from train_roads_hier_stage1_minimal import load_config, resolve_path
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


def score_matrix(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros((query_embeddings.shape[0], gallery_embeddings.shape[0]), dtype=np.float64)
    for i in range(query_embeddings.shape[0]):
        for j in range(gallery_embeddings.shape[0]):
            scores[i, j] = cosine_similarity(query_embeddings[i], gallery_embeddings[j])
    return scores


def sample_available_subtypes(
    available: Sequence[str],
    *,
    per_view: int | None,
    rng: np.random.Generator,
    drop_count: int = 0,
    force_drop: Sequence[str] = (),
) -> List[str]:
    names = [str(value) for value in available if str(value) not in set(force_drop)]
    if drop_count > 0 and len(names) > 1:
        actual_drop = min(int(drop_count), len(names) - 1)
        drop_indices = set(int(idx) for idx in rng.choice(len(names), size=actual_drop, replace=False))
        names = [name for idx, name in enumerate(names) if idx not in drop_indices]
    if not names:
        return []
    if per_view is None or int(per_view) >= len(names):
        return list(names)
    count = max(1, int(per_view))
    indices = rng.choice(len(names), size=count, replace=False)
    return [names[int(idx)] for idx in np.sort(indices)]


def build_tolerant_views(
    package_sample: Dict[str, object],
    *,
    config: Dict[str, Any],
    scenario: str,
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    device: Any,
    feature_noise_std: float = 0.0,
) -> Dict[str, Any]:
    line_protocol = config["line_protocol"]
    polygon_protocol = config["polygon_protocol"]
    point_protocol = config.get("point_protocol", {})
    line_depth_template = {int(key): int(value) for key, value in line_protocol["depth_template"].items()}
    polygon_depth_template = {int(key): int(value) for key, value in polygon_protocol["depth_template"].items()}
    point_depth_template = {int(key): int(value) for key, value in point_protocol.get("depth_template", {}).items()}

    available_line = [subtype for subtype in config["data"].get("line_subtypes", ()) if subtype in package_sample["line_samples"]]
    available_polygon = [
        subtype for subtype in config["data"].get("polygon_subtypes", ()) if subtype in package_sample["polygon_samples"]
    ]
    available_point = [
        subtype for subtype in config["data"].get("point_subtypes", ()) if subtype in package_sample.get("point_samples", {})
    ]

    line_drop = 0
    polygon_drop = 0
    point_drop = 0
    force_drop_point: Sequence[str] = ()
    if scenario == "minor_missing":
        line_drop = 1
        polygon_drop = 1
        point_drop = 1
    elif scenario == "partial_family_missing":
        point_drop = max(1, len(available_point) // 2) if available_point else 0
    elif scenario == "point_missing":
        force_drop_point = tuple(available_point)
    elif scenario != "natural_missing":
        raise ValueError(f"Unsupported scenario: {scenario}")

    line_a = sample_available_subtypes(
        available_line,
        per_view=config["data"].get("line_subtypes_per_view"),
        rng=rng_a,
        drop_count=line_drop,
    )
    line_b = sample_available_subtypes(
        available_line,
        per_view=config["data"].get("line_subtypes_per_view"),
        rng=rng_b,
        drop_count=line_drop,
    )
    polygon_a = sample_available_subtypes(
        available_polygon,
        per_view=config["data"].get("polygon_subtypes_per_view"),
        rng=rng_a,
        drop_count=polygon_drop,
    )
    polygon_b = sample_available_subtypes(
        available_polygon,
        per_view=config["data"].get("polygon_subtypes_per_view"),
        rng=rng_b,
        drop_count=polygon_drop,
    )
    point_a = sample_available_subtypes(
        available_point,
        per_view=config["data"].get("point_subtypes_per_view"),
        rng=rng_a,
        drop_count=point_drop,
        force_drop=force_drop_point,
    )
    point_b = sample_available_subtypes(
        available_point,
        per_view=config["data"].get("point_subtypes_per_view"),
        rng=rng_b,
        drop_count=point_drop,
        force_drop=force_drop_point,
    )
    if not line_a or not line_b or not polygon_a or not polygon_b:
        raise ValueError(f"Package {package_sample['package_id']} has no usable LP view under {scenario}.")

    line_view_a: Dict[str, object] = {}
    line_view_b: Dict[str, object] = {}
    polygon_view_a: Dict[str, object] = {}
    polygon_view_b: Dict[str, object] = {}
    point_view_a: Dict[str, object] = {}
    point_view_b: Dict[str, object] = {}

    for subtype in sorted(set(line_a) | set(line_b)):
        dual = build_fixed_budget_dual_views(
            package_sample["line_samples"][subtype],
            n_tiles=int(line_protocol["n_tiles"]),
            k_chunks=int(line_protocol["k_chunks_per_tile"]),
            depth_template=line_depth_template,
            rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
            rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
            feature_noise_std=feature_noise_std,
            device=device,
        )
        if subtype in line_a:
            line_view_a[subtype] = dual["view_a"]
        if subtype in line_b:
            line_view_b[subtype] = dual["view_b"]

    for subtype in sorted(set(polygon_a) | set(polygon_b)):
        dual = build_fixed_budget_polygon_views(
            package_sample["polygon_samples"][subtype],
            n_tiles=int(polygon_protocol["n_tiles"]),
            k_polygons=int(polygon_protocol["k_polygons_per_tile"]),
            depth_template=polygon_depth_template,
            rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
            rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
            feature_noise_std=feature_noise_std,
            device=device,
        )
        if subtype in polygon_a:
            polygon_view_a[subtype] = dual["view_a"]
        if subtype in polygon_b:
            polygon_view_b[subtype] = dual["view_b"]

    for subtype in sorted(set(point_a) | set(point_b)):
        dual = build_fixed_budget_point_views(
            package_sample["point_samples"][subtype],
            n_tiles=int(point_protocol["n_tiles"]),
            k_points=int(point_protocol["k_points_per_tile"]),
            depth_template=point_depth_template,
            rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
            rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
            feature_noise_std=feature_noise_std,
            device=device,
        )
        if subtype in point_a:
            point_view_a[subtype] = dual["view_a"]
        if subtype in point_b:
            point_view_b[subtype] = dual["view_b"]

    return {
        "line_view_a": line_view_a,
        "line_view_b": line_view_b,
        "polygon_view_a": polygon_view_a,
        "polygon_view_b": polygon_view_b,
        "point_view_a": point_view_a,
        "point_view_b": point_view_b,
        "line_available": available_line,
        "polygon_available": available_polygon,
        "point_available": available_point,
        "line_selected_a": line_a,
        "line_selected_b": line_b,
        "polygon_selected_a": polygon_a,
        "polygon_selected_b": polygon_b,
        "point_selected_a": point_a,
        "point_selected_b": point_b,
    }


def mean_point_embedding(
    point_evaluator: PointHybridDualViewEvaluator,
    point_view_map: Dict[str, object],
    *,
    prehash_space: str,
) -> np.ndarray | None:
    embeddings = [
        point_evaluator.encode_view(point_view_map[subtype], prehash_space=prehash_space)
        for subtype in sorted(point_view_map.keys())
    ]
    if not embeddings:
        return None
    return np.mean(np.stack(embeddings, axis=0), axis=0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate missing-subtype tolerant masked late score fusion.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_prototype.yaml")
    parser.add_argument("--lp_checkpoint", type=str, required=True)
    parser.add_argument("--point_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", type=str, default="natural_missing", choices=["natural_missing", "minor_missing", "partial_family_missing", "point_missing"])
    parser.add_argument("--alpha", type=float, default=0.5)
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
    if args.seed is not None:
        config["device"]["seed"] = int(args.seed)
    set_seed(config["device"]["seed"])

    if args.scenario == "point_missing":
        dataset = MixedLinePolygonPackageDataset(
            config["data"]["line_cache_root"],
            config["data"]["polygon_cache_root"],
            mode=args.dataset_mode,
            line_max_tiles=config["data"].get("line_max_tiles"),
            polygon_max_tiles=config["data"].get("polygon_max_tiles"),
            line_tile_selector=config["data"].get("line_tile_selector", "manifest_default"),
            polygon_tile_selector=config["data"].get("polygon_tile_selector", "manifest_default"),
            line_subtypes=config["data"].get("line_subtypes", ("roads", "railways", "waterways")),
            polygon_subtypes=config["data"].get("polygon_subtypes", ("building", "landuse", "natural")),
            require_complete_packages=False,
        )
        lpp_mode = False
    else:
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
            require_complete_packages=False,
        )
        lpp_mode = True

    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    lp_query: List[np.ndarray] = []
    lp_gallery: List[np.ndarray] = []
    point_query_by_row: List[np.ndarray | None] = []
    point_gallery_by_row: List[np.ndarray | None] = []
    package_ids: List[str] = []
    point_available_counts: List[int] = []
    point_used_counts: List[int] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if not lpp_mode:
            sample = {
                **sample,
                "point_samples": {},
                "point_subtypes": [],
            }
        try:
            views = build_tolerant_views(
                sample,
                config=config,
                scenario=args.scenario,
                rng_a=np.random.default_rng(config["device"]["seed"] + idx * 101),
                rng_b=np.random.default_rng(config["device"]["seed"] + idx * 101 + 1),
                device=lp_evaluator.device,
            )
        except ValueError:
            continue
        lp_a, _ = lp_evaluator.encode_package(views["line_view_a"], views["polygon_view_a"], args.prehash_space)
        lp_b, _ = lp_evaluator.encode_package(views["line_view_b"], views["polygon_view_b"], args.prehash_space)
        p_a = mean_point_embedding(point_evaluator, views["point_view_a"], prehash_space=args.prehash_space)
        p_b = mean_point_embedding(point_evaluator, views["point_view_b"], prehash_space=args.prehash_space)
        lp_query.append(lp_a)
        lp_gallery.append(lp_b)
        point_query_by_row.append(p_a)
        point_gallery_by_row.append(p_b)
        package_ids.append(str(sample["package_id"]))
        point_available_counts.append(len(views["point_available"]))
        point_used_counts.append(min(len(views["point_selected_a"]), len(views["point_selected_b"])))

    lp_scores = score_matrix(np.asarray(lp_query), np.asarray(lp_gallery))
    fused_scores = np.array(lp_scores, copy=True)
    point_rows = [
        idx
        for idx, (query, gallery) in enumerate(zip(point_query_by_row, point_gallery_by_row))
        if query is not None and gallery is not None
    ]
    if point_rows:
        point_query = np.stack([point_query_by_row[idx] for idx in point_rows], axis=0)
        point_gallery = np.stack([point_gallery_by_row[idx] for idx in point_rows], axis=0)
        point_scores_sub = score_matrix(point_query, point_gallery)
        for row_i, package_i in enumerate(point_rows):
            for row_j, package_j in enumerate(point_rows):
                fused_scores[package_i, package_j] = lp_scores[package_i, package_j] + float(args.alpha) * point_scores_sub[row_i, row_j]

    lp_metrics = metrics_from_score_matrix(lp_scores, package_ids)
    fused_metrics = metrics_from_score_matrix(fused_scores, package_ids)
    report = {
        "arguments": {
            "config": args.config,
            "lp_checkpoint": args.lp_checkpoint,
            "point_checkpoint": args.point_checkpoint,
            "line_cache_root": config["data"]["line_cache_root"],
            "polygon_cache_root": config["data"]["polygon_cache_root"],
            "point_cache_root": config["data"].get("point_cache_root"),
            "dataset_mode": args.dataset_mode,
            "scenario": args.scenario,
            "alpha": float(args.alpha),
            "seed": config["device"]["seed"],
            "prehash_space": args.prehash_space,
        },
        "n_packages": len(package_ids),
        "n_point_available_packages": len(point_rows),
        "point_available_rate": float(len(point_rows) / max(len(package_ids), 1)),
        "point_available_count_mean": float(np.mean(point_available_counts)) if point_available_counts else 0.0,
        "point_used_count_mean": float(np.mean(point_used_counts)) if point_used_counts else 0.0,
        "line_polygon": lp_metrics,
        "masked_late_fusion": fused_metrics,
    }
    print("=" * 90)
    print("Missing-Subtype Tolerant Masked Late Score Fusion")
    print("=" * 90)
    print(f"Scenario: {args.scenario}")
    print(f"Packages: {report['n_packages']}, point_available={report['n_point_available_packages']}")
    lp = report["line_polygon"]
    fused = report["masked_late_fusion"]
    print(
        "LP fallback: "
        f"AUC={lp['pairwise_metrics']['auc']:.6f}, EER={lp['pairwise_metrics']['eer']:.6f}, "
        f"R@1={lp['retrieval_metrics']['recall_at_1']:.6f}, R@5={lp['retrieval_metrics']['recall_at_5']:.6f}, "
        f"MRR={lp['retrieval_metrics']['mrr']:.6f}"
    )
    print(
        f"Masked LP + {args.alpha:g}*P: "
        f"AUC={fused['pairwise_metrics']['auc']:.6f}, EER={fused['pairwise_metrics']['eer']:.6f}, "
        f"R@1={fused['retrieval_metrics']['recall_at_1']:.6f}, R@5={fused['retrieval_metrics']['recall_at_5']:.6f}, "
        f"MRR={fused['retrieval_metrics']['mrr']:.6f}"
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
