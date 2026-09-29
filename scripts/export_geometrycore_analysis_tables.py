"""
导出 GeometryCore 文件级全局实验的三张分析表：
1. pair-level 结果表
2. experiment-level 配置与指标汇总表
3. file-level 原始文件统计表

当前脚本服务于“先把原始明细数据整理出来”的需求，
因此优先保证字段齐全、可追踪、可导出，而不是追求脚本最短。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

# 将项目根目录加入导入路径，保证从 scripts 目录直接运行时也能解析项目模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.geometry_core.crs import ensure_working_projected_crs
from preprocess.global_geometrycore_dataset import GeometryCoreGlobalDataset
from scripts.evaluate_geometrycore_global import (
    GeometryCoreGlobalSignatureEvaluator,
    build_config,
    resolve_path,
)
from utils.metrics import SignatureEvaluator, compute_ber, compute_normalized_correlation


@dataclass
class FileEvalItem:
    """保存单文件评估所需的中间结果。"""

    file_id: str
    file_path: str
    split: str
    family: str
    raw_token_count: int
    sampled_token_count: int
    embedding_ref_norm: float
    embedding_aug_norm: float
    projected_ref: np.ndarray
    projected_aug: np.ndarray
    bits_ref: np.ndarray
    bits_aug: np.ndarray


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Export GeometryCore analysis tables.")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/geometrycore_global.yaml",
        help="配置文件路径。",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="覆盖数据目录。")
    parser.add_argument("--file_pattern", type=str, default=None, help="文件过滤模式。")
    parser.add_argument("--max_files", type=int, default=None, help="最多加载多少个文件。")
    parser.add_argument("--max_geometries", type=int, default=None, help="每文件采样多少 geometry token。")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备。")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/analysis_tables",
        help="输出目录。",
    )
    return parser


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """计算连续向量的余弦相似度。"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def dominant_family(family_counts: Dict[str, int]) -> str:
    """返回文件中的主导几何家族。"""
    if not family_counts:
        return "unknown"
    return max(family_counts.items(), key=lambda x: x[1])[0]


def make_split_map(n_files: int, val_ratio: float, seed: int) -> Dict[int, str]:
    """复现训练时的 train/val 划分。"""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_files, generator=generator).tolist()
    val_size = max(1, int(n_files * val_ratio))
    train_size = n_files - val_size
    train_indices = set(indices[:train_size])

    split_map = {}
    for idx in range(n_files):
        split_map[idx] = "train" if idx in train_indices else "val"
    return split_map


def geom_family_from_type(geom_type: str) -> str:
    """按 geometry type 映射到 point/line/polygon 家族。"""
    if geom_type in {"Point", "MultiPoint"}:
        return "point"
    if geom_type in {"LineString", "MultiLineString", "LinearRing"}:
        return "line"
    if geom_type in {"Polygon", "MultiPolygon"}:
        return "polygon"
    return "other"


def count_feature_families(gdf: gpd.GeoDataFrame) -> Tuple[int, int, int]:
    """统计 point/line/polygon feature 数量。"""
    point_count = 0
    line_count = 0
    polygon_count = 0

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        family = geom_family_from_type(geom.geom_type)
        if family == "point":
            point_count += 1
        elif family == "line":
            line_count += 1
        elif family == "polygon":
            polygon_count += 1

    return point_count, line_count, polygon_count


def multipart_ratio(gdf: gpd.GeoDataFrame) -> float:
    """统计 multipart 比例。"""
    valid = 0
    multipart = 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        valid += 1
        if geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
            multipart += 1
    if valid == 0:
        return 0.0
    return float(multipart / valid)


def build_file_stats_row(
    sample: Dict[str, Any],
    config: Dict[str, Any],
    split: str,
    file_id: str,
) -> Dict[str, Any]:
    """生成 file-level 统计表的一行。"""
    shp_path = sample["file_path"]
    gdf = gpd.read_file(shp_path)
    gdf_projected, _ = ensure_working_projected_crs(
        gdf,
        source_crs_if_missing=config["data"].get("source_crs_if_missing"),
        working_crs=config["data"].get("working_crs"),
    )

    minx, miny, maxx, maxy = gdf_projected.total_bounds
    point_count, line_count, polygon_count = count_feature_families(gdf_projected)

    total_length = float(
        gdf_projected.geometry[gdf_projected.geometry.notnull()].length.sum()
    )
    total_area = float(
        gdf_projected.geometry[gdf_projected.geometry.notnull()].area.sum()
    )

    metadata = sample["metadata"]
    row = {
        "file_id": file_id,
        "file_path": shp_path,
        "split": split,
        "family": dominant_family(metadata["family_counts"]),
        "num_features": int(len(gdf)),
        "num_points": int(point_count),
        "num_lines": int(line_count),
        "num_polygons": int(polygon_count),
        "raw_token_count": int(sample["n_geometries"]),
        "sampled_token_count": int(sample["mask"].sum().item()),
        "bbox_width": float(maxx - minx),
        "bbox_height": float(maxy - miny),
        "total_length": total_length,
        "total_area": total_area,
        "multipart_ratio": multipart_ratio(gdf_projected),
    }
    return row


def hard_negative_type(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> str:
    """给 impostor 对打一个粗粒度 hard-negative 标签。"""
    if row_a["family"] == row_b["family"]:
        size_ratio = max(row_a["raw_token_count"], row_b["raw_token_count"]) / max(
            1, min(row_a["raw_token_count"], row_b["raw_token_count"])
        )
        if size_ratio <= 1.5:
            return "same_family_size_similar"
        return "same_family"
    return "cross_family"


def collect_projected_and_bits(
    evaluator: GeometryCoreGlobalSignatureEvaluator,
    geometries: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """收集 embedding、prehash 投影向量和 bit 签名。"""
    global_feat = evaluator.encode_global(geometries, mask)
    _, intermediates = evaluator.signature_generator.generate(global_feat, return_intermediate=True)
    embedding_np = global_feat.detach().cpu().numpy()
    projected_np = intermediates["projected"][0]
    bits_np = intermediates["bits"][0].astype(np.uint8)
    return embedding_np, projected_np, bits_np


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    config = build_config(args)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GeometryCoreGlobalDataset(
        data_dir=config["data"]["raw_dir"],
        file_pattern=config["data"].get("file_pattern", "*.shp"),
        max_files=config["data"].get("max_files"),
        source_crs_if_missing=config["data"].get("source_crs_if_missing"),
        working_crs=config["data"].get("working_crs"),
        max_geometries=config["data"].get("max_geometries", 128),
        min_geometries=config["data"].get("min_geometries", 10),
        fixed_points=config["data"].get("max_points", 128),
        include_holes=config["data"].get("include_holes", True),
        point_token_budget=config["data"].get("point_token_budget", 32),
        line_token_budget=config["data"].get("line_token_budget"),
        polygon_token_budget=config["data"].get("polygon_token_budget"),
        point_enabled=config["data"].get("point_enabled", True),
        line_enabled=config["data"].get("line_enabled", True),
        polygon_enabled=config["data"].get("polygon_enabled", True),
        line_eps_len=config["data"].get("line_eps_len", 1e-6),
        polygon_eps_area=config["data"].get("polygon_eps_area", 1e-8),
        line_densify_max_segment_length=config["data"].get("line_densify_max_segment_length"),
        verbose=config["data"].get("verbose_dataset", True),
    )

    evaluator = GeometryCoreGlobalSignatureEvaluator(
        checkpoint_path=str(resolve_path(args.checkpoint)),
        config=config,
        device=args.device,
    )
    metrics_evaluator = SignatureEvaluator(
        signature_bits=config.get("signature", {}).get("signature_bits", 256)
    )

    split_map = make_split_map(
        n_files=len(dataset),
        val_ratio=config["validation"]["val_ratio"],
        seed=config["device"]["seed"],
    )

    token_budget = int(config["data"].get("max_geometries", 128))
    aggregation_type = "attention_pooling"
    exp_id = f"{Path(args.checkpoint).stem}_{config['data'].get('dataset_backend', 'geometrycore')}"

    file_rows: List[Dict[str, Any]] = []
    file_items: List[FileEvalItem] = []

    # 先收集每个文件的原始 / 增强签名和原始统计。
    for idx in range(len(dataset)):
        sample = dataset[idx]
        file_id = f"file_{idx:04d}"
        split = split_map[idx]

        embedding_ref, projected_ref, bits_ref = collect_projected_and_bits(
            evaluator, sample["geometries"], sample["mask"]
        )
        aug_geometries = evaluator.augment_geometries(sample["geometries"], sample["mask"])
        embedding_aug, projected_aug, bits_aug = collect_projected_and_bits(
            evaluator, aug_geometries, sample["mask"]
        )

        file_items.append(
            FileEvalItem(
                file_id=file_id,
                file_path=sample["file_path"],
                split=split,
                family=dominant_family(sample["metadata"]["family_counts"]),
                raw_token_count=int(sample["n_geometries"]),
                sampled_token_count=int(sample["mask"].sum().item()),
                embedding_ref_norm=float(np.linalg.norm(embedding_ref)),
                embedding_aug_norm=float(np.linalg.norm(embedding_aug)),
                projected_ref=projected_ref,
                projected_aug=projected_aug,
                bits_ref=bits_ref,
                bits_aug=bits_aug,
            )
        )

        file_rows.append(build_file_stats_row(sample, config, split, file_id))

    pair_rows: List[Dict[str, Any]] = []

    # genuine pairs
    for idx, item in enumerate(file_items):
        ber = compute_ber(item.bits_ref, item.bits_aug)
        metrics_evaluator.add_genuine_pair(item.bits_ref, item.bits_aug)

        pair_rows.append(
            {
                "pair_id": f"{exp_id}_pair_{len(pair_rows):06d}",
                "file_a": item.file_path,
                "file_b": item.file_path,
                "label": "genuine",
                "family": item.family,
                "split": item.split,
                "prehash_score": cosine_score(item.projected_ref, item.projected_aug),
                "ber": float(ber),
                "same_source": True,
                "aug_view_a": "base",
                "aug_view_b": "aug_1",
                "token_budget": token_budget,
                "aggregation_type": aggregation_type,
                "hard_negative_type": "self_augmented",
            }
        )

    # impostor pairs
    file_row_by_id = {row["file_id"]: row for row in file_rows}
    for item_a, item_b in combinations(file_items, 2):
        ber = compute_ber(item_a.bits_ref, item_b.bits_ref)
        metrics_evaluator.add_impostor_pair(item_a.bits_ref, item_b.bits_ref)

        split_name = f"{item_a.split}-{item_b.split}" if item_a.split != item_b.split else item_a.split
        hard_type = hard_negative_type(
            file_row_by_id[item_a.file_id],
            file_row_by_id[item_b.file_id],
        )

        pair_rows.append(
            {
                "pair_id": f"{exp_id}_pair_{len(pair_rows):06d}",
                "file_a": item_a.file_path,
                "file_b": item_b.file_path,
                "label": "impostor",
                "family": item_a.family if item_a.family == item_b.family else f"{item_a.family}|{item_b.family}",
                "split": split_name,
                "prehash_score": cosine_score(item_a.projected_ref, item_b.projected_ref),
                "ber": float(ber),
                "same_source": False,
                "aug_view_a": "base",
                "aug_view_b": "base",
                "token_budget": token_budget,
                "aggregation_type": aggregation_type,
                "hard_negative_type": hard_type,
            }
        )

    metrics = metrics_evaluator.compute_metrics()
    exp_rows = [
        {
            "exp_id": exp_id,
            "token_budget": token_budget,
            "aggregation_type": aggregation_type,
            "batch_size": int(config["training"]["batch_size_files"]),
            "temperature": float(config["loss"]["temperature"]),
            "projection_dim": int(config["projector"]["output_dim"]),
            "hash_bits": int(config["signature"]["signature_bits"]),
            "auc": float(metrics.get("auc", np.nan)),
            "eer": float(metrics.get("eer", np.nan)),
            "separability": float(metrics.get("separability", np.nan)),
            "genuine_mean": float(metrics.get("genuine_ber_mean", np.nan)),
            "impostor_mean": float(metrics.get("impostor_ber_mean", np.nan)),
        }
    ]

    pair_df = pd.DataFrame(pair_rows)
    exp_df = pd.DataFrame(exp_rows)
    file_df = pd.DataFrame(file_rows)

    pair_path = output_dir / "pair_level_results.csv"
    exp_path = output_dir / "experiment_summary.csv"
    file_path = output_dir / "file_level_stats.csv"

    pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")
    exp_df.to_csv(exp_path, index=False, encoding="utf-8-sig")
    file_df.to_csv(file_path, index=False, encoding="utf-8-sig")

    manifest = {
        "pair_level_results": str(pair_path),
        "experiment_summary": str(exp_path),
        "file_level_stats": str(file_path),
        "n_pairs": int(len(pair_df)),
        "n_experiments": int(len(exp_df)),
        "n_files": int(len(file_df)),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("GeometryCore Analysis Tables Exported")
    print("=" * 90)
    print(f"Pair-level: {pair_path}")
    print(f"Experiment summary: {exp_path}")
    print(f"File-level stats: {file_path}")
    print(f"Manifest: {manifest_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
