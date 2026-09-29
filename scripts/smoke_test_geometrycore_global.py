"""
GeometryCore file-level global dataset 最小烟雾测试。

这个脚本不训练模型，只验证新的数据入口是否已经满足最基本要求：
1. 能否按文件级样本加载 shapefile
2. Point / Line / Polygon 家族是否都能进入新前端
3. 输出张量形状是否兼容现有 global trainer
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from preprocess.global_geometrycore_dataset import (
    GeometryCoreGlobalContrastiveDataset,
    GeometryCoreGlobalDataset,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Smoke test the GeometryCore-based file-level global dataset."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="原始 shapefile 根目录。")
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.shp",
        help="文件名过滤模式，例如 *roads*.shp 或 *railways*.shp。",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=20,
        help="最多加载多少个文件，用于快速测试。",
    )
    parser.add_argument(
        "--max_geometries",
        type=int,
        default=128,
        help="每个文件最多保留多少个 geometry token。",
    )
    parser.add_argument(
        "--min_geometries",
        type=int,
        default=1,
        help="文件至少需要多少个有效 geometry token。",
    )
    parser.add_argument(
        "--fixed_points",
        type=int,
        default=128,
        help="每个 geometry token 的固定点数。",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=2,
        help="对比学习视图数量。",
    )
    parser.add_argument("--disable_point", action="store_true", help="关闭 point 家族。")
    parser.add_argument("--disable_line", action="store_true", help="关闭 line 家族。")
    parser.add_argument("--disable_polygon", action="store_true", help="关闭 polygon 家族。")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭 dataset 逐文件加载日志，只保留最终摘要。",
    )
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def summarize_base_dataset(dataset: GeometryCoreGlobalDataset) -> Dict:
    """汇总基础数据集信息，统计全部已加载样本。"""
    family_totals: Counter = Counter()
    token_counts: List[int] = []
    sample_rows: List[Dict] = []
    rejected_primitives_total = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        metadata = sample["metadata"]
        n_geometries = int(sample["n_geometries"])
        token_counts.append(n_geometries)
        family_totals.update(metadata["family_counts"])
        rejected_primitives_total += int(metadata["rejected_primitives"])

        # 只保留少量样本明细，避免输出文件过大。
        if idx < min(5, len(dataset)):
            sample_rows.append(
                {
                    "file_idx": int(sample["file_idx"]),
                    "file_path": sample["file_path"],
                    "n_geometries": n_geometries,
                    "family_counts": metadata["family_counts"],
                    "working_crs": metadata["working_crs"],
                    "rejected_primitives": int(metadata["rejected_primitives"]),
                    "geometries_shape": list(sample["geometries"].shape),
                    "mask_shape": list(sample["mask"].shape),
                    "valid_mask_count": int(sample["mask"].sum().item()),
                }
            )

    return {
        "loaded_files": len(dataset),
        "family_totals": dict(family_totals),
        "rejected_primitives_total": rejected_primitives_total,
        "token_count_stats": {
            "min": min(token_counts) if token_counts else None,
            "max": max(token_counts) if token_counts else None,
            "mean": (sum(token_counts) / len(token_counts)) if token_counts else None,
        },
        "samples": sample_rows,
    }


def summarize_contrastive_dataset(dataset: GeometryCoreGlobalContrastiveDataset) -> Dict:
    """验证对比学习视图格式是否符合预期。"""
    sample_rows: List[Dict] = []

    for idx in range(min(3, len(dataset))):
        sample = dataset[idx]
        sample_rows.append(
            {
                "file_idx": int(sample["file_idx"]),
                "file_path": sample["file_path"],
                "views_shape": list(sample["views"].shape),
                "masks_shape": list(sample["masks"].shape),
                "n_geometries": int(sample["n_geometries"]),
                "valid_mask_count_view0": int(sample["masks"][0].sum().item()),
            }
        )

    return {
        "inspected_files": len(sample_rows),
        "samples": sample_rows,
    }


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    # 这里的 max_files 和 file_pattern 会直接传给 dataset，
    # 因此烟雾测试不会再把全量文件全部读进来。
    base_dataset = GeometryCoreGlobalDataset(
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        max_files=args.max_files,
        max_geometries=args.max_geometries,
        min_geometries=args.min_geometries,
        fixed_points=args.fixed_points,
        point_enabled=not args.disable_point,
        line_enabled=not args.disable_line,
        polygon_enabled=not args.disable_polygon,
        verbose=not args.quiet,
    )

    contrastive_dataset = GeometryCoreGlobalContrastiveDataset(
        base_dataset=base_dataset,
        n_views=args.n_views,
    )

    base_summary = summarize_base_dataset(base_dataset)
    contrastive_summary = summarize_contrastive_dataset(contrastive_dataset)

    report = {
        "arguments": {
            "data_dir": args.data_dir,
            "file_pattern": args.file_pattern,
            "max_files": args.max_files,
            "max_geometries": args.max_geometries,
            "min_geometries": args.min_geometries,
            "fixed_points": args.fixed_points,
            "n_views": args.n_views,
        },
        "base_dataset": base_summary,
        "contrastive_dataset": contrastive_summary,
    }

    print("=" * 90)
    print("GeometryCore Global Dataset Smoke Test")
    print("=" * 90)
    print(f"File pattern: {args.file_pattern}")
    print(f"Loaded files: {base_summary['loaded_files']}")
    print(f"Family totals: {base_summary['family_totals']}")
    print(f"Rejected primitives total: {base_summary['rejected_primitives_total']}")
    print(f"Token count stats: {base_summary['token_count_stats']}")

    if base_summary["samples"]:
        first = base_summary["samples"][0]
        print("First sample:")
        print(f"  file_path: {first['file_path']}")
        print(f"  n_geometries: {first['n_geometries']}")
        print(f"  family_counts: {first['family_counts']}")
        print(f"  geometries_shape: {first['geometries_shape']}")
        print(f"  mask_shape: {first['mask_shape']}")

    if contrastive_summary["samples"]:
        first_contrastive = contrastive_summary["samples"][0]
        print("First contrastive sample:")
        print(f"  views_shape: {first_contrastive['views_shape']}")
        print(f"  masks_shape: {first_contrastive['masks_shape']}")

    print("=" * 90)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
