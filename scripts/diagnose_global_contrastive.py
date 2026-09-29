"""
诊断 GeometryCore / SGP 文件级全局训练中的对比学习退化问题。

这个脚本不训练模型，只做一次或少量 batch 的前向分析，重点回答：
1. 两个增强视图是否真的不同
2. 文件级 global feature 是否真的不同
3. projector 输出是否发生塌缩
4. NT-Xent 在当前 batch_size 下是否因为“没有负样本”而天然退化为 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from train_global import GlobalGSDTrainer, load_config


def resolve_path(path_str: str) -> Path:
    """兼容命令行和 IDE 的相对路径解析。"""
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = Path(__file__).resolve().parent.parent / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Diagnose global contrastive behavior.")
    parser.add_argument("--config", type=str, default="configs/sgp.yaml", help="配置文件路径。")
    parser.add_argument("--data_dir", type=str, default=None, help="覆盖数据目录。")
    parser.add_argument(
        "--dataset_backend",
        type=str,
        default="geometrycore",
        help="数据后端：sgp 或 geometrycore。",
    )
    parser.add_argument("--file_pattern", type=str, default="*.shp", help="文件过滤模式。")
    parser.add_argument("--max_files", type=int, default=4, help="最多加载多少个文件。")
    parser.add_argument("--max_geometries", type=int, default=64, help="每个文件最多保留多少个几何 token。")
    parser.add_argument("--batch_size", type=int, default=1, help="文件级 batch size。")
    parser.add_argument("--num_batches", type=int, default=1, help="最多诊断多少个 batch。")
    parser.add_argument("--no_teacher", action="store_true", help="关闭视觉教师。")
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    """汇总张量的基础统计量。"""
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """从配置文件和命令行参数构建最终配置。"""
    config = load_config(str(resolve_path(args.config)))

    if args.data_dir:
        config["data"]["raw_dir"] = args.data_dir

    config["data"]["dataset_backend"] = args.dataset_backend
    config["data"]["file_pattern"] = args.file_pattern
    config["data"]["max_files"] = args.max_files
    config["data"]["max_geometries"] = args.max_geometries
    config["training"]["batch_size_files"] = args.batch_size
    config["training"]["epochs"] = 1
    config["use_visual_teacher"] = not args.no_teacher

    return config


@torch.no_grad()
def diagnose_one_batch(trainer: GlobalGSDTrainer, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """对单个 batch 做前向诊断。"""
    views = batch["views"].to(trainer.device)
    masks = batch["masks"].to(trainer.device)

    batch_size = int(views.shape[0])
    n_views = int(views.shape[1])

    view1 = views[:, 0]
    view2 = views[:, 1]
    mask1 = masks[:, 0]
    mask2 = masks[:, 1]

    # 先看输入视图本身是否不同。
    view_abs_diff = (view1 - view2).abs()
    mean_view_diff = float(view_abs_diff.mean().item())
    max_view_diff = float(view_abs_diff.max().item())

    global_feat1, _ = trainer._encode_global(view1, mask1)
    global_feat2, _ = trainer._encode_global(view2, mask2)

    con_feat1 = trainer.contrastive_head(global_feat1)
    con_feat2 = trainer.contrastive_head(global_feat2)
    con_loss = trainer.contrastive_loss(con_feat1, con_feat2)

    feat_diff = global_feat1 - global_feat2
    proj_diff = con_feat1 - con_feat2

    # 对 NT-Xent 而言，每个 anchor 的有效负样本数为 2B-2。
    negatives_per_anchor = max(0, 2 * batch_size - 2)

    result = {
        "batch_size": batch_size,
        "n_views": n_views,
        "negatives_per_anchor": negatives_per_anchor,
        "mask_valid_count_view1": int(mask1.sum().item()),
        "mask_valid_count_view2": int(mask2.sum().item()),
        "mean_abs_view_diff": mean_view_diff,
        "max_abs_view_diff": max_view_diff,
        "global_feat1_stats": tensor_stats(global_feat1),
        "global_feat2_stats": tensor_stats(global_feat2),
        "contrastive_feat1_stats": tensor_stats(con_feat1),
        "contrastive_feat2_stats": tensor_stats(con_feat2),
        "mean_abs_global_diff": float(feat_diff.abs().mean().item()),
        "max_abs_global_diff": float(feat_diff.abs().max().item()),
        "mean_abs_projected_diff": float(proj_diff.abs().mean().item()),
        "max_abs_projected_diff": float(proj_diff.abs().max().item()),
        "global_cosine_mean": float(torch.nn.functional.cosine_similarity(global_feat1, global_feat2, dim=-1).mean().item()),
        "projected_cosine_mean": float(torch.nn.functional.cosine_similarity(con_feat1, con_feat2, dim=-1).mean().item()),
        "contrastive_loss": float(con_loss.item()),
        "explanation": (
            "当前 batch_size 下，每个 anchor 的有效负样本数为 2B-2。"
            " 如果这个值为 0，则 NT-Xent 会天然退化，loss 接近 0 是预期现象。"
        ),
    }
    return result


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    config = build_config(args)
    trainer = GlobalGSDTrainer(config)

    trainer.geometry_encoder.eval()
    trainer.aggregator.eval()
    trainer.projector.eval()
    trainer.contrastive_head.eval()

    results = []
    for batch_idx, batch in enumerate(trainer.train_loader):
        if batch_idx >= args.num_batches:
            break
        result = diagnose_one_batch(trainer, batch)
        result["batch_index"] = batch_idx
        results.append(result)

    report = {
        "arguments": {
            "config": args.config,
            "data_dir": args.data_dir,
            "dataset_backend": args.dataset_backend,
            "file_pattern": args.file_pattern,
            "max_files": args.max_files,
            "max_geometries": args.max_geometries,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "no_teacher": args.no_teacher,
        },
        "results": results,
    }

    print("=" * 90)
    print("Global Contrastive Diagnosis")
    print("=" * 90)
    for item in results:
        print(f"Batch {item['batch_index']}:")
        print(f"  batch_size: {item['batch_size']}")
        print(f"  negatives_per_anchor: {item['negatives_per_anchor']}")
        print(f"  mean_abs_view_diff: {item['mean_abs_view_diff']:.6f}")
        print(f"  mean_abs_global_diff: {item['mean_abs_global_diff']:.6f}")
        print(f"  mean_abs_projected_diff: {item['mean_abs_projected_diff']:.6f}")
        print(f"  global_cosine_mean: {item['global_cosine_mean']:.6f}")
        print(f"  projected_cosine_mean: {item['projected_cosine_mean']:.6f}")
        print(f"  contrastive_loss: {item['contrastive_loss']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
