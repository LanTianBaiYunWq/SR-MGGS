"""
评估 roads_hier_stage1 最小训练链的 file-level prehash 分离。

当前只看连续 embedding：
1. genuine: 同一文件的两个随机增强视图
2. impostor: 不同文件之间的文件级 embedding 比较
3. 不接 hash / BCH
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from models.projector import SimCLRProjector
from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset
from roads_hier_stage1.hard_negative_mining import (
    build_coarse_matched_pairs,
    build_size_matched_pairs,
    build_subtype_matched_pairs,
    build_subtype_size_matched_pairs,
    collect_impostor_pairs,
)
from roads_hier_stage1.minimal_signature_model import MinimalRoadsHierarchicalEncoder
from train_roads_hier_stage1_minimal import (
    build_augmented_tile_features,
    load_config,
    resolve_path,
)
from utils.metrics import compute_auc, compute_eer
from utils.seed import set_seed


def to_jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组递归转换成 JSON 可写类型。"""
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


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate minimal roads hierarchical stage1 checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="最小训练 checkpoint 路径。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/roads_hier_stage1_minimal.yaml",
        help="配置文件路径。",
    )
    parser.add_argument("--cache_root", type=str, default=None, help="覆盖 cache 根目录。")
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="eval",
        choices=["train", "eval", "all"],
        help="评估时使用的 tile 选择模式。",
    )
    parser.add_argument("--max_files", type=int, default=None, help="最多评估多少个文件。")
    parser.add_argument("--max_tiles", type=int, default=None, help="覆盖读取的 tile 上限。")
    parser.add_argument(
        "--prehash_space",
        type=str,
        default="encoder",
        choices=["encoder", "projector"],
        help="预哈希空间：encoder 输出或 projector 输出。",
    )
    parser.add_argument("--genuine_views", type=int, default=2, help="每个文件生成多少个 genuine 视图。")
    parser.add_argument("--max_impostor_pairs", type=int, default=None, help="最多评估多少个 impostor 对。")
    parser.add_argument(
        "--negative_mode",
        type=str,
        default="all",
        choices=["all", "size_matched", "coarse_matched", "subtype_matched", "subtype_size_matched"],
        help="impostor 对构造方式。",
    )
    parser.add_argument(
        "--hard_negative_topk",
        type=int,
        default=2,
        help="size-matched 模式下每个文件保留多少个最近邻负样本。",
    )
    parser.add_argument("--match_ratio_tol", type=float, default=0.25, help="coarse_matched 下尺寸统计的相对差异阈值。")
    parser.add_argument("--match_budget_tol", type=float, default=0.10, help="coarse_matched 下 budget_hit_ratio 的绝对差异阈值。")
    parser.add_argument("--match_diag_tol", type=float, default=0.25, help="coarse_matched 下 bbox diagonal 的相对差异阈值。")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备。")
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """计算余弦相似度。"""
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
    return float(np.dot(x, y) / denom)


def cosine_to_ber(score: float) -> float:
    """把余弦相似度映射到类 BER 距离，范围约为 [0, 1]。"""
    return float((1.0 - score) / 2.0)


def compute_separability(genuine_bers: np.ndarray, impostor_bers: np.ndarray) -> float:
    """与旧评估口径一致的 separability。"""
    return float(
        (np.mean(impostor_bers) - np.mean(genuine_bers))
        / (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8)
    )


class MinimalRoadsHierEvaluator:
    """最小 roads hierarchical checkpoint 评估器。"""

    def __init__(self, checkpoint_path: str, config: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config

        checkpoint = torch.load(resolve_path(checkpoint_path), map_location=self.device)
        model_cfg = checkpoint["config"]["model"]

        self.encoder = MinimalRoadsHierarchicalEncoder(
            chunk_feature_dim=model_cfg.get("chunk_feature_dim", 4),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            file_dim=model_cfg.get("file_dim", 128),
            num_heads=model_cfg.get("num_heads", 4),
            num_set_layers=model_cfg.get("num_set_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(self.device)
        self.projector = SimCLRProjector(
            input_dim=model_cfg.get("file_dim", 128),
            hidden_dim=model_cfg.get("file_dim", 128),
            output_dim=model_cfg.get("projector_dim", 128),
        ).to(self.device)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.projector.load_state_dict(checkpoint["projector"])
        self.encoder.eval()
        self.projector.eval()

    @torch.no_grad()
    def encode_view(self, sample: Dict[str, object], prehash_space: str) -> np.ndarray:
        """编码一个文件视图。"""
        aug_cfg = self.config["augmentation"]
        tile_feature_list = build_augmented_tile_features(
            sample,
            tile_keep_prob=aug_cfg.get("tile_keep_prob", 0.85),
            chunk_keep_prob=aug_cfg.get("chunk_keep_prob", 0.85),
            feature_noise_std=aug_cfg.get("feature_noise_std", 0.01),
            device=self.device,
        )
        file_embedding = self.encoder(tile_feature_list)
        if prehash_space == "projector":
            file_embedding = self.projector(file_embedding.unsqueeze(0))[0]
        return file_embedding.detach().cpu().numpy()


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cache_root:
        config["data"]["cache_root"] = args.cache_root
    if args.max_files is not None:
        config["data"]["max_files"] = args.max_files
    if args.max_tiles is not None:
        config["data"]["max_tiles"] = args.max_tiles

    set_seed(config["device"]["seed"])

    dataset = GeometryCoreHierDataset(
        config["data"]["cache_root"],
        mode=args.dataset_mode,
        max_tiles=config["data"].get("max_tiles"),
        tile_selector=config["data"].get("tile_selector", "manifest_default"),
    )
    num_files = len(dataset) if args.max_files is None else min(len(dataset), args.max_files)
    evaluator = MinimalRoadsHierEvaluator(args.checkpoint, config, device=args.device)

    file_rows: List[Dict[str, Any]] = []
    genuine_rows: List[Dict[str, Any]] = []
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []

    for idx in range(num_files):
        sample = dataset[idx]
        embeddings = [
            evaluator.encode_view(sample, prehash_space=args.prehash_space)
            for _ in range(max(args.genuine_views, 2))
        ]
        file_rows.append(
            {
                "file_idx": idx,
                "file_id": sample["file_id"],
                "file_path": sample["file_path"],
                "num_selected_tiles": sample["num_selected_tiles"],
                "embedding_norm_mean": float(np.mean([np.linalg.norm(x) for x in embeddings])),
            }
        )

        ref = embeddings[0]
        aug = embeddings[1]
        score = cosine_similarity(ref, aug)
        ber = cosine_to_ber(score)
        genuine_scores.append(score)
        genuine_bers.append(ber)
        genuine_rows.append(
            {
                "file_idx": idx,
                "file_id": sample["file_id"],
                "file_path": sample["file_path"],
                "prehash_score": score,
                "ber": ber,
                "num_selected_tiles": sample["num_selected_tiles"],
            }
        )

        file_rows[-1]["embedding_ref"] = ref

    impostor_rows: List[Dict[str, Any]] = []
    impostor_scores: List[float] = []
    impostor_bers: List[float] = []

    if args.negative_mode == "size_matched":
        impostor_pairs = build_size_matched_pairs(
            config["data"]["cache_root"],
            max_files=num_files,
            topk_per_file=args.hard_negative_topk,
            max_pairs=args.max_impostor_pairs,
        )
    elif args.negative_mode == "subtype_matched":
        impostor_pairs = build_subtype_matched_pairs(
            config["data"]["cache_root"],
            max_files=num_files,
            max_pairs=args.max_impostor_pairs,
        )
    elif args.negative_mode == "subtype_size_matched":
        impostor_pairs = build_subtype_size_matched_pairs(
            config["data"]["cache_root"],
            max_files=num_files,
            topk_per_file=args.hard_negative_topk,
            max_pairs=args.max_impostor_pairs,
        )
    elif args.negative_mode == "coarse_matched":
        impostor_pairs = build_coarse_matched_pairs(
            config["data"]["cache_root"],
            max_files=num_files,
            topk_per_file=args.hard_negative_topk,
            max_pairs=args.max_impostor_pairs,
            ratio_tolerance=args.match_ratio_tol,
            budget_tolerance=args.match_budget_tol,
            diag_tolerance=args.match_diag_tol,
        )
    else:
        impostor_pairs = collect_impostor_pairs(num_files, args.max_impostor_pairs)

    for i, j in impostor_pairs:
        emb_i = file_rows[i]["embedding_ref"]
        emb_j = file_rows[j]["embedding_ref"]
        score = cosine_similarity(emb_i, emb_j)
        ber = cosine_to_ber(score)
        impostor_scores.append(score)
        impostor_bers.append(ber)
        impostor_rows.append(
            {
                "file_idx_i": i,
                "file_idx_j": j,
                "file_id_i": file_rows[i]["file_id"],
                "file_id_j": file_rows[j]["file_id"],
                "file_path_i": file_rows[i]["file_path"],
                "file_path_j": file_rows[j]["file_path"],
                "prehash_score": score,
                "ber": ber,
            }
        )

    impostor_rows.sort(key=lambda row: row["ber"])
    genuine_scores_np = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores_np = np.asarray(impostor_scores, dtype=np.float64)
    genuine_bers_np = np.asarray(genuine_bers, dtype=np.float64)
    impostor_bers_np = np.asarray(impostor_bers, dtype=np.float64)

    metrics = {
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
            "genuine_views": args.genuine_views,
            "max_impostor_pairs": args.max_impostor_pairs,
            "negative_mode": args.negative_mode,
            "hard_negative_topk": args.hard_negative_topk,
            "match_ratio_tol": args.match_ratio_tol,
            "match_budget_tol": args.match_budget_tol,
            "match_diag_tol": args.match_diag_tol,
            "device": args.device,
        },
        "n_files": num_files,
        "n_genuine_pairs": len(genuine_rows),
        "n_impostor_pairs": len(impostor_rows),
        "metrics": metrics,
        "genuine_examples": genuine_rows[:10],
        "hard_impostors": impostor_rows[:10],
    }

    print("=" * 90)
    print("Roads Hier Stage1 Minimal Evaluation")
    print("=" * 90)
    print(f"Files: {report['n_files']}")
    print(f"Genuine pairs: {report['n_genuine_pairs']}")
    print(f"Impostor pairs: {report['n_impostor_pairs']}")
    print(f"Genuine mean:  {metrics['genuine_mean']:.6f}")
    print(f"Impostor mean: {metrics['impostor_mean']:.6f}")
    print(f"AUC:           {metrics['auc']:.6f}")
    print(f"EER:           {metrics['eer']:.6f}")
    print(f"Separability:  {metrics['separability']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
