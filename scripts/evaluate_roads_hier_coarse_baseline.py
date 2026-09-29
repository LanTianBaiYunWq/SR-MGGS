"""
评估 roads_hier_stage1 的 coarse-stat baseline。

目标：
1. 不使用神经网络
2. 直接从 Gate 1 cache 中提取文件级粗统计特征
3. 在 same-file vs cross-file 设定下，估计这些粗统计本身能分到什么程度
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from roads_hier_stage1.hard_negative_mining import (
    build_coarse_matched_pairs,
    build_size_matched_pairs,
    build_subtype_matched_pairs,
    build_subtype_size_matched_pairs,
    collect_impostor_pairs,
    load_cache_feature_rows,
    resolve_path,
)
from utils.metrics import compute_auc, compute_eer


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
    parser = argparse.ArgumentParser(description="Evaluate coarse-stat baseline on roads hierarchical cache.")
    parser.add_argument("--cache_root", type=str, required=True, help="Gate 1 cache 根目录。")
    parser.add_argument("--max_files", type=int, default=None, help="最多评估多少个文件。")
    parser.add_argument(
        "--genuine_jitter_std",
        type=float,
        default=0.01,
        help="same-file genuine 对中加入的标准化特征抖动强度，避免完全相同向量造成退化。",
    )
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
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """计算余弦相似度。"""
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
    return float(np.dot(x, y) / denom)


def cosine_to_ber(score: float) -> float:
    """把余弦相似度映射到类 BER 距离。"""
    return float((1.0 - score) / 2.0)


def compute_separability(genuine_bers: np.ndarray, impostor_bers: np.ndarray) -> float:
    """与主评估口径一致的 separability。"""
    return float(
        (np.mean(impostor_bers) - np.mean(genuine_bers))
        / (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8)
    )


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    cache_root = resolve_path(args.cache_root)
    feature_rows, feature_names, feature_matrix = load_cache_feature_rows(str(cache_root), args.max_files)

    rng = np.random.default_rng(42)
    genuine_rows: List[Dict[str, Any]] = []
    genuine_scores: List[float] = []
    genuine_bers: List[float] = []

    for row, feature in zip(feature_rows, feature_matrix):
        jitter_a = rng.normal(0.0, args.genuine_jitter_std, size=feature.shape)
        jitter_b = rng.normal(0.0, args.genuine_jitter_std, size=feature.shape)
        feat_a = feature + jitter_a
        feat_b = feature + jitter_b
        score = cosine_similarity(feat_a, feat_b)
        ber = cosine_to_ber(score)
        genuine_scores.append(score)
        genuine_bers.append(ber)
        genuine_rows.append(
            {
                "file_id": row["file_id"],
                "file_path": row["file_path"],
                "prehash_score": score,
                "ber": ber,
            }
        )

    impostor_rows: List[Dict[str, Any]] = []
    impostor_scores: List[float] = []
    impostor_bers: List[float] = []

    if args.negative_mode == "size_matched":
        impostor_pairs = build_size_matched_pairs(
            str(cache_root),
            max_files=len(feature_rows),
            topk_per_file=args.hard_negative_topk,
            max_pairs=args.max_impostor_pairs,
        )
    elif args.negative_mode == "subtype_matched":
        impostor_pairs = build_subtype_matched_pairs(
            str(cache_root),
            max_files=len(feature_rows),
            max_pairs=args.max_impostor_pairs,
        )
    elif args.negative_mode == "subtype_size_matched":
        impostor_pairs = build_subtype_size_matched_pairs(
            str(cache_root),
            max_files=len(feature_rows),
            topk_per_file=args.hard_negative_topk,
            max_pairs=args.max_impostor_pairs,
        )
    elif args.negative_mode == "coarse_matched":
        impostor_pairs = build_coarse_matched_pairs(
            str(cache_root),
            max_files=len(feature_rows),
            topk_per_file=args.hard_negative_topk,
            max_pairs=args.max_impostor_pairs,
            ratio_tolerance=args.match_ratio_tol,
            budget_tolerance=args.match_budget_tol,
            diag_tolerance=args.match_diag_tol,
        )
    else:
        impostor_pairs = collect_impostor_pairs(len(feature_rows), args.max_impostor_pairs)

    for i, j in impostor_pairs:
        feat_i = feature_matrix[i]
        feat_j = feature_matrix[j]
        score = cosine_similarity(feat_i, feat_j)
        ber = cosine_to_ber(score)
        impostor_scores.append(score)
        impostor_bers.append(ber)
        impostor_rows.append(
            {
                "file_id_i": feature_rows[i]["file_id"],
                "file_id_j": feature_rows[j]["file_id"],
                "file_path_i": feature_rows[i]["file_path"],
                "file_path_j": feature_rows[j]["file_path"],
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
            "cache_root": str(cache_root),
            "max_files": args.max_files,
            "genuine_jitter_std": args.genuine_jitter_std,
            "max_impostor_pairs": args.max_impostor_pairs,
            "negative_mode": args.negative_mode,
            "hard_negative_topk": args.hard_negative_topk,
            "match_ratio_tol": args.match_ratio_tol,
            "match_budget_tol": args.match_budget_tol,
            "match_diag_tol": args.match_diag_tol,
        },
        "n_files": len(feature_rows),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_genuine_pairs": len(genuine_rows),
        "n_impostor_pairs": len(impostor_rows),
        "metrics": metrics,
        "genuine_examples": genuine_rows[:10],
        "hard_impostors": impostor_rows[:10],
        "file_features": [
            {
                key: value
                for key, value in row.items()
                if key != "file_path"
            }
            for row in feature_rows[:10]
        ],
    }

    print("=" * 90)
    print("Roads Hier Coarse-Stat Baseline")
    print("=" * 90)
    print(f"Files: {report['n_files']}")
    print(f"Feature dims: {report['n_features']}")
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
