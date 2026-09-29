"""
roads_hier_stage1 hard negative 选择工具。

目标：
1. 从 Gate 1 cache 中提取文件级 size / density / budget 统计
2. 在这些粗统计空间里找到最相近的跨文件对
3. 为模型评估和 coarse baseline 复用同一套 hard negative pair
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def resolve_path(path_str: str) -> Path:
    """兼容相对路径和绝对路径。"""
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = Path(__file__).resolve().parents[1] / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def percentile(values: List[float], q: float) -> float:
    """简化版分位数统计。"""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def infer_line_subtype(file_path: str) -> str:
    """从 shapefile 路径推断 line subtype。"""
    normalized = str(file_path).lower()
    if "roads" in normalized:
        return "roads"
    if "railways" in normalized:
        return "railways"
    if "waterways" in normalized:
        return "waterways"
    return "unknown"


def summarize_tile_rows(tile_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """从 tiles.json 聚合文件级粗统计。"""
    raw_line_counts = [float(row.get("raw_line_count", 0.0)) for row in tile_rows]
    raw_chunk_counts = [float(row.get("raw_chunk_count", 0.0)) for row in tile_rows]
    budgeted_chunk_counts = [float(row.get("budgeted_chunk_count", 0.0)) for row in tile_rows]
    chunk_length_sums = [float(row.get("chunk_length_sum", 0.0)) for row in tile_rows]
    chunk_densities = [float(row.get("chunk_density", 0.0)) for row in tile_rows]
    tile_areas = [float(row.get("tile_area", 0.0)) for row in tile_rows]
    compression_ratios = [float(row.get("compression_ratio", 0.0)) for row in tile_rows]
    depths = [float(row.get("depth", 0.0)) for row in tile_rows]

    return {
        "tile_count": float(len(tile_rows)),
        "raw_line_sum": float(np.sum(raw_line_counts)),
        "raw_line_mean": float(np.mean(raw_line_counts)) if raw_line_counts else 0.0,
        "raw_line_p95": percentile(raw_line_counts, 95),
        "raw_chunk_sum": float(np.sum(raw_chunk_counts)),
        "raw_chunk_mean": float(np.mean(raw_chunk_counts)) if raw_chunk_counts else 0.0,
        "raw_chunk_p95": percentile(raw_chunk_counts, 95),
        "budgeted_chunk_sum": float(np.sum(budgeted_chunk_counts)),
        "budgeted_chunk_mean": float(np.mean(budgeted_chunk_counts)) if budgeted_chunk_counts else 0.0,
        "budgeted_chunk_p95": percentile(budgeted_chunk_counts, 95),
        "chunk_length_sum_total": float(np.sum(chunk_length_sums)),
        "chunk_length_sum_mean": float(np.mean(chunk_length_sums)) if chunk_length_sums else 0.0,
        "chunk_length_sum_p95": percentile(chunk_length_sums, 95),
        "chunk_density_mean": float(np.mean(chunk_densities)) if chunk_densities else 0.0,
        "chunk_density_p95": percentile(chunk_densities, 95),
        "tile_area_sum": float(np.sum(tile_areas)),
        "tile_area_mean": float(np.mean(tile_areas)) if tile_areas else 0.0,
        "compression_ratio_mean": float(np.mean(compression_ratios)) if compression_ratios else 0.0,
        "compression_ratio_p95": percentile(compression_ratios, 95),
        "depth_mean": float(np.mean(depths)) if depths else 0.0,
        "depth_max": float(np.max(depths)) if depths else 0.0,
    }


def load_cache_feature_rows(cache_root: str, max_files: int | None = None) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    """从 cache 中加载文件级统计特征。"""
    cache_root_path = resolve_path(cache_root)
    index_path = cache_root_path / "index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    items = index_data["files"]
    if max_files is not None:
        items = items[:max_files]

    feature_rows: List[Dict[str, Any]] = []
    feature_names: List[str] | None = None
    feature_vectors: List[np.ndarray] = []

    for item in items:
        file_dir = cache_root_path / item["file_id"]
        manifest_path = file_dir / "manifest.json"
        tiles_path = file_dir / "tiles.json"

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        with tiles_path.open("r", encoding="utf-8") as f:
            tile_rows = json.load(f)

        manifest_stats = manifest.get("tile_chunk_count_stats", {})
        tile_stats = summarize_tile_rows(tile_rows)

        feature_dict = {
            "file_idx": len(feature_rows),
            "file_id": item["file_id"],
            "file_path": item["file_path"],
            "family": str(manifest.get("family", "unknown")),
            "subtype": infer_line_subtype(item["file_path"]),
            "num_tiles_manifest": float(manifest.get("num_tiles", 0.0)),
            "num_chunks_manifest": float(manifest.get("num_chunks", 0.0)),
            "file_bbox_diagonal": float(manifest.get("file_bbox_diagonal", 0.0)),
            "chunk_p50_manifest": float(manifest_stats.get("p50", 0.0)),
            "chunk_p95_manifest": float(manifest_stats.get("p95", 0.0)),
            "chunk_max_manifest": float(manifest_stats.get("max", 0.0)),
            "compression_p50_manifest": float(manifest_stats.get("compression_ratio_p50", 0.0)),
            "compression_p95_manifest": float(manifest_stats.get("compression_ratio_p95", 0.0)),
            "budget_hit_ratio_manifest": float(manifest_stats.get("budget_hit_ratio", 0.0)),
            **tile_stats,
        }

        current_names = [key for key in feature_dict.keys() if key not in {"file_idx", "file_id", "file_path", "family", "subtype"}]
        if feature_names is None:
            feature_names = current_names

        feature_rows.append(feature_dict)
        feature_vectors.append(np.asarray([feature_dict[name] for name in feature_names], dtype=np.float64))

    if feature_names is None:
        raise ValueError("No files loaded from cache.")

    feature_matrix = np.vstack(feature_vectors)
    mean = np.mean(feature_matrix, axis=0, keepdims=True)
    std = np.std(feature_matrix, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    feature_matrix = (feature_matrix - mean) / std
    return feature_rows, feature_names, feature_matrix


def collect_impostor_pairs(num_files: int, max_pairs: int | None = None) -> List[Tuple[int, int]]:
    """返回所有或截断后的 impostor pair。"""
    all_pairs = list(combinations(range(num_files), 2))
    if max_pairs is None or len(all_pairs) <= max_pairs:
        return all_pairs
    return all_pairs[:max_pairs]


def relative_difference(a: float, b: float, eps: float = 1e-8) -> float:
    """对称相对差异，范围约为 [0, +inf)。"""
    scale = max(abs(a), abs(b), eps)
    return float(abs(a - b) / scale)


def is_coarse_match(
    row_i: Dict[str, Any],
    row_j: Dict[str, Any],
    *,
    ratio_tolerance: float,
    budget_tolerance: float,
    diag_tolerance: float,
) -> bool:
    """判断两份文件级 coarse stats 是否足够接近。"""
    ratio_keys = [
        "num_tiles_manifest",
        "num_chunks_manifest",
        "chunk_length_sum_total",
    ]
    for key in ratio_keys:
        if relative_difference(float(row_i[key]), float(row_j[key])) > ratio_tolerance:
            return False

    if abs(float(row_i["budget_hit_ratio_manifest"]) - float(row_j["budget_hit_ratio_manifest"])) > budget_tolerance:
        return False

    if relative_difference(float(row_i["file_bbox_diagonal"]), float(row_j["file_bbox_diagonal"])) > diag_tolerance:
        return False

    return True


def build_size_matched_pairs(
    cache_root: str,
    *,
    max_files: int | None = None,
    topk_per_file: int = 2,
    max_pairs: int | None = None,
) -> List[Tuple[int, int]]:
    """
    按文件级粗统计相似度构造 hard negative pairs。

    规则：
    - 先在标准化 coarse-stat 空间中计算欧氏距离
    - 每个文件选最相近的 top-k 其他文件
    - 去重后得到最终 pairs
    """
    feature_rows, _, feature_matrix = load_cache_feature_rows(cache_root, max_files=max_files)
    num_files = len(feature_rows)
    if num_files < 2:
        return []

    distances = np.linalg.norm(
        feature_matrix[:, None, :] - feature_matrix[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(distances, np.inf)

    pair_set: set[Tuple[int, int]] = set()
    for i in range(num_files):
        neighbor_order = np.argsort(distances[i])[: max(1, topk_per_file)]
        for j in neighbor_order:
            pair = (i, j) if i < j else (j, i)
            pair_set.add(pair)

    ranked_pairs = sorted(pair_set, key=lambda pair: distances[pair[0], pair[1]])
    if max_pairs is not None:
        ranked_pairs = ranked_pairs[:max_pairs]
    return ranked_pairs


def build_subtype_matched_pairs(
    cache_root: str,
    *,
    max_files: int | None = None,
    max_pairs: int | None = None,
) -> List[Tuple[int, int]]:
    """只保留同 family/subtype 内的 impostor pairs。"""
    feature_rows, _, _ = load_cache_feature_rows(cache_root, max_files=max_files)
    num_files = len(feature_rows)
    if num_files < 2:
        return []

    subtype_pairs = [
        (i, j)
        for i, j in combinations(range(num_files), 2)
        if feature_rows[i]["subtype"] == feature_rows[j]["subtype"]
    ]
    if max_pairs is None or len(subtype_pairs) <= max_pairs:
        return subtype_pairs
    return subtype_pairs[:max_pairs]


def build_subtype_size_matched_pairs(
    cache_root: str,
    *,
    max_files: int | None = None,
    topk_per_file: int = 2,
    max_pairs: int | None = None,
) -> List[Tuple[int, int]]:
    """先约束同 subtype，再按 coarse-stat 邻近度选 hard negatives。"""
    feature_rows, _, feature_matrix = load_cache_feature_rows(cache_root, max_files=max_files)
    num_files = len(feature_rows)
    if num_files < 2:
        return []

    distances = np.linalg.norm(
        feature_matrix[:, None, :] - feature_matrix[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(distances, np.inf)

    pair_set: set[Tuple[int, int]] = set()
    for i in range(num_files):
        same_subtype_js = [
            j
            for j in range(num_files)
            if i != j and feature_rows[i]["subtype"] == feature_rows[j]["subtype"]
        ]
        if not same_subtype_js:
            continue

        same_subtype_js.sort(key=lambda j: distances[i, j])
        for j in same_subtype_js[: max(1, topk_per_file)]:
            pair = (i, j) if i < j else (j, i)
            pair_set.add(pair)

    ranked_pairs = sorted(pair_set, key=lambda pair: distances[pair[0], pair[1]])
    if max_pairs is not None:
        ranked_pairs = ranked_pairs[:max_pairs]
    return ranked_pairs


def build_coarse_matched_pairs(
    cache_root: str,
    *,
    max_files: int | None = None,
    topk_per_file: int = 2,
    max_pairs: int | None = None,
    ratio_tolerance: float = 0.25,
    budget_tolerance: float = 0.10,
    diag_tolerance: float = 0.25,
) -> List[Tuple[int, int]]:
    """
    先按关键 coarse stats 做匹配筛选，再在候选里选最近邻。
    关键约束：
    - num_tiles / num_chunks / chunk_length_sum_total 的相对差异
    - budget_hit_ratio 的绝对差异
    - file_bbox_diagonal 的相对差异
    """
    feature_rows, _, feature_matrix = load_cache_feature_rows(cache_root, max_files=max_files)
    num_files = len(feature_rows)
    if num_files < 2:
        return []

    distances = np.linalg.norm(
        feature_matrix[:, None, :] - feature_matrix[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(distances, np.inf)

    pair_set: set[Tuple[int, int]] = set()
    for i in range(num_files):
        candidate_js: List[int] = []
        for j in range(num_files):
            if i == j:
                continue
            if is_coarse_match(
                feature_rows[i],
                feature_rows[j],
                ratio_tolerance=ratio_tolerance,
                budget_tolerance=budget_tolerance,
                diag_tolerance=diag_tolerance,
            ):
                candidate_js.append(j)

        if not candidate_js:
            continue

        candidate_js.sort(key=lambda j: distances[i, j])
        for j in candidate_js[: max(1, topk_per_file)]:
            pair = (i, j) if i < j else (j, i)
            pair_set.add(pair)

    ranked_pairs = sorted(pair_set, key=lambda pair: distances[pair[0], pair[1]])
    if max_pairs is not None:
        ranked_pairs = ranked_pairs[:max_pairs]
    return ranked_pairs
