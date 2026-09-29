"""
LineString 家族的最小 token 化实现。
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from .types import FlattenedPrimitive, PrimitiveTokens


def _resample_open_polyline(coords: np.ndarray, n_tokens: int) -> np.ndarray:
    """按弧长对开放折线进行重采样。"""
    if len(coords) == 1:
        return np.repeat(coords, n_tokens, axis=0).astype(np.float32)

    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]

    if total < 1e-8:
        return np.repeat(coords[:1], n_tokens, axis=0).astype(np.float32)

    targets = np.linspace(0.0, total, n_tokens)
    x = np.interp(targets, cumulative, coords[:, 0])
    y = np.interp(targets, cumulative, coords[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _densify_if_needed(coords: np.ndarray, max_segment_length: float) -> np.ndarray:
    """
    对线段做 densify。

    这里用 shapely.segmentize（若可用）优先；否则退回原始坐标。
    """
    try:
        line = LineString(coords)
        densified = line.segmentize(max_segment_length)
        return np.array(densified.coords, dtype=np.float32)
    except Exception:
        return coords.astype(np.float32)


def tokenize_line_family(
    primitive: FlattenedPrimitive,
    max_tokens: int = 128,
    eps_len: float = 1e-8,
    densify_max_segment_length: float | None = None,
) -> PrimitiveTokens:
    """
    对 line 家族做最小 token 化。

    规则：
    - 唯一点数至少为 2
    - 长度必须大于 eps_len
    - 合法后可选 densify，再重采样到固定 token 数
    """
    if primitive.family != "line":
        raise ValueError(f"tokenize_line_family 只接受 line 家族，当前是 {primitive.family}")

    coords = primitive.coords.astype(np.float32)
    unique_coords = np.unique(coords, axis=0)
    if len(unique_coords) < 2:
        raise ValueError("线原语的唯一点数不足 2。")

    line = LineString(coords)
    if line.length <= eps_len:
        raise ValueError(f"线原语长度过短：{line.length} <= {eps_len}")

    if densify_max_segment_length is not None:
        coords = _densify_if_needed(coords, densify_max_segment_length)

    token_coords = _resample_open_polyline(coords, max_tokens)
    mask = np.ones(max_tokens, dtype=bool)

    return PrimitiveTokens(
        family="line",
        role=primitive.role,
        coords=token_coords,
        mask=mask,
        parent_id=primitive.parent_id,
        part_id=primitive.part_id,
        geom_type=primitive.geom_type,
        quality_flag="ok",
    )

