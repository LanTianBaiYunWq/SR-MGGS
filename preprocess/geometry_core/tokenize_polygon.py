"""
Polygon 家族的最小 token 化实现。
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .types import FlattenedPrimitive, PrimitiveTokens


def _signed_area(coords: np.ndarray) -> float:
    """计算闭合环的有符号面积。"""
    if len(coords) < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() * 0.5)


def _canonicalize_ring(coords: np.ndarray, role: str) -> np.ndarray:
    """
    统一环方向与起点。

    - shell 统一为逆时针
    - hole 统一为顺时针
    - 起点选字典序最小点
    """
    ring = coords.copy().astype(np.float32)
    area = _signed_area(ring)

    if role == "shell" and area < 0:
        ring = ring[::-1]
    if role == "hole" and area > 0:
        ring = ring[::-1]

    rounded = np.round(ring, 6)
    start_idx = np.lexsort((rounded[:, 1], rounded[:, 0]))[0]
    return np.roll(ring, -start_idx, axis=0)


def _resample_closed_ring(coords: np.ndarray, n_tokens: int) -> np.ndarray:
    """按弧长对闭合环进行均匀重采样。"""
    closed = np.vstack([coords, coords[:1]])
    diffs = np.diff(closed, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]

    if total < 1e-8:
        return np.repeat(coords[:1], n_tokens, axis=0).astype(np.float32)

    targets = np.linspace(0.0, total, n_tokens + 1)[:-1]
    x = np.interp(targets, cumulative, closed[:, 0])
    y = np.interp(targets, cumulative, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def tokenize_polygon_family(
    primitive: FlattenedPrimitive,
    max_tokens: int = 128,
    eps_area: float = 1e-10,
) -> PrimitiveTokens:
    """
    对 polygon 家族做最小 token 化。

    当前最小规则：
    - 输入 primitive 必须是 shell 或 hole ring
    - 唯一点数至少为 3
    - 面积绝对值必须大于 eps_area
    - 做环方向与起点规范化
    - 再重采样到固定 token 数
    """
    if primitive.family != "polygon":
        raise ValueError(f"tokenize_polygon_family 只接受 polygon 家族，当前是 {primitive.family}")

    coords = primitive.coords.astype(np.float32)
    unique_coords = np.unique(coords, axis=0)
    if len(unique_coords) < 3:
        raise ValueError("面原语的唯一点数不足 3。")

    canonical = _canonicalize_ring(coords, role=primitive.role)
    area = abs(_signed_area(canonical))
    if area <= eps_area:
        raise ValueError(f"面原语面积过小：{area} <= {eps_area}")

    # 用 polygon 只是为了增加一个最低限度的几何合法性检查。
    polygon = Polygon(canonical)
    if not polygon.is_valid:
        raise ValueError("面原语在最小 polygon 检查中无效。")

    token_coords = _resample_closed_ring(canonical, max_tokens)
    mask = np.ones(max_tokens, dtype=bool)

    return PrimitiveTokens(
        family="polygon",
        role=primitive.role,
        coords=token_coords,
        mask=mask,
        parent_id=primitive.parent_id,
        part_id=primitive.part_id,
        geom_type=primitive.geom_type,
        quality_flag="ok",
    )

