"""
Point / MultiPoint 家族的最小 token 化实现。
"""

from __future__ import annotations

import numpy as np

from .types import FlattenedPrimitive, PrimitiveTokens


def tokenize_point_family(
    primitive: FlattenedPrimitive,
    max_tokens: int = 64,
) -> PrimitiveTokens:
    """
    对 point 家族做最小 token 化。

    规则：
    - 非空 point / multipoint 即可入模
    - 如果点数超过预算，则截断
    - 如果点数不足预算，则 padding
    """
    if primitive.family != "point":
        raise ValueError(f"tokenize_point_family 只接受 point 家族，当前是 {primitive.family}")

    coords = primitive.coords
    if len(coords) == 0:
        raise ValueError("空点集不能 token 化。")

    token_coords = np.zeros((max_tokens, 2), dtype=np.float32)
    mask = np.zeros(max_tokens, dtype=bool)

    n = min(len(coords), max_tokens)
    token_coords[:n] = coords[:n]
    mask[:n] = True

    return PrimitiveTokens(
        family="point",
        role=primitive.role,
        coords=token_coords,
        mask=mask,
        parent_id=primitive.parent_id,
        part_id=primitive.part_id,
        geom_type=primitive.geom_type,
        quality_flag="ok",
    )

