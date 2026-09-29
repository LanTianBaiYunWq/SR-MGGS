"""
GeometryCore 中使用的数据结构定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FlattenedPrimitive:
    """
    展开后的最小几何原语。

    设计目的：
    - Multi* 与 GeometryCollection 展开后，每个叶子几何都变成一个原语
    - 保留 parent_id / part_id，便于之后重新汇总到原始 feature 或文件级
    - family 表示该原语属于 point / line / polygon 三大家族中的哪一类
    """

    parent_id: int
    part_id: int
    family: str
    geom_type: str
    coords: np.ndarray
    role: str = "main"
    is_hole: bool = False


@dataclass
class PrimitiveTokens:
    """
    送给后续模型的最小 token 表示。

    当前版本只保留最必要的信息：
    - 坐标 token
    - 有效点 mask
    - family / role 元数据
    """

    family: str
    role: str
    coords: np.ndarray
    mask: np.ndarray
    parent_id: int
    part_id: int
    geom_type: str
    quality_flag: str = "ok"

