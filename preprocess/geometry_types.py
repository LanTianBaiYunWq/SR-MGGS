"""
几何类型定义和数据结构
===============================

本模块定义了 GSD（Geometry Signature Distillation）系统中使用的
几何数据的层次化表示结构。

核心设计理念：
    不把 Multi* / GeometryCollection 压成一条坐标序列，
    而是采用"层次化表示：先编码 primitive，再对集合聚合"的方式。

主要组件：
    1. 原语类型常量（PRIM_*）：定义三种基本几何元素
       - PRIM_POINTSET: 点集合（Point/MultiPoint）
       - PRIM_LINE: 线（LineString）
       - PRIM_RING: 环（Polygon的外环/内环）
    
    2. 角色常量（ROLE_*）：区分多边形的外环和内环
       - ROLE_NONE: 非环类型
       - ROLE_EXTERIOR: 外环
       - ROLE_INTERIOR: 内环（洞）
    
    3. 容器类型常量（CONTAINER_*）：标识原始几何类型
       - 支持所有7种 Shapely 几何类型
    
    4. GeomPart 数据类：表示一个原子级别的几何部件
       - 包含坐标、类型、角色、权重信息
    
    5. GeometryData 数据类：表示完整的几何对象
       - 包含多个部件和元数据

使用示例：
    >>> from preprocess.geometry_types import GeomPart, PRIM_LINE, ROLE_NONE
    >>> part = GeomPart(
    ...     coords=np.array([[0, 0], [1, 1]], dtype=np.float32),
    ...     prim_type=PRIM_LINE,
    ...     role=ROLE_NONE,
    ...     weight=1.414
    ... )

作者：GSD Team
版本：2.0（支持所有几何类型）
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


# ============================================
# 几何原语类型常量
# ============================================
PRIM_POINTSET = 0   # Point / MultiPoint 统一为点集合
PRIM_LINE = 1       # LineString
PRIM_RING = 2       # Polygon 的外环/内环


# ============================================
# 角色常量（用于区分外环和内环）
# ============================================
ROLE_NONE = 0       # 非环类型
ROLE_EXTERIOR = 1   # 外环
ROLE_INTERIOR = 2   # 内环（洞）


# ============================================
# 容器类型（用于采样均衡）
# ============================================
CONTAINER_POINT = 0
CONTAINER_MULTIPOINT = 1
CONTAINER_LINESTRING = 2
CONTAINER_MULTILINESTRING = 3
CONTAINER_POLYGON = 4
CONTAINER_MULTIPOLYGON = 5
CONTAINER_GEOMETRYCOLLECTION = 6


# 类型名称映射
PRIM_TYPE_NAMES = {
    PRIM_POINTSET: "PointSet",
    PRIM_LINE: "Line",
    PRIM_RING: "Ring"
}

ROLE_NAMES = {
    ROLE_NONE: "None",
    ROLE_EXTERIOR: "Exterior",
    ROLE_INTERIOR: "Interior"
}

CONTAINER_TYPE_NAMES = {
    CONTAINER_POINT: "Point",
    CONTAINER_MULTIPOINT: "MultiPoint",
    CONTAINER_LINESTRING: "LineString",
    CONTAINER_MULTILINESTRING: "MultiLineString",
    CONTAINER_POLYGON: "Polygon",
    CONTAINER_MULTIPOLYGON: "MultiPolygon",
    CONTAINER_GEOMETRYCOLLECTION: "GeometryCollection"
}


@dataclass
class GeomPart:
    """
    几何原语部件
    
    表示一个原子级别的几何部件：
    - 点集合（Point/MultiPoint）
    - 线（LineString）
    - 环（Polygon的外环或内环）
    
    Attributes:
        coords: 坐标数组 (N, 2)，float32
        prim_type: 原语类型 (PRIM_POINTSET, PRIM_LINE, PRIM_RING)
        role: 角色类型 (ROLE_NONE, ROLE_EXTERIOR, ROLE_INTERIOR)
        weight: 聚合权重（基于长度/周长）
    """
    coords: np.ndarray      # (N, 2) float32
    prim_type: int          # PRIM_*
    role: int               # ROLE_*
    weight: float           # 聚合权重
    
    def __post_init__(self):
        """确保坐标是 float32 类型"""
        if self.coords is not None and self.coords.dtype != np.float32:
            self.coords = self.coords.astype(np.float32)
    
    @property
    def num_points(self) -> int:
        """返回点数"""
        return len(self.coords) if self.coords is not None else 0
    
    @property
    def prim_type_name(self) -> str:
        """返回原语类型名称"""
        return PRIM_TYPE_NAMES.get(self.prim_type, "Unknown")
    
    @property
    def role_name(self) -> str:
        """返回角色名称"""
        return ROLE_NAMES.get(self.role, "Unknown")
    
    def __repr__(self) -> str:
        return (f"GeomPart(type={self.prim_type_name}, role={self.role_name}, "
                f"points={self.num_points}, weight={self.weight:.4f})")


@dataclass
class GeometryData:
    """
    完整几何数据
    
    包含一个几何对象的所有部件和元信息
    
    Attributes:
        parts: 几何部件列表
        container_type: 容器类型（原始几何类型）
        geometry_id: 几何ID
        file_id: 所属文件ID
    """
    parts: List[GeomPart]
    container_type: int
    geometry_id: Optional[int] = None
    file_id: Optional[int] = None
    
    @property
    def num_parts(self) -> int:
        """返回部件数"""
        return len(self.parts)
    
    @property
    def total_points(self) -> int:
        """返回总点数"""
        return sum(p.num_points for p in self.parts)
    
    @property
    def container_type_name(self) -> str:
        """返回容器类型名称"""
        return CONTAINER_TYPE_NAMES.get(self.container_type, "Unknown")
    
    def __repr__(self) -> str:
        return (f"GeometryData(type={self.container_type_name}, "
                f"parts={self.num_parts}, points={self.total_points})")

