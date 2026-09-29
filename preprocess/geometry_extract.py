"""
几何提取模块
===============================

本模块负责将 Shapely 几何对象转换为层次化部件表示。

支持的几何类型：
    - Point: 单点 → 点集合部件（1个点）
    - MultiPoint: 多点 → 点集合部件（多个点）
    - LineString: 线 → 线部件
    - MultiLineString: 多线 → 多个线部件
    - Polygon: 多边形 → 外环部件 + 内环部件（可选）
    - MultiPolygon: 多个多边形 → 多个环部件
    - GeometryCollection: 混合几何 → 递归展开为多个部件

核心函数：
    - extract_parts(): 从几何对象提取所有原语部件
    - get_container_type(): 获取几何对象的容器类型
    - geometry_to_data(): 将几何对象完整转换为 GeometryData

设计原则：
    1. 保留结构：Multi* 类型的每个子几何都作为独立部件
    2. 保留类型：每个部件都带有类型和角色标记
    3. 计算权重：根据长度/周长计算聚合权重

使用示例：
    >>> from shapely.geometry import Polygon
    >>> from preprocess.geometry_extract import extract_parts
    >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> parts = extract_parts(poly, include_holes=True)
    >>> print(len(parts))  # 1（只有外环）

作者：GSD Team
版本：2.0
"""

from typing import List, Optional
import numpy as np

from shapely.geometry import (
    Point, MultiPoint, LineString, MultiLineString,
    Polygon, MultiPolygon, GeometryCollection
)

from .geometry_types import (
    GeomPart, GeometryData,
    PRIM_POINTSET, PRIM_LINE, PRIM_RING,
    ROLE_NONE, ROLE_EXTERIOR, ROLE_INTERIOR,
    CONTAINER_POINT, CONTAINER_MULTIPOINT,
    CONTAINER_LINESTRING, CONTAINER_MULTILINESTRING,
    CONTAINER_POLYGON, CONTAINER_MULTIPOLYGON,
    CONTAINER_GEOMETRYCOLLECTION
)


def _path_length(coords: np.ndarray, closed: bool) -> float:
    """
    计算路径长度
    
    Args:
        coords: 坐标数组 (N, 2)
        closed: 是否闭合（环）
        
    Returns:
        路径长度（最小为 1e-6 避免除零）
    """
    if len(coords) < 2:
        return 1.0
    
    if closed:
        pts = np.vstack([coords, coords[:1]])
    else:
        pts = coords
    
    d = np.diff(pts, axis=0)
    length = float(np.sqrt((d * d).sum(axis=1)).sum())
    return length + 1e-6


def extract_parts(geometry, include_holes: bool = True) -> List[GeomPart]:
    """
    从 Shapely 几何对象提取所有原语部件
    
    支持的几何类型：
    - Point / MultiPoint → 点集合部件
    - LineString / MultiLineString → 线部件
    - Polygon / MultiPolygon → 环部件（外环+内环）
    - GeometryCollection → 递归展开混合类型
    
    Args:
        geometry: Shapely 几何对象
        include_holes: 是否包含 Polygon 的内环（洞）
        
    Returns:
        部件列表 List[GeomPart]
    """
    parts = []
    
    if geometry is None or geometry.is_empty:
        return parts
    
    # ==========================================
    # Point: 单个点 → 点集合部件
    # ==========================================
    if isinstance(geometry, Point):
        coords = np.array([[geometry.x, geometry.y]], dtype=np.float32)
        parts.append(GeomPart(
            coords=coords,
            prim_type=PRIM_POINTSET,
            role=ROLE_NONE,
            weight=1.0
        ))
        return parts
    
    # ==========================================
    # MultiPoint: 多个点 → 点集合部件
    # ==========================================
    if isinstance(geometry, MultiPoint):
        if len(geometry.geoms) > 0:
            coords = np.array([[p.x, p.y] for p in geometry.geoms], dtype=np.float32)
            parts.append(GeomPart(
                coords=coords,
                prim_type=PRIM_POINTSET,
                role=ROLE_NONE,
                weight=float(len(coords))
            ))
        return parts
    
    # ==========================================
    # LineString: 线 → 线部件
    # ==========================================
    if isinstance(geometry, LineString):
        coords = np.array(geometry.coords, dtype=np.float32)
        if len(coords) > 0:
            weight = _path_length(coords, closed=False)
            parts.append(GeomPart(
                coords=coords,
                prim_type=PRIM_LINE,
                role=ROLE_NONE,
                weight=weight
            ))
        return parts
    
    # ==========================================
    # MultiLineString: 多条线 → 多个线部件
    # ==========================================
    if isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            parts.extend(extract_parts(line, include_holes=include_holes))
        return parts
    
    # ==========================================
    # Polygon: 多边形 → 外环 + 内环部件
    # ==========================================
    if isinstance(geometry, Polygon):
        # 外环
        ext_coords = np.array(geometry.exterior.coords, dtype=np.float32)
        if len(ext_coords) > 1:
            # 去掉闭合重复点
            ext_coords = ext_coords[:-1]
            weight = _path_length(ext_coords, closed=True)
            parts.append(GeomPart(
                coords=ext_coords,
                prim_type=PRIM_RING,
                role=ROLE_EXTERIOR,
                weight=weight
            ))
        
        # 内环（洞）
        if include_holes:
            for ring in geometry.interiors:
                int_coords = np.array(ring.coords, dtype=np.float32)
                if len(int_coords) > 1:
                    int_coords = int_coords[:-1]
                    weight = _path_length(int_coords, closed=True)
                    parts.append(GeomPart(
                        coords=int_coords,
                        prim_type=PRIM_RING,
                        role=ROLE_INTERIOR,
                        weight=weight
                    ))
        return parts
    
    # ==========================================
    # MultiPolygon: 多个多边形 → 多个环部件
    # ==========================================
    if isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            parts.extend(extract_parts(poly, include_holes=include_holes))
        return parts
    
    # ==========================================
    # GeometryCollection: 递归展开混合类型
    # ==========================================
    if isinstance(geometry, GeometryCollection):
        for g in geometry.geoms:
            parts.extend(extract_parts(g, include_holes=include_holes))
        return parts
    
    # 其他未知类型返回空
    return parts


def get_container_type(geometry) -> int:
    """
    获取几何对象的容器类型
    
    Args:
        geometry: Shapely 几何对象
        
    Returns:
        容器类型常量
    """
    if isinstance(geometry, Point):
        return CONTAINER_POINT
    elif isinstance(geometry, MultiPoint):
        return CONTAINER_MULTIPOINT
    elif isinstance(geometry, LineString):
        return CONTAINER_LINESTRING
    elif isinstance(geometry, MultiLineString):
        return CONTAINER_MULTILINESTRING
    elif isinstance(geometry, Polygon):
        return CONTAINER_POLYGON
    elif isinstance(geometry, MultiPolygon):
        return CONTAINER_MULTIPOLYGON
    elif isinstance(geometry, GeometryCollection):
        return CONTAINER_GEOMETRYCOLLECTION
    else:
        return -1


def geometry_to_data(
    geometry,
    geometry_id: Optional[int] = None,
    file_id: Optional[int] = None,
    include_holes: bool = True
) -> Optional[GeometryData]:
    """
    将 Shapely 几何对象转换为 GeometryData
    
    Args:
        geometry: Shapely 几何对象
        geometry_id: 几何ID
        file_id: 文件ID
        include_holes: 是否包含内环
        
    Returns:
        GeometryData 或 None（如果无效）
    """
    if geometry is None or geometry.is_empty:
        return None
    
    parts = extract_parts(geometry, include_holes=include_holes)
    
    if not parts:
        return None
    
    container_type = get_container_type(geometry)
    
    return GeometryData(
        parts=parts,
        container_type=container_type,
        geometry_id=geometry_id,
        file_id=file_id
    )

