"""
几何规范化模块
===============================

本模块处理几何的等价表示问题，确保同一几何的不同表示方式
能够产生一致的签名。

解决的问题：
    1. 坐标尺度不一致：不同几何的坐标范围差异大
    2. 多边形起点不一致：同一多边形从不同顶点开始表示
    3. 多边形方向不一致：顺时针 vs 逆时针
    4. 线的方向不一致：A→B vs B→A

规范化策略：
    1. 全局坐标归一化（normalize_parts）：
       - 计算所有部件的全局中心
       - 使用 RMS 半径进行缩放（比 max 更稳定）
       - 归一化后坐标分布在 [-1, 1] 附近
    
    2. 环的规范化（canonicalize_ring）：
       - 外环统一为逆时针（CCW，正面积）
       - 内环统一为顺时针（CW，负面积）
       - 起点选择字典序最小的顶点
    
    3. 线的规范化（canonicalize_line）：
       - 让字典序较小的端点在前
       - 使 A→B 和 B→A 视为同一条线

使用示例：
    >>> from preprocess.geometry_normalize import normalize_geometry_data
    >>> data = normalize_geometry_data(data, normalize_coords=True)

作者：GSD Team
版本：2.0
"""

from typing import List, Optional
import numpy as np

from .geometry_types import (
    GeomPart, GeometryData,
    PRIM_POINTSET, PRIM_LINE, PRIM_RING,
    ROLE_EXTERIOR, ROLE_INTERIOR
)


# ============================================
# 全局坐标归一化
# ============================================

def normalize_parts(parts: List[GeomPart]) -> List[GeomPart]:
    """
    对所有部件进行全局坐标归一化
    
    使用所有部件的坐标计算统一的中心和尺度，
    然后对每个部件应用相同的变换，保持空间关系。
    
    归一化方式：
    - 中心化：减去全局中心
    - 缩放：除以 RMS 半径
    
    Args:
        parts: 部件列表
        
    Returns:
        归一化后的部件列表
    """
    if not parts:
        return parts
    
    # 收集所有坐标
    all_coords = []
    for p in parts:
        if p.coords is not None and len(p.coords) > 0:
            all_coords.append(p.coords)
    
    if not all_coords:
        return parts
    
    all_xy = np.vstack(all_coords).astype(np.float32)
    
    if len(all_xy) == 0:
        return parts
    
    # 计算全局中心
    center = all_xy.mean(axis=0, keepdims=True)
    
    # 计算 RMS 半径（比 max 更稳定）
    centered = all_xy - center
    scale = np.sqrt((centered * centered).sum(axis=1).mean()) + 1e-6
    
    # 应用归一化到每个部件
    for p in parts:
        if p.coords is not None and len(p.coords) > 0:
            p.coords = ((p.coords.astype(np.float32) - center) / scale).astype(np.float32)
    
    return parts


# ============================================
# 环的规范化（起点和方向）
# ============================================

def _signed_area(coords: np.ndarray) -> float:
    """
    计算多边形的有符号面积
    
    正值表示逆时针（CCW），负值表示顺时针（CW）
    
    Args:
        coords: 坐标数组 (N, 2)
        
    Returns:
        有符号面积
    """
    if len(coords) < 3:
        return 0.0
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Shoelace formula
    area = float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() * 0.5)
    return area


def canonicalize_ring(
    coords: np.ndarray,
    role: int = ROLE_EXTERIOR,
    round_decimals: int = 6
) -> np.ndarray:
    """
    规范化环的起点和方向
    
    规则：
    - 外环统一为逆时针（CCW）
    - 内环统一为顺时针（CW）
    - 起点选择字典序最小的点
    
    Args:
        coords: 坐标数组 (N, 2)
        role: 角色类型（外环/内环）
        round_decimals: 用于比较的小数位数
        
    Returns:
        规范化后的坐标
    """
    if len(coords) < 3:
        return coords
    
    c = coords.copy()
    
    # 统一方向
    area = _signed_area(c)
    if role == ROLE_EXTERIOR:
        # 外环应为 CCW（正面积）
        if area < 0:
            c = c[::-1]
    elif role == ROLE_INTERIOR:
        # 内环应为 CW（负面积）
        if area > 0:
            c = c[::-1]
    
    # 统一起点：选字典序最小点
    # 先 round 提升数值稳定性
    cr = np.round(c, round_decimals)
    
    # 按 (x, y) 字典序排序，取最小索引
    idx = np.lexsort((cr[:, 1], cr[:, 0]))[0]
    
    # 循环移位使最小点成为起点
    c = np.roll(c, -idx, axis=0)
    
    return c.astype(np.float32)


# ============================================
# 线的规范化（方向）
# ============================================

def canonicalize_line(
    coords: np.ndarray,
    round_decimals: int = 6
) -> np.ndarray:
    """
    规范化线的方向
    
    使 A→B 和 B→A 视为同一条线（多数地图要素是这样）
    规则：让字典序较小的端点在前
    
    Args:
        coords: 坐标数组 (N, 2)
        round_decimals: 用于比较的小数位数
        
    Returns:
        规范化后的坐标
    """
    if len(coords) < 2:
        return coords
    
    # 比较两端点
    a = np.round(coords[0], round_decimals)
    b = np.round(coords[-1], round_decimals)
    
    # 如果起点字典序大于终点，翻转
    if (a[0] > b[0]) or (a[0] == b[0] and a[1] > b[1]):
        return coords[::-1].copy().astype(np.float32)
    
    return coords.astype(np.float32)


# ============================================
# 对部件应用规范化
# ============================================

def canonicalize_parts(
    parts: List[GeomPart],
    canonicalize_rings: bool = True,
    canonicalize_lines: bool = True
) -> List[GeomPart]:
    """
    对所有部件应用规范化
    
    Args:
        parts: 部件列表
        canonicalize_rings: 是否规范化环
        canonicalize_lines: 是否规范化线
        
    Returns:
        规范化后的部件列表
    """
    for p in parts:
        if p.coords is None or len(p.coords) == 0:
            continue
        
        if p.prim_type == PRIM_RING and canonicalize_rings:
            p.coords = canonicalize_ring(p.coords, role=p.role)
        
        elif p.prim_type == PRIM_LINE and canonicalize_lines:
            p.coords = canonicalize_line(p.coords)
    
    return parts


# ============================================
# 完整的规范化流程
# ============================================

def normalize_geometry_data(
    data: GeometryData,
    normalize_coords: bool = True,
    canonicalize_rings: bool = True,
    canonicalize_lines: bool = True
) -> GeometryData:
    """
    对 GeometryData 应用完整规范化流程
    
    顺序：
    1. 全局坐标归一化
    2. 环的起点/方向规范化
    3. 线的方向规范化
    
    Args:
        data: 几何数据
        normalize_coords: 是否归一化坐标
        canonicalize_rings: 是否规范化环
        canonicalize_lines: 是否规范化线
        
    Returns:
        规范化后的几何数据
    """
    if data is None or not data.parts:
        return data
    
    # 1. 全局坐标归一化
    if normalize_coords:
        data.parts = normalize_parts(data.parts)
    
    # 2. 规范化（起点/方向）
    data.parts = canonicalize_parts(
        data.parts,
        canonicalize_rings=canonicalize_rings,
        canonicalize_lines=canonicalize_lines
    )
    
    return data

