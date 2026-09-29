"""
几何重采样模块
===============================

本模块控制几何部件的序列长度和复杂度，确保 Transformer 能够
高效处理不同复杂度的几何数据。

解决的问题：
    1. 序列长度差异：不同几何的点数差异巨大（从几个到几万个）
    2. 复杂度控制：避免过于复杂的几何占用过多计算资源
    3. 表示一致性：确保重采样后的几何保持形状特征

重采样策略：
    1. 弧长等距重采样（resample_polyline）：
       - 按累积弧长均匀采样
       - 保持几何形状的均匀表示
       - 支持闭合环和开放线
    
    2. 点集合采样（subsample_pointset）：
       - 训练时随机采样（增强泛化）
       - 推理时固定截断
    
    3. 复杂度限制（cap_and_resample）：
       - 限制每个几何的最大部件数
       - 按权重排序，保留最重要的部件
       - 对每个部件重采样到固定点数

配置参数：
    - max_points_line: 线的最大点数（默认128）
    - max_points_ring: 环的最大点数（默认128）
    - max_points_pointset: 点集合的最大点数（默认128）
    - max_parts: 最大部件数（默认32）

作者：GSD Team
版本：2.0
"""

from typing import List, Optional
import numpy as np

from .geometry_types import (
    GeomPart, GeometryData,
    PRIM_POINTSET, PRIM_LINE, PRIM_RING
)


def resample_polyline(
    coords: np.ndarray,
    n: int,
    closed: bool
) -> np.ndarray:
    """
    对折线/环进行弧长等距重采样
    
    Args:
        coords: 坐标数组 (N, 2)
        n: 目标点数
        closed: 是否闭合（环）
        
    Returns:
        重采样后的坐标 (n, 2)
    """
    if len(coords) == 0:
        return coords
    
    if len(coords) == 1:
        # 单点重复
        return np.repeat(coords, n, axis=0).astype(np.float32)
    
    # 处理闭合情况
    pts = coords
    if closed:
        pts = np.vstack([coords, coords[:1]])
    
    # 计算累积弧长
    d = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    
    if total < 1e-8:
        # 所有点重合
        return np.repeat(coords[:1], n, axis=0).astype(np.float32)
    
    # 目标采样位置
    if closed:
        # 环：n 个点均匀分布在 [0, total) 上
        t = np.linspace(0, total, n + 1)[:-1]
    else:
        # 线：n 个点均匀分布在 [0, total] 上
        t = np.linspace(0, total, n)
    
    # 插值
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    
    out = np.stack([x, y], axis=1).astype(np.float32)
    return out


def subsample_pointset(
    coords: np.ndarray,
    n: int,
    random_sample: bool = False
) -> np.ndarray:
    """
    对点集合进行采样/截断
    
    Args:
        coords: 坐标数组 (N, 2)
        n: 目标点数
        random_sample: 是否随机采样（训练时用），否则取前 n 个
        
    Returns:
        采样后的坐标
    """
    if len(coords) == 0:
        return coords
    
    if len(coords) <= n:
        return coords
    
    if random_sample:
        indices = np.random.choice(len(coords), n, replace=False)
        indices = np.sort(indices)  # 保持一定顺序
        return coords[indices].astype(np.float32)
    else:
        return coords[:n].astype(np.float32)


def cap_and_resample(
    parts: List[GeomPart],
    max_points_line: int = 128,
    max_points_ring: int = 128,
    max_points_pointset: int = 128,
    max_parts: int = 32,
    random_sample_pointset: bool = False
) -> List[GeomPart]:
    """
    控制复杂度：限制部件数和每个部件的点数
    
    策略：
    1. 按权重排序，保留最重要的 max_parts 个部件
    2. 对每个部件按类型重采样到固定点数
    
    Args:
        parts: 部件列表
        max_points_line: 线的最大点数
        max_points_ring: 环的最大点数
        max_points_pointset: 点集合的最大点数
        max_parts: 最大部件数
        random_sample_pointset: 对点集合是否随机采样
        
    Returns:
        处理后的部件列表
    """
    if not parts:
        return parts
    
    # 1. 按权重排序，保留最重要的部件
    parts = sorted(parts, key=lambda p: p.weight, reverse=True)[:max_parts]
    
    # 2. 对每个部件重采样
    for p in parts:
        if p.coords is None or len(p.coords) == 0:
            continue
        
        if p.prim_type == PRIM_POINTSET:
            # 点集合：采样/截断
            if len(p.coords) > max_points_pointset:
                p.coords = subsample_pointset(
                    p.coords, max_points_pointset,
                    random_sample=random_sample_pointset
                )
        
        elif p.prim_type == PRIM_LINE:
            # 线：弧长重采样
            if len(p.coords) != max_points_line:
                p.coords = resample_polyline(
                    p.coords, max_points_line, closed=False
                )
        
        elif p.prim_type == PRIM_RING:
            # 环：弧长重采样
            if len(p.coords) != max_points_ring:
                p.coords = resample_polyline(
                    p.coords, max_points_ring, closed=True
                )
    
    return parts


# ============================================
# 完整的预处理流程
# ============================================

def process_geometry_data(
    data: GeometryData,
    max_points_line: int = 128,
    max_points_ring: int = 128,
    max_points_pointset: int = 128,
    max_parts: int = 32,
    random_sample_pointset: bool = False
) -> Optional[GeometryData]:
    """
    对 GeometryData 应用重采样
    
    Args:
        data: 几何数据
        max_points_*: 各类型最大点数
        max_parts: 最大部件数
        random_sample_pointset: 点集合是否随机采样
        
    Returns:
        处理后的几何数据
    """
    if data is None or not data.parts:
        return None
    
    data.parts = cap_and_resample(
        data.parts,
        max_points_line=max_points_line,
        max_points_ring=max_points_ring,
        max_points_pointset=max_points_pointset,
        max_parts=max_parts,
        random_sample_pointset=random_sample_pointset
    )
    
    if not data.parts:
        return None
    
    return data

