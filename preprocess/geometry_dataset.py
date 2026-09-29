"""
几何数据集模块
===============================

本模块提供 PyTorch 兼容的数据集和数据加载器，支持层次化几何表示。

主要组件：
    1. GeometryAugmentation：几何数据增强器
       - 旋转：随机选择预定义角度（0°, 90°, 180°, 270°）
       - 噪声：添加高斯噪声
       - 环起点移位：随机移动多边形的起始顶点
       - 点集打乱：随机打乱 MultiPoint 的点顺序
       - 部件顺序打乱：随机打乱 Multi* 的子几何顺序
    
    2. UniversalGeometryDataset：通用几何数据集
       - 自动加载 Shapefile 文件
       - 自动转换坐标系（默认 WGS84）
       - 自动预处理（规范化、重采样）
       - 支持数据增强
       - 支持类型均衡采样
    
    3. TwoViewGeometryDataset：双视图几何数据集（训练用）
       - 返回 parts_q, parts_k 两个增强视图
       - 返回 canonical parts 用于渲染教师输入
       - 支持几何-视觉对齐训练
    
    4. collate_parts：批次整理函数
       - 将多个几何的部件展平
       - 创建掩码和索引映射
       - 返回模型可直接使用的张量字典
    
    5. collate_two_views：双视图批次整理函数
       - 分别处理 q/k 视图
       - 返回教师输入（图像或预计算特征）
    
    6. augment_parts：可控几何扰动函数
       - 确定性随机扰动（可设 seed）
       - 支持旋转/平移/缩放/噪声/抽稀

输出格式（collate_parts 返回）：
    {
        "coords": (P, L, 2),      # 所有部件的坐标
        "mask": (P, L),           # 有效点掩码
        "prim_type": (P,),        # 原语类型
        "role": (P,),             # 角色类型
        "weight": (P,),           # 聚合权重
        "part_to_geom": (P,),     # 部件到几何的映射
        "container_type": (B,),   # 容器类型
        "batch_size": int         # 批次大小
    }

使用示例：
    >>> dataset = UniversalGeometryDataset(shp_files, augmentation=aug)
    >>> loader = create_dataloader(dataset, batch_size=32, balanced_sampling=True)
    >>> for batch in loader:
    ...     embeddings = encoder(batch)

作者：GSD Team
版本：2.1（新增双视图训练支持）
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import copy

import numpy as np
import geopandas as gpd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .geometry_types import (
    GeomPart, GeometryData,
    PRIM_POINTSET, PRIM_LINE, PRIM_RING,
    ROLE_NONE, ROLE_EXTERIOR, ROLE_INTERIOR,
    CONTAINER_TYPE_NAMES
)
from .geometry_extract import geometry_to_data, extract_parts
from .geometry_normalize import normalize_geometry_data, canonicalize_ring, canonicalize_line
from .geometry_resample import cap_and_resample, process_geometry_data


# ============================================
# 可控几何扰动函数
# ============================================

def augment_parts(
    parts: List[GeomPart],
    rng: Optional[np.random.Generator] = None,
    rotation_range: Tuple[float, float] = (-15.0, 15.0),
    scale_range: Tuple[float, float] = (0.9, 1.1),
    translate_range: Tuple[float, float] = (-0.05, 0.05),
    noise_std: float = 0.002,
    subsample_prob: float = 0.3,
    subsample_ratio: float = 0.8,
    ring_shift_prob: float = 0.5,
    pointset_shuffle_prob: float = 0.5
) -> List[GeomPart]:
    """
    对 parts 列表应用可控的几何扰动
    
    注意：只改变几何坐标，不改变 prim_type/role/weight 结构
    
    Args:
        parts: 原始部件列表（会被深拷贝）
        rng: numpy 随机数生成器（None 则创建新的）
        rotation_range: 旋转角度范围（度）
        scale_range: 缩放比例范围
        translate_range: 平移范围（相对于归一化坐标）
        noise_std: 高斯噪声标准差
        subsample_prob: 抽稀概率
        subsample_ratio: 抽稀保留比例
        ring_shift_prob: 环起点移位概率
        pointset_shuffle_prob: 点集打乱概率
        
    Returns:
        扰动后的部件列表（新对象）
    """
    if not parts:
        return parts
    
    # 确定性随机
    if rng is None:
        rng = np.random.default_rng()
    
    # 深拷贝
    new_parts = []
    for p in parts:
        new_p = GeomPart(
            coords=p.coords.copy() if p.coords is not None else None,
            prim_type=p.prim_type,
            role=p.role,
            weight=p.weight
        )
        new_parts.append(new_p)
    
    # 1. 全局旋转
    angle = rng.uniform(*rotation_range)
    if abs(angle) > 0.01:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        for p in new_parts:
            if p.coords is not None and len(p.coords) > 0:
                x, y = p.coords[:, 0], p.coords[:, 1]
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
                p.coords = np.stack([new_x, new_y], axis=1).astype(np.float32)
    
    # 2. 全局缩放
    scale = rng.uniform(*scale_range)
    if abs(scale - 1.0) > 0.001:
        for p in new_parts:
            if p.coords is not None and len(p.coords) > 0:
                p.coords = (p.coords * scale).astype(np.float32)
    
    # 3. 全局平移
    tx = rng.uniform(*translate_range)
    ty = rng.uniform(*translate_range)
    if abs(tx) > 0.001 or abs(ty) > 0.001:
        offset = np.array([tx, ty], dtype=np.float32)
        for p in new_parts:
            if p.coords is not None and len(p.coords) > 0:
                p.coords = p.coords + offset
    
    # 4. 添加噪声
    if noise_std > 0:
        for p in new_parts:
            if p.coords is not None and len(p.coords) > 0:
                noise = rng.standard_normal(p.coords.shape).astype(np.float32) * noise_std
                p.coords = p.coords + noise
    
    # 5. 抽稀（仅对线和环，保持结构）
    if rng.random() < subsample_prob:
        for p in new_parts:
            if p.prim_type in (PRIM_LINE, PRIM_RING) and p.coords is not None and len(p.coords) > 4:
                n = len(p.coords)
                keep_n = max(3, int(n * subsample_ratio))
                # 保持首尾，随机删除中间点
                indices = [0] + sorted(rng.choice(range(1, n-1), keep_n-2, replace=False).tolist()) + [n-1]
                p.coords = p.coords[indices].astype(np.float32)
    
    # 6. 环起点移位
    for p in new_parts:
        if p.prim_type == PRIM_RING and p.coords is not None and len(p.coords) > 2:
            if rng.random() < ring_shift_prob:
                shift = rng.integers(0, len(p.coords))
                p.coords = np.roll(p.coords, shift, axis=0)
    
    # 7. 点集打乱
    for p in new_parts:
        if p.prim_type == PRIM_POINTSET and p.coords is not None and len(p.coords) > 1:
            if rng.random() < pointset_shuffle_prob:
                rng.shuffle(p.coords)
    
    return new_parts


class GeometryAugmentation:
    """
    几何数据增强
    
    支持的增强操作：
    - 旋转
    - 添加噪声
    - 随机抽稀
    - 环的起点移位（训练时等价表示增强）
    - 点集合打乱顺序
    - 部件顺序打乱
    """
    
    def __init__(
        self,
        rotation_angles: List[float] = [0, 90, 180, 270],
        noise_std: float = 0.001,
        subsample_ratio_range: Tuple[float, float] = (0.7, 0.95),
        enable_rotation: bool = True,
        enable_noise: bool = True,
        enable_subsample: bool = True,
        enable_ring_shift: bool = True,
        enable_pointset_shuffle: bool = True,
        enable_parts_shuffle: bool = True
    ):
        self.rotation_angles = rotation_angles
        self.noise_std = noise_std
        self.subsample_ratio_range = subsample_ratio_range
        
        self.enable_rotation = enable_rotation
        self.enable_noise = enable_noise
        self.enable_subsample = enable_subsample
        self.enable_ring_shift = enable_ring_shift
        self.enable_pointset_shuffle = enable_pointset_shuffle
        self.enable_parts_shuffle = enable_parts_shuffle
    
    def augment(self, data: GeometryData) -> GeometryData:
        """
        应用数据增强
        
        Args:
            data: 几何数据
            
        Returns:
            增强后的几何数据（深拷贝）
        """
        # 深拷贝避免修改原数据
        data = copy.deepcopy(data)
        
        if not data.parts:
            return data
        
        # 1. 随机旋转
        if self.enable_rotation and self.rotation_angles:
            angle = random.choice(self.rotation_angles)
            if angle != 0:
                data = self._rotate(data, angle)
        
        # 2. 添加噪声
        if self.enable_noise and self.noise_std > 0:
            data = self._add_noise(data)
        
        # 3. 环的起点随机移位
        if self.enable_ring_shift:
            data = self._shift_ring_starts(data)
        
        # 4. 点集合打乱顺序
        if self.enable_pointset_shuffle:
            data = self._shuffle_pointsets(data)
        
        # 5. 部件顺序打乱
        if self.enable_parts_shuffle:
            random.shuffle(data.parts)
        
        return data
    
    def _rotate(self, data: GeometryData, angle: float) -> GeometryData:
        """旋转所有坐标"""
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        for p in data.parts:
            if p.coords is not None and len(p.coords) > 0:
                x = p.coords[:, 0]
                y = p.coords[:, 1]
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
                p.coords = np.stack([new_x, new_y], axis=1).astype(np.float32)
        
        return data
    
    def _add_noise(self, data: GeometryData) -> GeometryData:
        """添加高斯噪声"""
        for p in data.parts:
            if p.coords is not None and len(p.coords) > 0:
                noise = np.random.randn(*p.coords.shape).astype(np.float32) * self.noise_std
                p.coords = p.coords + noise
        
        return data
    
    def _shift_ring_starts(self, data: GeometryData) -> GeometryData:
        """随机移位环的起点"""
        for p in data.parts:
            if p.prim_type == PRIM_RING and p.coords is not None and len(p.coords) > 2:
                shift = random.randint(0, len(p.coords) - 1)
                p.coords = np.roll(p.coords, shift, axis=0)
        
        return data
    
    def _shuffle_pointsets(self, data: GeometryData) -> GeometryData:
        """打乱点集合的顺序"""
        for p in data.parts:
            if p.prim_type == PRIM_POINTSET and p.coords is not None and len(p.coords) > 1:
                np.random.shuffle(p.coords)
        
        return data


class UniversalGeometryDataset(Dataset):
    """
    通用几何数据集
    
    支持所有几何类型的层次化表示
    每个样本返回一个 GeometryData 对象的字典表示
    """
    
    def __init__(
        self,
        shp_files: List[str],
        max_points_line: int = 128,
        max_points_ring: int = 128,
        max_points_pointset: int = 128,
        max_parts: int = 32,
        max_geometries_per_file: Optional[int] = None,
        min_parts: int = 1,
        include_holes: bool = True,
        normalize: bool = True,
        canonicalize: bool = True,
        augmentation: Optional[GeometryAugmentation] = None,
        crs: str = "EPSG:4326",
        cache_data: bool = True
    ):
        """
        Args:
            shp_files: Shapefile 路径列表
            max_points_*: 各类型最大点数
            max_parts: 每个几何的最大部件数
            max_geometries_per_file: 每个文件最大几何数（None 表示全部）
            min_parts: 最小部件数（少于此数的几何跳过）
            include_holes: 是否包含多边形的内环
            normalize: 是否归一化坐标
            canonicalize: 是否规范化（起点/方向）
            augmentation: 数据增强器
            crs: 目标坐标系
            cache_data: 是否缓存数据
        """
        super().__init__()
        
        self.max_points_line = max_points_line
        self.max_points_ring = max_points_ring
        self.max_points_pointset = max_points_pointset
        self.max_parts = max_parts
        self.min_parts = min_parts
        self.include_holes = include_holes
        self.normalize = normalize
        self.canonicalize = canonicalize
        self.augmentation = augmentation
        self.crs = crs
        
        # 加载数据
        self.geometries: List[GeometryData] = []
        self.container_type_counts: Dict[int, int] = {}
        
        self._load_data(shp_files, max_geometries_per_file, cache_data)
    
    def _load_data(
        self,
        shp_files: List[str],
        max_geometries_per_file: Optional[int],
        cache: bool
    ):
        """加载所有 Shapefile 数据"""
        for file_id, shp_path in enumerate(shp_files):
            try:
                gdf = gpd.read_file(shp_path)
                
                # 转换坐标系
                if gdf.crs is not None and gdf.crs.to_string() != self.crs:
                    gdf = gdf.to_crs(self.crs)
                
                # 限制每个文件的几何数
                geometries = list(gdf.geometry)
                if max_geometries_per_file is not None:
                    geometries = geometries[:max_geometries_per_file]
                
                for geom_id, geom in enumerate(geometries):
                    data = geometry_to_data(
                        geom,
                        geometry_id=geom_id,
                        file_id=file_id,
                        include_holes=self.include_holes
                    )
                    
                    if data is None:
                        continue
                    
                    # 规范化
                    if self.normalize or self.canonicalize:
                        data = normalize_geometry_data(
                            data,
                            normalize_coords=self.normalize,
                            canonicalize_rings=self.canonicalize,
                            canonicalize_lines=self.canonicalize
                        )
                    
                    # 重采样
                    data = process_geometry_data(
                        data,
                        max_points_line=self.max_points_line,
                        max_points_ring=self.max_points_ring,
                        max_points_pointset=self.max_points_pointset,
                        max_parts=self.max_parts
                    )
                    
                    if data is None or data.num_parts < self.min_parts:
                        continue
                    
                    self.geometries.append(data)
                    
                    # 统计容器类型
                    ct = data.container_type
                    self.container_type_counts[ct] = self.container_type_counts.get(ct, 0) + 1
                
            except Exception as e:
                print(f"Warning: Failed to load {shp_path}: {e}")
                continue
        
        print(f"Loaded {len(self.geometries)} geometries")
        for ct, count in self.container_type_counts.items():
            print(f"  {CONTAINER_TYPE_NAMES.get(ct, 'Unknown')}: {count}")
    
    def __len__(self) -> int:
        return len(self.geometries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.geometries[idx]
        
        # 应用数据增强
        if self.augmentation is not None:
            data = self.augmentation.augment(data)
        
        return {
            "parts": data.parts,
            "container_type": data.container_type,
            "geometry_id": data.geometry_id,
            "file_id": data.file_id,
            "num_parts": data.num_parts
        }
    
    def get_type_weights(self) -> List[float]:
        """
        获取类型均衡采样权重
        
        Returns:
            每个样本的采样权重
        """
        # 计算每个类型的权重（反比于数量）
        type_weights = {}
        total = sum(self.container_type_counts.values())
        for ct, count in self.container_type_counts.items():
            type_weights[ct] = total / (count * len(self.container_type_counts))
        
        # 为每个样本分配权重
        weights = []
        for data in self.geometries:
            w = type_weights.get(data.container_type, 1.0)
            weights.append(w)
        
        return weights


def collate_parts(batch: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    将批次数据整理成模型输入格式
    
    输出格式：
    - coords: (P, L, 2) float32 - 所有部件的坐标
    - mask: (P, L) bool - 有效点掩码
    - prim_type: (P,) int64 - 原语类型
    - role: (P,) int64 - 角色类型
    - weight: (P,) float32 - 聚合权重
    - part_to_geom: (P,) int64 - 部件到几何的映射
    - container_type: (B,) int64 - 容器类型
    - batch_size: int - 批次大小
    
    Args:
        batch: 批次数据列表
        
    Returns:
        整理后的字典，或 None（如果无有效数据）
    """
    # 收集所有部件
    all_parts = []
    part_to_geom = []
    container_types = []
    
    for gi, item in enumerate(batch):
        parts = item["parts"]
        container_types.append(item["container_type"])
        
        for p in parts:
            all_parts.append(p)
            part_to_geom.append(gi)
    
    P = len(all_parts)
    B = len(batch)
    
    if P == 0:
        return None
    
    # 找最大序列长度
    L = max(len(p.coords) for p in all_parts if p.coords is not None and len(p.coords) > 0)
    
    if L == 0:
        return None
    
    # 创建张量
    coords = torch.zeros((P, L, 2), dtype=torch.float32)
    mask = torch.zeros((P, L), dtype=torch.bool)
    prim_t = torch.tensor([p.prim_type for p in all_parts], dtype=torch.long)
    role_t = torch.tensor([p.role for p in all_parts], dtype=torch.long)
    w_t = torch.tensor([p.weight for p in all_parts], dtype=torch.float32)
    g_idx = torch.tensor(part_to_geom, dtype=torch.long)
    ct = torch.tensor(container_types, dtype=torch.long)
    
    # 填充坐标和掩码
    for i, p in enumerate(all_parts):
        if p.coords is not None and len(p.coords) > 0:
            n = len(p.coords)
            coords[i, :n] = torch.from_numpy(p.coords)
            mask[i, :n] = True
    
    return {
        "coords": coords,           # (P, L, 2)
        "mask": mask,               # (P, L)
        "prim_type": prim_t,        # (P,)
        "role": role_t,             # (P,)
        "weight": w_t,              # (P,)
        "part_to_geom": g_idx,      # (P,)
        "container_type": ct,       # (B,)
        "batch_size": B
    }


def create_dataloader(
    dataset: UniversalGeometryDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    balanced_sampling: bool = False,
    drop_last: bool = True
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        balanced_sampling: 是否使用类型均衡采样
        drop_last: 是否丢弃最后不完整批次
        
    Returns:
        DataLoader
    """
    sampler = None
    
    if balanced_sampling:
        weights = dataset.get_type_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False  # 使用 sampler 时不能 shuffle
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_parts,
        drop_last=drop_last,
        sampler=sampler
    )


# ============================================
# 双视图几何数据集（训练期几何-视觉对齐用）
# ============================================

class TwoViewGeometryDataset(Dataset):
    """
    双视图几何数据集
    
    用于训练期的几何-视觉对齐：
    - parts_q: canonical parts + 随机几何扰动（view A）
    - parts_k: canonical parts + 另一组随机几何扰动（view B）
    - canonical_parts: 未扰动的 canonical parts（用于渲染教师输入）
    
    注意：扰动只改变几何坐标，不破坏 prim_type/role/weight 结构
    """
    
    def __init__(
        self,
        shp_files: List[str],
        max_points_line: int = 128,
        max_points_ring: int = 128,
        max_points_pointset: int = 128,
        max_parts: int = 32,
        max_geometries_per_file: Optional[int] = None,
        min_parts: int = 1,
        include_holes: bool = True,
        normalize: bool = True,
        canonicalize: bool = True,
        crs: str = "EPSG:4326",
        # 扰动参数
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translate_range: Tuple[float, float] = (-0.05, 0.05),
        noise_std: float = 0.002,
        subsample_prob: float = 0.3,
        # 教师输入模式
        teacher_mode: str = "render",  # "render" 或 "offline_feat"
        offline_feat_dir: Optional[str] = None,
        image_size: int = 224
    ):
        """
        Args:
            shp_files: Shapefile 路径列表
            max_points_*: 各类型最大点数
            max_parts: 每个几何的最大部件数
            max_geometries_per_file: 每个文件最大几何数
            min_parts: 最小部件数
            include_holes: 是否包含内环
            normalize: 是否归一化坐标
            canonicalize: 是否规范化
            crs: 目标坐标系
            rotation_range: 旋转角度范围
            scale_range: 缩放比例范围
            translate_range: 平移范围
            noise_std: 噪声标准差
            subsample_prob: 抽稀概率
            teacher_mode: 教师输入模式 ("render" 或 "offline_feat")
            offline_feat_dir: 离线特征目录（teacher_mode="offline_feat" 时使用）
            image_size: 渲染图像尺寸
        """
        super().__init__()
        
        self.max_points_line = max_points_line
        self.max_points_ring = max_points_ring
        self.max_points_pointset = max_points_pointset
        self.max_parts = max_parts
        self.min_parts = min_parts
        self.include_holes = include_holes
        self.normalize = normalize
        self.canonicalize = canonicalize
        self.crs = crs
        
        # 扰动参数
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.noise_std = noise_std
        self.subsample_prob = subsample_prob
        
        # 教师输入
        self.teacher_mode = teacher_mode
        self.offline_feat_dir = offline_feat_dir
        self.image_size = image_size
        
        # 渲染器（延迟加载）
        self._renderer = None
        
        # 加载数据
        self.geometries: List[GeometryData] = []
        self.container_type_counts: Dict[int, int] = {}
        
        self._load_data(shp_files, max_geometries_per_file)
    
    def _load_data(
        self,
        shp_files: List[str],
        max_geometries_per_file: Optional[int]
    ):
        """加载所有 Shapefile 数据"""
        for file_id, shp_path in enumerate(shp_files):
            try:
                gdf = gpd.read_file(shp_path)
                
                if gdf.crs is not None and gdf.crs.to_string() != self.crs:
                    gdf = gdf.to_crs(self.crs)
                
                geometries = list(gdf.geometry)
                if max_geometries_per_file is not None:
                    geometries = geometries[:max_geometries_per_file]
                
                for geom_id, geom in enumerate(geometries):
                    data = geometry_to_data(
                        geom,
                        geometry_id=geom_id,
                        file_id=file_id,
                        include_holes=self.include_holes
                    )
                    
                    if data is None:
                        continue
                    
                    # 规范化
                    if self.normalize or self.canonicalize:
                        data = normalize_geometry_data(
                            data,
                            normalize_coords=self.normalize,
                            canonicalize_rings=self.canonicalize,
                            canonicalize_lines=self.canonicalize
                        )
                    
                    # 重采样
                    data = process_geometry_data(
                        data,
                        max_points_line=self.max_points_line,
                        max_points_ring=self.max_points_ring,
                        max_points_pointset=self.max_points_pointset,
                        max_parts=self.max_parts
                    )
                    
                    if data is None or data.num_parts < self.min_parts:
                        continue
                    
                    self.geometries.append(data)
                    
                    ct = data.container_type
                    self.container_type_counts[ct] = self.container_type_counts.get(ct, 0) + 1
                
            except Exception as e:
                print(f"Warning: Failed to load {shp_path}: {e}")
                continue
        
        print(f"[TwoViewDataset] Loaded {len(self.geometries)} geometries")
        for ct, count in self.container_type_counts.items():
            print(f"  {CONTAINER_TYPE_NAMES.get(ct, 'Unknown')}: {count}")
    
    @property
    def renderer(self):
        """延迟加载渲染器"""
        if self._renderer is None:
            from .render import GeometryRenderer
            self._renderer = GeometryRenderer(image_size=self.image_size)
        return self._renderer
    
    def __len__(self) -> int:
        return len(self.geometries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.geometries[idx]
        
        # Canonical parts（不扰动）
        canonical_parts = copy.deepcopy(data.parts)
        
        # 为每个视图创建独立的 rng（确定性可控）
        # 使用 idx 和当前时间确保每次获取不同扰动
        seed_q = (idx * 2) ^ (id(data) & 0xFFFFFFFF)
        seed_k = (idx * 2 + 1) ^ (id(data) & 0xFFFFFFFF)
        
        rng_q = np.random.default_rng(seed_q)
        rng_k = np.random.default_rng(seed_k)
        
        # 生成两个扰动视图
        parts_q = augment_parts(
            canonical_parts,
            rng=rng_q,
            rotation_range=self.rotation_range,
            scale_range=self.scale_range,
            translate_range=self.translate_range,
            noise_std=self.noise_std,
            subsample_prob=self.subsample_prob
        )
        
        parts_k = augment_parts(
            canonical_parts,
            rng=rng_k,
            rotation_range=self.rotation_range,
            scale_range=self.scale_range,
            translate_range=self.translate_range,
            noise_std=self.noise_std,
            subsample_prob=self.subsample_prob
        )
        
        # 教师输入
        teacher_input = self._get_teacher_input(canonical_parts, data.file_id, data.geometry_id)
        
        return {
            "parts_q": parts_q,
            "parts_k": parts_k,
            "canonical_parts": canonical_parts,
            "teacher_input": teacher_input,
            "container_type": data.container_type,
            "geometry_id": data.geometry_id,
            "file_id": data.file_id,
            "num_parts": data.num_parts
        }
    
    def _get_teacher_input(
        self,
        parts: List[GeomPart],
        file_id: int,
        geometry_id: int
    ):
        """
        获取教师输入
        
        Returns:
            - render 模式：渲染的图像张量 [3, H, W]
            - offline_feat 模式：预计算的特征向量
        """
        if self.teacher_mode == "offline_feat" and self.offline_feat_dir is not None:
            # 读取离线特征
            feat_path = os.path.join(
                self.offline_feat_dir,
                f"file{file_id}_geom{geometry_id}.pt"
            )
            if os.path.exists(feat_path):
                return torch.load(feat_path)
            # fallback to render
        
        # 渲染模式：将 parts 渲染为图像
        return self._render_parts(parts)
    
    def _render_parts(self, parts: List[GeomPart]) -> torch.Tensor:
        """
        将 parts 渲染为图像
        
        合并所有 parts 的坐标渲染到一张图像
        """
        from .render import GeometryRenderer
        from PIL import Image, ImageDraw
        from torchvision import transforms
        
        # 收集所有坐标
        all_coords = []
        for p in parts:
            if p.coords is not None and len(p.coords) > 0:
                all_coords.append(p.coords)
        
        if not all_coords:
            # 返回空白图像
            return torch.zeros(3, self.image_size, self.image_size)
        
        # 创建画布
        img = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 计算全局边界框用于坐标转换
        all_xy = np.vstack(all_coords)
        min_xy = all_xy.min(axis=0)
        max_xy = all_xy.max(axis=0)
        range_xy = max_xy - min_xy
        range_xy = np.maximum(range_xy, 1e-6)
        
        padding = 0.1
        effective_size = self.image_size * (1 - 2 * padding)
        offset = self.image_size * padding
        
        max_range = max(range_xy)
        scale = effective_size / max_range
        center_offset = (effective_size - range_xy * scale) / 2
        
        # 绘制每个 part
        for p in parts:
            if p.coords is None or len(p.coords) == 0:
                continue
            
            # 坐标转换
            pixel_coords = (p.coords - min_xy) * scale + offset + center_offset
            points = [(int(x), int(y)) for x, y in pixel_coords]
            
            if p.prim_type == PRIM_RING and len(points) >= 3:
                # 多边形
                draw.polygon(points, fill=(100, 100, 100), outline=(0, 0, 0))
            elif p.prim_type == PRIM_LINE and len(points) >= 2:
                # 线
                draw.line(points, fill=(0, 0, 0), width=2)
            elif p.prim_type == PRIM_POINTSET:
                # 点
                for px, py in points:
                    draw.ellipse([px-2, py-2, px+2, py+2], fill=(0, 0, 0))
        
        # 转换为张量并归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transform(img)
    
    def get_type_weights(self) -> List[float]:
        """获取类型均衡采样权重"""
        type_weights = {}
        total = sum(self.container_type_counts.values())
        for ct, count in self.container_type_counts.items():
            type_weights[ct] = total / (count * len(self.container_type_counts))
        
        weights = []
        for data in self.geometries:
            w = type_weights.get(data.container_type, 1.0)
            weights.append(w)
        
        return weights


def collate_two_views(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    双视图批次整理函数
    
    将 batch 中的 parts_q, parts_k 分别整理，
    同时处理 teacher_input。
    
    Args:
        batch: 批次数据列表
        
    Returns:
        {
            "q": batch_q dict,      # collate_parts 的输出
            "k": batch_k dict,      # collate_parts 的输出
            "teacher": tensor,      # 教师输入 [B, ...] 或 [B, 3, H, W]
            "container_type": tensor,
            "batch_size": int
        }
        或 None（如果数据无效）
    """
    if not batch:
        return None
    
    # 分离 q 和 k 视图
    batch_q_items = [{"parts": item["parts_q"], "container_type": item["container_type"]} for item in batch]
    batch_k_items = [{"parts": item["parts_k"], "container_type": item["container_type"]} for item in batch]
    
    # 调用 collate_parts
    batch_q = collate_parts(batch_q_items)
    batch_k = collate_parts(batch_k_items)
    
    if batch_q is None or batch_k is None:
        return None
    
    # 处理教师输入
    teacher_inputs = [item["teacher_input"] for item in batch]
    
    # 检查是否都是 tensor
    if all(isinstance(t, torch.Tensor) for t in teacher_inputs):
        teacher_batch = torch.stack(teacher_inputs)
    else:
        # 如果有非 tensor，尝试转换
        teacher_batch = torch.stack([
            t if isinstance(t, torch.Tensor) else torch.tensor(t)
            for t in teacher_inputs
        ])
    
    # 容器类型
    container_types = torch.tensor([item["container_type"] for item in batch], dtype=torch.long)
    
    return {
        "q": batch_q,
        "k": batch_k,
        "teacher": teacher_batch,
        "container_type": container_types,
        "batch_size": len(batch)
    }


def create_two_view_dataloader(
    dataset: TwoViewGeometryDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    balanced_sampling: bool = False,
    drop_last: bool = True
) -> DataLoader:
    """
    创建双视图 DataLoader
    
    Args:
        dataset: 双视图数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        balanced_sampling: 是否使用类型均衡采样
        drop_last: 是否丢弃最后不完整批次
        
    Returns:
        DataLoader
    """
    sampler = None
    
    if balanced_sampling:
        weights = dataset.get_type_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_two_views,
        drop_last=drop_last,
        sampler=sampler
    )

