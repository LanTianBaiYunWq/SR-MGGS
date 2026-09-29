"""
SGP (Shape Geometry Preprocessing) 几何数据预处理
===============================

本模块提供矢量几何数据的预处理功能，是 GSD 系统的数据输入层。

注意：本模块为旧版实现，主要支持 Polygon 和 LineString。
新版通用几何支持请使用 geometry_dataset.py 模块。

主要组件：
    1. GeometryNormalizer：几何规范化器
       - DP 简化（Douglas-Peucker 算法）
       - 等距重采样
       - 坐标归一化到 [0, 1]
       - 确定性起点选择
    
    2. GeometryAugmentor：几何数据增强器
       - 旋转增强
       - 高斯噪声
       - 随机抽稀
       - DP 简化增强
    
    3. ShapefileDataset：Shapefile 数据集
       - 自动加载和解析 Shapefile
       - 坐标系转换
       - 几何规范化
    
    4. ContrastiveDataset：对比学习数据集
       - 为每个几何生成多个增强视图
       - 用于自监督对比学习
    
    5. 全局签名相关类：
       - AdaptiveGeometryNormalizer：自适应规范化
       - ShapefileGlobalDataset：每个 shp 文件作为一个样本
       - GlobalContrastiveDataset：全局签名的对比学习

使用示例：
    >>> dataset = ShapefileDataset(shp_files, normalizer, augmentor)
    >>> loader = create_dataloader(dataset, batch_size=32)

作者：GSD Team
版本：1.0（保持向后兼容）
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import geopandas as gpd
from shapely.geometry import (
    LineString, MultiLineString, MultiPolygon, Polygon, 
    GeometryCollection, mapping
)
from shapely.affinity import rotate, scale, translate
from shapely.ops import transform
import torch
from torch.utils.data import Dataset, DataLoader


class GeometryNormalizer:
    """
    几何规范化器
    统一几何对象的表示方式
    """
    
    def __init__(
        self,
        max_points: int = 256,
        min_points: int = 16,
        dp_tolerance: float = 0.0001,
        resample_distance: Optional[float] = None
    ):
        """
        Args:
            max_points: 最大采样点数
            min_points: 最小点数阈值
            dp_tolerance: Douglas-Peucker 简化容差
            resample_distance: 等距重采样距离
        """
        self.max_points = max_points
        self.min_points = min_points
        self.dp_tolerance = dp_tolerance
        self.resample_distance = resample_distance
    
    def normalize(self, geometry) -> Optional[np.ndarray]:
        """
        规范化几何对象
        
        Args:
            geometry: Shapely几何对象
            
        Returns:
            规范化后的点序列 [N, 2]，或None（如果无效）
        """
        # 1. 提取坐标点
        coords = self._extract_coords(geometry)
        if coords is None or len(coords) < self.min_points:
            return None
        
        # 2. DP简化（如果点太多）
        if len(coords) > self.max_points * 2:
            coords = self._simplify(coords)
        
        # 3. 重采样
        coords = self._resample(coords)
        
        # 4. 归一化到 [0, 1]
        coords = self._normalize_coords(coords)
        
        # 5. 确定性起点选择
        coords = self._align_start_point(coords)
        
        return coords
    
    def _extract_coords(self, geometry) -> Optional[np.ndarray]:
        """提取几何对象的坐标点"""
        try:
            if isinstance(geometry, Polygon):
                # 提取外环坐标
                coords = np.array(geometry.exterior.coords)[:-1]  # 去除重复的闭合点
                return coords
            
            elif isinstance(geometry, LineString):
                return np.array(geometry.coords)
            
            elif isinstance(geometry, MultiPolygon):
                # 合并所有多边形的外环
                all_coords = []
                for poly in geometry.geoms:
                    coords = np.array(poly.exterior.coords)[:-1]
                    all_coords.append(coords)
                return np.vstack(all_coords)
            
            elif isinstance(geometry, MultiLineString):
                all_coords = []
                for line in geometry.geoms:
                    all_coords.append(np.array(line.coords))
                return np.vstack(all_coords)
            
            else:
                return None
                
        except Exception:
            return None
    
    def _simplify(self, coords: np.ndarray) -> np.ndarray:
        """使用Douglas-Peucker算法简化"""
        if len(coords) <= self.max_points:
            return coords
        
        # 创建临时LineString进行简化
        line = LineString(coords)
        simplified = line.simplify(self.dp_tolerance, preserve_topology=True)
        return np.array(simplified.coords)
    
    def _resample(self, coords: np.ndarray) -> np.ndarray:
        """重采样到固定点数"""
        n_points = len(coords)
        
        if n_points == self.max_points:
            return coords
        
        # 计算累积弧长
        diffs = np.diff(coords, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]
        
        if total_length < 1e-10:
            return coords[:self.max_points] if n_points > self.max_points else coords
        
        # 等距采样点
        target_distances = np.linspace(0, total_length, self.max_points)
        
        # 插值
        resampled = np.zeros((self.max_points, 2))
        for i, target in enumerate(target_distances):
            # 找到目标距离所在的线段
            idx = np.searchsorted(cumulative_length, target, side='right') - 1
            idx = max(0, min(idx, n_points - 2))
            
            # 线性插值
            segment_start = cumulative_length[idx]
            segment_end = cumulative_length[idx + 1]
            segment_length = segment_end - segment_start
            
            if segment_length > 1e-10:
                t = (target - segment_start) / segment_length
            else:
                t = 0
            
            resampled[i] = coords[idx] * (1 - t) + coords[idx + 1] * t
        
        return resampled
    
    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """归一化坐标到 [0, 1]"""
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = max_vals - min_vals
        
        # 避免除零
        range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)
        
        # 保持宽高比
        max_range = range_vals.max()
        normalized = (coords - min_vals) / max_range
        
        # 居中
        center_offset = (1 - (range_vals / max_range)) / 2
        normalized = normalized + center_offset
        
        return normalized
    
    def _align_start_point(self, coords: np.ndarray) -> np.ndarray:
        """
        确定性起点选择
        选择最左下角的点作为起点
        """
        # 计算每个点的"左下角程度"：x + y 最小的点
        scores = coords[:, 0] + coords[:, 1]
        start_idx = np.argmin(scores)
        
        # 旋转序列使起点为第一个
        return np.roll(coords, -start_idx, axis=0)


class GeometryAugmentor:
    """
    几何数据增强器
    生成多视图用于对比学习
    """
    
    def __init__(
        self,
        rotation_angles: List[float] = [0, 90, 180, 270],
        noise_std: float = 0.001,
        simplify_tolerance_range: Tuple[float, float] = (0.00005, 0.0002),
        subsample_ratio_range: Tuple[float, float] = (0.7, 0.95)
    ):
        """
        Args:
            rotation_angles: 旋转角度列表
            noise_std: 噪声标准差
            simplify_tolerance_range: 简化容差范围
            subsample_ratio_range: 抽稀比例范围
        """
        self.rotation_angles = rotation_angles
        self.noise_std = noise_std
        self.simplify_tolerance_range = simplify_tolerance_range
        self.subsample_ratio_range = subsample_ratio_range
    
    def augment(
        self,
        coords: np.ndarray,
        augment_type: Optional[str] = None
    ) -> np.ndarray:
        """
        对坐标进行增强
        
        Args:
            coords: 输入坐标 [N, 2]
            augment_type: 增强类型，None表示随机选择
            
        Returns:
            增强后的坐标 [N, 2]
        """
        if augment_type is None:
            augment_type = random.choice(['rotation', 'noise', 'subsample', 'none'])
        
        if augment_type == 'rotation':
            return self.rotate(coords)
        elif augment_type == 'noise':
            return self.add_noise(coords)
        elif augment_type == 'subsample':
            return self.subsample(coords)
        elif augment_type == 'simplify':
            return self.simplify(coords)
        else:
            return coords.copy()
    
    def rotate(self, coords: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """旋转增强"""
        if angle is None:
            angle = random.choice(self.rotation_angles)
        
        # 转换为弧度
        theta = np.radians(angle)
        
        # 中心点
        center = coords.mean(axis=0)
        
        # 旋转矩阵
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # 旋转
        centered = coords - center
        rotated = centered @ rot_matrix.T
        
        return rotated + center
    
    def add_noise(self, coords: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        """添加高斯噪声"""
        if std is None:
            std = self.noise_std
        
        noise = np.random.randn(*coords.shape) * std
        return coords + noise
    
    def subsample(
        self, 
        coords: np.ndarray, 
        ratio: Optional[float] = None
    ) -> np.ndarray:
        """随机抽稀"""
        if ratio is None:
            ratio = random.uniform(*self.subsample_ratio_range)
        
        n_points = len(coords)
        n_keep = max(int(n_points * ratio), 8)  # 至少保留8个点
        
        # 均匀抽样保持形状
        indices = np.linspace(0, n_points - 1, n_keep).astype(int)
        return coords[indices]
    
    def simplify(
        self, 
        coords: np.ndarray, 
        tolerance: Optional[float] = None
    ) -> np.ndarray:
        """DP简化"""
        if tolerance is None:
            tolerance = random.uniform(*self.simplify_tolerance_range)
        
        # 使用shapely进行简化
        line = LineString(coords)
        simplified = line.simplify(tolerance, preserve_topology=True)
        return np.array(simplified.coords)
    
    def generate_views(
        self, 
        coords: np.ndarray, 
        n_views: int = 4
    ) -> List[np.ndarray]:
        """
        生成多个视图
        
        Args:
            coords: 输入坐标
            n_views: 视图数量
            
        Returns:
            视图列表
        """
        views = [coords.copy()]  # 原始视图
        
        augment_types = ['rotation', 'noise', 'subsample']
        for i in range(n_views - 1):
            aug_type = augment_types[i % len(augment_types)]
            views.append(self.augment(coords, aug_type))
        
        return views


class ShapefileDataset(Dataset):
    """
    Shapefile数据集
    """
    
    def __init__(
        self,
        data_dir: str,
        normalizer: GeometryNormalizer,
        augmentor: Optional[GeometryAugmentor] = None,
        file_pattern: str = "*.shp",
        target_crs: str = "EPSG:4326",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_dir: 数据目录
            normalizer: 几何规范化器
            augmentor: 几何增强器
            file_pattern: 文件匹配模式
            target_crs: 目标坐标系
            max_samples: 最大样本数
            cache_dir: 缓存目录
        """
        self.data_dir = Path(data_dir)
        self.normalizer = normalizer
        self.augmentor = augmentor
        self.target_crs = target_crs
        self.max_samples = max_samples
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 加载所有几何数据
        self.geometries = []
        self.labels = []
        self.file_sources = []
        
        self._load_data(file_pattern)
    
    def _load_data(self, file_pattern: str):
        """加载所有shapefile数据"""
        shp_files = list(self.data_dir.glob(f"**/{file_pattern}"))
        
        if not shp_files:
            print(f"Warning: No shapefile found in {self.data_dir}")
            return
        
        print(f"Found {len(shp_files)} shapefiles")
        
        sample_count = 0
        file_id = 0
        
        for shp_path in shp_files:
            try:
                gdf = gpd.read_file(shp_path)
                
                # 坐标系转换
                if gdf.crs is not None and gdf.crs != self.target_crs:
                    gdf = gdf.to_crs(self.target_crs)
                
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    
                    # 规范化
                    coords = self.normalizer.normalize(geom)
                    if coords is not None:
                        self.geometries.append(coords)
                        self.labels.append(file_id)
                        self.file_sources.append(str(shp_path))
                        sample_count += 1
                        
                        if self.max_samples and sample_count >= self.max_samples:
                            break
                
                file_id += 1
                
                if self.max_samples and sample_count >= self.max_samples:
                    break
                    
            except Exception as e:
                print(f"Error loading {shp_path}: {e}")
                continue
        
        print(f"Loaded {len(self.geometries)} geometries from {file_id} files")
    
    def __len__(self) -> int:
        return len(self.geometries)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取样本
        
        Returns:
            {
                'coords': 坐标张量 [max_points, 2],
                'label': 类别标签,
                'idx': 样本索引
            }
        """
        coords = self.geometries[idx]
        label = self.labels[idx]
        
        # 数据增强
        if self.augmentor is not None:
            coords = self.augmentor.augment(coords)
        
        # 确保点数一致
        coords = self._pad_or_truncate(coords)
        
        return {
            'coords': torch.FloatTensor(coords),
            'label': label,
            'idx': idx
        }
    
    def _pad_or_truncate(self, coords: np.ndarray) -> np.ndarray:
        """填充或截断到固定长度"""
        target_len = self.normalizer.max_points
        current_len = len(coords)
        
        if current_len == target_len:
            return coords
        elif current_len > target_len:
            # 均匀采样
            indices = np.linspace(0, current_len - 1, target_len).astype(int)
            return coords[indices]
        else:
            # 填充
            padded = np.zeros((target_len, 2), dtype=coords.dtype)
            padded[:current_len] = coords
            # 用最后一个点填充
            padded[current_len:] = coords[-1]
            return padded
    
    def get_original_geometry(self, idx: int) -> np.ndarray:
        """获取原始（未增强）几何"""
        return self.geometries[idx].copy()


class ContrastiveDataset(Dataset):
    """
    对比学习数据集
    为每个样本生成多个视图
    """
    
    def __init__(
        self,
        base_dataset: ShapefileDataset,
        n_views: int = 4,
        augmentor: Optional[GeometryAugmentor] = None
    ):
        """
        Args:
            base_dataset: 基础数据集
            n_views: 每个样本的视图数
            augmentor: 增强器
        """
        self.base_dataset = base_dataset
        self.n_views = n_views
        self.augmentor = augmentor or GeometryAugmentor()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取多视图样本
        
        Returns:
            {
                'views': 多视图张量 [n_views, max_points, 2],
                'label': 类别标签,
                'idx': 样本索引
            }
        """
        coords = self.base_dataset.get_original_geometry(idx)
        label = self.base_dataset.labels[idx]
        
        # 生成多视图
        views = self.augmentor.generate_views(coords, self.n_views)
        
        # 规范化每个视图
        processed_views = []
        for view in views:
            view = self.base_dataset._pad_or_truncate(view)
            processed_views.append(view)
        
        views_tensor = torch.FloatTensor(np.stack(processed_views))
        
        return {
            'views': views_tensor,
            'label': label,
            'idx': idx
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_points: int = 256,
    n_views: int = 4,
    shuffle: bool = True,
    contrastive: bool = True,
    **kwargs
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作线程数
        max_points: 最大点数
        n_views: 视图数
        shuffle: 是否打乱
        contrastive: 是否使用对比学习模式
        
    Returns:
        DataLoader
    """
    normalizer = GeometryNormalizer(max_points=max_points)
    augmentor = GeometryAugmentor()
    
    base_dataset = ShapefileDataset(
        data_dir=data_dir,
        normalizer=normalizer,
        augmentor=None if contrastive else augmentor,
        **kwargs
    )
    
    if contrastive:
        dataset = ContrastiveDataset(
            base_dataset=base_dataset,
            n_views=n_views,
            augmentor=augmentor
        )
    else:
        dataset = base_dataset
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


# 扰动函数（用于评估）
def apply_perturbation(
    coords: np.ndarray,
    perturbation_type: str,
    param: float
) -> np.ndarray:
    """
    应用指定的扰动
    
    Args:
        coords: 输入坐标
        perturbation_type: 扰动类型
        param: 扰动参数
        
    Returns:
        扰动后的坐标
    """
    augmentor = GeometryAugmentor()
    
    if perturbation_type == "rotation":
        return augmentor.rotate(coords, angle=param)
    elif perturbation_type == "noise":
        return augmentor.add_noise(coords, std=param)
    elif perturbation_type == "simplify":
        return augmentor.simplify(coords, tolerance=param)
    elif perturbation_type == "subsample":
        return augmentor.subsample(coords, ratio=param)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


# ============================================
# 全局签名数据集（每个shp文件一个样本）
# ============================================

class AdaptiveGeometryNormalizer(GeometryNormalizer):
    """
    自适应几何规范化器
    根据几何复杂度动态调整采样点数
    """
    
    def __init__(
        self,
        max_points: int = 256,
        min_points: int = 64,
        dp_tolerance: float = 0.0001,
        adaptive: bool = True
    ):
        super().__init__(max_points, min_points, dp_tolerance)
        self.adaptive = adaptive
    
    def normalize(self, geometry, target_points: Optional[int] = None) -> Optional[np.ndarray]:
        """
        规范化几何对象，支持自适应点数
        
        Args:
            geometry: Shapely几何对象
            target_points: 目标点数（None表示自适应）
        """
        coords = self._extract_coords(geometry)
        if coords is None or len(coords) < self.min_points:
            return None
        
        # 自适应计算目标点数
        if target_points is None and self.adaptive:
            target_points = self._compute_adaptive_points(geometry, coords)
        elif target_points is None:
            target_points = self.max_points
        
        # DP简化
        if len(coords) > target_points * 2:
            coords = self._simplify(coords)
        
        # 重采样到目标点数
        original_max = self.max_points
        self.max_points = target_points
        coords = self._resample(coords)
        self.max_points = original_max
        
        # 归一化
        coords = self._normalize_coords(coords)
        coords = self._align_start_point(coords)
        
        return coords
    
    def _compute_adaptive_points(self, geometry, coords: np.ndarray) -> int:
        """根据几何复杂度计算采样点数"""
        try:
            # 基于周长计算
            if hasattr(geometry, 'length'):
                length = geometry.length
            else:
                length = LineString(coords).length
            
            # 基于复杂度的点数计算
            # 简单几何：64-128点，复杂几何：128-256点
            if length < 0.01:  # 非常简单
                num_points = self.min_points
            elif length < 0.1:  # 简单
                num_points = min(128, self.max_points)
            else:  # 复杂
                num_points = self.max_points
            
            return max(self.min_points, min(num_points, self.max_points))
        except:
            return self.max_points


class ShapefileGlobalDataset(Dataset):
    """
    全局Shapefile数据集
    每个shp文件作为一个样本，输出该文件所有几何的集合
    用于生成整个shp文件的全局签名
    """
    
    def __init__(
        self,
        data_dir: str,
        normalizer: Optional[GeometryNormalizer] = None,
        augmentor: Optional[GeometryAugmentor] = None,
        file_pattern: str = "*.shp",
        target_crs: str = "EPSG:4326",
        max_geometries: int = 500,
        min_geometries: int = 10,
        fixed_points: int = 128,
        n_views: int = 2
    ):
        """
        Args:
            data_dir: 数据目录
            normalizer: 几何规范化器
            augmentor: 几何增强器
            file_pattern: 文件匹配模式
            target_crs: 目标坐标系
            max_geometries: 每个shp最大采样几何数
            min_geometries: 最小几何数（少于此数的文件跳过）
            fixed_points: 每个几何的固定点数
            n_views: 视图数（用于对比学习）
        """
        self.data_dir = Path(data_dir)
        self.normalizer = normalizer or AdaptiveGeometryNormalizer(max_points=fixed_points)
        self.augmentor = augmentor or GeometryAugmentor()
        self.target_crs = target_crs
        self.max_geometries = max_geometries
        self.min_geometries = min_geometries
        self.fixed_points = fixed_points
        self.n_views = n_views
        
        # 存储每个shp文件的信息
        self.shp_files = []
        self.shp_geometries = []  # 每个shp的所有几何
        
        self._load_shapefiles(file_pattern)
    
    def _load_shapefiles(self, file_pattern: str):
        """加载所有shapefile"""
        shp_files = list(self.data_dir.glob(f"**/{file_pattern}"))
        
        if not shp_files:
            print(f"Warning: No shapefile found in {self.data_dir}")
            return
        
        print(f"Found {len(shp_files)} shapefiles, loading...")
        
        for shp_path in shp_files:
            try:
                gdf = gpd.read_file(shp_path)
                
                # 坐标系转换
                if gdf.crs is not None and str(gdf.crs) != self.target_crs:
                    gdf = gdf.to_crs(self.target_crs)
                
                # 提取所有有效几何
                geometries = []
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    
                    coords = self.normalizer.normalize(geom)
                    if coords is not None:
                        # 确保点数一致
                        coords = self._pad_or_truncate(coords, self.fixed_points)
                        geometries.append(coords)
                
                # 检查是否有足够的几何
                if len(geometries) >= self.min_geometries:
                    self.shp_files.append(str(shp_path))
                    self.shp_geometries.append(geometries)
                    print(f"  Loaded {shp_path.name}: {len(geometries)} geometries")
                else:
                    print(f"  Skipped {shp_path.name}: only {len(geometries)} geometries")
                    
            except Exception as e:
                print(f"Error loading {shp_path}: {e}")
                continue
        
        print(f"Total: {len(self.shp_files)} shapefiles loaded")
    
    def _pad_or_truncate(self, coords: np.ndarray, target_len: int) -> np.ndarray:
        """填充或截断到固定长度"""
        current_len = len(coords)
        
        if current_len == target_len:
            return coords
        elif current_len > target_len:
            indices = np.linspace(0, current_len - 1, target_len).astype(int)
            return coords[indices]
        else:
            padded = np.zeros((target_len, 2), dtype=coords.dtype)
            padded[:current_len] = coords
            padded[current_len:] = coords[-1]
            return padded
    
    def _sample_geometries(
        self, 
        geometries: List[np.ndarray],
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样几何体
        
        Args:
            geometries: 所有几何列表
            n_samples: 采样数量（None表示自适应）
            
        Returns:
            (采样的几何数组, 有效掩码)
        """
        n_total = len(geometries)
        
        # 自适应采样数量
        if n_samples is None:
            n_samples = min(n_total, self.max_geometries)
        
        # 采样或使用全部
        if n_total <= n_samples:
            # 使用全部，需要填充
            sampled = geometries.copy()
            # 填充到 max_geometries
            while len(sampled) < self.max_geometries:
                # 循环填充
                sampled.append(geometries[len(sampled) % n_total])
            sampled = sampled[:self.max_geometries]
            mask = np.zeros(self.max_geometries, dtype=bool)
            mask[:n_total] = True
        else:
            # 随机采样
            indices = np.random.choice(n_total, n_samples, replace=False)
            sampled = [geometries[i] for i in indices]
            # 填充到 max_geometries
            while len(sampled) < self.max_geometries:
                idx = np.random.randint(0, n_total)
                sampled.append(geometries[idx])
            sampled = sampled[:self.max_geometries]
            mask = np.ones(self.max_geometries, dtype=bool)
        
        return np.array(sampled), mask
    
    def __len__(self) -> int:
        return len(self.shp_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个shp文件的所有几何
        
        Returns:
            {
                'geometries': 几何数组 [max_geometries, fixed_points, 2],
                'mask': 有效掩码 [max_geometries],
                'n_geometries': 实际几何数量,
                'file_path': 文件路径,
                'file_idx': 文件索引
            }
        """
        geometries = self.shp_geometries[idx]
        file_path = self.shp_files[idx]
        
        # 采样几何
        sampled_geoms, mask = self._sample_geometries(geometries)
        
        return {
            'geometries': torch.FloatTensor(sampled_geoms),
            'mask': torch.BoolTensor(mask),
            'n_geometries': len(geometries),
            'file_path': file_path,
            'file_idx': idx
        }
    
    def get_all_geometries(self, idx: int) -> np.ndarray:
        """获取一个shp文件的所有几何（不采样）"""
        return np.array(self.shp_geometries[idx])


class GlobalContrastiveDataset(Dataset):
    """
    全局对比学习数据集
    为每个shp文件生成多个视图用于对比学习
    """
    
    def __init__(
        self,
        base_dataset: ShapefileGlobalDataset,
        n_views: int = 2,
        augmentor: Optional[GeometryAugmentor] = None
    ):
        """
        Args:
            base_dataset: 基础全局数据集
            n_views: 视图数
            augmentor: 增强器
        """
        self.base_dataset = base_dataset
        self.n_views = n_views
        self.augmentor = augmentor or GeometryAugmentor()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def _augment_geometries(self, geometries: np.ndarray) -> np.ndarray:
        """对一组几何进行增强"""
        augmented = []
        for geom in geometries:
            aug_geom = self.augmentor.augment(geom)
            # 确保点数一致
            aug_geom = self.base_dataset._pad_or_truncate(
                aug_geom, self.base_dataset.fixed_points
            )
            augmented.append(aug_geom)
        return np.array(augmented)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取多视图样本
        
        Returns:
            {
                'views': 多视图 [n_views, max_geometries, fixed_points, 2],
                'masks': 掩码 [n_views, max_geometries],
                'n_geometries': 实际几何数量,
                'file_path': 文件路径,
                'file_idx': 文件索引
            }
        """
        # 获取原始数据
        original = self.base_dataset[idx]
        geometries_np = original['geometries'].numpy()
        mask_np = original['mask'].numpy()
        
        views = [geometries_np]  # 第一个是原始视图
        masks = [mask_np]
        
        # 生成增强视图
        for _ in range(self.n_views - 1):
            aug_geoms = self._augment_geometries(geometries_np)
            views.append(aug_geoms)
            masks.append(mask_np)  # 掩码相同
        
        return {
            'views': torch.FloatTensor(np.stack(views)),
            'masks': torch.BoolTensor(np.stack(masks)),
            'n_geometries': original['n_geometries'],
            'file_path': original['file_path'],
            'file_idx': original['file_idx']
        }


def create_global_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    max_geometries: int = 500,
    fixed_points: int = 128,
    n_views: int = 2,
    shuffle: bool = True,
    contrastive: bool = True,
    **kwargs
) -> DataLoader:
    """
    创建全局签名数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小（shp文件数）
        num_workers: 工作线程数
        max_geometries: 每个shp最大几何数
        fixed_points: 每个几何的固定点数
        n_views: 视图数
        shuffle: 是否打乱
        contrastive: 是否使用对比学习模式
        
    Returns:
        DataLoader
    """
    base_dataset = ShapefileGlobalDataset(
        data_dir=data_dir,
        max_geometries=max_geometries,
        fixed_points=fixed_points,
        n_views=n_views,
        **kwargs
    )
    
    if contrastive:
        dataset = GlobalContrastiveDataset(
            base_dataset=base_dataset,
            n_views=n_views
        )
    else:
        dataset = base_dataset
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 不丢弃最后一个batch
    )

