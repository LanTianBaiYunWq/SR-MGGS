"""
几何渲染模块
===============================

本模块将矢量几何坐标渲染为栅格图像，供视觉教师模型（ResNet50）使用。

核心功能：
    1. 坐标到图像的转换：将归一化坐标映射到像素空间
    2. 多种渲染风格：支持填充、描边、抗锯齿
    3. 空间对齐：确保几何特征与视觉特征的空间一致性

主要组件：
    1. GeometryRenderer：基础几何渲染器
       - 将点序列渲染为图像
       - 支持多边形填充和线条绘制
       - 可配置颜色、线宽、填充等
    
    2. MultiViewRenderer：多视图渲染器
       - 生成多个增强视图
       - 用于对比学习
    
    3. CachedRenderer：缓存渲染器
       - 缓存渲染结果
       - 提高训练效率
    
    4. SpatialAlignedRenderer：空间对齐渲染器
       - 精确的坐标到像素映射
       - 返回空间映射信息
    
    5. MultiAngleRenderer：多角度渲染器
       - 从不同角度渲染几何
       - 增强模型的旋转不变性

便捷函数：
    - coords_to_image(): 单个几何渲染
    - batch_coords_to_images(): 批量渲染
    - render_with_spatial_info(): 带空间信息的渲染

配置参数：
    - image_size: 渲染图像尺寸（默认 224，匹配 ResNet）
    - line_width: 线条宽度
    - fill: 是否填充多边形
    - padding: 边界填充比例

作者：GSD Team
版本：1.0
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms


class GeometryRenderer:
    """
    几何渲染器
    将点序列渲染为图像
    """
    
    def __init__(
        self,
        image_size: int = 224,
        line_width: int = 2,
        fill: bool = True,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        fill_color: Tuple[int, int, int] = (100, 100, 100),
        edge_color: Tuple[int, int, int] = (0, 0, 0),
        padding: float = 0.1,
        antialias: bool = True
    ):
        """
        Args:
            image_size: 输出图像尺寸
            line_width: 线宽
            fill: 是否填充多边形
            background_color: 背景色
            fill_color: 填充色
            edge_color: 边缘色
            padding: 边界填充比例
            antialias: 是否抗锯齿
        """
        self.image_size = image_size
        self.line_width = line_width
        self.fill = fill
        self.background_color = background_color
        self.fill_color = fill_color
        self.edge_color = edge_color
        self.padding = padding
        self.antialias = antialias
        
        # 如果需要抗锯齿，使用更大的画布然后缩小
        self.render_scale = 2 if antialias else 1
        self.render_size = image_size * self.render_scale
        
        # 图像预处理（用于ResNet）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def render(
        self,
        coords: np.ndarray,
        as_tensor: bool = True,
        normalize: bool = True
    ) -> Union[Image.Image, torch.Tensor]:
        """
        渲染几何为图像
        
        Args:
            coords: 坐标点 [N, 2]，假设已归一化到 [0, 1]
            as_tensor: 是否返回tensor
            normalize: 是否应用ImageNet归一化
            
        Returns:
            渲染后的图像
        """
        # 创建画布
        img = Image.new('RGB', (self.render_size, self.render_size), self.background_color)
        draw = ImageDraw.Draw(img)
        
        # 坐标转换：[0, 1] -> 像素坐标
        effective_size = self.render_size * (1 - 2 * self.padding)
        offset = self.render_size * self.padding
        
        pixel_coords = coords * effective_size + offset
        
        # 转换为整数坐标列表
        points = [(int(x), int(y)) for x, y in pixel_coords]
        
        # 绘制
        if len(points) >= 3:
            if self.fill:
                # 填充多边形
                draw.polygon(points, fill=self.fill_color, outline=self.edge_color)
            else:
                # 只绘制边缘
                draw.polygon(points, outline=self.edge_color)
        elif len(points) >= 2:
            # 绘制折线
            draw.line(points, fill=self.edge_color, width=self.line_width * self.render_scale)
        
        # 抗锯齿缩放
        if self.antialias:
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        if as_tensor:
            if normalize:
                return self.transform(img)
            else:
                return transforms.ToTensor()(img)
        
        return img
    
    def render_batch(
        self,
        coords_batch: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        批量渲染
        
        Args:
            coords_batch: 坐标批次 [batch_size, N, 2]
            normalize: 是否应用ImageNet归一化
            
        Returns:
            图像张量 [batch_size, 3, H, W]
        """
        if isinstance(coords_batch, torch.Tensor):
            coords_batch = coords_batch.cpu().numpy()
        
        images = []
        for coords in coords_batch:
            img_tensor = self.render(coords, as_tensor=True, normalize=normalize)
            images.append(img_tensor)
        
        return torch.stack(images)
    
    def render_with_augmentation(
        self,
        coords: np.ndarray,
        augment: bool = True
    ) -> torch.Tensor:
        """
        带图像增强的渲染
        
        Args:
            coords: 坐标点
            augment: 是否进行图像增强
            
        Returns:
            渲染后的图像张量
        """
        img = self.render(coords, as_tensor=False, normalize=False)
        
        if augment:
            # 图像级别增强
            augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            return augment_transform(img)
        else:
            return self.transform(img)


class MultiViewRenderer:
    """
    多视图渲染器
    为对比学习生成多个视图
    """
    
    def __init__(
        self,
        base_renderer: Optional[GeometryRenderer] = None,
        n_views: int = 2
    ):
        """
        Args:
            base_renderer: 基础渲染器
            n_views: 视图数量
        """
        self.renderer = base_renderer or GeometryRenderer()
        self.n_views = n_views
    
    def render_views(
        self,
        coords: np.ndarray
    ) -> List[torch.Tensor]:
        """
        渲染多个视图
        
        Args:
            coords: 坐标点
            
        Returns:
            视图列表
        """
        views = []
        for i in range(self.n_views):
            # 第一个视图不增强，后续视图增强
            augment = i > 0
            view = self.renderer.render_with_augmentation(coords, augment=augment)
            views.append(view)
        
        return views
    
    def render_batch_views(
        self,
        coords_batch: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        批量渲染多视图
        
        Args:
            coords_batch: 坐标批次 [batch_size, N, 2]
            
        Returns:
            视图张量 [batch_size, n_views, 3, H, W]
        """
        if isinstance(coords_batch, torch.Tensor):
            coords_batch = coords_batch.cpu().numpy()
        
        batch_views = []
        for coords in coords_batch:
            views = self.render_views(coords)
            batch_views.append(torch.stack(views))
        
        return torch.stack(batch_views)


class CachedRenderer:
    """
    带缓存的渲染器
    避免重复渲染相同几何
    """
    
    def __init__(
        self,
        renderer: Optional[GeometryRenderer] = None,
        cache_size: int = 10000
    ):
        """
        Args:
            renderer: 基础渲染器
            cache_size: 缓存大小
        """
        self.renderer = renderer or GeometryRenderer()
        self.cache_size = cache_size
        self.cache = {}
        self.access_order = []
    
    def _get_cache_key(self, coords: np.ndarray) -> str:
        """生成缓存键"""
        return hash(coords.tobytes())
    
    def render(
        self,
        coords: np.ndarray,
        **kwargs
    ) -> torch.Tensor:
        """渲染（带缓存）"""
        key = self._get_cache_key(coords)
        
        if key in self.cache:
            # 更新访问顺序
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # 渲染
        result = self.renderer.render(coords, **kwargs)
        
        # 缓存
        if len(self.cache) >= self.cache_size:
            # 移除最久未访问的
            old_key = self.access_order.pop(0)
            del self.cache[old_key]
        
        self.cache[key] = result
        self.access_order.append(key)
        
        return result
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()


def coords_to_image(
    coords: np.ndarray,
    image_size: int = 224,
    fill: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    便捷函数：将坐标转换为图像张量
    
    Args:
        coords: 坐标点 [N, 2]
        image_size: 图像尺寸
        fill: 是否填充
        normalize: 是否归一化
        
    Returns:
        图像张量 [3, H, W]
    """
    renderer = GeometryRenderer(image_size=image_size, fill=fill)
    return renderer.render(coords, as_tensor=True, normalize=normalize)


def batch_coords_to_images(
    coords_batch: Union[np.ndarray, torch.Tensor],
    image_size: int = 224,
    fill: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    便捷函数：批量将坐标转换为图像
    
    Args:
        coords_batch: 坐标批次 [batch_size, N, 2]
        image_size: 图像尺寸
        fill: 是否填充
        normalize: 是否归一化
        
    Returns:
        图像张量 [batch_size, 3, H, W]
    """
    renderer = GeometryRenderer(image_size=image_size, fill=fill)
    return renderer.render_batch(coords_batch, normalize=normalize)


# ============================================
# 空间对齐渲染器（改进版）
# ============================================

class SpatialAlignedRenderer:
    """
    空间对齐渲染器
    确保几何坐标与图像像素的精确空间对应
    返回空间映射信息用于特征对齐
    """
    
    def __init__(
        self,
        image_size: int = 224,
        line_width: int = 2,
        fill: bool = True,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        fill_color: Tuple[int, int, int] = (100, 100, 100),
        edge_color: Tuple[int, int, int] = (0, 0, 0),
        padding: float = 0.1
    ):
        """
        Args:
            image_size: 输出图像尺寸
            line_width: 线宽
            fill: 是否填充
            background_color: 背景色
            fill_color: 填充色
            edge_color: 边缘色
            padding: 边界填充比例
        """
        self.image_size = image_size
        self.line_width = line_width
        self.fill = fill
        self.background_color = background_color
        self.fill_color = fill_color
        self.edge_color = edge_color
        self.padding = padding
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def compute_spatial_mapping(
        self,
        coords: np.ndarray
    ) -> dict:
        """
        计算几何坐标到像素坐标的空间映射
        
        Args:
            coords: 原始坐标 [N, 2]
            
        Returns:
            空间映射信息字典
        """
        # 计算边界框
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        
        # 计算范围
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        # 避免除零
        range_x = max(range_x, 1e-10)
        range_y = max(range_y, 1e-10)
        
        # 保持宽高比
        max_range = max(range_x, range_y)
        
        # 计算有效渲染区域
        effective_size = self.image_size * (1 - 2 * self.padding)
        offset = self.image_size * self.padding
        
        # 比例因子
        scale = effective_size / max_range
        
        # 居中偏移
        center_offset_x = (effective_size - range_x * scale) / 2
        center_offset_y = (effective_size - range_y * scale) / 2
        
        return {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'range_x': range_x,
            'range_y': range_y,
            'scale': scale,
            'offset_x': offset + center_offset_x,
            'offset_y': offset + center_offset_y,
            'image_size': self.image_size
        }
    
    def coords_to_pixels(
        self,
        coords: np.ndarray,
        mapping: dict
    ) -> np.ndarray:
        """
        将几何坐标转换为像素坐标
        
        Args:
            coords: 几何坐标 [N, 2]
            mapping: 空间映射信息
            
        Returns:
            像素坐标 [N, 2]
        """
        pixel_x = (coords[:, 0] - mapping['min_x']) * mapping['scale'] + mapping['offset_x']
        pixel_y = (coords[:, 1] - mapping['min_y']) * mapping['scale'] + mapping['offset_y']
        return np.stack([pixel_x, pixel_y], axis=1)
    
    def pixels_to_coords(
        self,
        pixels: np.ndarray,
        mapping: dict
    ) -> np.ndarray:
        """
        将像素坐标转换回几何坐标
        
        Args:
            pixels: 像素坐标 [N, 2]
            mapping: 空间映射信息
            
        Returns:
            几何坐标 [N, 2]
        """
        coord_x = (pixels[:, 0] - mapping['offset_x']) / mapping['scale'] + mapping['min_x']
        coord_y = (pixels[:, 1] - mapping['offset_y']) / mapping['scale'] + mapping['min_y']
        return np.stack([coord_x, coord_y], axis=1)
    
    def render_with_mapping(
        self,
        coords: np.ndarray,
        as_tensor: bool = True,
        normalize: bool = True
    ) -> Tuple[Union[Image.Image, torch.Tensor], dict]:
        """
        渲染几何并返回空间映射信息
        
        Args:
            coords: 坐标点 [N, 2]（原始坐标，非归一化）
            as_tensor: 是否返回tensor
            normalize: 是否应用ImageNet归一化
            
        Returns:
            (渲染图像, 空间映射信息)
        """
        # 计算空间映射
        mapping = self.compute_spatial_mapping(coords)
        
        # 转换为像素坐标
        pixel_coords = self.coords_to_pixels(coords, mapping)
        
        # 创建画布
        img = Image.new('RGB', (self.image_size, self.image_size), self.background_color)
        draw = ImageDraw.Draw(img)
        
        # 绘制
        points = [(int(x), int(y)) for x, y in pixel_coords]
        
        if len(points) >= 3:
            if self.fill:
                draw.polygon(points, fill=self.fill_color, outline=self.edge_color)
            else:
                draw.polygon(points, outline=self.edge_color)
        elif len(points) >= 2:
            draw.line(points, fill=self.edge_color, width=self.line_width)
        
        if as_tensor:
            if normalize:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)
            return img_tensor, mapping
        
        return img, mapping
    
    def generate_spatial_attention_mask(
        self,
        coords: np.ndarray,
        mapping: dict,
        sigma: float = 5.0
    ) -> np.ndarray:
        """
        生成空间注意力掩码
        几何点密集的区域权重更高
        
        Args:
            coords: 坐标点
            mapping: 空间映射
            sigma: 高斯核大小
            
        Returns:
            注意力掩码 [H, W]
        """
        from scipy.ndimage import gaussian_filter
        
        # 转换为像素坐标
        pixel_coords = self.coords_to_pixels(coords, mapping)
        
        # 创建稀疏掩码
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        for px, py in pixel_coords:
            x, y = int(px), int(py)
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                mask[y, x] = 1.0
        
        # 高斯平滑
        mask = gaussian_filter(mask, sigma=sigma)
        
        # 归一化到 [0, 1]
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask


class MultiAngleRenderer:
    """
    多角度渲染器
    从不同视角渲染几何体，用于增强对比学习
    """
    
    def __init__(
        self,
        base_renderer: Optional[SpatialAlignedRenderer] = None,
        angles: List[float] = [0, 90, 180, 270],
        scales: List[float] = [1.0],
        translations: List[Tuple[float, float]] = [(0, 0)]
    ):
        """
        Args:
            base_renderer: 基础渲染器
            angles: 旋转角度列表
            scales: 缩放比例列表
            translations: 平移列表
        """
        self.renderer = base_renderer or SpatialAlignedRenderer()
        self.angles = angles
        self.scales = scales
        self.translations = translations
    
    def rotate_coords(
        self,
        coords: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """旋转坐标"""
        theta = np.radians(angle)
        center = coords.mean(axis=0)
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ])
        
        centered = coords - center
        rotated = centered @ rotation_matrix.T
        return rotated + center
    
    def scale_coords(
        self,
        coords: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """缩放坐标"""
        center = coords.mean(axis=0)
        return (coords - center) * scale + center
    
    def translate_coords(
        self,
        coords: np.ndarray,
        tx: float,
        ty: float
    ) -> np.ndarray:
        """平移坐标"""
        return coords + np.array([tx, ty])
    
    def render_multi_angle(
        self,
        coords: np.ndarray,
        n_views: int = 4
    ) -> List[Tuple[torch.Tensor, dict]]:
        """
        从多个角度渲染
        
        Args:
            coords: 坐标点
            n_views: 视图数量
            
        Returns:
            [(图像, 映射信息), ...]
        """
        views = []
        
        # 原始视图
        img, mapping = self.renderer.render_with_mapping(coords)
        views.append((img, mapping))
        
        # 旋转视图
        for i in range(1, min(n_views, len(self.angles))):
            angle = self.angles[i]
            rotated = self.rotate_coords(coords, angle)
            img, mapping = self.renderer.render_with_mapping(rotated)
            mapping['rotation'] = angle
            views.append((img, mapping))
        
        return views
    
    def render_augmented_views(
        self,
        coords: np.ndarray,
        n_views: int = 2
    ) -> List[torch.Tensor]:
        """
        渲染增强视图（简化版，只返回图像）
        
        Args:
            coords: 坐标点
            n_views: 视图数量
            
        Returns:
            [图像tensor, ...]
        """
        views = []
        
        # 原始视图
        img, _ = self.renderer.render_with_mapping(coords)
        views.append(img)
        
        # 增强视图
        for i in range(1, n_views):
            # 随机选择变换
            if i < len(self.angles):
                angle = self.angles[i]
                transformed = self.rotate_coords(coords, angle)
            else:
                angle = np.random.choice(self.angles)
                transformed = self.rotate_coords(coords, angle)
            
            img, _ = self.renderer.render_with_mapping(transformed)
            views.append(img)
        
        return views


def render_with_spatial_info(
    coords: np.ndarray,
    image_size: int = 224
) -> Tuple[torch.Tensor, dict]:
    """
    便捷函数：渲染并返回空间信息
    
    Args:
        coords: 坐标点
        image_size: 图像尺寸
        
    Returns:
        (图像tensor, 空间映射)
    """
    renderer = SpatialAlignedRenderer(image_size=image_size)
    return renderer.render_with_mapping(coords)


def batch_render_with_spatial_info(
    coords_batch: Union[np.ndarray, torch.Tensor],
    image_size: int = 224
) -> Tuple[torch.Tensor, List[dict]]:
    """
    批量渲染并返回空间信息
    
    Args:
        coords_batch: 坐标批次 [B, N, 2]
        image_size: 图像尺寸
        
    Returns:
        (图像tensor [B, 3, H, W], 空间映射列表)
    """
    if isinstance(coords_batch, torch.Tensor):
        coords_batch = coords_batch.cpu().numpy()
    
    renderer = SpatialAlignedRenderer(image_size=image_size)
    
    images = []
    mappings = []
    
    for coords in coords_batch:
        img, mapping = renderer.render_with_mapping(coords)
        images.append(img)
        mappings.append(mapping)
    
    return torch.stack(images), mappings

