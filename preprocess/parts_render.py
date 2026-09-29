"""
确定性 Parts 渲染器
===============================

本模块将 GeomPart 列表渲染为图像，用于视觉教师。

设计要点：
    1. 确定性渲染：相同输入产生完全相同的输出
    2. 支持所有几何类型：Ring/Line/PointSet
    3. 与 canonical pipeline 配合使用
    4. 输出标准 ImageNet 预处理格式

主要函数：
    - render_parts_to_image: 单个几何渲染
    - batch_render_parts: 批量渲染

使用示例：
    >>> from preprocess.parts_render import render_parts_to_image
    >>> image = render_parts_to_image(parts, image_size=224)
    >>> teacher_feat = teacher(image.unsqueeze(0))

作者：GSD Team
版本：1.0
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from .geometry_types import GeomPart, PRIM_POINTSET, PRIM_LINE, PRIM_RING


class PartsRenderer:
    """
    确定性 Parts 渲染器
    
    将 GeomPart 列表渲染为图像
    """
    
    def __init__(
        self,
        image_size: int = 224,
        line_width: int = 2,
        point_radius: int = 3,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        ring_fill_color: Tuple[int, int, int] = (100, 100, 100),
        ring_edge_color: Tuple[int, int, int] = (0, 0, 0),
        line_color: Tuple[int, int, int] = (0, 0, 128),
        point_color: Tuple[int, int, int] = (128, 0, 0),
        padding: float = 0.1,
        normalize: bool = True
    ):
        """
        Args:
            image_size: 输出图像尺寸
            line_width: 线宽
            point_radius: 点半径
            background_color: 背景色
            ring_fill_color: 环填充色
            ring_edge_color: 环边缘色
            line_color: 线颜色
            point_color: 点颜色
            padding: 边界填充比例
            normalize: 是否应用 ImageNet 归一化
        """
        self.image_size = image_size
        self.line_width = line_width
        self.point_radius = point_radius
        self.background_color = background_color
        self.ring_fill_color = ring_fill_color
        self.ring_edge_color = ring_edge_color
        self.line_color = line_color
        self.point_color = point_color
        self.padding = padding
        self.normalize = normalize
        
        # ImageNet 归一化
        if normalize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.ToTensor()
    
    def render(
        self,
        parts: List[GeomPart],
        as_tensor: bool = True
    ) -> Union[Image.Image, torch.Tensor]:
        """
        渲染 parts 列表为图像
        
        Args:
            parts: GeomPart 列表
            as_tensor: 是否返回张量
            
        Returns:
            渲染后的图像（PIL Image 或 Tensor）
        """
        # 创建画布
        img = Image.new('RGB', (self.image_size, self.image_size), self.background_color)
        draw = ImageDraw.Draw(img)
        
        if not parts:
            if as_tensor:
                return self.transform(img)
            return img
        
        # 收集所有坐标计算全局边界
        all_coords = []
        for p in parts:
            if p.coords is not None and len(p.coords) > 0:
                all_coords.append(p.coords)
        
        if not all_coords:
            if as_tensor:
                return self.transform(img)
            return img
        
        # 计算坐标变换参数
        all_xy = np.vstack(all_coords)
        min_xy = all_xy.min(axis=0)
        max_xy = all_xy.max(axis=0)
        range_xy = max_xy - min_xy
        range_xy = np.maximum(range_xy, 1e-6)
        
        effective_size = self.image_size * (1 - 2 * self.padding)
        offset = self.image_size * self.padding
        
        max_range = max(range_xy)
        scale = effective_size / max_range
        center_offset = (effective_size - range_xy * scale) / 2
        
        # 坐标转换函数
        def to_pixel(coords: np.ndarray) -> List[Tuple[int, int]]:
            pixel = (coords - min_xy) * scale + offset + center_offset
            return [(int(x), int(y)) for x, y in pixel]
        
        # 按类型分组绘制（Ring 先绘制，保证被 Line/Point 覆盖）
        rings = [p for p in parts if p.prim_type == PRIM_RING]
        lines = [p for p in parts if p.prim_type == PRIM_LINE]
        points = [p for p in parts if p.prim_type == PRIM_POINTSET]
        
        # 1. 绘制 Ring（多边形）
        for p in rings:
            if p.coords is None or len(p.coords) < 3:
                continue
            pts = to_pixel(p.coords)
            draw.polygon(pts, fill=self.ring_fill_color, outline=self.ring_edge_color)
        
        # 2. 绘制 Line（折线）
        for p in lines:
            if p.coords is None or len(p.coords) < 2:
                continue
            pts = to_pixel(p.coords)
            draw.line(pts, fill=self.line_color, width=self.line_width)
        
        # 3. 绘制 PointSet（点）
        for p in points:
            if p.coords is None or len(p.coords) == 0:
                continue
            pts = to_pixel(p.coords)
            r = self.point_radius
            for px, py in pts:
                draw.ellipse([px-r, py-r, px+r, py+r], fill=self.point_color)
        
        if as_tensor:
            return self.transform(img)
        return img
    
    def render_batch(
        self,
        parts_batch: List[List[GeomPart]]
    ) -> torch.Tensor:
        """
        批量渲染
        
        Args:
            parts_batch: 批次 parts 列表
            
        Returns:
            图像张量 [B, 3, H, W]
        """
        images = [self.render(parts, as_tensor=True) for parts in parts_batch]
        return torch.stack(images)


# 全局渲染器实例（复用）
_default_renderer: Optional[PartsRenderer] = None


def get_default_renderer(image_size: int = 224) -> PartsRenderer:
    """获取默认渲染器（单例模式）"""
    global _default_renderer
    if _default_renderer is None or _default_renderer.image_size != image_size:
        _default_renderer = PartsRenderer(image_size=image_size)
    return _default_renderer


def render_parts_to_image(
    parts: List[GeomPart],
    image_size: int = 224,
    as_tensor: bool = True,
    normalize: bool = True
) -> Union[Image.Image, torch.Tensor]:
    """
    便捷函数：渲染 parts 为图像
    
    Args:
        parts: GeomPart 列表
        image_size: 图像尺寸
        as_tensor: 是否返回张量
        normalize: 是否 ImageNet 归一化
        
    Returns:
        渲染后的图像
    """
    renderer = PartsRenderer(image_size=image_size, normalize=normalize)
    return renderer.render(parts, as_tensor=as_tensor)


def batch_render_parts(
    parts_batch: List[List[GeomPart]],
    image_size: int = 224
) -> torch.Tensor:
    """
    便捷函数：批量渲染 parts
    
    Args:
        parts_batch: 批次 parts 列表
        image_size: 图像尺寸
        
    Returns:
        图像张量 [B, 3, H, W]
    """
    renderer = get_default_renderer(image_size)
    return renderer.render_batch(parts_batch)
