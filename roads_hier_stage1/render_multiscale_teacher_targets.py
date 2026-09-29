"""
render_multiscale_teacher_targets

职责：
1. 将 tile-level chunks 渲染为教师目标图像
2. 将 file-level tile 汇总渲染为全局教师目标图像

说明：
- 这是 Gate 1 的独立步骤，不在训练期在线生成。
- 第一步默认可关闭，只在离线构建脚本显式传参时执行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


def _normalize_to_canvas(coords: np.ndarray, image_size: int) -> np.ndarray:
    """将二维坐标归一化到渲染画布。"""
    if len(coords) == 0:
        return coords

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    norm = (coords - min_xy) / span
    norm[:, 1] = 1.0 - norm[:, 1]
    return norm * float(image_size - 1)


def render_polyline_chunks(
    chunks: Sequence[np.ndarray],
    output_path: Path,
    *,
    image_size: int = 224,
    line_width: int = 2,
) -> None:
    """渲染单个 tile 或 file 的折线集合。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(canvas)

    for coords in chunks:
        if len(coords) < 2:
            continue
        pts = _normalize_to_canvas(np.asarray(coords, dtype=np.float32), image_size)
        draw.line([tuple(point.tolist()) for point in pts], fill=0, width=line_width)

    canvas.save(output_path)

