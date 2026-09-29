"""
GeometryCore 最小实现。

这个子模块的目标不是一次性替换全部旧流程，而是先提供一套
“可检查、可展开、可按几何家族分别 token 化”的基础前端。

当前最小版本包含：
1. 文件与图层清单检查
2. CRS 工作坐标系选择与投影
3. Multi* / GeometryCollection 展开
4. Point / Line / Polygon 三大家族的最小 token 化
"""

from .types import FlattenedPrimitive, PrimitiveTokens
from .inspect import inspect_vector_file
from .crs import ensure_working_projected_crs
from .flatten import flatten_geometry
from .tokenize_point import tokenize_point_family
from .tokenize_line import tokenize_line_family
from .tokenize_polygon import tokenize_polygon_family

__all__ = [
    "FlattenedPrimitive",
    "PrimitiveTokens",
    "inspect_vector_file",
    "ensure_working_projected_crs",
    "flatten_geometry",
    "tokenize_point_family",
    "tokenize_line_family",
    "tokenize_polygon_family",
]
