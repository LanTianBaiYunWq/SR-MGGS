"""
原始矢量文件清单检查。
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Dict, List, Optional

import geopandas as gpd


def _geometry_type_name(geom) -> str:
    """返回稳定的几何类型名称。"""
    if geom is None:
        return "None"
    if geom.is_empty:
        return "Empty"
    return geom.geom_type


def _count_coordinates(geom) -> int:
    """递归统计几何对象包含的坐标数量。"""
    if geom is None or geom.is_empty:
        return 0

    geom_type = geom.geom_type
    if geom_type == "Point":
        return 1
    if geom_type in {"LineString", "LinearRing"}:
        return len(geom.coords)
    if geom_type == "Polygon":
        total = len(geom.exterior.coords)
        for ring in geom.interiors:
            total += len(ring.coords)
        return total
    if hasattr(geom, "geoms"):
        return sum(_count_coordinates(part) for part in geom.geoms)
    return 0


def _safe_has_z(geom) -> bool:
    """安全读取 Z 维标记。"""
    try:
        return bool(getattr(geom, "has_z", False))
    except Exception:
        return False


def _safe_has_m(geom) -> bool:
    """安全读取 M 维标记。"""
    try:
        return bool(getattr(geom, "has_m", False))
    except Exception:
        return False


def _quantile(values: List[int], q: float) -> Optional[float]:
    """计算简单分位数，避免引入额外依赖。"""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def inspect_vector_file(file_path: str, layer: Optional[str] = None) -> Dict:
    """
    检查单个矢量文件或其中指定图层的原始数据统计。

    这个函数只做“看清数据”，不做任何预处理改写。
    """
    read_kwargs = {}
    if layer is not None:
        read_kwargs["layer"] = layer

    gdf = gpd.read_file(file_path, **read_kwargs)
    total_features = int(len(gdf))

    if "geometry" not in gdf.columns:
        return {
            "file_path": file_path,
            "layer": layer,
            "is_spatial": False,
            "feature_count": total_features,
        }

    type_counts: Counter = Counter()
    empty_or_null_count = 0
    invalid_count = 0
    multipart_count = 0
    geometry_collection_count = 0
    has_z_count = 0
    has_m_count = 0
    coord_counts: List[int] = []

    for geom in gdf.geometry:
        geom_type = _geometry_type_name(geom)
        type_counts[geom_type] += 1

        if geom is None or geom.is_empty:
            empty_or_null_count += 1
            continue

        if not geom.is_valid:
            invalid_count += 1
        if geom_type.startswith("Multi"):
            multipart_count += 1
        if geom_type == "GeometryCollection":
            geometry_collection_count += 1
        if _safe_has_z(geom):
            has_z_count += 1
        if _safe_has_m(geom):
            has_m_count += 1

        coord_counts.append(_count_coordinates(geom))

    non_empty = max(total_features - empty_or_null_count, 1)
    return {
        "file_path": file_path,
        "layer": layer,
        "is_spatial": True,
        "feature_count": total_features,
        "empty_or_null_count": empty_or_null_count,
        "raw_type_counts": dict(type_counts),
        "invalid_ratio": round(invalid_count / non_empty, 6),
        "multipart_ratio": round(multipart_count / non_empty, 6),
        "geometry_collection_ratio": round(geometry_collection_count / non_empty, 6),
        "has_z_ratio": round(has_z_count / non_empty, 6),
        "has_m_ratio": round(has_m_count / non_empty, 6),
        "coordinate_count_distribution": {
            "min": float(min(coord_counts)) if coord_counts else None,
            "median": float(median(coord_counts)) if coord_counts else None,
            "p95": _quantile(coord_counts, 0.95),
            "max": float(max(coord_counts)) if coord_counts else None,
        },
    }

