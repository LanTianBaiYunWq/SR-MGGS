"""
CRS 相关工具。
"""

from __future__ import annotations

from typing import Optional, Tuple

import geopandas as gpd


def ensure_working_projected_crs(
    gdf: gpd.GeoDataFrame,
    source_crs_if_missing: Optional[str] = None,
    working_crs: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """
    确保几何清洗在投影坐标系中进行。

    设计原则：
    - set_crs 只用于“原始数据缺少 CRS 但来源明确”的情况
    - 真正的坐标转换必须用 to_crs
    - 如果未指定工作坐标系，则优先使用 estimate_utm_crs()
    """
    if gdf.crs is None:
        if source_crs_if_missing is None:
            raise ValueError("输入数据缺少 CRS，且未提供 source_crs_if_missing。")
        gdf = gdf.set_crs(source_crs_if_missing)

    if working_crs is None:
        try:
            estimated = gdf.estimate_utm_crs()
            if estimated is not None:
                working_crs = str(estimated)
        except Exception:
            working_crs = None

    if working_crs is None:
        # 如果无法估计本地投影 CRS，则退回 Web Mercator。
        # 这不是最理想的工作坐标系，但至少比在 geographic CRS 中直接做长度/面积更稳。
        working_crs = "EPSG:3857"

    if str(gdf.crs) != working_crs:
        gdf = gdf.to_crs(working_crs)

    return gdf, working_crs

