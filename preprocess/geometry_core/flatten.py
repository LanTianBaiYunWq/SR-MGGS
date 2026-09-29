"""
将复杂几何展开为 point / line / polygon 三大家族的最小原语。
"""

from __future__ import annotations

from typing import List

import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from .types import FlattenedPrimitive


def flatten_geometry(geometry, parent_id: int) -> List[FlattenedPrimitive]:
    """
    将任意 shapely 几何展开为可追踪的原语列表。

    关键设计：
    - Multi* 不再直接失败，而是逐子几何展开
    - GeometryCollection 递归展开
    - Polygon 会拆成 exterior ring 与 interior holes
    - 每个原语保留 parent_id / part_id，方便之后文件级聚合
    """
    flattened: List[FlattenedPrimitive] = []

    def _append(family: str, geom_type: str, coords: np.ndarray, role: str, is_hole: bool):
        flattened.append(
            FlattenedPrimitive(
                parent_id=parent_id,
                part_id=len(flattened),
                family=family,
                geom_type=geom_type,
                coords=coords.astype(np.float32),
                role=role,
                is_hole=is_hole,
            )
        )

    def _walk(geom):
        if geom is None or geom.is_empty:
            return

        if isinstance(geom, Point):
            _append(
                family="point",
                geom_type="Point",
                coords=np.array([[geom.x, geom.y]], dtype=np.float32),
                role="main",
                is_hole=False,
            )
            return

        if isinstance(geom, MultiPoint):
            coords = np.array([[p.x, p.y] for p in geom.geoms], dtype=np.float32)
            _append(
                family="point",
                geom_type="MultiPoint",
                coords=coords,
                role="main",
                is_hole=False,
            )
            return

        if isinstance(geom, LineString):
            _append(
                family="line",
                geom_type="LineString",
                coords=np.array(geom.coords, dtype=np.float32),
                role="main",
                is_hole=False,
            )
            return

        if isinstance(geom, MultiLineString):
            for part in geom.geoms:
                _walk(part)
            return

        if isinstance(geom, Polygon):
            exterior = np.array(geom.exterior.coords, dtype=np.float32)
            if len(exterior) > 1:
                _append(
                    family="polygon",
                    geom_type="Polygon",
                    coords=exterior[:-1],
                    role="shell",
                    is_hole=False,
                )
            for ring in geom.interiors:
                hole = np.array(ring.coords, dtype=np.float32)
                if len(hole) > 1:
                    _append(
                        family="polygon",
                        geom_type="Polygon",
                        coords=hole[:-1],
                        role="hole",
                        is_hole=True,
                    )
            return

        if isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                _walk(part)
            return

        if isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                _walk(part)
            return

    _walk(geometry)
    return flattened

