"""
GSD 数据预处理模块
"""

from .sgp import (
    GeometryNormalizer,
    GeometryAugmentor,
    ShapefileDataset,
    ContrastiveDataset,
    create_dataloader,
    apply_perturbation,
    # 全局签名相关
    AdaptiveGeometryNormalizer,
    ShapefileGlobalDataset,
    GlobalContrastiveDataset,
    create_global_dataloader
)
from .render import (
    GeometryRenderer,
    MultiViewRenderer,
    CachedRenderer,
    coords_to_image,
    batch_coords_to_images,
    SpatialAlignedRenderer,
    MultiAngleRenderer,
    render_with_spatial_info,
    batch_render_with_spatial_info
)

# 新的通用几何类型支持
from .geometry_types import (
    GeomPart,
    GeometryData,
    PRIM_POINTSET,
    PRIM_LINE,
    PRIM_RING,
    ROLE_NONE,
    ROLE_EXTERIOR,
    ROLE_INTERIOR,
    CONTAINER_POINT,
    CONTAINER_MULTIPOINT,
    CONTAINER_LINESTRING,
    CONTAINER_MULTILINESTRING,
    CONTAINER_POLYGON,
    CONTAINER_MULTIPOLYGON,
    CONTAINER_GEOMETRYCOLLECTION
)
from .geometry_extract import (
    extract_parts,
    get_container_type,
    geometry_to_data
)
from .geometry_normalize import (
    normalize_parts,
    canonicalize_ring,
    canonicalize_line,
    canonicalize_parts,
    normalize_geometry_data
)
from .geometry_resample import (
    resample_polyline,
    subsample_pointset,
    cap_and_resample,
    process_geometry_data
)
from .geometry_dataset import (
    GeometryAugmentation,
    UniversalGeometryDataset,
    collate_parts,
    create_dataloader as create_universal_dataloader,
    # 新增：双视图训练支持
    augment_parts,
    TwoViewGeometryDataset,
    collate_two_views,
    create_two_view_dataloader
)
from .parts_render import (
    PartsRenderer,
    render_parts_to_image,
    batch_render_parts
)

__all__ = [
    # 旧的 API（保持兼容）
    'GeometryNormalizer',
    'GeometryAugmentor',
    'ShapefileDataset',
    'ContrastiveDataset',
    'create_dataloader',
    'apply_perturbation',
    'AdaptiveGeometryNormalizer',
    'ShapefileGlobalDataset',
    'GlobalContrastiveDataset',
    'create_global_dataloader',
    'GeometryRenderer',
    'MultiViewRenderer',
    'CachedRenderer',
    'coords_to_image',
    'batch_coords_to_images',
    'SpatialAlignedRenderer',
    'MultiAngleRenderer',
    'render_with_spatial_info',
    'batch_render_with_spatial_info',
    
    # 新的通用几何类型 API
    'GeomPart',
    'GeometryData',
    'PRIM_POINTSET',
    'PRIM_LINE',
    'PRIM_RING',
    'ROLE_NONE',
    'ROLE_EXTERIOR',
    'ROLE_INTERIOR',
    'CONTAINER_POINT',
    'CONTAINER_MULTIPOINT',
    'CONTAINER_LINESTRING',
    'CONTAINER_MULTILINESTRING',
    'CONTAINER_POLYGON',
    'CONTAINER_MULTIPOLYGON',
    'CONTAINER_GEOMETRYCOLLECTION',
    'extract_parts',
    'get_container_type',
    'geometry_to_data',
    'normalize_parts',
    'canonicalize_ring',
    'canonicalize_line',
    'canonicalize_parts',
    'normalize_geometry_data',
    'resample_polyline',
    'subsample_pointset',
    'cap_and_resample',
    'process_geometry_data',
    'GeometryAugmentation',
    'UniversalGeometryDataset',
    'collate_parts',
    'create_universal_dataloader',
    # 新增：双视图训练支持
    'augment_parts',
    'TwoViewGeometryDataset',
    'collate_two_views',
    'create_two_view_dataloader',
    # 新增：Parts 渲染器
    'PartsRenderer',
    'render_parts_to_image',
    'batch_render_parts'
]

