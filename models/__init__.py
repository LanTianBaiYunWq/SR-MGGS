"""
GSD 模型模块
"""

from .geometry_encoder import (
    GeometryEncoder,
    GeometryEncoderWithStats,
    create_geometry_encoder
)
from .visual_teacher import (
    VisualTeacher,
    MultiScaleVisualTeacher,
    create_visual_teacher
)
from .projector import (
    MLPProjector,
    DistillProjector,
    SimCLRProjector,
    GSDModel,
    create_projector
)
from .aggregator import (
    AttentionPooling,
    SetTransformerAggregator,
    GlobalGeometryEncoder,
    create_aggregator
)
from .spatial_align import (
    SpatialProjector,
    SpatialSemanticAlignment,
    GeometryVisualAligner,
    SpatialWeightedLoss,
    create_spatial_projector
)

# 新的通用几何编码器
try:
    from .universal_encoder import (
        PartEncoder,
        GeomAggregator,
        UniversalGeometryEncoder,
        create_universal_encoder
    )
except ModuleNotFoundError:
    PartEncoder = None
    GeomAggregator = None
    UniversalGeometryEncoder = None
    create_universal_encoder = None

# 教师投影器
from .teacher_projector import (
    TeacherProjector,
    BidirectionalProjector,
    create_teacher_projector
)

__all__ = [
    # 旧的 API（保持兼容）
    'GeometryEncoder',
    'GeometryEncoderWithStats',
    'create_geometry_encoder',
    'VisualTeacher',
    'MultiScaleVisualTeacher',
    'create_visual_teacher',
    'MLPProjector',
    'DistillProjector',
    'SimCLRProjector',
    'GSDModel',
    'create_projector',
    'AttentionPooling',
    'SetTransformerAggregator',
    'GlobalGeometryEncoder',
    'create_aggregator',
    'SpatialProjector',
    'SpatialSemanticAlignment',
    'GeometryVisualAligner',
    'SpatialWeightedLoss',
    'create_spatial_projector',
    
    # 新的通用几何编码器
    'PartEncoder',
    'GeomAggregator',
    'UniversalGeometryEncoder',
    'create_universal_encoder',
    # 教师投影器
    'TeacherProjector',
    'BidirectionalProjector',
    'create_teacher_projector'
]

