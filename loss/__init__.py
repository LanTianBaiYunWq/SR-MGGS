"""
GSD 损失函数模块
"""

from .distill_loss import (
    DistillationLoss,
    FeatureMatchingLoss,
    RelationalDistillationLoss,
    CombinedDistillationLoss,
    create_distill_loss
)
from .contrastive import (
    InfoNCELoss,
    SymmetricInfoNCELoss,
    NTXentLoss,
    SupervisedContrastiveLoss,
    MultiViewContrastiveLoss,
    GSDContrastiveLoss,
    SpatialWeightedContrastiveLoss,
    MultiViewSpatialContrastiveLoss,
    create_contrastive_loss
)
from .align import (
    GeometryVisualAlignLoss,
    SingleViewAlignLoss,
    SmoothAlignLoss,
    MSEAlignLoss,
    create_align_loss
)
from .bit_margin import (
    BitMarginLoss,
    AdaptiveMarginLoss,
    create_bit_margin_loss
)

__all__ = [
    'DistillationLoss',
    'FeatureMatchingLoss',
    'RelationalDistillationLoss',
    'CombinedDistillationLoss',
    'create_distill_loss',
    'InfoNCELoss',
    'SymmetricInfoNCELoss',
    'NTXentLoss',
    'SupervisedContrastiveLoss',
    'MultiViewContrastiveLoss',
    'GSDContrastiveLoss',
    'SpatialWeightedContrastiveLoss',
    'MultiViewSpatialContrastiveLoss',
    'create_contrastive_loss',
    # 新增：几何-视觉对齐
    'GeometryVisualAlignLoss',
    'SingleViewAlignLoss',
    'SmoothAlignLoss',
    'MSEAlignLoss',
    'create_align_loss',
    # 新增：符号边界
    'BitMarginLoss',
    'AdaptiveMarginLoss',
    'create_bit_margin_loss'
]

