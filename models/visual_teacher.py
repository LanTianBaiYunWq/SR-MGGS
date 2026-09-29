"""
视觉教师模型
基于预训练ResNet的图像特征提取器
仅在训练阶段使用，推理时不需要
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class VisualTeacher(nn.Module):
    """
    视觉教师模型
    使用预训练的ResNet50提取图像特征
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        frozen: bool = True,
        output_dim: int = 2048
    ):
        """
        Args:
            backbone: 骨干网络名称
            pretrained: 是否使用预训练权重
            frozen: 是否冻结参数
            output_dim: 输出维度
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.frozen = frozen
        self.output_dim = output_dim
        
        # 加载预训练模型
        self.backbone = self._load_backbone(backbone, pretrained)
        
        # 移除分类头
        if hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            in_features = output_dim
        
        # 如果输出维度不匹配，添加投影层
        if in_features != output_dim:
            self.projection = nn.Linear(in_features, output_dim)
        else:
            self.projection = nn.Identity()
        
        # 冻结参数
        if frozen:
            self._freeze_backbone()
    
    def _load_backbone(self, name: str, pretrained: bool) -> nn.Module:
        """加载骨干网络"""
        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnet50(weights=weights)
        elif name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnet101(weights=weights)
        elif name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet18(weights=weights)
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {name}")
    
    def _freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def train(self, mode: bool = True):
        """
        重写train方法，保持backbone冻结
        """
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 图像张量 [batch_size, 3, H, W]
            
        Returns:
            图像特征 [batch_size, output_dim]
        """
        with torch.no_grad() if self.frozen else torch.enable_grad():
            features = self.backbone(images)
        
        # 投影
        features = self.projection(features)
        
        return features
    
    def get_intermediate_features(
        self,
        images: torch.Tensor,
        layer_names: Optional[list] = None
    ) -> dict:
        """
        获取中间层特征（用于多尺度蒸馏）
        
        Args:
            images: 图像张量
            layer_names: 要提取的层名称
            
        Returns:
            特征字典
        """
        if layer_names is None:
            layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        features = {}
        x = images
        
        # ResNet forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        if 'layer1' in layer_names:
            features['layer1'] = x
        
        x = self.backbone.layer2(x)
        if 'layer2' in layer_names:
            features['layer2'] = x
        
        x = self.backbone.layer3(x)
        if 'layer3' in layer_names:
            features['layer3'] = x
        
        x = self.backbone.layer4(x)
        if 'layer4' in layer_names:
            features['layer4'] = x
        
        return features


class MultiScaleVisualTeacher(nn.Module):
    """
    多尺度视觉教师
    提取多尺度特征用于更丰富的蒸馏
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        frozen: bool = True,
        output_dim: int = 2048
    ):
        super().__init__()
        
        self.teacher = VisualTeacher(backbone, pretrained, frozen, output_dim)
        
        # 多尺度特征投影
        # ResNet各层特征维度
        if backbone in ["resnet50", "resnet101"]:
            layer_dims = {
                'layer1': 256,
                'layer2': 512,
                'layer3': 1024,
                'layer4': 2048
            }
        else:
            layer_dims = {
                'layer1': 64,
                'layer2': 128,
                'layer3': 256,
                'layer4': 512
            }
        
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, output_dim)
            )
            for name, dim in layer_dims.items()
        })
    
    def forward(
        self,
        images: torch.Tensor,
        return_multiscale: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        前向传播
        
        Args:
            images: 图像张量
            return_multiscale: 是否返回多尺度特征
            
        Returns:
            (最终特征, 多尺度特征字典)
        """
        with torch.no_grad():
            intermediate = self.teacher.get_intermediate_features(images)
            final_feat = self.teacher(images)
        
        if return_multiscale:
            multiscale_feats = {}
            for name, feat in intermediate.items():
                multiscale_feats[name] = self.projections[name](feat)
            return final_feat, multiscale_feats
        
        return final_feat, None


def create_visual_teacher(config: dict) -> nn.Module:
    """
    根据配置创建视觉教师
    
    Args:
        config: 配置字典
        
    Returns:
        视觉教师模型
    """
    return VisualTeacher(
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True),
        frozen=config.get('frozen', True),
        output_dim=config.get('output_dim', 2048)
    )

