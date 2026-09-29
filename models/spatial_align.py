"""
空间对齐模块
确保几何特征与视觉特征在空间上保持一致
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialProjector(nn.Module):
    """
    空间感知投影器
    在投影之前进行空间对齐处理
    """
    
    def __init__(
        self,
        student_dim: int = 512,
        teacher_dim: int = 2048,
        hidden_dim: Optional[int] = None,
        use_spatial_attention: bool = True
    ):
        """
        Args:
            student_dim: 学生特征维度
            teacher_dim: 教师特征维度
            hidden_dim: 隐藏层维度
            use_spatial_attention: 是否使用空间注意力
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = (student_dim + teacher_dim) // 2
        
        self.use_spatial_attention = use_spatial_attention
        
        # 空间特征增强
        if use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Linear(student_dim, student_dim // 4),
                nn.ReLU(),
                nn.Linear(student_dim // 4, student_dim),
                nn.Sigmoid()
            )
        
        # 主投影器
        self.projector = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        student_feat: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_feat: 学生特征 [B, D]
            normalize: 是否L2归一化
            
        Returns:
            投影后特征 [B, teacher_dim]
        """
        # 空间注意力增强
        if self.use_spatial_attention:
            attention = self.spatial_attention(student_feat)
            student_feat = student_feat * attention
        
        # 投影
        projected = self.projector(student_feat)
        
        if normalize:
            projected = F.normalize(projected, dim=-1)
        
        return projected


class SpatialSemanticAlignment(nn.Module):
    """
    空间语义对齐模块
    增强几何特征与视觉特征之间的空间对应关系
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_spatial_tokens: int = 16
    ):
        """
        Args:
            feature_dim: 特征维度
            num_spatial_tokens: 空间token数量
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_spatial_tokens = num_spatial_tokens
        
        # 空间token（用于位置敏感的特征提取）
        self.spatial_tokens = nn.Parameter(
            torch.randn(1, num_spatial_tokens, feature_dim) * 0.02
        )
        
        # 空间交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 空间位置编码
        self.position_encoding = self._create_position_encoding(num_spatial_tokens)
        
        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)
    
    def _create_position_encoding(self, num_positions: int) -> nn.Parameter:
        """创建2D位置编码"""
        # 假设 num_positions 是完全平方数
        grid_size = int(math.sqrt(num_positions))
        
        pe = torch.zeros(num_positions, self.feature_dim)
        
        for i in range(num_positions):
            row = i // grid_size
            col = i % grid_size
            
            for j in range(0, self.feature_dim, 4):
                div_term = 10000 ** (j / self.feature_dim)
                pe[i, j] = math.sin(row / div_term)
                pe[i, j + 1] = math.cos(row / div_term)
                pe[i, j + 2] = math.sin(col / div_term)
                pe[i, j + 3] = math.cos(col / div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(
        self,
        geometry_features: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            geometry_features: 几何特征 [B, D] 或 [B, N, D]
            visual_features: 视觉特征 [B, D]（可选）
            
        Returns:
            空间对齐后的特征
        """
        batch_size = geometry_features.size(0)
        
        # 确保是3D
        if geometry_features.dim() == 2:
            geometry_features = geometry_features.unsqueeze(1)
        
        # 扩展空间token
        spatial_tokens = self.spatial_tokens.expand(batch_size, -1, -1)
        spatial_tokens = spatial_tokens + self.position_encoding
        
        # 交叉注意力：空间token查询几何特征
        aligned_features, _ = self.cross_attention(
            query=spatial_tokens,
            key=geometry_features,
            value=geometry_features
        )
        
        # 聚合（平均池化）
        output = aligned_features.mean(dim=1)
        output = self.output_proj(output)
        
        return output


class GeometryVisualAligner(nn.Module):
    """
    几何-视觉特征对齐器
    联合优化几何特征和视觉特征的对齐
    """
    
    def __init__(
        self,
        geometry_dim: int = 512,
        visual_dim: int = 2048,
        hidden_dim: int = 512,
        num_heads: int = 8
    ):
        """
        Args:
            geometry_dim: 几何特征维度
            visual_dim: 视觉特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()
        
        # 几何特征投影到共享空间
        self.geometry_proj = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 视觉特征投影到共享空间
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 交叉注意力对齐
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 对齐后投影回各自空间
        self.geometry_output = nn.Linear(hidden_dim, geometry_dim)
        self.visual_output = nn.Linear(hidden_dim, visual_dim)
    
    def forward(
        self,
        geometry_feat: torch.Tensor,
        visual_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            geometry_feat: 几何特征 [B, geometry_dim]
            visual_feat: 视觉特征 [B, visual_dim]
            
        Returns:
            (对齐的几何特征, 对齐的视觉特征)
        """
        # 投影到共享空间
        geo_hidden = self.geometry_proj(geometry_feat).unsqueeze(1)
        vis_hidden = self.visual_proj(visual_feat).unsqueeze(1)
        
        # 双向交叉注意力
        geo_aligned, _ = self.cross_attention(
            query=geo_hidden,
            key=vis_hidden,
            value=vis_hidden
        )
        
        vis_aligned, _ = self.cross_attention(
            query=vis_hidden,
            key=geo_hidden,
            value=geo_hidden
        )
        
        # 残差连接并投影回原空间
        geo_out = self.geometry_output(geo_aligned.squeeze(1)) + geometry_feat
        vis_out = self.visual_output(vis_aligned.squeeze(1)) + visual_feat
        
        return geo_out, vis_out


class SpatialWeightedLoss(nn.Module):
    """
    空间加权损失
    根据空间重要性对损失进行加权
    """
    
    def __init__(
        self,
        base_loss: str = "mse",
        use_importance_weighting: bool = True
    ):
        """
        Args:
            base_loss: 基础损失类型
            use_importance_weighting: 是否使用重要性加权
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.use_importance_weighting = use_importance_weighting
    
    def compute_importance_weights(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算特征重要性权重
        
        Args:
            features: 特征 [B, D]
            
        Returns:
            权重 [B]
        """
        # 基于特征范数计算重要性
        norms = torch.norm(features, dim=-1)
        
        # Softmax归一化
        weights = F.softmax(norms, dim=0)
        
        return weights
    
    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算空间加权损失
        
        Args:
            student_feat: 学生特征 [B, D]
            teacher_feat: 教师特征 [B, D]
            spatial_weights: 空间权重 [B]（可选）
            
        Returns:
            损失值
        """
        # 归一化
        student_feat = F.normalize(student_feat, dim=-1)
        teacher_feat = F.normalize(teacher_feat, dim=-1)
        
        # 计算基础损失
        if self.base_loss == "mse":
            element_loss = F.mse_loss(student_feat, teacher_feat, reduction='none')
            element_loss = element_loss.mean(dim=-1)  # [B]
        elif self.base_loss == "cosine":
            cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
            element_loss = 1 - cos_sim
        else:
            raise ValueError(f"Unknown loss type: {self.base_loss}")
        
        # 应用空间权重
        if spatial_weights is not None:
            loss = (element_loss * spatial_weights).sum() / (spatial_weights.sum() + 1e-8)
        elif self.use_importance_weighting:
            weights = self.compute_importance_weights(teacher_feat)
            loss = (element_loss * weights).sum()
        else:
            loss = element_loss.mean()
        
        return loss


def create_spatial_projector(
    student_dim: int = 512,
    teacher_dim: int = 2048,
    use_spatial_attention: bool = True
) -> nn.Module:
    """
    创建空间感知投影器
    """
    return SpatialProjector(
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        use_spatial_attention=use_spatial_attention
    )

