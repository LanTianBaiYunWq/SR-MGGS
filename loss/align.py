"""
几何-视觉对齐损失
===============================

本模块实现训练期几何特征与视觉特征的对齐损失。

设计要点：
    1. 使用余弦距离（更稳定，不受尺度影响）
    2. 所有向量先 L2 归一化
    3. 支持双视图对齐（E_q 和 E_k 都与 V 对齐）

损失公式：
    L_kd = (1 - cos(E_q, V)) + (1 - cos(E_k, V))
         = 2 - cos(E_q, V) - cos(E_k, V)

其中：
    - E_q: 几何编码器对 view A 的输出
    - E_k: 几何编码器对 view B 的输出
    - V: 投影后的视觉教师特征

使用示例：
    >>> loss_fn = GeometryVisualAlignLoss()
    >>> loss = loss_fn(E_q, E_k, V)

作者：GSD Team
版本：1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryVisualAlignLoss(nn.Module):
    """
    几何-视觉对齐损失
    
    使用余弦相似度让几何特征与视觉特征对齐
    """
    
    def __init__(
        self,
        normalize: bool = True,
        reduction: str = "mean"
    ):
        """
        Args:
            normalize: 是否 L2 归一化（推荐 True）
            reduction: 归约方式 ("mean", "sum", "none")
        """
        super().__init__()
        self.normalize = normalize
        self.reduction = reduction
    
    def forward(
        self,
        E_q: torch.Tensor,
        E_k: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对齐损失
        
        Args:
            E_q: 几何特征 view A [B, d]
            E_k: 几何特征 view B [B, d]
            V: 视觉特征（已投影到学生空间）[B, d]
            
        Returns:
            对齐损失
        """
        # L2 归一化
        if self.normalize:
            E_q = F.normalize(E_q, dim=-1)
            E_k = F.normalize(E_k, dim=-1)
            V = F.normalize(V, dim=-1)
        
        # 余弦相似度
        cos_q = (E_q * V).sum(dim=-1)  # [B]
        cos_k = (E_k * V).sum(dim=-1)  # [B]
        
        # 损失 = 1 - cos（余弦距离）
        loss_q = 1 - cos_q
        loss_k = 1 - cos_k
        
        # 总损失
        loss = loss_q + loss_k
        
        # 归约
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SingleViewAlignLoss(nn.Module):
    """
    单视图对齐损失
    
    用于验证或单视图场景
    """
    
    def __init__(
        self,
        normalize: bool = True,
        reduction: str = "mean"
    ):
        super().__init__()
        self.normalize = normalize
        self.reduction = reduction
    
    def forward(
        self,
        E: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        计算单视图对齐损失
        
        Args:
            E: 几何特征 [B, d]
            V: 视觉特征 [B, d]
            
        Returns:
            对齐损失
        """
        if self.normalize:
            E = F.normalize(E, dim=-1)
            V = F.normalize(V, dim=-1)
        
        cos_sim = (E * V).sum(dim=-1)
        loss = 1 - cos_sim
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SmoothAlignLoss(nn.Module):
    """
    平滑对齐损失
    
    使用 SmoothL1 距离而非余弦距离
    对离群值更鲁棒
    """
    
    def __init__(
        self,
        normalize: bool = True,
        beta: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Args:
            normalize: 是否 L2 归一化
            beta: SmoothL1 的 beta 参数
            reduction: 归约方式
        """
        super().__init__()
        self.normalize = normalize
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        E_q: torch.Tensor,
        E_k: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        计算平滑对齐损失
        """
        if self.normalize:
            E_q = F.normalize(E_q, dim=-1)
            E_k = F.normalize(E_k, dim=-1)
            V = F.normalize(V, dim=-1)
        
        # SmoothL1 距离
        loss_q = F.smooth_l1_loss(E_q, V, beta=self.beta, reduction=self.reduction)
        loss_k = F.smooth_l1_loss(E_k, V, beta=self.beta, reduction=self.reduction)
        
        return loss_q + loss_k


class MSEAlignLoss(nn.Module):
    """
    MSE 对齐损失
    
    简单的均方误差对齐
    """
    
    def __init__(
        self,
        normalize: bool = True,
        reduction: str = "mean"
    ):
        super().__init__()
        self.normalize = normalize
        self.reduction = reduction
    
    def forward(
        self,
        E_q: torch.Tensor,
        E_k: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 MSE 对齐损失
        """
        if self.normalize:
            E_q = F.normalize(E_q, dim=-1)
            E_k = F.normalize(E_k, dim=-1)
            V = F.normalize(V, dim=-1)
        
        loss_q = F.mse_loss(E_q, V, reduction=self.reduction)
        loss_k = F.mse_loss(E_k, V, reduction=self.reduction)
        
        return loss_q + loss_k


def create_align_loss(config: dict) -> nn.Module:
    """
    根据配置创建对齐损失
    
    Args:
        config: 配置字典
        
    Returns:
        对齐损失模块
    """
    loss_type = config.get('type', 'cosine')
    
    if loss_type == 'cosine':
        return GeometryVisualAlignLoss(
            normalize=config.get('normalize', True),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'smooth':
        return SmoothAlignLoss(
            normalize=config.get('normalize', True),
            beta=config.get('beta', 1.0),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'mse':
        return MSEAlignLoss(
            normalize=config.get('normalize', True),
            reduction=config.get('reduction', 'mean')
        )
    else:
        raise ValueError(f"Unknown align loss type: {loss_type}")
