"""
蒸馏损失函数
用于知识蒸馏训练
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    基础蒸馏损失
    L_KD = || W * g' - f ||^2
    """
    
    def __init__(
        self,
        normalize: bool = True,
        loss_type: str = "mse"
    ):
        """
        Args:
            normalize: 是否L2归一化特征
            loss_type: 损失类型 ("mse", "smooth_l1", "cosine")
        """
        super().__init__()
        self.normalize = normalize
        self.loss_type = loss_type
    
    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        计算蒸馏损失
        
        Args:
            student_feat: 学生特征 [batch_size, dim]
            teacher_feat: 教师特征 [batch_size, dim]
            
        Returns:
            蒸馏损失
        """
        if self.normalize:
            student_feat = F.normalize(student_feat, dim=-1)
            teacher_feat = F.normalize(teacher_feat, dim=-1)
        
        if self.loss_type == "mse":
            loss = F.mse_loss(student_feat, teacher_feat)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(student_feat, teacher_feat)
        elif self.loss_type == "cosine":
            # 余弦相似度损失
            cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
            loss = (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class FeatureMatchingLoss(nn.Module):
    """
    特征匹配损失
    支持多尺度特征匹配
    """
    
    def __init__(
        self,
        normalize: bool = True,
        weights: Optional[dict] = None
    ):
        """
        Args:
            normalize: 是否归一化
            weights: 各尺度权重字典
        """
        super().__init__()
        self.normalize = normalize
        self.weights = weights or {}
    
    def forward(
        self,
        student_feats: dict,
        teacher_feats: dict
    ) -> torch.Tensor:
        """
        计算多尺度特征匹配损失
        
        Args:
            student_feats: 学生特征字典
            teacher_feats: 教师特征字典
            
        Returns:
            总损失
        """
        total_loss = 0.0
        
        for key in teacher_feats:
            if key not in student_feats:
                continue
            
            s_feat = student_feats[key]
            t_feat = teacher_feats[key]
            
            if self.normalize:
                s_feat = F.normalize(s_feat, dim=-1)
                t_feat = F.normalize(t_feat, dim=-1)
            
            weight = self.weights.get(key, 1.0)
            loss = F.mse_loss(s_feat, t_feat) * weight
            total_loss = total_loss + loss
        
        return total_loss


class SoftTargetLoss(nn.Module):
    """
    软标签蒸馏损失
    使用KL散度匹配软化的输出分布
    """
    
    def __init__(
        self,
        temperature: float = 4.0
    ):
        """
        Args:
            temperature: 温度参数，控制软化程度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算软标签损失
        
        Args:
            student_logits: 学生logits
            teacher_logits: 教师logits
            
        Returns:
            KL散度损失
        """
        # 软化
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL散度
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # 温度平方缩放
        loss = loss * (self.temperature ** 2)
        
        return loss


class RelationalDistillationLoss(nn.Module):
    """
    关系蒸馏损失
    保持样本间的关系结构
    """
    
    def __init__(
        self,
        distance_type: str = "l2"
    ):
        """
        Args:
            distance_type: 距离类型 ("l2", "cosine")
        """
        super().__init__()
        self.distance_type = distance_type
    
    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        计算关系蒸馏损失
        
        Args:
            student_feat: 学生特征 [batch_size, dim]
            teacher_feat: 教师特征 [batch_size, dim]
            
        Returns:
            关系损失
        """
        # 计算样本间距离矩阵
        student_dist = self._compute_distance_matrix(student_feat)
        teacher_dist = self._compute_distance_matrix(teacher_feat)
        
        # 归一化距离矩阵
        student_dist = student_dist / (student_dist.mean() + 1e-8)
        teacher_dist = teacher_dist / (teacher_dist.mean() + 1e-8)
        
        # 匹配距离矩阵
        loss = F.mse_loss(student_dist, teacher_dist)
        
        return loss
    
    def _compute_distance_matrix(self, feat: torch.Tensor) -> torch.Tensor:
        """计算距离矩阵"""
        if self.distance_type == "l2":
            # L2距离
            diff = feat.unsqueeze(1) - feat.unsqueeze(0)
            dist = torch.norm(diff, dim=-1)
        elif self.distance_type == "cosine":
            # 余弦距离
            feat_norm = F.normalize(feat, dim=-1)
            dist = 1 - torch.mm(feat_norm, feat_norm.t())
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return dist


class CombinedDistillationLoss(nn.Module):
    """
    组合蒸馏损失
    结合多种蒸馏策略
    """
    
    def __init__(
        self,
        feature_weight: float = 1.0,
        relation_weight: float = 0.5,
        normalize: bool = True
    ):
        """
        Args:
            feature_weight: 特征匹配权重
            relation_weight: 关系蒸馏权重
            normalize: 是否归一化
        """
        super().__init__()
        
        self.feature_loss = DistillationLoss(normalize=normalize)
        self.relation_loss = RelationalDistillationLoss()
        
        self.feature_weight = feature_weight
        self.relation_weight = relation_weight
    
    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> dict:
        """
        计算组合损失
        
        Args:
            student_feat: 学生特征
            teacher_feat: 教师特征
            
        Returns:
            损失字典
        """
        # 特征匹配损失
        feat_loss = self.feature_loss(student_feat, teacher_feat)
        
        # 关系蒸馏损失
        rel_loss = self.relation_loss(student_feat, teacher_feat)
        
        # 总损失
        total_loss = (
            self.feature_weight * feat_loss +
            self.relation_weight * rel_loss
        )
        
        return {
            'total': total_loss,
            'feature': feat_loss,
            'relation': rel_loss
        }


def create_distill_loss(config: dict) -> nn.Module:
    """
    根据配置创建蒸馏损失
    
    Args:
        config: 配置字典
        
    Returns:
        损失函数
    """
    loss_type = config.get('type', 'feature')
    
    if loss_type == 'feature':
        return DistillationLoss(
            normalize=config.get('normalize', True),
            loss_type=config.get('loss_type', 'mse')
        )
    elif loss_type == 'combined':
        return CombinedDistillationLoss(
            feature_weight=config.get('feature_weight', 1.0),
            relation_weight=config.get('relation_weight', 0.5),
            normalize=config.get('normalize', True)
        )
    else:
        raise ValueError(f"Unknown distillation loss type: {loss_type}")

