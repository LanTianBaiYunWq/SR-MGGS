"""
对比学习损失函数
InfoNCE及其变体
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比损失
    用于自监督对比学习
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True
    ):
        """
        Args:
            temperature: 温度参数，控制分布锐度
            normalize: 是否L2归一化
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        对于同一样本的不同视图（query和key），最大化它们的相似度
        同时最小化与其他样本的相似度
        
        Args:
            query: 查询特征 [batch_size, dim]
            key: 键特征 [batch_size, dim]
            labels: 样本标签（可选，用于监督对比）
            
        Returns:
            InfoNCE损失
        """
        batch_size = query.size(0)
        
        if self.normalize:
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
        
        # 计算相似度矩阵
        # logits[i, j] = query[i] · key[j]
        logits = torch.mm(query, key.t()) / self.temperature
        
        # 对角线是正样本对
        # labels[i] = i，即第i个query的正样本是第i个key
        if labels is None:
            labels = torch.arange(batch_size, device=query.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SymmetricInfoNCELoss(nn.Module):
    """
    对称InfoNCE损失
    同时计算query->key和key->query方向的损失
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.criterion = InfoNCELoss(temperature, normalize)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对称InfoNCE损失
        
        Args:
            query: 查询特征
            key: 键特征
            
        Returns:
            对称损失
        """
        loss_qk = self.criterion(query, key)
        loss_kq = self.criterion(key, query)
        
        return (loss_qk + loss_kq) / 2


class NTXentLoss(nn.Module):
    """
    NT-Xent损失 (Normalized Temperature-scaled Cross Entropy)
    SimCLR使用的对比损失
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        normalize: bool = True
    ):
        """
        Args:
            temperature: 温度参数
            normalize: 是否归一化
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        计算NT-Xent损失
        
        Args:
            z_i: 第一个视图的特征 [batch_size, dim]
            z_j: 第二个视图的特征 [batch_size, dim]
            
        Returns:
            NT-Xent损失
        """
        batch_size = z_i.size(0)
        
        if self.normalize:
            z_i = F.normalize(z_i, dim=-1)
            z_j = F.normalize(z_j, dim=-1)
        
        # 拼接两个视图
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]
        
        # 计算相似度矩阵
        sim = torch.mm(z, z.t()) / self.temperature
        
        # 创建掩码，排除自身
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))
        
        # 正样本对的位置
        # z_i[k] 的正样本是 z_j[k]，即位置 batch_size + k
        # z_j[k] 的正样本是 z_i[k]，即位置 k
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ], dim=0).to(z.device)
        
        # 交叉熵
        loss = F.cross_entropy(sim, labels)
        
        return loss


class HardNegativeRankingLoss(nn.Module):
    """
    Hard-negative ranking loss for retrieval-oriented training.
    For each anchor, enforce the positive pair score to exceed the hardest
    in-batch negative score by a margin.
    """

    def __init__(
        self,
        margin: float = 0.05,
        normalize: bool = True,
        same_group_only: bool = True,
    ):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.same_group_only = same_group_only

    def _pairwise_scores(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        return torch.mm(x, y.t())

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = z_i.size(0)
        if batch_size <= 1:
            return z_i.new_zeros(()), {
                "valid_anchors": 0.0,
                "mean_hard_negative": 0.0,
                "mean_positive": 0.0,
            }

        sim_ij = self._pairwise_scores(z_i, z_j)
        sim_ji = sim_ij.t()
        device = z_i.device
        eye_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        if group_ids is not None and self.same_group_only:
            group_ids = group_ids.view(-1, 1)
            neg_mask = torch.eq(group_ids, group_ids.t()) & (~eye_mask)
        else:
            neg_mask = ~eye_mask

        losses = []
        valid_anchor_count = 0
        hard_negatives = []
        positives = []

        for sim_matrix in (sim_ij, sim_ji):
            positives_diag = sim_matrix.diag()
            masked_negatives = sim_matrix.masked_fill(~neg_mask, float("-inf"))
            hardest_negatives, _ = masked_negatives.max(dim=1)
            valid_mask = torch.isfinite(hardest_negatives)
            if valid_mask.any():
                valid_anchor_count += int(valid_mask.sum().item())
                pos_valid = positives_diag[valid_mask]
                neg_valid = hardest_negatives[valid_mask]
                losses.append(F.relu(self.margin + neg_valid - pos_valid))
                hard_negatives.append(neg_valid.detach())
                positives.append(pos_valid.detach())

        if not losses:
            return z_i.new_zeros(()), {
                "valid_anchors": 0.0,
                "mean_hard_negative": 0.0,
                "mean_positive": 0.0,
            }

        loss = torch.cat(losses, dim=0).mean()
        return loss, {
            "valid_anchors": float(valid_anchor_count),
            "mean_hard_negative": float(torch.cat(hard_negatives).mean().item()),
            "mean_positive": float(torch.cat(positives).mean().item()),
        }


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比损失
    使用标签信息，同类样本为正样本对
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True
    ):
        """
        Args:
            temperature: 温度参数
            normalize: 是否归一化
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算监督对比损失
        
        Args:
            features: 特征 [batch_size, dim]
            labels: 标签 [batch_size]
            
        Returns:
            监督对比损失
        """
        batch_size = features.size(0)
        device = features.device
        
        if self.normalize:
            features = F.normalize(features, dim=-1)
        
        # 相似度矩阵
        sim = torch.mm(features, features.t()) / self.temperature
        
        # 标签掩码：同类为1，异类为0
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        
        # 排除自身
        self_mask = torch.eye(batch_size, device=device)
        mask = mask - self_mask
        
        # 计算损失
        # 对于每个anchor，正样本是同类的其他样本
        exp_sim = torch.exp(sim) * (1 - self_mask)  # 排除自身
        
        # 分母：所有负样本的exp-sim之和
        neg_mask = 1 - mask - self_mask
        neg_exp = (exp_sim * neg_mask).sum(dim=1, keepdim=True)
        
        # 对每个正样本对计算损失
        pos_mask = mask
        n_pos = pos_mask.sum(dim=1)
        
        # 避免没有正样本的情况
        valid = n_pos > 0
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        log_prob = sim - torch.log(exp_sim + neg_exp + 1e-8)
        
        # 只对正样本求平均
        loss = -(log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)
        loss = loss[valid].mean()
        
        return loss


class MultiViewContrastiveLoss(nn.Module):
    """
    多视图对比损失
    处理多于2个视图的情况
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        n_views: int = 4
    ):
        """
        Args:
            temperature: 温度参数
            normalize: 是否归一化
            n_views: 视图数量
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.n_views = n_views
    
    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多视图对比损失
        
        Args:
            features: 多视图特征 [batch_size, n_views, dim]
            
        Returns:
            对比损失
        """
        batch_size = features.size(0)
        n_views = features.size(1)
        device = features.device
        
        # 展平
        features = features.view(batch_size * n_views, -1)
        
        if self.normalize:
            features = F.normalize(features, dim=-1)
        
        # 相似度矩阵
        sim = torch.mm(features, features.t()) / self.temperature
        
        # 创建标签：同一样本的不同视图应该相似
        labels = torch.arange(batch_size, device=device).repeat_interleave(n_views)
        
        # 创建正样本掩码
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        
        # 排除自身
        self_mask = torch.eye(batch_size * n_views, device=device)
        mask = mask - self_mask
        
        # 计算损失（类似SupCon）
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 对正样本求平均
        n_pos = mask.sum(dim=1)
        loss = -(log_prob * mask).sum(dim=1) / (n_pos + 1e-8)
        loss = loss.mean()
        
        return loss


class GSDContrastiveLoss(nn.Module):
    """
    GSD专用对比损失
    结合几何特征的对比学习
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        hard_negative_weight: float = 1.0
    ):
        """
        Args:
            temperature: 温度参数
            normalize: 是否归一化
            hard_negative_weight: 困难负样本权重
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.hard_negative_weight = hard_negative_weight
        
        self.base_loss = InfoNCELoss(temperature, normalize)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hard_negatives: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算GSD对比损失
        
        Args:
            query: 查询特征
            key: 键特征
            labels: 样本标签
            hard_negatives: 困难负样本特征
            
        Returns:
            (总损失, 损失详情字典)
        """
        # 基础InfoNCE
        base_loss = self.base_loss(query, key, labels)
        
        losses = {'base': base_loss}
        total_loss = base_loss
        
        # 困难负样本挖掘
        if hard_negatives is not None and self.hard_negative_weight > 0:
            if self.normalize:
                query = F.normalize(query, dim=-1)
                hard_negatives = F.normalize(hard_negatives, dim=-1)
            
            # 与困难负样本的相似度应该低
            hard_sim = torch.mm(query, hard_negatives.t()) / self.temperature
            hard_loss = torch.logsumexp(hard_sim, dim=1).mean()
            
            losses['hard_negative'] = hard_loss
            total_loss = total_loss + self.hard_negative_weight * hard_loss
        
        losses['total'] = total_loss
        
        return total_loss, losses


class SpatialWeightedContrastiveLoss(nn.Module):
    """
    空间语义加权对比损失
    根据空间重要性对不同区域/样本进行加权
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        importance_type: str = "attention"
    ):
        """
        Args:
            temperature: 温度参数
            normalize: 是否归一化
            importance_type: 重要性计算方式 ("attention", "norm", "entropy")
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.importance_type = importance_type
    
    def compute_importance_weights(
        self,
        features: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算样本重要性权重
        
        Args:
            features: 特征 [B, D]
            attention_weights: 注意力权重 [B, N]（来自聚合器）
            
        Returns:
            重要性权重 [B]
        """
        if self.importance_type == "attention" and attention_weights is not None:
            # 使用注意力权重的熵作为重要性
            # 熵低 = 注意力集中 = 更重要
            entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
            weights = 1.0 / (entropy + 1.0)
        elif self.importance_type == "norm":
            # 使用特征范数
            weights = torch.norm(features, dim=-1)
        elif self.importance_type == "entropy":
            # 使用特征分布的熵
            probs = F.softmax(features, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            weights = 1.0 / (entropy + 1.0)
        else:
            weights = torch.ones(features.size(0), device=features.device)
        
        # 归一化权重
        weights = weights / (weights.sum() + 1e-8) * features.size(0)
        
        return weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_weights: Optional[torch.Tensor] = None,
        key_weights: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算空间加权对比损失
        
        Args:
            query: 查询特征 [B, D]
            key: 键特征 [B, D]
            query_weights: 查询权重 [B]
            key_weights: 键权重 [B]
            attention_weights: 注意力权重 [B, N]
            
        Returns:
            加权对比损失
        """
        batch_size = query.size(0)
        device = query.device
        
        if self.normalize:
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
        
        # 计算重要性权重
        if query_weights is None:
            query_weights = self.compute_importance_weights(query, attention_weights)
        if key_weights is None:
            key_weights = self.compute_importance_weights(key, attention_weights)
        
        # 合并权重
        weights = (query_weights + key_weights) / 2
        
        # 计算相似度矩阵
        logits = torch.mm(query, key.t()) / self.temperature
        
        # 正样本对（对角线）
        labels = torch.arange(batch_size, device=device)
        
        # 加权交叉熵
        log_probs = F.log_softmax(logits, dim=1)
        loss_per_sample = -log_probs[torch.arange(batch_size), labels]
        
        # 应用权重
        weighted_loss = (loss_per_sample * weights).sum() / (weights.sum() + 1e-8)
        
        return weighted_loss


class MultiViewSpatialContrastiveLoss(nn.Module):
    """
    多视图空间对比损失
    结合多角度渲染和空间语义加权
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        normalize: bool = True,
        use_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.1
    ):
        """
        Args:
            temperature: 温度参数
            normalize: 是否归一化
            use_hard_negatives: 是否使用困难负样本
            hard_negative_ratio: 困难负样本比例
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
    
    def mine_hard_negatives(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        挖掘困难负样本
        
        Args:
            features: 所有特征 [N, D]
            labels: 标签 [N]
            
        Returns:
            困难负样本掩码
        """
        # 计算相似度矩阵
        sim = torch.mm(features, features.t())
        
        # 创建负样本掩码
        labels = labels.view(-1, 1)
        neg_mask = torch.ne(labels, labels.t()).float()
        
        # 负样本中相似度最高的作为困难负样本
        neg_sim = sim * neg_mask - 1e9 * (1 - neg_mask)
        
        # 选择top-k
        n_hard = max(1, int(neg_mask.sum(dim=1).mean() * self.hard_negative_ratio))
        _, hard_indices = neg_sim.topk(n_hard, dim=1)
        
        return hard_indices
    
    def forward(
        self,
        features: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算多视图空间对比损失
        
        Args:
            features: 多视图特征 [B, n_views, D]
            spatial_weights: 空间权重 [B, n_views]
            
        Returns:
            (损失, 详情字典)
        """
        batch_size, n_views, dim = features.shape
        device = features.device
        
        if self.normalize:
            features = F.normalize(features, dim=-1)
        
        # 展平
        features_flat = features.view(batch_size * n_views, dim)
        
        # 创建标签
        labels = torch.arange(batch_size, device=device).repeat_interleave(n_views)
        
        # 相似度矩阵
        sim = torch.mm(features_flat, features_flat.t()) / self.temperature
        
        # 正样本掩码
        pos_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        self_mask = torch.eye(batch_size * n_views, device=device)
        pos_mask = pos_mask - self_mask
        
        # 计算损失
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 基础损失
        n_pos = pos_mask.sum(dim=1)
        base_loss = -(log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)
        
        # 应用空间权重
        if spatial_weights is not None:
            weights_flat = spatial_weights.view(-1)
            loss = (base_loss * weights_flat).sum() / (weights_flat.sum() + 1e-8)
        else:
            loss = base_loss.mean()
        
        # 困难负样本损失
        details = {'base': loss}
        
        if self.use_hard_negatives:
            hard_indices = self.mine_hard_negatives(features_flat, labels)
            hard_sim = sim.gather(1, hard_indices)
            hard_loss = torch.logsumexp(hard_sim, dim=1).mean()
            details['hard_negative'] = hard_loss
            loss = loss + 0.1 * hard_loss
        
        details['total'] = loss
        
        return loss, details


def create_contrastive_loss(config: dict) -> nn.Module:
    """
    根据配置创建对比损失
    
    Args:
        config: 配置字典
        
    Returns:
        损失函数
    """
    loss_type = config.get('type', 'infonce')
    temperature = config.get('temperature', 0.07)
    normalize = config.get('normalize', True)
    
    if loss_type == 'infonce':
        return InfoNCELoss(temperature, normalize)
    elif loss_type == 'symmetric':
        return SymmetricInfoNCELoss(temperature, normalize)
    elif loss_type == 'ntxent':
        return NTXentLoss(temperature, normalize)
    elif loss_type == 'supcon':
        return SupervisedContrastiveLoss(temperature, normalize)
    elif loss_type == 'gsd':
        return GSDContrastiveLoss(
            temperature, 
            normalize,
            config.get('hard_negative_weight', 1.0)
        )
    elif loss_type == 'spatial_weighted':
        return SpatialWeightedContrastiveLoss(
            temperature,
            normalize,
            config.get('importance_type', 'attention')
        )
    elif loss_type == 'multiview_spatial':
        return MultiViewSpatialContrastiveLoss(
            temperature,
            normalize,
            config.get('use_hard_negatives', True)
        )
    else:
        raise ValueError(f"Unknown contrastive loss type: {loss_type}")

