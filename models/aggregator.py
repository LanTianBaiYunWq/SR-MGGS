"""
全局特征聚合模块
将多个几何体的嵌入聚合为一个全局表示
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    注意力池化
    使用可学习的查询向量对多个几何嵌入进行加权聚合
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: dropout率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 可学习的全局查询向量
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            embeddings: 多个几何的嵌入 [batch_size, num_geometries, embed_dim]
            mask: 填充掩码 [batch_size, num_geometries]，True表示有效位置
            
        Returns:
            (全局嵌入 [batch_size, embed_dim], 注意力权重 [batch_size, num_geometries])
        """
        batch_size, num_geoms, _ = embeddings.shape
        
        # 扩展全局查询
        query = self.global_query.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # 构建key_padding_mask（True表示忽略的位置）
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # 反转，因为attention期望True=忽略
        
        # 注意力计算
        # query: [B, 1, D], key/value: [B, N, D]
        global_feat, attn_weights = self.attention(
            query=query,
            key=embeddings,
            value=embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # 移除query维度
        global_feat = global_feat.squeeze(1)  # [B, D]
        attn_weights = attn_weights.squeeze(1)  # [B, N]
        
        # 层归一化 + 残差
        global_feat = self.norm(global_feat)
        
        # 输出投影
        global_feat = self.output_proj(global_feat)
        
        return global_feat, attn_weights


class SetTransformerAggregator(nn.Module):
    """
    Set Transformer 聚合器
    更强大的集合聚合方法
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_inducing_points: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_inducing_points: 诱导点数量
            num_layers: 层数
            dropout: dropout率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 诱导点（Inducing Points）
        self.inducing_points = nn.Parameter(
            torch.randn(1, num_inducing_points, embed_dim) * 0.02
        )
        
        # ISAB层（Induced Set Attention Block）
        self.isab_layers = nn.ModuleList([
            InducedSetAttentionBlock(embed_dim, num_heads, num_inducing_points, dropout)
            for _ in range(num_layers)
        ])
        
        # PMA（Pooling by Multihead Attention）
        self.pma = PoolingMultiheadAttention(embed_dim, num_heads, num_seeds=1)
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            embeddings: 多个几何的嵌入 [batch_size, num_geometries, embed_dim]
            mask: 填充掩码
            
        Returns:
            (全局嵌入, 注意力权重)
        """
        batch_size = embeddings.shape[0]
        
        # ISAB处理
        x = embeddings
        for isab in self.isab_layers:
            x = isab(x, mask)
        
        # PMA聚合
        global_feat, attn_weights = self.pma(x, mask)
        global_feat = global_feat.squeeze(1)  # [B, D]
        
        # 输出投影
        global_feat = self.output_proj(global_feat)
        
        return global_feat, attn_weights


class InducedSetAttentionBlock(nn.Module):
    """ISAB: 使用诱导点的集合注意力块"""
    
    def __init__(self, embed_dim, num_heads, num_inducing, dropout):
        super().__init__()
        
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, embed_dim) * 0.02)
        
        self.mab1 = MultiheadAttentionBlock(embed_dim, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(embed_dim, num_heads, dropout)
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        inducing = self.inducing.expand(batch_size, -1, -1)
        
        # X -> Inducing
        h = self.mab1(inducing, x, mask)
        # Inducing -> X
        return self.mab2(x, h)


class PoolingMultiheadAttention(nn.Module):
    """PMA: 多头注意力池化"""
    
    def __init__(self, embed_dim, num_heads, num_seeds=1):
        super().__init__()
        
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, embed_dim) * 0.02)
        self.mab = MultiheadAttentionBlock(embed_dim, num_heads, 0.0)
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seeds = self.seeds.expand(batch_size, -1, -1)
        return self.mab(seeds, x, mask, return_weights=True)


class MultiheadAttentionBlock(nn.Module):
    """MAB: 多头注意力块"""
    
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, query, kv, mask=None, return_weights=False):
        # Attention
        key_padding_mask = ~mask if mask is not None else None
        attn_out, attn_weights = self.attention(
            query, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = self.norm1(query + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        if return_weights:
            return x, attn_weights.squeeze(1)
        return x


class GlobalGeometryEncoder(nn.Module):
    """
    全局几何编码器
    将多个几何体编码并聚合为一个全局表示
    """
    
    def __init__(
        self,
        geometry_encoder: nn.Module,
        aggregator_type: str = "attention",
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            geometry_encoder: 单几何编码器
            aggregator_type: 聚合类型 ("attention", "set_transformer", "mean")
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: dropout率
        """
        super().__init__()
        
        self.geometry_encoder = geometry_encoder
        self.aggregator_type = aggregator_type
        
        if aggregator_type == "attention":
            self.aggregator = AttentionPooling(embed_dim, num_heads, dropout)
        elif aggregator_type == "set_transformer":
            self.aggregator = SetTransformerAggregator(embed_dim, num_heads)
        elif aggregator_type == "mean":
            self.aggregator = None  # 简单平均
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")
    
    def forward(
        self,
        geometries: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            geometries: 多个几何 [batch_size, num_geometries, num_points, 2]
            mask: 有效几何掩码 [batch_size, num_geometries]
            
        Returns:
            (全局嵌入 [batch_size, embed_dim], 注意力权重)
        """
        batch_size, num_geoms, num_points, coord_dim = geometries.shape
        
        # 展平为 [B*N, P, 2] 进行批量编码
        flat_geoms = geometries.view(batch_size * num_geoms, num_points, coord_dim)
        
        # 编码每个几何
        flat_embeddings = self.geometry_encoder(flat_geoms)  # [B*N, D]
        
        # 恢复形状 [B, N, D]
        embeddings = flat_embeddings.view(batch_size, num_geoms, -1)
        
        # 聚合
        if self.aggregator_type == "mean":
            if mask is not None:
                # 掩码平均
                mask_expanded = mask.unsqueeze(-1).float()
                global_feat = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                global_feat = embeddings.mean(dim=1)
            attn_weights = None
        else:
            global_feat, attn_weights = self.aggregator(embeddings, mask)
        
        return global_feat, attn_weights
    
    def encode_single_geometry(self, geometry: torch.Tensor) -> torch.Tensor:
        """编码单个几何（用于调试）"""
        return self.geometry_encoder(geometry)


def create_aggregator(
    aggregator_type: str = "attention",
    embed_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1
) -> nn.Module:
    """
    创建聚合器
    
    Args:
        aggregator_type: 聚合类型
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: dropout率
        
    Returns:
        聚合器模块
    """
    if aggregator_type == "attention":
        return AttentionPooling(embed_dim, num_heads, dropout)
    elif aggregator_type == "set_transformer":
        return SetTransformerAggregator(embed_dim, num_heads)
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")

