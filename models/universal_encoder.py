"""
通用几何编码器
===============================

本模块实现支持所有几何类型的层次化编码器。

设计理念：
    采用"两层编码"架构，先对每个原语部件独立编码，
    再对同一几何的所有部件进行置换不变聚合。

架构组件：
    1. SinusoidalPositionalEncoding：正弦位置编码
       - 为序列添加位置信息
       - 仅用于线和环，点集合不使用
    
    2. PartEncoder：部件编码器
       - 输入：坐标序列 + 类型 + 角色
       - 特点：点集合不使用位置编码（置换不变）
       - 输出：每个部件的嵌入向量
       - 组成：坐标投影 + 类型嵌入 + Transformer + Mean Pooling
    
    3. GeomAggregator：几何聚合器
       - 输入：多个部件嵌入
       - 方式：加权平均（稳定）或注意力聚合（表达力强）
       - 特点：置换不变，部件顺序不影响结果
       - 输出：几何级别的嵌入向量
    
    4. UniversalGeometryEncoder：完整编码器
       - 整合 PartEncoder 和 GeomAggregator
       - 直接接收 collate_parts 的输出

关键设计：
    - 点集合（PointSet）不使用位置编码，实现置换不变性
    - 使用原语类型嵌入和角色嵌入，让模型知道处理的是什么
    - 使用加权聚合，重要部件（长度/周长大）贡献更多

使用示例：
    >>> encoder = UniversalGeometryEncoder(d_model=256)
    >>> batch = collate_parts(dataset_batch)
    >>> embeddings = encoder(batch)  # (B, d_model)

作者：GSD Team
版本：2.0
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess.geometry_types import PRIM_POINTSET, PRIM_LINE, PRIM_RING


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns:
            位置编码 (seq_len, d_model)
        """
        return self.pe[:seq_len]


class PartEncoder(nn.Module):
    """
    部件编码器
    
    对单个原语部件（点集合/线/环）进行编码
    
    关键设计：
    - 点集合（PointSet）不使用位置编码 → 置换不变性
    - 线/环使用位置编码 → 保留顺序信息
    - 使用原语类型嵌入和角色嵌入
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_len: int = 256,
        n_prim_types: int = 3,
        n_roles: int = 3,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            n_prim_types: 原语类型数量
            n_roles: 角色类型数量
            num_heads: 注意力头数
            num_layers: Transformer 层数
            dim_feedforward: FFN 隐藏层维度
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_model = d_model
        
        # 坐标投影：(x, y) → d_model
        self.coord_proj = nn.Linear(2, d_model)
        
        # 类型嵌入
        self.prim_emb = nn.Embedding(n_prim_types, d_model)
        self.role_emb = nn.Embedding(n_roles, d_model)
        
        # 位置编码
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.coord_proj.weight)
        nn.init.zeros_(self.coord_proj.bias)
        nn.init.normal_(self.prim_emb.weight, std=0.02)
        nn.init.normal_(self.role_emb.weight, std=0.02)
    
    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        prim_type: torch.Tensor,
        role: torch.Tensor
    ) -> torch.Tensor:
        """
        编码所有部件
        
        Args:
            coords: (P, L, 2) 坐标
            mask: (P, L) 有效点掩码 (True = 有效)
            prim_type: (P,) 原语类型
            role: (P,) 角色类型
            
        Returns:
            部件嵌入 (P, d_model)
        """
        P, L, _ = coords.shape
        device = coords.device
        
        # 1. 坐标投影
        x = self.coord_proj(coords)  # (P, L, d_model)
        
        # 2. 添加类型嵌入（广播到每个位置）
        prim_embed = self.prim_emb(prim_type)[:, None, :]  # (P, 1, d_model)
        role_embed = self.role_emb(role)[:, None, :]      # (P, 1, d_model)
        x = x + prim_embed + role_embed
        
        # 3. 添加位置编码（仅对线/环，点集合不加）
        pos = self.pos_encoding(L).unsqueeze(0)  # (1, L, d_model)
        
        # 创建位置编码掩码：PointSet 不使用位置编码
        is_pointset = (prim_type == PRIM_POINTSET)[:, None, None]  # (P, 1, 1)
        pos_to_add = torch.where(is_pointset, torch.zeros_like(pos), pos)
        x = x + pos_to_add
        
        # 4. Transformer 编码
        # key_padding_mask: True 表示要忽略的位置
        key_padding_mask = ~mask  # (P, L)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # 5. Masked Mean Pooling
        x = self.norm(x)
        
        # 扩展掩码用于加权平均
        mask_expanded = mask.unsqueeze(-1).float()  # (P, L, 1)
        
        # 计算每个部件的平均嵌入
        sum_embed = (x * mask_expanded).sum(dim=1)  # (P, d_model)
        count = mask_expanded.sum(dim=1).clamp(min=1)  # (P, 1)
        
        part_embed = sum_embed / count  # (P, d_model)
        
        return part_embed


class GeomAggregator(nn.Module):
    """
    几何聚合器
    
    对同一几何的多个部件进行置换不变聚合
    生成几何级别的嵌入
    
    支持两种聚合方式：
    - weighted_mean: 加权平均（稳定，推荐第一版）
    - attention: 注意力聚合（更强表达力）
    """
    
    def __init__(
        self,
        d_model: int = 256,
        aggregation_type: str = "weighted_mean"
    ):
        """
        Args:
            d_model: 模型维度
            aggregation_type: 聚合类型 ("weighted_mean" 或 "attention")
        """
        super().__init__()
        
        self.d_model = d_model
        self.aggregation_type = aggregation_type
        
        if aggregation_type == "attention":
            self.attention = nn.Linear(d_model, 1)
    
    def forward(
        self,
        part_embed: torch.Tensor,
        part_to_geom: torch.Tensor,
        weight: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        聚合部件嵌入为几何嵌入
        
        Args:
            part_embed: (P, d_model) 部件嵌入
            part_to_geom: (P,) 部件到几何的映射
            weight: (P,) 聚合权重
            batch_size: 批次中几何数量
            
        Returns:
            几何嵌入 (B, d_model)
        """
        device = part_embed.device
        
        if self.aggregation_type == "weighted_mean":
            return self._weighted_mean_aggregation(
                part_embed, part_to_geom, weight, batch_size
            )
        elif self.aggregation_type == "attention":
            return self._attention_aggregation(
                part_embed, part_to_geom, weight, batch_size
            )
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
    
    def _weighted_mean_aggregation(
        self,
        part_embed: torch.Tensor,
        part_to_geom: torch.Tensor,
        weight: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """加权平均聚合"""
        device = part_embed.device
        
        # 权重归一化
        w = weight.clamp(min=1e-6).unsqueeze(-1)  # (P, 1)
        
        # 加权嵌入
        weighted_embed = part_embed * w  # (P, d_model)
        
        # 按几何索引聚合
        out = torch.zeros((batch_size, self.d_model), device=device)
        den = torch.zeros((batch_size, 1), device=device)
        
        out.index_add_(0, part_to_geom, weighted_embed)
        den.index_add_(0, part_to_geom, w)
        
        # 归一化
        geom_embed = out / den.clamp(min=1e-6)
        
        return geom_embed
    
    def _attention_aggregation(
        self,
        part_embed: torch.Tensor,
        part_to_geom: torch.Tensor,
        weight: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """注意力聚合"""
        device = part_embed.device
        P = part_embed.size(0)
        
        # 计算注意力分数
        attn_scores = self.attention(part_embed).squeeze(-1)  # (P,)
        
        # 按几何分组 softmax
        # 创建组掩码
        out = torch.zeros((batch_size, self.d_model), device=device)
        
        for g in range(batch_size):
            mask = (part_to_geom == g)
            if mask.sum() == 0:
                continue
            
            # 获取该几何的部件
            group_embed = part_embed[mask]  # (n, d_model)
            group_scores = attn_scores[mask]  # (n,)
            group_weight = weight[mask]  # (n,)
            
            # 结合原始权重和学习的注意力
            combined_scores = group_scores + torch.log(group_weight.clamp(min=1e-6))
            
            # Softmax
            attn_weights = F.softmax(combined_scores, dim=0)  # (n,)
            
            # 加权求和
            out[g] = (group_embed * attn_weights.unsqueeze(-1)).sum(dim=0)
        
        return out


class UniversalGeometryEncoder(nn.Module):
    """
    通用几何编码器
    
    整合 PartEncoder 和 GeomAggregator，
    接收 collate_parts 的输出，返回几何级别的嵌入
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_len: int = 256,
        n_prim_types: int = 3,
        n_roles: int = 3,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        aggregation_type: str = "weighted_mean"
    ):
        super().__init__()
        
        self.part_encoder = PartEncoder(
            d_model=d_model,
            max_len=max_len,
            n_prim_types=n_prim_types,
            n_roles=n_roles,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.aggregator = GeomAggregator(
            d_model=d_model,
            aggregation_type=aggregation_type
        )
        
        self.d_model = d_model
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        编码批次数据
        
        Args:
            batch: collate_parts 的输出字典
            
        Returns:
            几何嵌入 (B, d_model)
        """
        # 1. 编码所有部件
        part_embed = self.part_encoder(
            coords=batch["coords"],
            mask=batch["mask"],
            prim_type=batch["prim_type"],
            role=batch["role"]
        )
        
        # 2. 聚合为几何嵌入
        geom_embed = self.aggregator(
            part_embed=part_embed,
            part_to_geom=batch["part_to_geom"],
            weight=batch["weight"],
            batch_size=batch["batch_size"]
        )
        
        return geom_embed
    
    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_universal_encoder(config: dict) -> UniversalGeometryEncoder:
    """
    从配置创建通用几何编码器
    
    Args:
        config: 配置字典
        
    Returns:
        UniversalGeometryEncoder
    """
    encoder_cfg = config.get("geometry_encoder", {})
    
    return UniversalGeometryEncoder(
        d_model=encoder_cfg.get("embed_dim", 256),
        max_len=encoder_cfg.get("max_len", 256),
        n_prim_types=3,
        n_roles=3,
        num_heads=encoder_cfg.get("num_heads", 4),
        num_layers=encoder_cfg.get("num_layers", 4),
        dim_feedforward=int(encoder_cfg.get("embed_dim", 256) * encoder_cfg.get("mlp_ratio", 2.0)),
        dropout=encoder_cfg.get("dropout", 0.1),
        aggregation_type=encoder_cfg.get("aggregation_type", "weighted_mean")
    )

