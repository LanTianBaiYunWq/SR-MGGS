"""
几何编码器
基于Transformer的矢量几何特征提取器
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码
    为序列添加位置信息
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GeometryEmbedding(nn.Module):
    """
    几何嵌入层
    将2D坐标映射到高维空间
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 输入维度 (x, y)
            embed_dim: 嵌入维度
            dropout: dropout率
        """
        super().__init__()
        
        # 线性投影
        self.linear = nn.Linear(input_dim, embed_dim)
        
        # 额外的几何特征
        self.use_extra_features = True
        if self.use_extra_features:
            # 添加局部几何特征：差分、角度等
            self.feature_dim = input_dim + 4  # dx, dy, dist, angle
            self.linear = nn.Linear(self.feature_dim, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: 坐标 [batch_size, seq_len, 2]
            
        Returns:
            嵌入向量 [batch_size, seq_len, embed_dim]
        """
        if self.use_extra_features:
            # 计算差分特征
            diff = torch.zeros_like(coords)
            diff[:, 1:, :] = coords[:, 1:, :] - coords[:, :-1, :]
            
            # 计算距离
            dist = torch.norm(diff, dim=-1, keepdim=True)
            
            # 计算角度
            angle = torch.atan2(diff[:, :, 1:2], diff[:, :, 0:1] + 1e-8)
            
            # 拼接特征
            features = torch.cat([coords, diff, dist, angle], dim=-1)
        else:
            features = coords
        
        x = self.linear(features)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            src_mask: 注意力掩码
            src_key_padding_mask: padding掩码
            
        Returns:
            输出 [batch_size, seq_len, d_model]
        """
        # Pre-norm架构
        # 自注意力
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout(attn_out)
        
        # 前馈网络
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


class GeometryEncoder(nn.Module):
    """
    几何编码器
    完整的Transformer架构用于几何特征提取
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_points: int = 256,
        use_pos_encoding: bool = True,
        pooling: str = "cls"
    ):
        """
        Args:
            input_dim: 输入维度
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            mlp_ratio: MLP扩展比例
            dropout: dropout率
            max_points: 最大点数
            use_pos_encoding: 是否使用位置编码
            pooling: 池化方式 ("cls", "mean", "max")
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pooling = pooling
        
        # 嵌入层
        self.embedding = GeometryEmbedding(input_dim, embed_dim, dropout)
        
        # CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = LearnablePositionalEncoding(
                embed_dim, max_len=max_points + 1, dropout=dropout
            )
        
        # Transformer编码器层
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            coords: 坐标 [batch_size, seq_len, 2]
            mask: padding掩码 [batch_size, seq_len]
            
        Returns:
            几何嵌入 [batch_size, embed_dim]
        """
        batch_size = coords.size(0)
        
        # 嵌入
        x = self.embedding(coords)
        
        # 添加CLS token
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # 更新mask
            if mask is not None:
                cls_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # 位置编码
        if self.use_pos_encoding:
            x = self.pos_encoding(x)
        
        # Transformer编码
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        
        x = self.norm(x)
        
        # 池化
        if self.pooling == "cls":
            return x[:, 0]
        elif self.pooling == "mean":
            if mask is not None:
                # 排除padding位置
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x = x.masked_fill(mask_expanded.bool(), 0)
                lengths = (~mask.bool()).sum(dim=1, keepdim=True).float()
                return x.sum(dim=1) / lengths.clamp(min=1)
            return x.mean(dim=1)
        elif self.pooling == "max":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x = x.masked_fill(mask_expanded.bool(), float('-inf'))
            return x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def get_attention_maps(
        self,
        coords: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        获取注意力图（用于可视化）
        
        Args:
            coords: 坐标
            layer_idx: 层索引
            
        Returns:
            注意力图
        """
        batch_size = coords.size(0)
        x = self.embedding(coords)
        
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        if self.use_pos_encoding:
            x = self.pos_encoding(x)
        
        # 运行到指定层
        target_layer = self.layers[layer_idx]
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) + layer_idx if layer_idx < 0 else layer_idx:
                x_norm = layer.norm1(x)
                _, attn_weights = layer.self_attn(
                    x_norm, x_norm, x_norm,
                    need_weights=True,
                    average_attn_weights=False
                )
                return attn_weights
            x = layer(x)
        
        return None


class GeometryEncoderWithStats(nn.Module):
    """
    带统计特征的几何编码器
    结合Transformer特征和手工统计特征
    """
    
    def __init__(
        self,
        transformer_config: dict,
        use_stats: bool = True,
        stats_dim: int = 32
    ):
        """
        Args:
            transformer_config: Transformer配置
            use_stats: 是否使用统计特征
            stats_dim: 统计特征维度
        """
        super().__init__()
        
        self.transformer = GeometryEncoder(**transformer_config)
        self.use_stats = use_stats
        
        embed_dim = transformer_config.get('embed_dim', 512)
        
        if use_stats:
            # 统计特征提取
            self.stats_proj = nn.Sequential(
                nn.Linear(10, stats_dim),  # 10个手工特征
                nn.LayerNorm(stats_dim),
                nn.GELU(),
                nn.Linear(stats_dim, stats_dim)
            )
            
            # 融合层
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim + stats_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        
        self.embed_dim = embed_dim
    
    def compute_stats(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算统计特征
        
        Args:
            coords: 坐标 [batch_size, seq_len, 2]
            
        Returns:
            统计特征 [batch_size, 10]
        """
        batch_size = coords.size(0)
        stats = []
        
        for i in range(batch_size):
            c = coords[i]  # [seq_len, 2]
            
            # 中心点
            center = c.mean(dim=0)
            
            # 边界框
            min_vals = c.min(dim=0)[0]
            max_vals = c.max(dim=0)[0]
            bbox_size = max_vals - min_vals
            
            # 面积估计（使用shoelace公式）
            x, y = c[:, 0], c[:, 1]
            area = 0.5 * torch.abs(
                (x[:-1] * y[1:]).sum() - (x[1:] * y[:-1]).sum() +
                x[-1] * y[0] - x[0] * y[-1]
            )
            
            # 周长
            diff = c[1:] - c[:-1]
            perimeter = torch.norm(diff, dim=1).sum()
            
            # 紧凑度
            compactness = 4 * math.pi * area / (perimeter ** 2 + 1e-8)
            
            # 长宽比
            aspect_ratio = bbox_size[0] / (bbox_size[1] + 1e-8)
            
            # 到中心的平均距离
            distances = torch.norm(c - center, dim=1)
            mean_dist = distances.mean()
            std_dist = distances.std()
            
            sample_stats = torch.stack([
                center[0], center[1],
                bbox_size[0], bbox_size[1],
                area, perimeter,
                compactness, aspect_ratio,
                mean_dist, std_dist
            ])
            stats.append(sample_stats)
        
        return torch.stack(stats)
    
    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            coords: 坐标
            mask: padding掩码
            
        Returns:
            几何嵌入
        """
        # Transformer特征
        transformer_feat = self.transformer(coords, mask)
        
        if self.use_stats:
            # 统计特征
            stats = self.compute_stats(coords)
            stats_feat = self.stats_proj(stats)
            
            # 融合
            combined = torch.cat([transformer_feat, stats_feat], dim=-1)
            output = self.fusion(combined)
            return output
        
        return transformer_feat


def create_geometry_encoder(config: dict) -> nn.Module:
    """
    根据配置创建几何编码器
    
    Args:
        config: 配置字典
        
    Returns:
        几何编码器模型
    """
    encoder = GeometryEncoder(
        input_dim=config.get('input_dim', 2),
        embed_dim=config.get('embed_dim', 512),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
        max_points=config.get('max_points', 256),
        use_pos_encoding=config.get('use_pos_encoding', True),
        pooling=config.get('pooling', 'cls')
    )
    
    return encoder

