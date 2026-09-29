"""
roads_hier_stage1 最小签名模型。

目标：
1. 只读 Gate 1 cache
2. 不接 graph
3. 不接视觉蒸馏
4. 先把 tile -> file 的最小 contrastive 训练链跑通
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class AttentionPool1D(nn.Module):
    """对变长集合做简单注意力池化。"""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [N, D], got {tuple(x.shape)}")
        if x.shape[0] == 1:
            return x[0]

        attn = torch.softmax(self.score(x).squeeze(-1), dim=0)
        return torch.sum(x * attn.unsqueeze(-1), dim=0)


class SetSelfAttentionBlock(nn.Module):
    """最小 self-attention block，用于 file-level tile 集合编码。"""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class MinimalRoadsHierarchicalEncoder(nn.Module):
    """
    最小 roads line-only 分层编码器。

    输入：
    - 一个文件包含若干 tile
    - 每个 tile 包含若干 chunk feature

    输出：
    - file embedding
    """

    def __init__(
        self,
        chunk_feature_dim: int = 4,
        hidden_dim: int = 128,
        file_dim: int = 128,
        num_heads: int = 4,
        num_set_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tile_pool = AttentionPool1D(hidden_dim)
        self.tile_proj = nn.Sequential(
            nn.Linear(hidden_dim, file_dim),
            nn.LayerNorm(file_dim),
            nn.GELU(),
        )
        self.file_blocks = nn.ModuleList(
            [SetSelfAttentionBlock(file_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_set_layers)]
        )
        self.file_pool = AttentionPool1D(file_dim)

    def encode_tile(self, chunk_features: torch.Tensor) -> torch.Tensor:
        """编码一个 tile。"""
        chunk_emb = self.chunk_encoder(chunk_features)
        tile_emb = self.tile_pool(chunk_emb)
        return self.tile_proj(tile_emb)

    def forward(self, tile_feature_list: List[torch.Tensor]) -> torch.Tensor:
        """编码一个文件。"""
        if not tile_feature_list:
            raise ValueError("tile_feature_list is empty")

        tile_embeddings = [self.encode_tile(tile_features) for tile_features in tile_feature_list]
        x = torch.stack(tile_embeddings, dim=0).unsqueeze(0)
        for block in self.file_blocks:
            x = block(x)
        return self.file_pool(x[0])
