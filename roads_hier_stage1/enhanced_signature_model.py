"""
Stronger dual-view encoder with chunk-level and file-level Transformer aggregation.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class TransformerSetBlock(nn.Module):
    """Lightweight Transformer block for unordered set elements."""

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


class EnhancedRoadsHierarchicalEncoder(nn.Module):
    """
    Stronger fixed-budget dual-view encoder.

    Differences from the minimal encoder:
    - chunk-level Transformer with a tile CLS token
    - explicit tile slot embedding
    - explicit tile depth embedding
    - file-level Transformer with a file CLS token
    """

    def __init__(
        self,
        chunk_feature_dim: int = 10,
        hidden_dim: int = 192,
        file_dim: int = 192,
        n_tiles: int = 8,
        max_depth: int = 8,
        chunk_num_heads: int = 4,
        file_num_heads: int = 4,
        num_chunk_layers: int = 2,
        num_file_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tiles = int(n_tiles)
        self.max_depth = int(max_depth)

        self.chunk_input = nn.Sequential(
            nn.Linear(chunk_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tile_cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.chunk_blocks = nn.ModuleList(
            [
                TransformerSetBlock(hidden_dim, num_heads=chunk_num_heads, dropout=dropout)
                for _ in range(num_chunk_layers)
            ]
        )
        self.tile_proj = nn.Sequential(
            nn.Linear(hidden_dim, file_dim),
            nn.LayerNorm(file_dim),
            nn.GELU(),
        )

        self.tile_slot_embed = nn.Embedding(self.n_tiles, file_dim)
        self.depth_embed = nn.Embedding(self.max_depth + 2, file_dim)
        self.file_cls = nn.Parameter(torch.randn(1, 1, file_dim) * 0.02)
        self.file_blocks = nn.ModuleList(
            [
                TransformerSetBlock(file_dim, num_heads=file_num_heads, dropout=dropout)
                for _ in range(num_file_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(file_dim)

    def _encode_tile(self, chunk_features: torch.Tensor) -> torch.Tensor:
        chunk_emb = self.chunk_input(chunk_features)
        cls = self.tile_cls.expand(1, -1, -1)
        x = torch.cat([cls, chunk_emb.unsqueeze(0)], dim=1)
        for block in self.chunk_blocks:
            x = block(x)
        tile_cls = x[0, 0]
        return self.tile_proj(tile_cls)

    def _depth_to_index(self, depth: int) -> int:
        if depth < 0:
            return 0
        return min(int(depth) + 1, self.max_depth + 1)

    def forward(
        self,
        tile_feature_list: List[torch.Tensor],
        tile_depths: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if not tile_feature_list:
            raise ValueError("tile_feature_list is empty")
        if len(tile_feature_list) > self.n_tiles:
            raise ValueError(f"Expected at most {self.n_tiles} tiles, got {len(tile_feature_list)}")

        tile_embeddings = []
        for tile_idx, chunk_features in enumerate(tile_feature_list):
            tile_emb = self._encode_tile(chunk_features)
            slot_idx = torch.tensor(tile_idx, dtype=torch.long, device=tile_emb.device)
            tile_emb = tile_emb + self.tile_slot_embed(slot_idx)
            if tile_depths is not None:
                depth_idx = torch.tensor(
                    self._depth_to_index(int(tile_depths[tile_idx])),
                    dtype=torch.long,
                    device=tile_emb.device,
                )
                tile_emb = tile_emb + self.depth_embed(depth_idx)
            tile_embeddings.append(tile_emb)

        tiles = torch.stack(tile_embeddings, dim=0).unsqueeze(0)
        file_cls = self.file_cls.expand(1, -1, -1)
        x = torch.cat([file_cls, tiles], dim=1)
        for block in self.file_blocks:
            x = block(x)
        return self.final_norm(x[0, 0])
