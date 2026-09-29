"""Point branch encoders: tile local set encoder + file Transformer aggregator."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class TransformerSetBlock(nn.Module):
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
        return self.norm2(x + self.ffn(x))


class PointNetTileEncoder(nn.Module):
    """Local point-set encoder used only inside a tile."""

    def __init__(self, point_feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        point_hidden = self.mlp(point_features)
        pooled_max = point_hidden.max(dim=0).values
        pooled_mean = point_hidden.mean(dim=0)
        return self.out_proj(torch.cat([pooled_mean, pooled_max], dim=0))


class SetTransformerTileEncoder(nn.Module):
    """Tile encoder with self-attention over points before pooling."""

    def __init__(
        self,
        point_feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [TransformerSetBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(point_features).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = x[0]
        pooled_max = x.max(dim=0).values
        pooled_mean = x.mean(dim=0)
        return self.out_proj(torch.cat([pooled_mean, pooled_max], dim=0))


class PointHierarchicalEncoder(nn.Module):
    """point -> tile -> file encoder aligned with the line branch interface."""

    def __init__(
        self,
        point_feature_dim: int = 8,
        hidden_dim: int = 192,
        file_dim: int = 192,
        n_tiles: int = 8,
        max_depth: int = 8,
        tile_encoder_type: str = "pointnet",
        tile_num_heads: int = 4,
        num_tile_layers: int = 2,
        file_num_heads: int = 4,
        num_file_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tiles = int(n_tiles)
        self.max_depth = int(max_depth)

        tile_encoder_type = str(tile_encoder_type).lower()
        if tile_encoder_type == "pointnet":
            self.tile_encoder = PointNetTileEncoder(point_feature_dim, hidden_dim, dropout)
        elif tile_encoder_type == "set_transformer":
            self.tile_encoder = SetTransformerTileEncoder(
                point_feature_dim,
                hidden_dim,
                num_heads=tile_num_heads,
                num_layers=num_tile_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported tile_encoder_type: {tile_encoder_type}")

        self.tile_proj = nn.Sequential(
            nn.Linear(hidden_dim, file_dim),
            nn.LayerNorm(file_dim),
            nn.GELU(),
        )

        self.tile_slot_embed = nn.Embedding(self.n_tiles, file_dim)
        self.depth_embed = nn.Embedding(self.max_depth + 2, file_dim)
        self.file_cls = nn.Parameter(torch.randn(1, 1, file_dim) * 0.02)
        self.file_blocks = nn.ModuleList(
            [TransformerSetBlock(file_dim, num_heads=file_num_heads, dropout=dropout) for _ in range(num_file_layers)]
        )
        self.final_norm = nn.LayerNorm(file_dim)

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
        for tile_idx, point_features in enumerate(tile_feature_list):
            tile_emb = self.tile_proj(self.tile_encoder(point_features))
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


class PointHybridSignatureEncoder(nn.Module):
    """Fuse learned point geometry with view-visible point statistics."""

    def __init__(
        self,
        point_feature_dim: int = 8,
        visible_feature_dim: int = 24,
        hidden_dim: int = 192,
        file_dim: int = 192,
        visible_hidden_dim: int = 128,
        visible_dim: int = 128,
        fusion_hidden_dim: int = 256,
        fusion_dim: int = 192,
        n_tiles: int = 8,
        max_depth: int = 8,
        tile_encoder_type: str = "set_transformer",
        tile_num_heads: int = 4,
        num_tile_layers: int = 2,
        file_num_heads: int = 4,
        num_file_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_encoder = PointHierarchicalEncoder(
            point_feature_dim=point_feature_dim,
            hidden_dim=hidden_dim,
            file_dim=file_dim,
            n_tiles=n_tiles,
            max_depth=max_depth,
            tile_encoder_type=tile_encoder_type,
            tile_num_heads=tile_num_heads,
            num_tile_layers=num_tile_layers,
            file_num_heads=file_num_heads,
            num_file_layers=num_file_layers,
            dropout=dropout,
        )
        self.visible_encoder = nn.Sequential(
            nn.Linear(visible_feature_dim, visible_hidden_dim),
            nn.LayerNorm(visible_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visible_hidden_dim, visible_dim),
            nn.LayerNorm(visible_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(file_dim + visible_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self,
        tile_feature_list: List[torch.Tensor],
        tile_depths: Sequence[int] | None,
        visible_stats: torch.Tensor,
    ) -> torch.Tensor:
        if visible_stats.ndim == 1:
            visible_stats = visible_stats.unsqueeze(0)
        if visible_stats.ndim != 2 or visible_stats.shape[0] != 1:
            raise ValueError(f"Expected visible_stats shape [D] or [1, D], got {tuple(visible_stats.shape)}")

        point_embedding = self.point_encoder(tile_feature_list, tile_depths)
        visible_embedding = self.visible_encoder(visible_stats)[0]
        return self.fusion(torch.cat([point_embedding, visible_embedding], dim=0))
