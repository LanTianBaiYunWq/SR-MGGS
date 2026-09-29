"""
roads / line-only / hierarchical signature 模型骨架。

这一版提供最小模块切分：

1. line chunk encoder
2. local road subgraph encoder
3. tile pooling
4. Set Transformer file encoder
5. tile/file 双层蒸馏头

当前重点是把主线结构固定下来，便于后续逐步替换具体实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry_encoder import create_geometry_encoder


class LocalGraphConv(nn.Module):
    """最小局部图卷积层。"""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return x + out


class LocalRoadSubgraphEncoder(nn.Module):
    """tile 内局部 road subgraph 编码器。"""

    def __init__(self, hidden_dim: int, edge_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([LocalGraphConv(hidden_dim, edge_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class MultiheadAttentionBlock(nn.Module):
    """Set Transformer 的基础注意力块。"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, key, value, need_weights=False)
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class SetTransformerFileEncoder(nn.Module):
    """将 tile embeddings 聚合为 file embedding。"""

    def __init__(self, dim: int, num_heads: int = 4, num_seeds: int = 1, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.sab_layers = nn.ModuleList(
            [MultiheadAttentionBlock(dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, dim))
        self.pma = MultiheadAttentionBlock(dim, num_heads, dropout=dropout)
        self.output_proj = nn.Linear(dim * num_seeds, dim)

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        x = tile_embeddings
        for sab in self.sab_layers:
            x = sab(x, x, x)

        batch_size = x.shape[0]
        seeds = self.seed_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        pooled = self.pma(seeds, x, x)
        return self.output_proj(pooled.reshape(batch_size, -1))


class TileDistillHead(nn.Module):
    """tile-level 蒸馏头。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class FileDistillHead(nn.Module):
    """file-level 蒸馏头。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


@dataclass
class HierarchicalSignatureOutput:
    """统一输出结构。"""

    tile_node_embeddings: Optional[torch.Tensor]
    tile_embeddings: torch.Tensor
    file_embedding: torch.Tensor
    tile_teacher_projection: Optional[torch.Tensor]
    file_teacher_projection: Optional[torch.Tensor]


class HierarchicalRoadSignatureModel(nn.Module):
    """
    roads / line-only / hierarchical signature 主模型。

    输入级别：
    - chunk
    - tile
    - file
    """

    def __init__(
        self,
        geometry_encoder_config: Dict,
        chunk_hidden_dim: int = 256,
        edge_dim: int = 2,
        graph_layers: int = 2,
        set_num_heads: int = 4,
        set_num_layers: int = 2,
        teacher_dim: int = 512,
    ):
        super().__init__()
        self.chunk_encoder = create_geometry_encoder(geometry_encoder_config)
        chunk_dim = geometry_encoder_config["embed_dim"]

        self.node_proj = nn.Linear(chunk_dim + 5, chunk_hidden_dim)
        self.local_graph = LocalRoadSubgraphEncoder(chunk_hidden_dim, edge_dim, num_layers=graph_layers)
        self.tile_pool = nn.Sequential(
            nn.Linear(chunk_hidden_dim, chunk_hidden_dim),
            nn.ReLU(),
            nn.Linear(chunk_hidden_dim, chunk_hidden_dim),
        )
        self.file_encoder = SetTransformerFileEncoder(
            dim=chunk_hidden_dim,
            num_heads=set_num_heads,
            num_layers=set_num_layers,
        )
        self.tile_distill_head = TileDistillHead(chunk_hidden_dim, teacher_dim)
        self.file_distill_head = FileDistillHead(chunk_hidden_dim, teacher_dim)

    def encode_tile(
        self,
        chunk_coords: torch.Tensor,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_geom = self.chunk_encoder(chunk_coords)
        node_input = torch.cat([chunk_geom, chunk_features], dim=-1)
        node_state = self.node_proj(node_input)
        node_state = self.local_graph(node_state, edge_index, edge_attr)
        tile_embedding = self.tile_pool(node_state.mean(dim=0))
        return node_state, tile_embedding

    def forward(
        self,
        tile_chunk_coords: torch.Tensor,
        tile_chunk_features: torch.Tensor,
        tile_edge_index: torch.Tensor,
        tile_edge_attr: torch.Tensor,
        tile_to_file_index: Optional[torch.Tensor] = None,
    ) -> HierarchicalSignatureOutput:
        """
        当前最小前向约定：

        - tile_chunk_coords: [num_tiles, num_chunks, num_points, 2]
        - tile_chunk_features: [num_tiles, num_chunks, 5]
        - tile_edge_index: [num_tiles, 2, num_edges]
        - tile_edge_attr: [num_tiles, num_edges, edge_dim]
        """
        tile_embeddings = []
        tile_node_embeddings = []

        num_tiles = tile_chunk_coords.shape[0]
        for tile_idx in range(num_tiles):
            node_state, tile_embedding = self.encode_tile(
                tile_chunk_coords[tile_idx],
                tile_chunk_features[tile_idx],
                tile_edge_index[tile_idx],
                tile_edge_attr[tile_idx],
            )
            tile_node_embeddings.append(node_state)
            tile_embeddings.append(tile_embedding)

        tile_embeddings_tensor = torch.stack(tile_embeddings, dim=0).unsqueeze(0)
        file_embedding = self.file_encoder(tile_embeddings_tensor)

        tile_teacher_projection = self.tile_distill_head(torch.stack(tile_embeddings, dim=0))
        file_teacher_projection = self.file_distill_head(file_embedding)

        return HierarchicalSignatureOutput(
            tile_node_embeddings=torch.stack(tile_node_embeddings, dim=0),
            tile_embeddings=torch.stack(tile_embeddings, dim=0),
            file_embedding=file_embedding,
            tile_teacher_projection=tile_teacher_projection,
            file_teacher_projection=file_teacher_projection,
        )
