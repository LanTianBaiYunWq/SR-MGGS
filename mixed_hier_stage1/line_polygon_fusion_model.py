"""Line + polygon mixed V1 gated fusion encoder."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from polygon_hier_stage1.polygon_signature_model import PolygonHierarchicalEncoder
from roads_hier_stage1.enhanced_signature_model import EnhancedRoadsHierarchicalEncoder


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


class BranchAggregator(nn.Module):
    def __init__(self, dim: int, num_types: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerSetBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, embeddings: Sequence[torch.Tensor], type_ids: Sequence[int]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("embeddings is empty")
        rows: List[torch.Tensor] = []
        for emb, type_id in zip(embeddings, type_ids):
            type_index = torch.tensor(int(type_id), dtype=torch.long, device=emb.device)
            rows.append(emb + self.type_embed(type_index))
        x = torch.stack(rows, dim=0).unsqueeze(0)
        cls = self.cls.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x[0, 0])


class MixedLinePolygonFusionEncoder(nn.Module):
    def __init__(
        self,
        *,
        line_chunk_feature_dim: int,
        polygon_feature_dim: int,
        line_hidden_dim: int = 192,
        line_file_dim: int = 192,
        polygon_hidden_dim: int = 192,
        polygon_file_dim: int = 192,
        line_n_tiles: int = 8,
        polygon_n_tiles: int = 8,
        line_max_depth: int = 8,
        polygon_max_depth: int = 8,
        polygon_tile_encoder_type: str = "set_transformer",
        line_chunk_num_heads: int = 4,
        line_file_num_heads: int = 4,
        line_num_chunk_layers: int = 2,
        line_num_file_layers: int = 3,
        polygon_tile_num_heads: int = 4,
        polygon_num_tile_layers: int = 2,
        polygon_file_num_heads: int = 4,
        polygon_num_file_layers: int = 3,
        branch_num_heads: int = 4,
        branch_num_layers: int = 2,
        fusion_dim: int = 192,
        dropout: float = 0.1,
        line_subtypes: Sequence[str] = ("roads", "railways", "waterways"),
        polygon_subtypes: Sequence[str] = ("building", "landuse", "natural", "water"),
    ):
        super().__init__()
        self.line_subtypes = tuple(str(value) for value in line_subtypes)
        self.polygon_subtypes = tuple(str(value) for value in polygon_subtypes)
        self.line_type_to_id = {name: idx for idx, name in enumerate(self.line_subtypes)}
        self.polygon_type_to_id = {name: idx for idx, name in enumerate(self.polygon_subtypes)}

        self.line_encoder = EnhancedRoadsHierarchicalEncoder(
            chunk_feature_dim=int(line_chunk_feature_dim),
            hidden_dim=int(line_hidden_dim),
            file_dim=int(line_file_dim),
            n_tiles=int(line_n_tiles),
            max_depth=int(line_max_depth),
            chunk_num_heads=int(line_chunk_num_heads),
            file_num_heads=int(line_file_num_heads),
            num_chunk_layers=int(line_num_chunk_layers),
            num_file_layers=int(line_num_file_layers),
            dropout=float(dropout),
        )
        self.polygon_encoder = PolygonHierarchicalEncoder(
            polygon_feature_dim=int(polygon_feature_dim),
            hidden_dim=int(polygon_hidden_dim),
            file_dim=int(polygon_file_dim),
            n_tiles=int(polygon_n_tiles),
            max_depth=int(polygon_max_depth),
            tile_encoder_type=str(polygon_tile_encoder_type),
            tile_num_heads=int(polygon_tile_num_heads),
            num_tile_layers=int(polygon_num_tile_layers),
            file_num_heads=int(polygon_file_num_heads),
            num_file_layers=int(polygon_num_file_layers),
            dropout=float(dropout),
        )

        self.line_branch = BranchAggregator(
            dim=int(line_file_dim),
            num_types=len(self.line_subtypes),
            num_heads=int(branch_num_heads),
            num_layers=int(branch_num_layers),
            dropout=float(dropout),
        )
        self.polygon_branch = BranchAggregator(
            dim=int(polygon_file_dim),
            num_types=len(self.polygon_subtypes),
            num_heads=int(branch_num_heads),
            num_layers=int(branch_num_layers),
            dropout=float(dropout),
        )
        self.line_proj = nn.Sequential(
            nn.Linear(int(line_file_dim), int(fusion_dim)),
            nn.LayerNorm(int(fusion_dim)),
            nn.GELU(),
        )
        self.polygon_proj = nn.Sequential(
            nn.Linear(int(polygon_file_dim), int(fusion_dim)),
            nn.LayerNorm(int(fusion_dim)),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(int(fusion_dim) * 2, int(fusion_dim)),
            nn.GELU(),
            nn.Linear(int(fusion_dim), int(fusion_dim)),
            nn.Sigmoid(),
        )
        self.final_norm = nn.LayerNorm(int(fusion_dim))

    def encode_line_package(self, line_view_map: Dict[str, object]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        type_ids: List[int] = []
        for subtype in self.line_subtypes:
            if subtype not in line_view_map:
                continue
            dual_view = line_view_map[subtype]
            emb = self.line_encoder(dual_view.tile_feature_list, dual_view.tile_depths)
            embeddings.append(emb)
            type_ids.append(self.line_type_to_id[subtype])
        return self.line_branch(embeddings, type_ids)

    def encode_polygon_package(self, polygon_view_map: Dict[str, object]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        type_ids: List[int] = []
        for subtype in self.polygon_subtypes:
            if subtype not in polygon_view_map:
                continue
            dual_view = polygon_view_map[subtype]
            emb = self.polygon_encoder(dual_view.tile_feature_list, dual_view.tile_depths)
            embeddings.append(emb)
            type_ids.append(self.polygon_type_to_id[subtype])
        return self.polygon_branch(embeddings, type_ids)

    def forward(self, line_view_map: Dict[str, object], polygon_view_map: Dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        line_repr = self.line_proj(self.encode_line_package(line_view_map))
        polygon_repr = self.polygon_proj(self.encode_polygon_package(polygon_view_map))
        gate = self.gate(torch.cat([line_repr, polygon_repr], dim=-1))
        fused = gate * line_repr + (1.0 - gate) * polygon_repr
        return self.final_norm(fused), gate
