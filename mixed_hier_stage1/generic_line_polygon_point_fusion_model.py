"""Generic line + polygon + optional point mixed encoder.

Unlike the subtype-aware prototype, this encoder does not use subtype
embeddings inside each geometry family. It pools all available subtypes as a
generic family set and can skip point entirely when point data is unavailable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from mixed_hier_stage1.line_polygon_fusion_model import TransformerSetBlock
from point_hier_stage1.dual_view_protocol import visible_point_hybrid_stats_from_tile_features
from point_hier_stage1.point_signature_model import PointHybridSignatureEncoder
from polygon_hier_stage1.polygon_signature_model import PolygonHierarchicalEncoder
from roads_hier_stage1.enhanced_signature_model import EnhancedRoadsHierarchicalEncoder


class GenericSetAggregator(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerSetBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("embeddings is empty")
        x = torch.stack(list(embeddings), dim=0).unsqueeze(0)
        x = torch.cat([self.cls.expand(1, -1, -1), x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x[0, 0])


class OptionalModalityAggregator(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.modality_embed = nn.Embedding(3, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerSetBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, embeddings: Sequence[torch.Tensor], modality_ids: Sequence[int]) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        for emb, modality_id in zip(embeddings, modality_ids):
            type_index = torch.tensor(int(modality_id), dtype=torch.long, device=emb.device)
            rows.append(emb + self.modality_embed(type_index))
        if not rows:
            raise ValueError("embeddings is empty")
        x = torch.stack(rows, dim=0).unsqueeze(0)
        x = torch.cat([self.cls.expand(1, -1, -1), x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x[0, 0])


class MixedGenericLinePolygonPointFusionEncoder(nn.Module):
    def __init__(
        self,
        *,
        line_chunk_feature_dim: int,
        polygon_feature_dim: int,
        point_feature_dim: int,
        point_visible_feature_dim: int,
        line_hidden_dim: int = 192,
        line_file_dim: int = 192,
        polygon_hidden_dim: int = 192,
        polygon_file_dim: int = 192,
        point_hidden_dim: int = 192,
        point_file_dim: int = 192,
        point_visible_hidden_dim: int = 160,
        point_visible_dim: int = 128,
        point_fusion_hidden_dim: int = 256,
        point_fusion_dim: int = 192,
        line_n_tiles: int = 8,
        polygon_n_tiles: int = 8,
        point_n_tiles: int = 8,
        line_max_depth: int = 8,
        polygon_max_depth: int = 8,
        point_max_depth: int = 8,
        polygon_tile_encoder_type: str = "set_transformer",
        point_tile_encoder_type: str = "set_transformer",
        line_chunk_num_heads: int = 4,
        line_file_num_heads: int = 4,
        line_num_chunk_layers: int = 2,
        line_num_file_layers: int = 3,
        polygon_tile_num_heads: int = 4,
        polygon_num_tile_layers: int = 2,
        polygon_file_num_heads: int = 4,
        polygon_num_file_layers: int = 3,
        point_tile_num_heads: int = 4,
        point_num_tile_layers: int = 2,
        point_file_num_heads: int = 4,
        point_num_file_layers: int = 3,
        branch_num_heads: int = 4,
        branch_num_layers: int = 2,
        modality_num_heads: int = 4,
        modality_num_layers: int = 2,
        fusion_dim: int = 192,
        dropout: float = 0.1,
        point_include_local_structure: bool = True,
        point_local_grid_size: int = 4,
        point_local_radial_bins: int = 4,
    ):
        super().__init__()
        self.point_include_local_structure = bool(point_include_local_structure)
        self.point_local_grid_size = int(point_local_grid_size)
        self.point_local_radial_bins = int(point_local_radial_bins)

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
        self.point_encoder = PointHybridSignatureEncoder(
            point_feature_dim=int(point_feature_dim),
            visible_feature_dim=int(point_visible_feature_dim),
            hidden_dim=int(point_hidden_dim),
            file_dim=int(point_file_dim),
            visible_hidden_dim=int(point_visible_hidden_dim),
            visible_dim=int(point_visible_dim),
            fusion_hidden_dim=int(point_fusion_hidden_dim),
            fusion_dim=int(point_fusion_dim),
            n_tiles=int(point_n_tiles),
            max_depth=int(point_max_depth),
            tile_encoder_type=str(point_tile_encoder_type),
            tile_num_heads=int(point_tile_num_heads),
            num_tile_layers=int(point_num_tile_layers),
            file_num_heads=int(point_file_num_heads),
            num_file_layers=int(point_num_file_layers),
            dropout=float(dropout),
        )

        self.line_branch = GenericSetAggregator(int(line_file_dim), branch_num_heads, branch_num_layers, dropout)
        self.polygon_branch = GenericSetAggregator(int(polygon_file_dim), branch_num_heads, branch_num_layers, dropout)
        self.point_branch = GenericSetAggregator(int(point_fusion_dim), branch_num_heads, branch_num_layers, dropout)
        self.line_proj = nn.Sequential(nn.Linear(int(line_file_dim), int(fusion_dim)), nn.LayerNorm(int(fusion_dim)), nn.GELU())
        self.polygon_proj = nn.Sequential(nn.Linear(int(polygon_file_dim), int(fusion_dim)), nn.LayerNorm(int(fusion_dim)), nn.GELU())
        self.point_proj = nn.Sequential(nn.Linear(int(point_fusion_dim), int(fusion_dim)), nn.LayerNorm(int(fusion_dim)), nn.GELU())
        self.modality_fusion = OptionalModalityAggregator(int(fusion_dim), modality_num_heads, modality_num_layers, dropout)

        self.register_buffer("point_feature_mean", torch.zeros(int(point_visible_feature_dim), dtype=torch.float32))
        self.register_buffer("point_feature_std", torch.ones(int(point_visible_feature_dim), dtype=torch.float32))

    def set_point_feature_stats(self, mean: torch.Tensor | np.ndarray, std: torch.Tensor | np.ndarray) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=self.point_feature_mean.device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=self.point_feature_std.device)
        if tuple(mean_tensor.shape) != tuple(self.point_feature_mean.shape):
            raise ValueError(f"point mean shape mismatch: {tuple(mean_tensor.shape)} vs {tuple(self.point_feature_mean.shape)}")
        if tuple(std_tensor.shape) != tuple(self.point_feature_std.shape):
            raise ValueError(f"point std shape mismatch: {tuple(std_tensor.shape)} vs {tuple(self.point_feature_std.shape)}")
        self.point_feature_mean.copy_(mean_tensor)
        self.point_feature_std.copy_(std_tensor)

    def encode_line_package(self, line_view_map: Dict[str, object]) -> torch.Tensor:
        embeddings = [
            self.line_encoder(dual_view.tile_feature_list, dual_view.tile_depths)
            for _, dual_view in sorted(line_view_map.items())
        ]
        return self.line_branch(embeddings)

    def encode_polygon_package(self, polygon_view_map: Dict[str, object]) -> torch.Tensor:
        embeddings = [
            self.polygon_encoder(dual_view.tile_feature_list, dual_view.tile_depths)
            for _, dual_view in sorted(polygon_view_map.items())
        ]
        return self.polygon_branch(embeddings)

    def _normalized_point_stats(self, dual_view: object) -> torch.Tensor:
        _, stats = visible_point_hybrid_stats_from_tile_features(
            dual_view.tile_feature_list,
            include_local_structure=self.point_include_local_structure,
            grid_size=self.point_local_grid_size,
            radial_bins=self.point_local_radial_bins,
        )
        stats_tensor = torch.as_tensor(stats, dtype=torch.float32, device=self.point_feature_mean.device)
        return (stats_tensor - self.point_feature_mean) / self.point_feature_std

    def encode_point_package(self, point_view_map: Dict[str, object]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        for _, dual_view in sorted(point_view_map.items()):
            visible_stats = self._normalized_point_stats(dual_view)
            embeddings.append(self.point_encoder(dual_view.tile_feature_list, dual_view.tile_depths, visible_stats))
        return self.point_branch(embeddings)

    def forward(
        self,
        line_view_map: Dict[str, object],
        polygon_view_map: Dict[str, object],
        point_view_map: Dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings: List[torch.Tensor] = [
            self.line_proj(self.encode_line_package(line_view_map)),
            self.polygon_proj(self.encode_polygon_package(polygon_view_map)),
        ]
        modality_ids: List[int] = [0, 1]
        if point_view_map:
            embeddings.append(self.point_proj(self.encode_point_package(point_view_map)))
            modality_ids.append(2)
        fused = self.modality_fusion(embeddings, modality_ids)
        return fused, torch.stack(embeddings, dim=0)
