"""
Fixed-budget dual-view protocol utilities for roads_hier_stage1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def infer_line_subtype(file_path: str) -> str:
    """Infer subtype from shapefile path."""
    normalized = str(file_path).lower()
    if "roads" in normalized:
        return "roads"
    if "railways" in normalized:
        return "railways"
    if "waterways" in normalized:
        return "waterways"
    return "unknown"


def build_depth_slots(depth_template: Dict[int, int], n_tiles: int) -> List[int]:
    """Expand a depth->quota template into a fixed ordered slot list."""
    slots: List[int] = []
    for depth in sorted(int(key) for key in depth_template.keys()):
        slots.extend([depth] * int(depth_template[depth]))
    if len(slots) != n_tiles:
        raise ValueError(f"Depth template expands to {len(slots)} slots, expected n_tiles={n_tiles}.")
    return slots


def _choose_source_depth(
    available_depths: Sequence[int],
    target_depth: int,
) -> int:
    """Pick the closest available depth to a requested target depth."""
    if not available_depths:
        raise ValueError("No available depths to sample from.")
    return min(available_depths, key=lambda depth: (abs(int(depth) - int(target_depth)), int(depth)))


def _sample_tile_for_depth(
    tiles_by_depth: Dict[int, List[Dict[str, Any]]],
    used_indices: Dict[int, set[int]],
    target_depth: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample one tile following a fixed depth slot, reusing tiles only if necessary."""
    source_depth = _choose_source_depth(list(tiles_by_depth.keys()), target_depth)
    candidates = tiles_by_depth[source_depth]
    if not candidates:
        raise ValueError(f"No candidates at resolved source depth {source_depth}.")

    used = used_indices.setdefault(source_depth, set())
    unused = [idx for idx in range(len(candidates)) if idx not in used]
    if unused:
        idx = int(rng.choice(np.asarray(unused, dtype=np.int64)))
        used.add(idx)
        return candidates[idx]

    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def _sample_chunk_rows(
    tile: Dict[str, Any],
    k_chunks: int,
    rng: np.random.Generator,
    feature_noise_std: float,
) -> tuple[np.ndarray, List[str]]:
    """Sample a fixed number of chunks from one tile."""
    chunk_features = np.asarray(tile["chunk_features"], dtype=np.float32)
    chunk_meta = np.asarray(tile["chunk_meta"], dtype=object)
    if chunk_features.ndim != 2 or chunk_features.shape[0] == 0:
        raise ValueError(f"Tile {tile['tile_id']} has no chunk features.")

    if chunk_features.shape[0] >= k_chunks:
        indices = rng.choice(chunk_features.shape[0], size=k_chunks, replace=False)
    else:
        indices = rng.choice(chunk_features.shape[0], size=k_chunks, replace=True)

    sampled = np.asarray(chunk_features[indices], dtype=np.float32)
    if feature_noise_std > 0:
        noise = rng.normal(0.0, feature_noise_std, size=sampled.shape).astype(np.float32)
        sampled = sampled + noise

    chunk_ids = [str(chunk_meta[int(i)][0]) for i in indices]
    return sampled, chunk_ids


@dataclass
class DualView:
    """One sampled fixed-budget view."""

    tile_feature_list: List[torch.Tensor]
    tile_ids: List[str]
    chunk_ids: List[str]
    tile_depths: List[int]


def sample_fixed_budget_view(
    sample: Dict[str, Any],
    *,
    n_tiles: int,
    k_chunks: int,
    depth_template: Dict[int, int],
    rng: np.random.Generator,
    feature_noise_std: float,
    device: torch.device | None = None,
) -> DualView:
    """Sample one fixed-budget dual-view protocol instance."""
    tiles = list(sample["tiles"])
    if not tiles:
        raise ValueError(f"File {sample['file_id']} has no readable tiles.")

    depth_slots = build_depth_slots(depth_template, n_tiles)
    tiles_by_depth: Dict[int, List[Dict[str, Any]]] = {}
    for tile in tiles:
        depth = int(tile.get("depth", -1))
        tiles_by_depth.setdefault(depth, []).append(tile)

    used_indices: Dict[int, set[int]] = {}
    tile_feature_list: List[torch.Tensor] = []
    tile_ids: List[str] = []
    chunk_ids: List[str] = []
    tile_depths: List[int] = []

    for target_depth in depth_slots:
        tile = _sample_tile_for_depth(tiles_by_depth, used_indices, target_depth, rng)
        sampled_features, sampled_chunk_ids = _sample_chunk_rows(tile, k_chunks, rng, feature_noise_std)
        tensor = torch.from_numpy(sampled_features)
        if device is not None:
            tensor = tensor.to(device)
        tile_feature_list.append(tensor)
        tile_ids.append(str(tile["tile_id"]))
        tile_depths.append(int(tile.get("depth", -1)))
        chunk_ids.extend(sampled_chunk_ids)

    return DualView(
        tile_feature_list=tile_feature_list,
        tile_ids=tile_ids,
        chunk_ids=chunk_ids,
        tile_depths=tile_depths,
    )


def build_fixed_budget_dual_views(
    sample: Dict[str, Any],
    *,
    n_tiles: int,
    k_chunks: int,
    depth_template: Dict[int, int],
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    feature_noise_std: float,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Sample two independent fixed-budget views from the same file."""
    view_a = sample_fixed_budget_view(
        sample,
        n_tiles=n_tiles,
        k_chunks=k_chunks,
        depth_template=depth_template,
        rng=rng_a,
        feature_noise_std=feature_noise_std,
        device=device,
    )
    view_b = sample_fixed_budget_view(
        sample,
        n_tiles=n_tiles,
        k_chunks=k_chunks,
        depth_template=depth_template,
        rng=rng_b,
        feature_noise_std=feature_noise_std,
        device=device,
    )

    tile_set_a = set(view_a.tile_ids)
    tile_set_b = set(view_b.tile_ids)
    chunk_set_a = set(view_a.chunk_ids)
    chunk_set_b = set(view_b.chunk_ids)

    tile_overlap = float(len(tile_set_a & tile_set_b) / max(len(tile_set_a | tile_set_b), 1))
    chunk_overlap = float(len(chunk_set_a & chunk_set_b) / max(len(chunk_set_a | chunk_set_b), 1))

    return {
        "view_a": view_a,
        "view_b": view_b,
        "tile_overlap_ratio": tile_overlap,
        "chunk_overlap_ratio": chunk_overlap,
    }


def visible_chunk_stats_from_tile_features(
    tile_feature_list: Sequence[np.ndarray | torch.Tensor],
) -> tuple[List[str], np.ndarray]:
    """Build a view-visible coarse baseline from only the visible chunk features."""
    if not tile_feature_list:
        raise ValueError("tile_feature_list is empty.")

    chunk_arrays: List[np.ndarray] = []
    for tile in tile_feature_list:
        if isinstance(tile, torch.Tensor):
            chunk_arrays.append(tile.detach().cpu().numpy().astype(np.float32, copy=False))
        else:
            chunk_arrays.append(np.asarray(tile, dtype=np.float32))
    chunk_matrix = np.concatenate(chunk_arrays, axis=0)
    feature_dim = int(chunk_matrix.shape[1])
    names: List[str] = []
    values: List[float] = []
    for dim in range(feature_dim):
        arr = chunk_matrix[:, dim].astype(np.float64)
        names.extend([f"dim{dim}_mean", f"dim{dim}_std", f"dim{dim}_p95"])
        values.extend([
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.percentile(arr, 95)),
        ])
    return names, np.asarray(values, dtype=np.float64)
