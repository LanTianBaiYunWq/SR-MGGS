"""Fixed-budget dual-view utilities for polygon-family files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def infer_polygon_subtype(file_path: str) -> str:
    normalized = Path(str(file_path)).name.lower()
    if "building" in normalized:
        return "building"
    if "landuse" in normalized:
        return "landuse"
    if "natural" in normalized:
        return "natural"
    if "water" in normalized:
        return "water"
    return "polygon"


def build_depth_slots(depth_template: Dict[int, int], n_tiles: int) -> List[int]:
    slots: List[int] = []
    for depth in sorted(int(key) for key in depth_template.keys()):
        slots.extend([depth] * int(depth_template[depth]))
    if len(slots) != n_tiles:
        raise ValueError(f"Depth template expands to {len(slots)} slots, expected n_tiles={n_tiles}.")
    return slots


def _choose_source_depth(available_depths: Sequence[int], target_depth: int) -> int:
    if not available_depths:
        raise ValueError("No available depths to sample from.")
    return min(available_depths, key=lambda depth: (abs(int(depth) - int(target_depth)), int(depth)))


def _sample_tile_for_depth(
    tiles_by_depth: Dict[int, List[Dict[str, Any]]],
    used_indices: Dict[int, set[int]],
    target_depth: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    source_depth = _choose_source_depth(list(tiles_by_depth.keys()), target_depth)
    candidates = tiles_by_depth[source_depth]
    used = used_indices.setdefault(source_depth, set())
    unused = [idx for idx in range(len(candidates)) if idx not in used]
    if unused:
        idx = int(rng.choice(np.asarray(unused, dtype=np.int64)))
        used.add(idx)
        return candidates[idx]
    return candidates[int(rng.integers(0, len(candidates)))]


def _sample_polygon_rows(
    tile: Dict[str, Any],
    k_polygons: int,
    rng: np.random.Generator,
    feature_noise_std: float,
) -> tuple[np.ndarray, List[str]]:
    polygon_features = np.asarray(tile["polygon_features"], dtype=np.float32)
    polygon_meta = np.asarray(tile["polygon_meta"], dtype=object)
    if polygon_features.ndim != 2 or polygon_features.shape[0] == 0:
        raise ValueError(f"Tile {tile['tile_id']} has no polygon features.")

    replace = polygon_features.shape[0] < k_polygons
    indices = rng.choice(polygon_features.shape[0], size=k_polygons, replace=replace)
    sampled = np.asarray(polygon_features[indices], dtype=np.float32)
    if feature_noise_std > 0:
        sampled = sampled + rng.normal(0.0, feature_noise_std, size=sampled.shape).astype(np.float32)

    polygon_ids = [str(polygon_meta[int(i)][0]) for i in indices]
    return sampled, polygon_ids


@dataclass
class PolygonDualView:
    tile_feature_list: List[torch.Tensor]
    tile_ids: List[str]
    polygon_ids: List[str]
    tile_depths: List[int]


def sample_fixed_budget_view(
    sample: Dict[str, Any],
    *,
    n_tiles: int,
    k_polygons: int,
    depth_template: Dict[int, int],
    rng: np.random.Generator,
    feature_noise_std: float,
    device: torch.device | None = None,
) -> PolygonDualView:
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
    polygon_ids: List[str] = []
    tile_depths: List[int] = []

    for target_depth in depth_slots:
        tile = _sample_tile_for_depth(tiles_by_depth, used_indices, target_depth, rng)
        sampled_features, sampled_polygon_ids = _sample_polygon_rows(tile, k_polygons, rng, feature_noise_std)
        tensor = torch.from_numpy(sampled_features)
        if device is not None:
            tensor = tensor.to(device)
        tile_feature_list.append(tensor)
        tile_ids.append(str(tile["tile_id"]))
        tile_depths.append(int(tile.get("depth", -1)))
        polygon_ids.extend(sampled_polygon_ids)

    return PolygonDualView(
        tile_feature_list=tile_feature_list,
        tile_ids=tile_ids,
        polygon_ids=polygon_ids,
        tile_depths=tile_depths,
    )


def build_fixed_budget_polygon_views(
    sample: Dict[str, Any],
    *,
    n_tiles: int,
    k_polygons: int,
    depth_template: Dict[int, int],
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    feature_noise_std: float,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    view_a = sample_fixed_budget_view(
        sample,
        n_tiles=n_tiles,
        k_polygons=k_polygons,
        depth_template=depth_template,
        rng=rng_a,
        feature_noise_std=feature_noise_std,
        device=device,
    )
    view_b = sample_fixed_budget_view(
        sample,
        n_tiles=n_tiles,
        k_polygons=k_polygons,
        depth_template=depth_template,
        rng=rng_b,
        feature_noise_std=feature_noise_std,
        device=device,
    )

    tile_set_a = set(view_a.tile_ids)
    tile_set_b = set(view_b.tile_ids)
    poly_set_a = set(view_a.polygon_ids)
    poly_set_b = set(view_b.polygon_ids)

    tile_overlap = float(len(tile_set_a & tile_set_b) / max(len(tile_set_a | tile_set_b), 1))
    polygon_overlap = float(len(poly_set_a & poly_set_b) / max(len(poly_set_a | poly_set_b), 1))

    return {
        "view_a": view_a,
        "view_b": view_b,
        "tile_overlap_ratio": tile_overlap,
        "polygon_overlap_ratio": polygon_overlap,
    }


def visible_polygon_stats_from_tile_features(
    tile_feature_list: Sequence[np.ndarray | torch.Tensor],
) -> tuple[List[str], np.ndarray]:
    if not tile_feature_list:
        raise ValueError("tile_feature_list is empty.")

    polygon_arrays: List[np.ndarray] = []
    for tile in tile_feature_list:
        if isinstance(tile, torch.Tensor):
            polygon_arrays.append(tile.detach().cpu().numpy().astype(np.float32, copy=False))
        else:
            polygon_arrays.append(np.asarray(tile, dtype=np.float32))
    polygon_matrix = np.concatenate(polygon_arrays, axis=0)

    names: List[str] = []
    values: List[float] = []
    for dim in range(int(polygon_matrix.shape[1])):
        arr = polygon_matrix[:, dim].astype(np.float64)
        names.extend([f"dim{dim}_mean", f"dim{dim}_std", f"dim{dim}_p95"])
        values.extend([float(np.mean(arr)), float(np.std(arr)), float(np.percentile(arr, 95))])
    return names, np.asarray(values, dtype=np.float64)
