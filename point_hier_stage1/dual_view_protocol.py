"""Fixed-budget dual-view utilities for point-family files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def infer_point_subtype(file_path: str) -> str:
    normalized = str(file_path).lower()
    if "poi" in normalized:
        return "poi"
    if "facility" in normalized:
        return "facility"
    if "station" in normalized:
        return "station"
    return "point"


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


def _sample_point_rows(
    tile: Dict[str, Any],
    k_points: int,
    rng: np.random.Generator,
    feature_noise_std: float,
) -> tuple[np.ndarray, List[str]]:
    point_features = np.asarray(tile["point_features"], dtype=np.float32)
    point_meta = np.asarray(tile["point_meta"], dtype=object)
    if point_features.ndim != 2 or point_features.shape[0] == 0:
        raise ValueError(f"Tile {tile['tile_id']} has no point features.")

    replace = point_features.shape[0] < k_points
    indices = rng.choice(point_features.shape[0], size=k_points, replace=replace)
    sampled = np.asarray(point_features[indices], dtype=np.float32)
    if feature_noise_std > 0:
        noise = rng.normal(0.0, feature_noise_std, size=sampled.shape).astype(np.float32)
        sampled = sampled + noise

    point_ids = [str(point_meta[int(i)][0]) for i in indices]
    return sampled, point_ids


@dataclass
class PointDualView:
    tile_feature_list: List[torch.Tensor]
    tile_ids: List[str]
    point_ids: List[str]
    tile_depths: List[int]


def sample_fixed_budget_view(
    sample: Dict[str, Any],
    *,
    n_tiles: int,
    k_points: int,
    depth_template: Dict[int, int],
    rng: np.random.Generator,
    feature_noise_std: float,
    device: torch.device | None = None,
) -> PointDualView:
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
    point_ids: List[str] = []
    tile_depths: List[int] = []

    for target_depth in depth_slots:
        tile = _sample_tile_for_depth(tiles_by_depth, used_indices, target_depth, rng)
        sampled_features, sampled_point_ids = _sample_point_rows(tile, k_points, rng, feature_noise_std)
        tensor = torch.from_numpy(sampled_features)
        if device is not None:
            tensor = tensor.to(device)
        tile_feature_list.append(tensor)
        tile_ids.append(str(tile["tile_id"]))
        tile_depths.append(int(tile.get("depth", -1)))
        point_ids.extend(sampled_point_ids)

    return PointDualView(
        tile_feature_list=tile_feature_list,
        tile_ids=tile_ids,
        point_ids=point_ids,
        tile_depths=tile_depths,
    )


def build_fixed_budget_point_views(
    sample: Dict[str, Any],
    *,
    n_tiles: int,
    k_points: int,
    depth_template: Dict[int, int],
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    feature_noise_std: float,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    view_a = sample_fixed_budget_view(
        sample,
        n_tiles=n_tiles,
        k_points=k_points,
        depth_template=depth_template,
        rng=rng_a,
        feature_noise_std=feature_noise_std,
        device=device,
    )
    view_b = sample_fixed_budget_view(
        sample,
        n_tiles=n_tiles,
        k_points=k_points,
        depth_template=depth_template,
        rng=rng_b,
        feature_noise_std=feature_noise_std,
        device=device,
    )

    tile_set_a = set(view_a.tile_ids)
    tile_set_b = set(view_b.tile_ids)
    point_set_a = set(view_a.point_ids)
    point_set_b = set(view_b.point_ids)

    tile_overlap = float(len(tile_set_a & tile_set_b) / max(len(tile_set_a | tile_set_b), 1))
    point_overlap = float(len(point_set_a & point_set_b) / max(len(point_set_a | point_set_b), 1))

    return {
        "view_a": view_a,
        "view_b": view_b,
        "tile_overlap_ratio": tile_overlap,
        "point_overlap_ratio": point_overlap,
    }


def visible_point_stats_from_tile_features(
    tile_feature_list: Sequence[np.ndarray | torch.Tensor],
) -> tuple[List[str], np.ndarray]:
    if not tile_feature_list:
        raise ValueError("tile_feature_list is empty.")

    point_arrays: List[np.ndarray] = []
    for tile in tile_feature_list:
        if isinstance(tile, torch.Tensor):
            point_arrays.append(tile.detach().cpu().numpy().astype(np.float32, copy=False))
        else:
            point_arrays.append(np.asarray(tile, dtype=np.float32))
    point_matrix = np.concatenate(point_arrays, axis=0)

    names: List[str] = []
    values: List[float] = []
    for dim in range(int(point_matrix.shape[1])):
        arr = point_matrix[:, dim].astype(np.float64)
        names.extend([f"dim{dim}_mean", f"dim{dim}_std", f"dim{dim}_p95"])
        values.extend([float(np.mean(arr)), float(np.std(arr)), float(np.percentile(arr, 95))])
    return names, np.asarray(values, dtype=np.float64)


def visible_point_local_structure_stats_from_tile_features(
    tile_feature_list: Sequence[np.ndarray | torch.Tensor],
    *,
    grid_size: int = 4,
    radial_bins: int = 4,
    nn_bins: Sequence[float] = (0.0, 0.025, 0.05, 0.1, 0.2, 1.0),
) -> tuple[List[str], np.ndarray]:
    """View-level spatial structure features derived from sampled point coordinates."""

    if not tile_feature_list:
        raise ValueError("tile_feature_list is empty.")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}.")
    if radial_bins <= 0:
        raise ValueError(f"radial_bins must be positive, got {radial_bins}.")

    tile_arrays: List[np.ndarray] = []
    for tile in tile_feature_list:
        if isinstance(tile, torch.Tensor):
            arr = tile.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(tile, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
            continue
        tile_arrays.append(arr)

    if not tile_arrays:
        raise ValueError("No non-empty point tiles for local structure stats.")

    grid_vectors: List[np.ndarray] = []
    radial_values: List[np.ndarray] = []
    nn_values: List[np.ndarray] = []
    for arr in tile_arrays:
        coords = np.clip(np.asarray(arr[:, :2], dtype=np.float64), 0.0, 1.0)
        grid_x = np.minimum((coords[:, 0] * grid_size).astype(np.int64), grid_size - 1)
        grid_y = np.minimum((coords[:, 1] * grid_size).astype(np.int64), grid_size - 1)
        flat_idx = grid_y * grid_size + grid_x
        counts = np.bincount(flat_idx, minlength=grid_size * grid_size).astype(np.float64)
        grid_vectors.append(counts / max(float(coords.shape[0]), 1.0))

        radial = np.linalg.norm(coords - 0.5, axis=1) / np.sqrt(0.5)
        radial_values.append(np.clip(radial, 0.0, 1.0))

        if coords.shape[0] > 1:
            deltas = coords[:, None, :] - coords[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            np.fill_diagonal(distances, np.inf)
            nn_values.append(np.min(distances, axis=1))
        else:
            nn_values.append(np.asarray([1.0], dtype=np.float64))

    grid_matrix = np.vstack(grid_vectors)
    all_coords = np.vstack([np.clip(arr[:, :2], 0.0, 1.0) for arr in tile_arrays])
    all_grid_x = np.minimum((all_coords[:, 0] * grid_size).astype(np.int64), grid_size - 1)
    all_grid_y = np.minimum((all_coords[:, 1] * grid_size).astype(np.int64), grid_size - 1)
    all_flat_idx = all_grid_y * grid_size + all_grid_x
    global_grid = np.bincount(all_flat_idx, minlength=grid_size * grid_size).astype(np.float64)
    global_grid = global_grid / max(float(all_coords.shape[0]), 1.0)

    radial_all = np.concatenate(radial_values)
    radial_hist, _ = np.histogram(radial_all, bins=radial_bins, range=(0.0, 1.0), density=False)
    radial_hist = radial_hist.astype(np.float64) / max(float(radial_all.shape[0]), 1.0)

    nn_all = np.concatenate(nn_values)
    nn_edges = np.asarray(nn_bins, dtype=np.float64)
    nn_hist, _ = np.histogram(np.clip(nn_all, nn_edges[0], nn_edges[-1]), bins=nn_edges, density=False)
    nn_hist = nn_hist.astype(np.float64) / max(float(nn_all.shape[0]), 1.0)

    names: List[str] = []
    values: List[float] = []
    for idx, value in enumerate(global_grid):
        names.append(f"local_grid_global_{idx}")
        values.append(float(value))
    for idx, value in enumerate(np.mean(grid_matrix, axis=0)):
        names.append(f"local_grid_tile_mean_{idx}")
        values.append(float(value))
    for idx, value in enumerate(np.std(grid_matrix, axis=0)):
        names.append(f"local_grid_tile_std_{idx}")
        values.append(float(value))
    for idx, value in enumerate(radial_hist):
        names.append(f"local_radial_hist_{idx}")
        values.append(float(value))
    for idx, value in enumerate(nn_hist):
        names.append(f"local_nn_hist_{idx}")
        values.append(float(value))
    for stat_name, value in (
        ("local_nn_mean", np.mean(nn_all)),
        ("local_nn_std", np.std(nn_all)),
        ("local_nn_p50", np.percentile(nn_all, 50)),
        ("local_nn_p95", np.percentile(nn_all, 95)),
    ):
        names.append(stat_name)
        values.append(float(value))

    return names, np.asarray(values, dtype=np.float64)


def visible_point_hybrid_stats_from_tile_features(
    tile_feature_list: Sequence[np.ndarray | torch.Tensor],
    *,
    include_local_structure: bool = False,
    grid_size: int = 4,
    radial_bins: int = 4,
) -> tuple[List[str], np.ndarray]:
    names, values = visible_point_stats_from_tile_features(tile_feature_list)
    if not include_local_structure:
        return names, values

    local_names, local_values = visible_point_local_structure_stats_from_tile_features(
        tile_feature_list,
        grid_size=grid_size,
        radial_bins=radial_bins,
    )
    return names + local_names, np.concatenate([values, local_values]).astype(np.float64)
