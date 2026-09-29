"""Helpers for extracting point visible coarse features under fixed-budget dual-view."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from point_hier_stage1.dual_view_protocol import (
    build_fixed_budget_point_views,
    infer_point_subtype,
    visible_point_stats_from_tile_features,
)


def sample_visible_dual_view_features(
    sample: Dict[str, Any],
    *,
    n_tiles: int,
    k_points: int,
    depth_template: Dict[int, int],
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    feature_noise_std: float = 0.0,
    device=None,
) -> Dict[str, Any]:
    dual_views = build_fixed_budget_point_views(
        sample,
        n_tiles=n_tiles,
        k_points=k_points,
        depth_template=depth_template,
        rng_a=rng_a,
        rng_b=rng_b,
        feature_noise_std=feature_noise_std,
        device=device,
    )
    feature_names_a, feat_a = visible_point_stats_from_tile_features(dual_views["view_a"].tile_feature_list)
    feature_names_b, feat_b = visible_point_stats_from_tile_features(dual_views["view_b"].tile_feature_list)
    if feature_names_a != feature_names_b:
        raise ValueError("Visible point feature names differ across the two views.")

    return {
        "file_id": str(sample["file_id"]),
        "file_path": str(sample["file_path"]),
        "subtype": infer_point_subtype(sample["file_path"]),
        "feature_names": feature_names_a,
        "feat_a": feat_a,
        "feat_b": feat_b,
        "tile_overlap_ratio": float(dual_views["tile_overlap_ratio"]),
        "point_overlap_ratio": float(dual_views["point_overlap_ratio"]),
    }
