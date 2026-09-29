"""Partial-package dual-view protocol for line + polygon + point mixed prototype."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from mixed_hier_stage1.line_polygon_protocol import _sample_subtypes
from point_hier_stage1.dual_view_protocol import build_fixed_budget_point_views
from polygon_hier_stage1.dual_view_protocol import build_fixed_budget_polygon_views
from roads_hier_stage1.dual_view_protocol import build_fixed_budget_dual_views


def build_partial_line_polygon_point_views(
    package_sample: Dict[str, object],
    *,
    line_subtypes: Sequence[str],
    polygon_subtypes: Sequence[str],
    point_subtypes: Sequence[str],
    line_subtypes_per_view: int | None,
    polygon_subtypes_per_view: int | None,
    point_subtypes_per_view: int | None,
    line_n_tiles: int,
    line_k_chunks: int,
    line_depth_template: Dict[int, int],
    polygon_n_tiles: int,
    polygon_k_polygons: int,
    polygon_depth_template: Dict[int, int],
    point_n_tiles: int,
    point_k_points: int,
    point_depth_template: Dict[int, int],
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    feature_noise_std: float,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    line_selected_a = _sample_subtypes(line_subtypes, line_subtypes_per_view, rng_a)
    line_selected_b = _sample_subtypes(line_subtypes, line_subtypes_per_view, rng_b)
    polygon_selected_a = _sample_subtypes(polygon_subtypes, polygon_subtypes_per_view, rng_a)
    polygon_selected_b = _sample_subtypes(polygon_subtypes, polygon_subtypes_per_view, rng_b)
    point_selected_a = _sample_subtypes(point_subtypes, point_subtypes_per_view, rng_a)
    point_selected_b = _sample_subtypes(point_subtypes, point_subtypes_per_view, rng_b)

    line_view_a: Dict[str, object] = {}
    line_view_b: Dict[str, object] = {}
    polygon_view_a: Dict[str, object] = {}
    polygon_view_b: Dict[str, object] = {}
    point_view_a: Dict[str, object] = {}
    point_view_b: Dict[str, object] = {}

    line_tile_overlaps: List[float] = []
    line_chunk_overlaps: List[float] = []
    polygon_tile_overlaps: List[float] = []
    polygon_object_overlaps: List[float] = []
    point_tile_overlaps: List[float] = []
    point_object_overlaps: List[float] = []

    for subtype in line_subtypes:
        dual = build_fixed_budget_dual_views(
            package_sample["line_samples"][subtype],
            n_tiles=int(line_n_tiles),
            k_chunks=int(line_k_chunks),
            depth_template=line_depth_template,
            rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
            rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
            feature_noise_std=feature_noise_std,
            device=device,
        )
        if subtype in line_selected_a:
            line_view_a[subtype] = dual["view_a"]
        if subtype in line_selected_b:
            line_view_b[subtype] = dual["view_b"]
        if subtype in line_selected_a and subtype in line_selected_b:
            line_tile_overlaps.append(float(dual["tile_overlap_ratio"]))
            line_chunk_overlaps.append(float(dual["chunk_overlap_ratio"]))

    for subtype in polygon_subtypes:
        dual = build_fixed_budget_polygon_views(
            package_sample["polygon_samples"][subtype],
            n_tiles=int(polygon_n_tiles),
            k_polygons=int(polygon_k_polygons),
            depth_template=polygon_depth_template,
            rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
            rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
            feature_noise_std=feature_noise_std,
            device=device,
        )
        if subtype in polygon_selected_a:
            polygon_view_a[subtype] = dual["view_a"]
        if subtype in polygon_selected_b:
            polygon_view_b[subtype] = dual["view_b"]
        if subtype in polygon_selected_a and subtype in polygon_selected_b:
            polygon_tile_overlaps.append(float(dual["tile_overlap_ratio"]))
            polygon_object_overlaps.append(float(dual["polygon_overlap_ratio"]))

    for subtype in point_subtypes:
        dual = build_fixed_budget_point_views(
            package_sample["point_samples"][subtype],
            n_tiles=int(point_n_tiles),
            k_points=int(point_k_points),
            depth_template=point_depth_template,
            rng_a=np.random.default_rng(int(rng_a.integers(0, 2**31 - 1))),
            rng_b=np.random.default_rng(int(rng_b.integers(0, 2**31 - 1))),
            feature_noise_std=feature_noise_std,
            device=device,
        )
        if subtype in point_selected_a:
            point_view_a[subtype] = dual["view_a"]
        if subtype in point_selected_b:
            point_view_b[subtype] = dual["view_b"]
        if subtype in point_selected_a and subtype in point_selected_b:
            point_tile_overlaps.append(float(dual["tile_overlap_ratio"]))
            point_object_overlaps.append(float(dual["point_overlap_ratio"]))

    line_overlap = len(set(line_selected_a) & set(line_selected_b)) / max(len(set(line_selected_a) | set(line_selected_b)), 1)
    polygon_overlap = len(set(polygon_selected_a) & set(polygon_selected_b)) / max(len(set(polygon_selected_a) | set(polygon_selected_b)), 1)
    point_overlap = len(set(point_selected_a) & set(point_selected_b)) / max(len(set(point_selected_a) | set(point_selected_b)), 1)

    return {
        "line_view_a": line_view_a,
        "line_view_b": line_view_b,
        "polygon_view_a": polygon_view_a,
        "polygon_view_b": polygon_view_b,
        "point_view_a": point_view_a,
        "point_view_b": point_view_b,
        "line_selected_a": list(line_selected_a),
        "line_selected_b": list(line_selected_b),
        "polygon_selected_a": list(polygon_selected_a),
        "polygon_selected_b": list(polygon_selected_b),
        "point_selected_a": list(point_selected_a),
        "point_selected_b": list(point_selected_b),
        "line_subtype_overlap": float(line_overlap),
        "polygon_subtype_overlap": float(polygon_overlap),
        "point_subtype_overlap": float(point_overlap),
        "line_tile_overlap": float(np.mean(line_tile_overlaps)) if line_tile_overlaps else 0.0,
        "line_chunk_overlap": float(np.mean(line_chunk_overlaps)) if line_chunk_overlaps else 0.0,
        "polygon_tile_overlap": float(np.mean(polygon_tile_overlaps)) if polygon_tile_overlaps else 0.0,
        "polygon_overlap": float(np.mean(polygon_object_overlaps)) if polygon_object_overlaps else 0.0,
        "point_tile_overlap": float(np.mean(point_tile_overlaps)) if point_tile_overlaps else 0.0,
        "point_overlap": float(np.mean(point_object_overlaps)) if point_object_overlaps else 0.0,
    }
