"""
Read-only point hierarchical dataset backed by an offline cache.

Expected cache layout:

cache_root/
  index.json
  <file_id>/
    manifest.json
    tiles.json
    points/
      <tile_id>.npz

Each tile npz is expected to contain:
- point_features: [num_points, feature_dim]
- point_meta: object array with point ids in column 0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset


def resolve_path(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = Path(__file__).resolve().parents[1] / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


class PointHierDataset(Dataset):
    """Dataset for point-family fixed-budget dual-view experiments."""

    def __init__(
        self,
        cache_root: str,
        *,
        mode: str = "train",
        max_tiles: int | None = None,
        tile_selector: str = "manifest_default",
    ):
        self.cache_root = resolve_path(cache_root)
        index_path = self.cache_root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Point cache index not found: {index_path}")
        if mode not in {"train", "eval", "all"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if tile_selector not in {"manifest_default", "top_selection_score", "all"}:
            raise ValueError(f"Unsupported tile_selector: {tile_selector}")

        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.items: List[Dict[str, object]] = index_data["files"]
        self.mode = mode
        self.max_tiles = max_tiles
        self.tile_selector = tile_selector

    def __len__(self) -> int:
        return len(self.items)

    def _select_tiles(
        self,
        tiles: List[Dict[str, object]],
        manifest: Dict[str, object],
    ) -> List[Dict[str, object]]:
        if self.mode == "all" or self.tile_selector == "all":
            selected = list(tiles)
        elif self.tile_selector == "manifest_default":
            flag_key = "selected_for_train" if self.mode == "train" else "selected_for_eval"
            selected = [tile for tile in tiles if bool(tile.get(flag_key, False))]

            if not selected:
                selection = manifest.get("tile_selection", {})
                max_tiles = self.max_tiles
                if max_tiles is None:
                    fallback_key = "train_max_tiles" if self.mode == "train" else "eval_max_tiles"
                    raw_value = selection.get(fallback_key)
                    max_tiles = int(raw_value) if raw_value is not None else None
                ranked = sorted(
                    tiles,
                    key=lambda row: (
                        float(row.get("selection_score", 0.0)),
                        -float(row.get("selection_rank", 10**9)),
                    ),
                    reverse=True,
                )
                selected = ranked[:max_tiles] if max_tiles is not None else ranked
        else:
            selected = sorted(
                tiles,
                key=lambda row: (
                    float(row.get("selection_score", 0.0)),
                    -float(row.get("selection_rank", 10**9)),
                ),
                reverse=True,
            )

        if self.max_tiles is not None:
            selected = selected[: self.max_tiles]

        return sorted(
            selected,
            key=lambda row: (
                float(row.get("selection_rank", 10**9)),
                -float(row.get("selection_score", 0.0)),
            ),
        )

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[idx]
        file_dir = self.cache_root / str(item["file_id"])
        manifest_path = file_dir / "manifest.json"
        tile_manifest_path = file_dir / "tiles.json"

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        with tile_manifest_path.open("r", encoding="utf-8") as f:
            tiles = json.load(f)

        selected_tiles = self._select_tiles(tiles, manifest)

        tile_payloads: List[Dict[str, object]] = []
        skipped_empty_tiles = 0
        for tile in selected_tiles:
            if int(tile.get("budgeted_point_count", tile.get("raw_point_count", 0))) <= 0:
                skipped_empty_tiles += 1
                continue

            npz_path = file_dir / "points" / f"{tile['tile_id']}.npz"
            if not npz_path.exists():
                skipped_empty_tiles += 1
                continue

            data = np.load(npz_path, allow_pickle=True)
            point_features = np.asarray(data["point_features"], dtype=np.float32)
            if point_features.ndim != 2 or point_features.shape[0] == 0:
                skipped_empty_tiles += 1
                continue

            tile_payloads.append(
                {
                    "tile_id": tile["tile_id"],
                    "depth": tile.get("depth"),
                    "tile_rank": tile.get("selection_rank"),
                    "tile_score": tile.get("selection_score"),
                    "tile_bbox": tile["bbox"],
                    "raw_point_count": tile.get("raw_point_count"),
                    "budgeted_point_count": tile.get("budgeted_point_count", point_features.shape[0]),
                    "tile_area": tile.get("tile_area"),
                    "point_density": tile.get("point_density"),
                    "point_features": point_features,
                    "point_meta": data["point_meta"],
                }
            )

        return {
            "file_id": item["file_id"],
            "file_path": item["file_path"],
            "family": item.get("family", "point"),
            "manifest": manifest,
            "tiles": tile_payloads,
            "num_selected_tiles": len(tile_payloads),
            "num_skipped_empty_tiles": skipped_empty_tiles,
        }
