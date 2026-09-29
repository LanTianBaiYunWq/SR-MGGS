"""
geometrycore_hier_dataset

训练期 dataset 只读离线缓存产物：
- global index
- file manifest
- tile manifest
- chunk npz

不在 dataset.__init__ 或 __getitem__ 中解析 shapefile、切 tile、切 chunk、构图或渲染。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset


def resolve_path(path_str: str) -> Path:
    """兼容相对路径和绝对路径。"""
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


class GeometryCoreHierDataset(Dataset):
    """只读 manifest/cache 的分层数据集。"""

    def __init__(
        self,
        cache_root: str,
        *,
        mode: str = "train",
        max_tiles: int | None = None,
        tile_selector: str = "manifest_default",
    ):
        self.cache_root = resolve_path(cache_root)
        self.index_path = self.cache_root / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"Cache index not found: {self.index_path}")
        if mode not in {"train", "eval", "all"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if tile_selector not in {"manifest_default", "top_selection_score", "all"}:
            raise ValueError(f"Unsupported tile_selector: {tile_selector}")

        with self.index_path.open("r", encoding="utf-8") as f:
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
        """按 manifest 或显式策略选择 file-level top-M tiles。"""
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
            ranked = sorted(
                tiles,
                key=lambda row: (
                    float(row.get("selection_score", 0.0)),
                    -float(row.get("selection_rank", 10**9)),
                ),
                reverse=True,
            )
            selected = ranked

        if self.max_tiles is not None:
            selected = selected[: self.max_tiles]

        selected = sorted(
            selected,
            key=lambda row: (
                float(row.get("selection_rank", 10**9)),
                -float(row.get("selection_score", 0.0)),
            ),
        )
        return selected

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[idx]
        file_dir = self.cache_root / item["file_id"]
        manifest_path = file_dir / "manifest.json"
        tile_manifest_path = file_dir / "tiles.json"

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        with tile_manifest_path.open("r", encoding="utf-8") as f:
            tiles = json.load(f)

        selected_tiles = self._select_tiles(tiles, manifest)

        tile_payloads = []
        skipped_empty_tiles = 0
        for tile in selected_tiles:
            if int(tile.get("budgeted_chunk_count", 0)) <= 0:
                skipped_empty_tiles += 1
                continue
            npz_path = file_dir / "chunks" / f"{tile['tile_id']}.npz"
            if not npz_path.exists():
                skipped_empty_tiles += 1
                continue
            # 当前 Gate 1 cache 使用 object array 保存变长 chunk 坐标和元数据。
            # 训练期只读取本地离线缓存，因此这里显式允许 pickle。
            data = np.load(npz_path, allow_pickle=True)
            tile_payloads.append(
                {
                    "tile_id": tile["tile_id"],
                    "depth": tile.get("depth"),
                    "tile_rank": tile.get("selection_rank"),
                    "tile_score": tile.get("selection_score"),
                    "tile_bbox": tile["bbox"],
                    "raw_chunk_count": tile.get("raw_chunk_count"),
                    "budgeted_chunk_count": tile.get("budgeted_chunk_count"),
                    "chunk_length_sum": tile.get("chunk_length_sum"),
                    "tile_area": tile.get("tile_area"),
                    "chunk_density": tile.get("chunk_density"),
                    "compression_ratio": tile.get("compression_ratio"),
                    "chunk_coords": data["chunk_coords"],
                    "chunk_features": data["chunk_features"],
                    "chunk_meta": data["chunk_meta"],
                }
            )

        return {
            "file_id": item["file_id"],
            "file_path": item["file_path"],
            "family": item["family"],
            "manifest": manifest,
            "tiles": tile_payloads,
            "num_selected_tiles": len(tile_payloads),
            "num_skipped_empty_tiles": skipped_empty_tiles,
        }
