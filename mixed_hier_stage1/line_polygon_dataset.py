"""Package-level dataset for line + polygon mixed V1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from torch.utils.data import Dataset

from polygon_hier_stage1.dual_view_protocol import infer_polygon_subtype
from polygon_hier_stage1.polygon_hier_dataset import PolygonHierDataset
from roads_hier_stage1.dual_view_protocol import infer_line_subtype
from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset


def package_key_from_file_path(file_path: str) -> str:
    path = Path(str(file_path))
    parent_name = path.parent.name.strip().lower()
    if parent_name == "shape" and path.parent.parent.name:
        return path.parent.parent.name.strip().lower()
    if parent_name:
        return parent_name
    return path.stem.strip().lower()


class MixedLinePolygonPackageDataset(Dataset):
    def __init__(
        self,
        line_cache_root: str,
        polygon_cache_root: str,
        *,
        mode: str = "train",
        line_max_tiles: int | None = None,
        polygon_max_tiles: int | None = None,
        line_tile_selector: str = "manifest_default",
        polygon_tile_selector: str = "manifest_default",
        line_subtypes: Sequence[str] = ("roads", "railways", "waterways"),
        polygon_subtypes: Sequence[str] = ("building", "landuse", "natural", "water"),
        max_packages: int | None = None,
        require_complete_packages: bool = True,
    ):
        self.line_dataset = GeometryCoreHierDataset(
            line_cache_root,
            mode=mode,
            max_tiles=line_max_tiles,
            tile_selector=line_tile_selector,
        )
        self.polygon_dataset = PolygonHierDataset(
            polygon_cache_root,
            mode=mode,
            max_tiles=polygon_max_tiles,
            tile_selector=polygon_tile_selector,
        )
        self.line_subtypes = tuple(str(value) for value in line_subtypes)
        self.polygon_subtypes = tuple(str(value) for value in polygon_subtypes)
        self.mode = mode

        line_groups = self._group_line_items()
        polygon_groups = self._group_polygon_items()
        package_keys = sorted(set(line_groups.keys()) & set(polygon_groups.keys()))

        package_records: List[Dict[str, object]] = []
        for package_key in package_keys:
            line_map = line_groups[package_key]
            polygon_map = polygon_groups[package_key]
            if require_complete_packages:
                if any(subtype not in line_map for subtype in self.line_subtypes):
                    continue
                if any(subtype not in polygon_map for subtype in self.polygon_subtypes):
                    continue
            package_records.append(
                {
                    "package_id": package_key,
                    "line_indices": {subtype: int(line_map[subtype]) for subtype in self.line_subtypes if subtype in line_map},
                    "polygon_indices": {
                        subtype: int(polygon_map[subtype]) for subtype in self.polygon_subtypes if subtype in polygon_map
                    },
                }
            )

        if max_packages is not None:
            package_records = package_records[: int(max_packages)]

        self.package_records = package_records

    def _group_line_items(self) -> Dict[str, Dict[str, int]]:
        groups: Dict[str, Dict[str, int]] = {}
        for idx, item in enumerate(self.line_dataset.items):
            package_key = package_key_from_file_path(str(item["file_path"]))
            subtype = infer_line_subtype(str(item["file_path"]))
            groups.setdefault(package_key, {})[subtype] = idx
        return groups

    def _group_polygon_items(self) -> Dict[str, Dict[str, int]]:
        groups: Dict[str, Dict[str, int]] = {}
        for idx, item in enumerate(self.polygon_dataset.items):
            package_key = package_key_from_file_path(str(item["file_path"]))
            subtype = infer_polygon_subtype(str(item["file_path"]))
            groups.setdefault(package_key, {})[subtype] = idx
        return groups

    def __len__(self) -> int:
        return len(self.package_records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.package_records[idx]
        line_samples = {
            subtype: self.line_dataset[int(sample_idx)]
            for subtype, sample_idx in dict(record["line_indices"]).items()
        }
        polygon_samples = {
            subtype: self.polygon_dataset[int(sample_idx)]
            for subtype, sample_idx in dict(record["polygon_indices"]).items()
        }
        return {
            "package_id": str(record["package_id"]),
            "line_samples": line_samples,
            "polygon_samples": polygon_samples,
            "line_subtypes": list(line_samples.keys()),
            "polygon_subtypes": list(polygon_samples.keys()),
        }
