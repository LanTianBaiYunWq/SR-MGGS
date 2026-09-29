"""Package-level dataset for line + polygon + point mixed prototype."""

from __future__ import annotations

from typing import Dict, List, Sequence

from torch.utils.data import Dataset

from mixed_hier_stage1.line_polygon_dataset import package_key_from_file_path
from point_hier_stage1.point_hier_dataset import PointHierDataset
from polygon_hier_stage1.dual_view_protocol import infer_polygon_subtype
from polygon_hier_stage1.polygon_hier_dataset import PolygonHierDataset
from roads_hier_stage1.dual_view_protocol import infer_line_subtype
from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset


def infer_point_mixed_subtype(file_path: str) -> str:
    normalized = str(file_path).lower()
    name = normalized.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if "pois" in name or name == "pois.shp":
        return "pois"
    if "places" in name or name == "places.shp":
        return "places"
    if "traffic" in name or name == "traffic.shp":
        return "traffic"
    if "transport" in name or name == "transport.shp":
        return "transport"
    if "pofw" in name or name == "pofw.shp":
        return "pofw"
    return "point"


class MixedLinePolygonPointPackageDataset(Dataset):
    def __init__(
        self,
        line_cache_root: str,
        polygon_cache_root: str,
        point_cache_root: str,
        *,
        mode: str = "train",
        line_max_tiles: int | None = None,
        polygon_max_tiles: int | None = None,
        point_max_tiles: int | None = None,
        line_tile_selector: str = "manifest_default",
        polygon_tile_selector: str = "manifest_default",
        point_tile_selector: str = "manifest_default",
        line_subtypes: Sequence[str] = ("roads", "railways", "waterways"),
        polygon_subtypes: Sequence[str] = ("building", "landuse", "natural"),
        point_subtypes: Sequence[str] = ("pois", "places", "traffic", "transport", "pofw"),
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
        self.point_dataset = PointHierDataset(
            point_cache_root,
            mode=mode,
            max_tiles=point_max_tiles,
            tile_selector=point_tile_selector,
        )
        self.line_subtypes = tuple(str(value) for value in line_subtypes)
        self.polygon_subtypes = tuple(str(value) for value in polygon_subtypes)
        self.point_subtypes = tuple(str(value) for value in point_subtypes)

        line_groups = self._group_line_items()
        polygon_groups = self._group_polygon_items()
        point_groups = self._group_point_items()
        package_keys = sorted(set(line_groups.keys()) & set(polygon_groups.keys()) & set(point_groups.keys()))

        package_records: List[Dict[str, object]] = []
        for package_key in package_keys:
            line_map = line_groups[package_key]
            polygon_map = polygon_groups[package_key]
            point_map = point_groups[package_key]
            if require_complete_packages:
                if any(subtype not in line_map for subtype in self.line_subtypes):
                    continue
                if any(subtype not in polygon_map for subtype in self.polygon_subtypes):
                    continue
                if any(subtype not in point_map for subtype in self.point_subtypes):
                    continue
            package_records.append(
                {
                    "package_id": package_key,
                    "line_indices": {subtype: int(line_map[subtype]) for subtype in self.line_subtypes if subtype in line_map},
                    "polygon_indices": {
                        subtype: int(polygon_map[subtype]) for subtype in self.polygon_subtypes if subtype in polygon_map
                    },
                    "point_indices": {
                        subtype: int(point_map[subtype]) for subtype in self.point_subtypes if subtype in point_map
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

    def _group_point_items(self) -> Dict[str, Dict[str, int]]:
        groups: Dict[str, Dict[str, int]] = {}
        for idx, item in enumerate(self.point_dataset.items):
            package_key = package_key_from_file_path(str(item["file_path"]))
            subtype = infer_point_mixed_subtype(str(item["file_path"]))
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
        point_samples = {
            subtype: self.point_dataset[int(sample_idx)]
            for subtype, sample_idx in dict(record["point_indices"]).items()
        }
        return {
            "package_id": str(record["package_id"]),
            "line_samples": line_samples,
            "polygon_samples": polygon_samples,
            "point_samples": point_samples,
            "line_subtypes": list(line_samples.keys()),
            "polygon_subtypes": list(polygon_samples.keys()),
            "point_subtypes": list(point_samples.keys()),
        }


class MixedLinePolygonOptionalPointPackageDataset(Dataset):
    """Package dataset for line+polygon with optional point coverage.

    This is used by the generic mixed backbone, where missing point subtypes or
    even a completely missing point family should degrade to the available LP
    signal instead of filtering the package out.
    """

    def __init__(
        self,
        line_cache_root: str,
        polygon_cache_root: str,
        point_cache_root: str | None = None,
        *,
        mode: str = "train",
        line_max_tiles: int | None = None,
        polygon_max_tiles: int | None = None,
        point_max_tiles: int | None = None,
        line_tile_selector: str = "manifest_default",
        polygon_tile_selector: str = "manifest_default",
        point_tile_selector: str = "manifest_default",
        line_subtypes: Sequence[str] = ("roads", "railways", "waterways"),
        polygon_subtypes: Sequence[str] = ("building", "landuse", "natural"),
        point_subtypes: Sequence[str] = ("pois", "places", "traffic", "transport", "pofw"),
        max_packages: int | None = None,
        require_complete_lp: bool = False,
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
        self.point_dataset = (
            PointHierDataset(
                point_cache_root,
                mode=mode,
                max_tiles=point_max_tiles,
                tile_selector=point_tile_selector,
            )
            if point_cache_root
            else None
        )
        self.line_subtypes = tuple(str(value) for value in line_subtypes)
        self.polygon_subtypes = tuple(str(value) for value in polygon_subtypes)
        self.point_subtypes = tuple(str(value) for value in point_subtypes)

        line_groups = self._group_line_items()
        polygon_groups = self._group_polygon_items()
        point_groups = self._group_point_items() if self.point_dataset is not None else {}
        package_keys = sorted(set(line_groups.keys()) & set(polygon_groups.keys()))

        package_records: List[Dict[str, object]] = []
        for package_key in package_keys:
            line_map = line_groups[package_key]
            polygon_map = polygon_groups[package_key]
            point_map = point_groups.get(package_key, {})
            if require_complete_lp:
                if any(subtype not in line_map for subtype in self.line_subtypes):
                    continue
                if any(subtype not in polygon_map for subtype in self.polygon_subtypes):
                    continue
            if not any(subtype in line_map for subtype in self.line_subtypes):
                continue
            if not any(subtype in polygon_map for subtype in self.polygon_subtypes):
                continue
            package_records.append(
                {
                    "package_id": package_key,
                    "line_indices": {subtype: int(line_map[subtype]) for subtype in self.line_subtypes if subtype in line_map},
                    "polygon_indices": {
                        subtype: int(polygon_map[subtype]) for subtype in self.polygon_subtypes if subtype in polygon_map
                    },
                    "point_indices": {
                        subtype: int(point_map[subtype]) for subtype in self.point_subtypes if subtype in point_map
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

    def _group_point_items(self) -> Dict[str, Dict[str, int]]:
        groups: Dict[str, Dict[str, int]] = {}
        if self.point_dataset is None:
            return groups
        for idx, item in enumerate(self.point_dataset.items):
            package_key = package_key_from_file_path(str(item["file_path"]))
            subtype = infer_point_mixed_subtype(str(item["file_path"]))
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
        point_samples = {
            subtype: self.point_dataset[int(sample_idx)]
            for subtype, sample_idx in dict(record["point_indices"]).items()
        } if self.point_dataset is not None else {}
        return {
            "package_id": str(record["package_id"]),
            "line_samples": line_samples,
            "polygon_samples": polygon_samples,
            "point_samples": point_samples,
            "line_subtypes": list(line_samples.keys()),
            "polygon_subtypes": list(polygon_samples.keys()),
            "point_subtypes": list(point_samples.keys()),
        }
