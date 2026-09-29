"""
为原始矢量数据宇宙构建文件/图层/几何清单。

脚本存在的原因
----------------------
在更改模型或进行预处理之前，我们需要了解数据实际包含的内容。该脚本检查原始矢量资产，并生成描述以下内容的清单：

- 存在哪些文件
- 每个文件中存在哪些图层
- 图层是空间的还是非空间的
- 几何类型组成
- 空/空值比例
- 无效比例
- 多部分比例
- 几何集合比例
- Z/M 存在情况
- 坐标数量分布

这本质上是一个检查工具。它不会规范化或更改数据。在做建模假设之前应先运行它。

典型用法
-----------
python scripts/build_vector_manifest.py --data_dir data/raw
python scripts/build_vector_manifest.py --data_dir data/raw --output reports/vector_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_input_path(path_str: str) -> Path:
    """Resolve paths robustly for both terminal and IDE execution."""
    raw_path = Path(path_str)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = Path.cwd() / raw_path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = PROJECT_ROOT / raw_path
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def collect_vector_files(data_dir: str) -> List[Path]:
    """
    Collect actual vector files under the raw data directory.

    Right now the project is centered on shapefiles, so this script only scans
    real `.shp` files. Directories whose names also end with `.shp` are ignored.
    """
    base = resolve_input_path(data_dir)
    return sorted(path for path in base.glob("**/*.shp") if path.is_file())


def safe_has_z(geom) -> bool:
    """Return geometry.has_z when available, otherwise False."""
    try:
        return bool(getattr(geom, "has_z", False))
    except Exception:
        return False


def safe_has_m(geom) -> bool:
    """Return geometry.has_m when available, otherwise False."""
    try:
        return bool(getattr(geom, "has_m", False))
    except Exception:
        return False


def geometry_type_name(geom) -> str:
    """Return a stable geometry type label including null/empty cases."""
    if geom is None:
        return "None"
    if geom.is_empty:
        return "Empty"
    return geom.geom_type


def is_multipart_type(geom_type: str) -> bool:
    """Classify multipart geometry families by type name."""
    return geom_type.startswith("Multi")


def count_coordinates(geom) -> int:
    """
    Count coordinates recursively for all common geometry families.

    This is a raw inspection utility. It does not alter geometry semantics.
    """
    if geom is None or geom.is_empty:
        return 0

    geom_type = geom.geom_type
    if geom_type == "Point":
        return 1
    if geom_type in {"LineString", "LinearRing"}:
        return len(geom.coords)
    if geom_type == "Polygon":
        total = len(geom.exterior.coords)
        for ring in geom.interiors:
            total += len(ring.coords)
        return total
    if hasattr(geom, "geoms"):
        return sum(count_coordinates(part) for part in geom.geoms)

    # Unknown geometry family: return a best-effort zero rather than failing the
    # whole manifest.
    return 0


def quantile(values: List[int], q: float) -> Optional[float]:
    """Small helper for p95-style summaries without adding heavy dependencies."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def summarize_coordinate_distribution(coord_counts: List[int]) -> Dict[str, Optional[float]]:
    """Summarize coordinate counts with a compact descriptive statistics block."""
    if not coord_counts:
        return {
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "min": float(min(coord_counts)),
        "median": float(median(coord_counts)),
        "p95": quantile(coord_counts, 0.95),
        "max": float(max(coord_counts)),
    }


def list_layers_for_file(vector_path: Path) -> List[Dict]:
    """
    List layers using geopandas.list_layers when available.

    For shapefiles this usually returns a single spatial layer, but keeping this
    structure is useful because the next stage of the project may eventually
    inspect other vector containers too.
    """
    try:
        list_layers_fn = getattr(gpd, "list_layers", None)
        if list_layers_fn is None:
            return [
                {
                    "name": vector_path.stem,
                    "geometry_type": "unknown",
                }
            ]

        layers_df = list_layers_fn(vector_path)
        layers = []
        for _, row in layers_df.iterrows():
            layers.append(
                {
                    "name": row.get("name"),
                    "geometry_type": row.get("geometry_type"),
                }
            )
        return layers
    except Exception:
        return [
            {
                "name": vector_path.stem,
                "geometry_type": "unknown",
            }
        ]


def analyze_layer(vector_path: Path, layer_name: Optional[str]) -> Dict:
    """
    Inspect one spatial layer and compute file-level geometry statistics.

    The result is intentionally descriptive. It does not apply any project
    preprocessing rules.
    """
    read_kwargs = {}
    if layer_name:
        read_kwargs["layer"] = layer_name

    gdf = gpd.read_file(vector_path, **read_kwargs)

    total_features = int(len(gdf))
    if "geometry" not in gdf.columns:
        return {
            "layer_name": layer_name,
            "is_spatial": False,
            "feature_count": total_features,
            "empty_or_null_count": None,
            "raw_type_counts": {},
            "has_z_ratio": None,
            "has_m_ratio": None,
            "multipart_ratio": None,
            "geometry_collection_ratio": None,
            "invalid_ratio": None,
            "coordinate_count_distribution": {
                "min": None,
                "median": None,
                "p95": None,
                "max": None,
            },
        }

    type_counts: Counter = Counter()
    empty_or_null_count = 0
    invalid_count = 0
    multipart_count = 0
    geometry_collection_count = 0
    has_z_count = 0
    has_m_count = 0
    coord_counts: List[int] = []

    for geom in gdf.geometry:
        geom_type = geometry_type_name(geom)
        type_counts[geom_type] += 1

        if geom is None or geom.is_empty:
            empty_or_null_count += 1
            continue

        if not geom.is_valid:
            invalid_count += 1

        if safe_has_z(geom):
            has_z_count += 1
        if safe_has_m(geom):
            has_m_count += 1
        if is_multipart_type(geom.geom_type):
            multipart_count += 1
        if geom.geom_type == "GeometryCollection":
            geometry_collection_count += 1

        coord_counts.append(count_coordinates(geom))

    non_empty_features = max(total_features - empty_or_null_count, 1)
    return {
        "layer_name": layer_name,
        "is_spatial": True,
        "feature_count": total_features,
        "empty_or_null_count": empty_or_null_count,
        "raw_type_counts": dict(type_counts),
        "has_z_ratio": round(has_z_count / non_empty_features, 6),
        "has_m_ratio": round(has_m_count / non_empty_features, 6),
        "multipart_ratio": round(multipart_count / non_empty_features, 6),
        "geometry_collection_ratio": round(geometry_collection_count / non_empty_features, 6),
        "invalid_ratio": round(invalid_count / non_empty_features, 6),
        "coordinate_count_distribution": summarize_coordinate_distribution(coord_counts),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line interface for the manifest builder."""
    parser = argparse.ArgumentParser(
        description="Build a raw-data manifest for vector geometry files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing raw vector data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def print_manifest_summary(file_records: List[Dict]) -> None:
    """Print a compact summary that is useful before opening the full JSON."""
    total_files = len(file_records)
    total_layers = sum(len(record["layers"]) for record in file_records)
    spatial_layers = sum(
        1 for record in file_records for layer in record["layers"] if layer["is_spatial"]
    )

    overall_types: Counter = Counter()
    for record in file_records:
        for layer in record["layers"]:
            overall_types.update(layer["raw_type_counts"])

    print("=" * 90)
    print("Vector Manifest Summary")
    print("=" * 90)
    print(f"Files scanned: {total_files}")
    print(f"Layers discovered: {total_layers}")
    print(f"Spatial layers: {spatial_layers}")
    print("Top raw geometry types:")
    for geom_type, count in overall_types.most_common(10):
        print(f"  {geom_type:<18} {count}")
    print("=" * 90)


def main() -> None:
    """Entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    vector_files = collect_vector_files(args.data_dir)
    if not vector_files:
        raise ValueError(f"No shapefiles found under: {args.data_dir}")

    print(f"Found {len(vector_files)} vector files to inspect.")

    file_records: List[Dict] = []
    start_time = time.time()

    for index, vector_path in enumerate(vector_files, start=1):
        layer_infos = list_layers_for_file(vector_path)
        layers = []
        for layer_info in layer_infos:
            layer_name = layer_info.get("name")
            geometry_type = layer_info.get("geometry_type")
            is_non_spatial = geometry_type is None

            if is_non_spatial:
                layers.append(
                    {
                        "layer_name": layer_name,
                        "declared_geometry_type": geometry_type,
                        "is_spatial": False,
                        "feature_count": None,
                        "empty_or_null_count": None,
                        "raw_type_counts": {},
                        "has_z_ratio": None,
                        "has_m_ratio": None,
                        "multipart_ratio": None,
                        "geometry_collection_ratio": None,
                        "invalid_ratio": None,
                        "coordinate_count_distribution": {
                            "min": None,
                            "median": None,
                            "p95": None,
                            "max": None,
                        },
                    }
                )
                continue

            layer_result = analyze_layer(vector_path, layer_name)
            layer_result["declared_geometry_type"] = geometry_type
            layers.append(layer_result)

        file_records.append(
            {
                "file_path": str(vector_path),
                "layers": layers,
            }
        )

        if index == 1 or index % 20 == 0 or index == len(vector_files):
            elapsed = time.time() - start_time
            print(f"[{index}/{len(vector_files)}] elapsed={elapsed:.1f}s")

    print_manifest_summary(file_records)

    report = {
        "files": file_records,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved manifest to: {output_path}")


if __name__ == "__main__":
    main()
