"""
统计原始矢量数据集中各类 shapefile 的数量。

输出重点：
1. 总 shapefile 数量
2. 按 layer 名称统计，例如 roads / railways / waterways
3. 按几何 schema 统计，例如 LineString / Polygon
4. 按几何家族统计，例如 line / polygon / point
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

try:
    from pyogrio import read_info  # type: ignore
except Exception:  # pragma: no cover
    read_info = None

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None


def resolve_path(path_str: str) -> Path:
    """兼容相对路径和绝对路径。"""
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = Path(__file__).resolve().parent.parent / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def collect_shapefiles(data_dir: str) -> List[Path]:
    """递归收集真实的 shapefile 文件。"""
    root = resolve_path(data_dir)
    return sorted(path for path in root.glob("**/*.shp") if path.is_file())


def parse_layer_name(shp_path: Path) -> str:
    """
    从类似 gis_osm_roads_free_1.shp 中提取 roads。
    不满足该模式时退化为 stem。
    """
    stem = shp_path.stem
    prefix = "gis_osm_"
    suffix = "_free_1"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def geometry_family(schema_geometry: str) -> str:
    """把 schema geometry 映射到 point/line/polygon/unknown。"""
    geom = (schema_geometry or "Unknown").lower()
    if "point" in geom:
        return "point"
    if "line" in geom or "arc" in geom:
        return "line"
    if "polygon" in geom or "area" in geom:
        return "polygon"
    return "unknown"


def build_report(shp_files: List[Path]) -> Dict[str, object]:
    """扫描每个文件并聚合统计。"""
    layer_counter: Counter[str] = Counter()
    schema_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    family_by_layer: Dict[str, Counter[str]] = defaultdict(Counter)
    examples_by_layer: Dict[str, str] = {}

    file_rows: List[Dict[str, str]] = []

    for shp_path in shp_files:
        layer = parse_layer_name(shp_path)
        try:
            schema_geometry = read_schema_geometry(shp_path)
        except Exception as exc:
            schema_geometry = f"ReadError:{type(exc).__name__}"

        family = geometry_family(schema_geometry)

        layer_counter[layer] += 1
        schema_counter[schema_geometry] += 1
        family_counter[family] += 1
        family_by_layer[layer][family] += 1
        examples_by_layer.setdefault(layer, str(shp_path))

        file_rows.append(
            {
                "file_path": str(shp_path),
                "region_dir": shp_path.parent.name,
                "layer_name": layer,
                "schema_geometry": schema_geometry,
                "geometry_family": family,
            }
        )

    layers = []
    for layer, count in layer_counter.most_common():
        family_breakdown = dict(family_by_layer[layer])
        layers.append(
            {
                "layer_name": layer,
                "count": count,
                "geometry_families": family_breakdown,
                "example_path": examples_by_layer[layer],
            }
        )

    return {
        "total_shapefiles": len(shp_files),
        "layer_counts": layers,
        "schema_geometry_counts": dict(schema_counter.most_common()),
        "geometry_family_counts": dict(family_counter.most_common()),
        "files": file_rows,
    }


def read_schema_geometry(shp_path: Path) -> str:
    """尽量便宜地读取 schema geometry。"""
    if read_info is not None:
        info = read_info(shp_path)
        geometry_name = info.get("geometry_name")
        if geometry_name:
            return str(geometry_name)

    if gpd is not None:
        gdf = gpd.read_file(shp_path, rows=1)
        if len(gdf) == 0:
            return "Unknown"
        geom = gdf.geometry.iloc[0]
        if geom is None:
            return "None"
        if geom.is_empty:
            return "Empty"
        return str(geom.geom_type)

    raise RuntimeError("Neither pyogrio nor geopandas is available for schema inspection.")


def print_report(report: Dict[str, object]) -> None:
    """打印摘要。"""
    print("=" * 90)
    print("Vector Dataset Type Count")
    print("=" * 90)
    print(f"Total shapefiles: {report['total_shapefiles']}")
    print()

    print("Geometry families:")
    for family, count in report["geometry_family_counts"].items():
        print(f"  {family:<10} {count}")
    print()

    print("Top layer counts:")
    for row in report["layer_counts"]:
        fam_text = ", ".join(f"{k}:{v}" for k, v in sorted(row["geometry_families"].items()))
        print(f"  {row['layer_name']:<20} {row['count']:<4} {fam_text}")
    print()

    print("Schema geometry counts:")
    for schema_name, count in report["schema_geometry_counts"].items():
        print(f"  {schema_name:<24} {count}")
    print("=" * 90)


def build_argument_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Count vector dataset types under raw shapefiles.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="原始 shapefile 根目录。")
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    shp_files = collect_shapefiles(args.data_dir)
    if not shp_files:
        raise ValueError(f"No shapefiles found under: {args.data_dir}")

    report = build_report(shp_files)
    print_report(report)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
