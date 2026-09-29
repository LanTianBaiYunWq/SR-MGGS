"""Create attacked raw vector datasets for zero-watermark robustness tests.

The output keeps the original package/shape/*.shp layout so existing cache
builders can rebuild line, polygon, and point query caches from attacked data.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable

import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import box
from shapely.ops import transform
try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover - compatibility fallback
    make_valid = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    candidate = Path.cwd() / raw
    if candidate.exists():
        return candidate
    return PROJECT_ROOT / raw


def load_package_subset(split_path: Path, split_name: str) -> set[str]:
    with split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    values = data.get(split_name)
    if values is None:
        raise KeyError(f"Split '{split_name}' not found in {split_path}")
    return {str(value).strip().lower() for value in values}


def package_id_from_dir(path: Path) -> str:
    return path.name.strip().lower()


def is_vector_package_dir(path: Path) -> bool:
    shape_dir = path / "shape"
    if shape_dir.is_dir() and any(shape_dir.glob("*.shp")):
        return True
    return any(path.glob("*.shp"))


def iter_package_dirs(raw_root: Path, package_ids: set[str]) -> list[Path]:
    dirs = [
        path
        for path in raw_root.rglob("*")
        if path.is_dir() and is_vector_package_dir(path) and package_id_from_dir(path) in package_ids
    ]
    return sorted(dirs, key=lambda path: path.name.lower())


def safe_estimate_working_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        return None
    try:
        estimated = gdf.estimate_utm_crs()
        if estimated is not None:
            return estimated
    except Exception:
        pass
    if gdf.crs.is_geographic:
        return "EPSG:3857"
    return gdf.crs


def finite_bounds(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float] | None:
    if gdf.empty:
        return None
    bounds = tuple(float(value) for value in gdf.total_bounds)
    if any(not math.isfinite(value) for value in bounds):
        return None
    minx, miny, maxx, maxy = bounds
    if maxx <= minx or maxy <= miny:
        return None
    return bounds


def transform_coords(geom, fn: Callable[[np.ndarray], np.ndarray]):
    if geom is None or geom.is_empty:
        return geom

    def mapper(x, y, z=None):
        coords = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
        out = fn(coords)
        if z is None:
            return out[:, 0], out[:, 1]
        return out[:, 0], out[:, 1], z

    return transform(mapper, geom)


def repair_geometry(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        if geom.is_valid:
            return geom
    except Exception:
        return geom
    try:
        if make_valid is not None:
            return make_valid(geom)
    except Exception:
        pass
    try:
        return geom.buffer(0)
    except Exception:
        return geom


def repair_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.geometry = out.geometry.apply(repair_geometry)
    return out[out.geometry.notna() & ~out.geometry.is_empty].copy()


def apply_coord_noise(gdf: gpd.GeoDataFrame, rng: np.random.Generator, ratio: float) -> gpd.GeoDataFrame:
    bounds = finite_bounds(gdf)
    if bounds is None:
        return gdf
    minx, miny, maxx, maxy = bounds
    diag = max(float(math.hypot(maxx - minx, maxy - miny)), 1e-9)
    sigma = float(ratio) * diag

    def noisy(coords: np.ndarray) -> np.ndarray:
        return coords + rng.normal(0.0, sigma, size=coords.shape)

    out = gdf.copy()
    out.geometry = out.geometry.apply(lambda geom: transform_coords(geom, noisy))
    return out


def apply_random_delete(gdf: gpd.GeoDataFrame, rng: np.random.Generator, drop_ratio: float) -> gpd.GeoDataFrame:
    if len(gdf) <= 1:
        return gdf
    keep_count = max(1, int(round(len(gdf) * (1.0 - float(drop_ratio)))))
    indices = np.asarray(gdf.index)
    keep = set(rng.choice(indices, size=keep_count, replace=False).tolist())
    return gdf.loc[[idx for idx in gdf.index if idx in keep]].copy()


def apply_center_crop(gdf: gpd.GeoDataFrame, keep_ratio: float) -> gpd.GeoDataFrame:
    bounds = finite_bounds(gdf)
    if bounds is None:
        return gdf
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    scale = math.sqrt(float(keep_ratio))
    half_w = (maxx - minx) * scale / 2.0
    half_h = (maxy - miny) * scale / 2.0
    crop_geom = box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    out = repair_gdf(gdf)
    out.geometry = out.geometry.intersection(crop_geom)
    return repair_gdf(out)


def apply_simplify(gdf: gpd.GeoDataFrame, tolerance_ratio: float) -> gpd.GeoDataFrame:
    bounds = finite_bounds(gdf)
    if bounds is None:
        return gdf
    minx, miny, maxx, maxy = bounds
    diag = max(float(math.hypot(maxx - minx, maxy - miny)), 1e-9)
    tolerance = float(tolerance_ratio) * diag
    out = gdf.copy()
    out.geometry = out.geometry.simplify(tolerance, preserve_topology=True)
    return out[~out.geometry.is_empty & out.geometry.notna()].copy()


def apply_affine(gdf: gpd.GeoDataFrame, attack: str) -> gpd.GeoDataFrame:
    bounds = finite_bounds(gdf)
    if bounds is None:
        return gdf
    minx, miny, maxx, maxy = bounds
    origin = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    diag = max(float(math.hypot(maxx - minx, maxy - miny)), 1e-9)
    out = gdf.copy()
    if attack == "rotate_10":
        out.geometry = out.geometry.apply(lambda geom: affinity.rotate(geom, 10.0, origin=origin))
    elif attack == "scale_110":
        out.geometry = out.geometry.apply(lambda geom: affinity.scale(geom, xfact=1.10, yfact=1.10, origin=origin))
    elif attack == "translate_1pct":
        out.geometry = out.geometry.apply(lambda geom: affinity.translate(geom, xoff=0.01 * diag, yoff=-0.01 * diag))
    else:
        raise ValueError(f"Unsupported affine attack: {attack}")
    return out


def attack_projected(gdf: gpd.GeoDataFrame, attack: str, rng: np.random.Generator) -> gpd.GeoDataFrame:
    if attack == "clean":
        return gdf.copy()
    if attack == "random_delete_30":
        return apply_random_delete(gdf, rng, 0.30)
    if attack == "random_delete_50":
        return apply_random_delete(gdf, rng, 0.50)
    if attack == "crop_50":
        return apply_center_crop(gdf, 0.50)
    if attack == "simplify_light":
        return apply_simplify(gdf, 0.0005)
    if attack == "simplify_strong":
        return apply_simplify(gdf, 0.0015)
    if attack == "coord_noise_0p001":
        return apply_coord_noise(gdf, rng, 0.001)
    if attack == "coord_noise_0p003":
        return apply_coord_noise(gdf, rng, 0.003)
    if attack in {"rotate_10", "scale_110", "translate_1pct"}:
        return apply_affine(gdf, attack)
    raise ValueError(f"Unsupported attack: {attack}")


def write_attacked_shp(src_path: Path, dst_path: Path, attack: str, rng: np.random.Generator) -> dict[str, object]:
    gdf = gpd.read_file(src_path)
    original_crs = gdf.crs
    original_count = int(len(gdf))
    if "geometry" not in gdf.columns or gdf.empty:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(dst_path)
        return {"input": str(src_path), "output": str(dst_path), "before": original_count, "after": int(len(gdf))}

    working_crs = safe_estimate_working_crs(gdf)
    projected = gdf.to_crs(working_crs) if working_crs is not None and gdf.crs is not None else gdf.copy()
    projected = repair_gdf(projected)
    attacked = attack_projected(projected, attack, rng)
    attacked = repair_gdf(attacked)
    if original_crs is not None and attacked.crs is not None and attacked.crs != original_crs:
        attacked = attacked.to_crs(original_crs)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    attacked.to_file(dst_path)
    return {
        "input": str(src_path),
        "output": str(dst_path),
        "before": original_count,
        "after": int(len(attacked)),
        "crs": str(original_crs) if original_crs is not None else None,
    }


def copy_sidecars(src_shp: Path, dst_shp: Path) -> None:
    for src in src_shp.parent.glob(src_shp.stem + ".*"):
        if src.suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
            continue
        dst = dst_shp.with_suffix(src.suffix)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create attacked raw vector packages.")
    parser.add_argument("--raw_root", type=str, default="data/raw")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--package_split", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="heldout_packages")
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--max_packages", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    raw_root = resolve_path(args.raw_root)
    output_root = resolve_path(args.output_root)
    split_path = resolve_path(args.package_split)
    package_ids = load_package_subset(split_path, args.split_name)
    package_dirs = iter_package_dirs(raw_root, package_ids)
    if args.max_packages is not None:
        package_dirs = package_dirs[: int(args.max_packages)]

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            print(f"Output already exists, resume missing shapefiles: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    manifest: list[dict[str, object]] = []
    for package_dir in package_dirs:
        for shp_path in sorted(package_dir.glob("**/*.shp")):
            rel = shp_path.relative_to(raw_root)
            dst_path = output_root / rel
            if dst_path.exists() and not args.overwrite:
                copy_sidecars(shp_path, dst_path)
                stats = {
                    "input": str(shp_path),
                    "output": str(dst_path),
                    "skipped_existing": True,
                }
            else:
                stats = write_attacked_shp(shp_path, dst_path, args.attack, rng)
                copy_sidecars(shp_path, dst_path)
            manifest.append(stats)

    with (output_root / f"attack_manifest_{args.attack}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "attack": args.attack,
                "raw_root": str(raw_root),
                "output_root": str(output_root),
                "split": str(split_path),
                "split_name": args.split_name,
                "packages": [path.name for path in package_dirs],
                "files": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote attacked dataset: {output_root}")
    print(f"Packages: {len(package_dirs)}, shapefiles: {len(manifest)}")


if __name__ == "__main__":
    main()
