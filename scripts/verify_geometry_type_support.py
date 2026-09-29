"""
Geometry-type support verification for the file-level global watermark pipeline.

This script answers a very specific question for the current project:

    "Which geometry types are actually supported by the current global pipeline?"

Support is checked in three layers:
1. Raw data layer:
   - What geometry types exist in the shapefiles?
   - Are files homogeneous or mixed?
2. Preprocessing layer:
   - Which features survive the current global normalizer?
   - Which files still have enough valid geometries to enter the global dataset?
3. Optional model/signature layer:
   - Can the current global encoder produce a file-level embedding?
   - Can the signature module generate a watermark?
   - Is the signature at least minimally stable under a small synthetic perturbation?

The script is intentionally conservative:
- It reports what the CURRENT implementation supports.
- It does not assume all geometry types are semantically useful.
- It separates "can be processed" from "is actually useful for recognition".

Usage examples
--------------
1. Only inspect data and preprocessing support:
   python scripts/verify_geometry_type_support.py --data_dir data/raw

2. Also verify model/signature generation:
   python scripts/verify_geometry_type_support.py --data_dir data/raw --checkpoint checkpoints/best_global.pth

3. Save report to JSON:
   python scripts/verify_geometry_type_support.py --data_dir data/raw --output reports/geometry_support.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import torch
import yaml

from models.aggregator import AttentionPooling
from models.geometry_encoder import create_geometry_encoder
from preprocess.sgp import AdaptiveGeometryNormalizer
from utils.signature import SignatureGenerator, compute_ber


# Only these geometry families are currently handled by preprocess/sgp.py.
# The old global pipeline does not support Point or MultiPoint in normalize().
SUPPORTED_BY_NORMALIZER = {
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
}

# Repository root. The script lives under scripts/, so its parent parent is the
# project root. This is used to make relative paths behave sensibly in IDEs
# such as PyCharm, where the working directory may not be the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FileAnalysis:
    """Per-file summary used to build both human-readable and JSON reports."""

    file_path: str
    total_features: int
    valid_features_after_preprocess: int
    empty_or_none_features: int
    homogeneous_type: Optional[str]
    dominant_type: Optional[str]
    raw_type_counts: Dict[str, int]
    preprocess_success_counts: Dict[str, int]
    preprocess_failure_counts: Dict[str, int]
    qualifies_for_global_pipeline: bool
    model_forward_ok: Optional[bool] = None
    signature_ok: Optional[bool] = None
    perturbation_ber: Optional[float] = None
    error: Optional[str] = None


class GlobalSignatureProbe:
    """
    Small wrapper around the current file-level global pipeline.

    This reuses the existing project logic:
    - encode each geometry independently
    - aggregate geometry embeddings into a file-level embedding
    - convert the file-level embedding into a binary signature
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get("config", self._default_config())

        self.geometry_encoder = create_geometry_encoder(
            self.config["geometry_encoder"]
        ).to(self.device)
        self.aggregator = AttentionPooling(
            embed_dim=self.config["geometry_encoder"]["embed_dim"],
            num_heads=self.config["geometry_encoder"].get("num_heads", 8),
            dropout=self.config["geometry_encoder"].get("dropout", 0.1),
        ).to(self.device)

        self.geometry_encoder.load_state_dict(checkpoint["geometry_encoder"])
        self.aggregator.load_state_dict(checkpoint["aggregator"])

        self.geometry_encoder.eval()
        self.aggregator.eval()

        sig_cfg = self.config.get("signature", {})
        self.signature_generator = SignatureGenerator(
            embed_dim=self.config["geometry_encoder"]["embed_dim"],
            signature_bits=sig_cfg.get("signature_bits", 256),
            quantization=sig_cfg.get("quantization", "sign"),
            use_bch=sig_cfg.get("use_bch", True),
            bch_poly=sig_cfg.get("bch_poly", 137),
            bch_bits=sig_cfg.get("bch_bits", 5),
            use_hmac=sig_cfg.get("use_hmac", True),
            hmac_key=sig_cfg.get("hmac_key", "gsd_secret_key"),
            device=str(self.device),
        )

        data_cfg = self.config.get("data", {})
        self.fixed_points = data_cfg.get("max_points", 128)
        self.max_geometries = data_cfg.get("max_geometries", 500)

    @staticmethod
    def _default_config() -> dict:
        """Fallback config for older checkpoints that may not store config."""
        return {
            "geometry_encoder": {
                "input_dim": 2,
                "embed_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "dropout": 0.0,
                "max_points": 128,
                "pooling": "cls",
            },
            "data": {
                "max_points": 128,
                "max_geometries": 500,
            },
            "signature": {
                "signature_bits": 256,
                "use_bch": True,
                "use_hmac": True,
            },
        }

    def _pad_or_truncate(self, coords: np.ndarray) -> np.ndarray:
        """
        Match the current global pipeline behavior:
        every geometry is forced to the same number of sampled points.
        """
        target_len = self.fixed_points
        current_len = len(coords)

        if current_len == target_len:
            return coords
        if current_len > target_len:
            indices = np.linspace(0, current_len - 1, target_len).astype(int)
            return coords[indices]

        padded = np.zeros((target_len, 2), dtype=coords.dtype)
        padded[:current_len] = coords
        padded[current_len:] = coords[-1]
        return padded

    def _fit_to_max_geometries(
        self, geometries: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match the current global inference behavior:
        - downsample geometry count if too large
        - cyclically pad if too small
        """
        n_original = len(geometries)
        if n_original == 0:
            raise ValueError("No valid geometries available for the global probe.")

        if n_original > self.max_geometries:
            # Deterministic sampling keeps the validation script reproducible.
            indices = np.linspace(0, n_original - 1, self.max_geometries).astype(int)
            sampled = [geometries[i] for i in indices]
            mask = np.ones(self.max_geometries, dtype=bool)
        else:
            sampled = geometries.copy()
            mask = np.zeros(self.max_geometries, dtype=bool)
            mask[:n_original] = True
            while len(sampled) < self.max_geometries:
                sampled.append(geometries[len(sampled) % n_original])

        return np.array(sampled[: self.max_geometries]), mask

    @torch.no_grad()
    def get_global_embedding(
        self, geometries: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recreate the current file-level embedding path:
        geometry encoder per geometry -> attention pooling over geometries.
        """
        fitted_geometries, mask = self._fit_to_max_geometries(geometries)

        geom_tensor = torch.FloatTensor(fitted_geometries).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        num_geoms, num_points = geom_tensor.shape[1], geom_tensor.shape[2]
        flat_geoms = geom_tensor.view(num_geoms, num_points, -1)
        geometry_embeddings = self.geometry_encoder(flat_geoms)
        geometry_embeddings = geometry_embeddings.unsqueeze(0)

        global_feat, attn_weights = self.aggregator(geometry_embeddings, mask_tensor)
        return global_feat.cpu().numpy()[0], attn_weights.cpu().numpy()[0]

    @torch.no_grad()
    def generate_signature(self, geometries: List[np.ndarray]) -> bytes:
        """Generate the current global watermark signature from valid geometries."""
        global_embedding, _ = self.get_global_embedding(geometries)
        embedding_tensor = torch.FloatTensor(global_embedding).to(self.device)
        return self.signature_generator.generate(embedding_tensor)


def geometry_type_name(geom) -> str:
    """Return a robust geometry type label, including null/empty cases."""
    if geom is None:
        return "None"
    if geom.is_empty:
        return "Empty"
    return geom.geom_type


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for both lightweight and model-aware checks."""
    parser = argparse.ArgumentParser(
        description="Verify which geometry types are truly supported by the current global shapefile pipeline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing shapefiles.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional global model checkpoint. When provided, the script also verifies file-level embedding/signature generation.",
    )
    parser.add_argument(
        "--target_crs",
        type=str,
        default="EPSG:4326",
        help="Target CRS used before normalization.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=128,
        help="Fixed number of points per geometry during preprocessing checks.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=16,
        help="Minimum point count accepted by the current normalizer.",
    )
    parser.add_argument(
        "--min_geometries",
        type=int,
        default=10,
        help="Minimum valid geometry count required for a file to qualify for the current global pipeline.",
    )
    parser.add_argument(
        "--max_files_per_type",
        type=int,
        default=5,
        help="When checkpoint is provided, limit model/signature probing to this many files per dominant type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the optional model probe.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path for the full report.",
    )
    return parser


def collect_shapefiles(data_dir: str) -> List[Path]:
    """
    Discover all real shapefile files recursively under the given root.

    This project's raw data may use a nested structure like:

        data/raw/
            anhui/
                anhui-latest-free.shp/
                    gis_osm_roads_free_1.shp
                    gis_osm_buildings_a_free_1.shp

    The middle directory itself ends with ".shp", but it is still a directory.
    We must therefore filter candidates with is_file(), otherwise geopandas
    would be asked to read a directory path and the verification report would
    be polluted by false read errors.
    """
    data_path = resolve_input_path(data_dir)
    return sorted(path for path in data_path.glob("**/*.shp") if path.is_file())


def resolve_input_path(path_str: str) -> Path:
    """
    Resolve an input path robustly across terminal and IDE execution modes.

    Resolution order:
    1. Absolute path, if already absolute
    2. Relative to current working directory
    3. Relative to repository root

    This avoids a common PyCharm issue where the script is launched correctly,
    but the working directory is not the project root, so paths like "data/raw"
    appear missing even though the data exists inside the repository.
    """
    raw_path = Path(path_str)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = Path.cwd() / raw_path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = PROJECT_ROOT / raw_path
    if project_candidate.exists():
        return project_candidate

    # Fall back to the CWD-relative path so downstream error messages remain
    # concrete and easy to understand.
    return cwd_candidate


def analyze_single_file(
    shp_path: Path,
    normalizer: AdaptiveGeometryNormalizer,
    target_crs: str,
    min_geometries: int,
) -> FileAnalysis:
    """
    Run raw-type and preprocessing checks for one shapefile.

    This function deliberately mirrors the current global pipeline:
    any geometry that cannot be normalized into a coordinate sequence is counted
    as unsupported at the preprocessing layer.
    """
    try:
        gdf = gpd.read_file(shp_path)

        if gdf.crs is not None and str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)

        raw_type_counts: Counter = Counter()
        preprocess_success_counts: Counter = Counter()
        preprocess_failure_counts: Counter = Counter()

        valid_feature_count = 0
        empty_or_none_count = 0

        for _, row in gdf.iterrows():
            geom = row.geometry
            geom_type = geometry_type_name(geom)
            raw_type_counts[geom_type] += 1

            if geom is None or geom.is_empty:
                empty_or_none_count += 1
                continue

            coords = normalizer.normalize(geom)
            if coords is None:
                preprocess_failure_counts[geom_type] += 1
                continue

            preprocess_success_counts[geom_type] += 1
            valid_feature_count += 1

        non_empty_types = [
            geom_type for geom_type in raw_type_counts.keys()
            if geom_type not in {"None", "Empty"}
        ]
        homogeneous_type = (
            non_empty_types[0] if len(non_empty_types) == 1 else None
        )
        dominant_type = None
        if non_empty_types:
            dominant_type = max(
                non_empty_types,
                key=lambda geom_type: raw_type_counts[geom_type],
            )

        return FileAnalysis(
            file_path=str(shp_path),
            total_features=int(len(gdf)),
            valid_features_after_preprocess=valid_feature_count,
            empty_or_none_features=empty_or_none_count,
            homogeneous_type=homogeneous_type,
            dominant_type=dominant_type,
            raw_type_counts=dict(raw_type_counts),
            preprocess_success_counts=dict(preprocess_success_counts),
            preprocess_failure_counts=dict(preprocess_failure_counts),
            qualifies_for_global_pipeline=valid_feature_count >= min_geometries,
        )
    except Exception as exc:
        return FileAnalysis(
            file_path=str(shp_path),
            total_features=0,
            valid_features_after_preprocess=0,
            empty_or_none_features=0,
            homogeneous_type=None,
            dominant_type=None,
            raw_type_counts={},
            preprocess_success_counts={},
            preprocess_failure_counts={},
            qualifies_for_global_pipeline=False,
            error=str(exc),
        )


def prepare_valid_geometries_for_probe(
    shp_path: str,
    normalizer: AdaptiveGeometryNormalizer,
    target_crs: str,
    fixed_points: int,
) -> List[np.ndarray]:
    """
    Load one shapefile and return only the geometries that survive the current
    normalization path. This is the exact input expected by the global probe.
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    valid_geometries: List[np.ndarray] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        coords = normalizer.normalize(geom)
        if coords is None:
            continue

        current_len = len(coords)
        if current_len > fixed_points:
            indices = np.linspace(0, current_len - 1, fixed_points).astype(int)
            coords = coords[indices]
        elif current_len < fixed_points:
            padded = np.zeros((fixed_points, 2), dtype=coords.dtype)
            padded[:current_len] = coords
            padded[current_len:] = coords[-1]
            coords = padded

        valid_geometries.append(coords)

    return valid_geometries


def add_model_probe_results(
    analyses: List[FileAnalysis],
    checkpoint_path: str,
    target_crs: str,
    max_points: int,
    min_points: int,
    max_files_per_type: int,
    device: str,
) -> None:
    """
    Add an optional model/signature probe.

    To keep this script practical on large datasets, probing is limited to a
    few files per dominant geometry type. The goal is not exhaustive benchmark
    evaluation; the goal is to verify whether the current global pipeline can
    produce meaningful signatures for that type family at all.
    """
    probe = GlobalSignatureProbe(checkpoint_path=checkpoint_path, device=device)
    normalizer = AdaptiveGeometryNormalizer(max_points=max_points, min_points=min_points)

    picked_per_type: Dict[str, int] = defaultdict(int)

    for analysis in analyses:
        if not analysis.qualifies_for_global_pipeline or analysis.error:
            continue

        dominant_type = analysis.dominant_type or "Unknown"
        if picked_per_type[dominant_type] >= max_files_per_type:
            continue

        picked_per_type[dominant_type] += 1

        try:
            valid_geometries = prepare_valid_geometries_for_probe(
                shp_path=analysis.file_path,
                normalizer=normalizer,
                target_crs=target_crs,
                fixed_points=probe.fixed_points,
            )

            if not valid_geometries:
                analysis.model_forward_ok = False
                analysis.signature_ok = False
                analysis.error = "No valid geometries survived preprocessing for model probe."
                continue

            # Verify that the current file-level encoder path runs end-to-end.
            global_embedding, _ = probe.get_global_embedding(valid_geometries)
            analysis.model_forward_ok = bool(np.isfinite(global_embedding).all())

            if not analysis.model_forward_ok:
                analysis.signature_ok = False
                analysis.error = "Global embedding contains non-finite values."
                continue

            signature_a = probe.generate_signature(valid_geometries)
            signature_b = probe.generate_signature(valid_geometries)
            analysis.signature_ok = signature_a == signature_b

            # Add a minimal perturbation sanity check:
            # shift coordinates slightly to see whether the resulting BER stays finite.
            perturbed = []
            for coords in valid_geometries:
                # Small synthetic perturbation; this is not a full robustness test,
                # only a cheap way to ensure the signature path reacts sensibly.
                perturbed.append((coords + 1e-4).astype(coords.dtype))

            signature_p = probe.generate_signature(perturbed)
            match, ber = probe.signature_generator.verify(signature_a, signature_p, threshold=1.0)
            _ = match  # The report stores BER directly; thresholded decision is not the focus here.
            analysis.perturbation_ber = float(ber)
        except Exception as exc:
            analysis.model_forward_ok = False
            analysis.signature_ok = False
            analysis.error = str(exc)


def build_summary(analyses: List[FileAnalysis], min_geometries: int) -> Dict:
    """
    Convert per-file analyses into the aggregate view needed to judge support.

    The report distinguishes:
    - raw presence of a geometry type
    - preprocessing success
    - file qualification for the current global pipeline
    - optional model/signature probe outcomes
    """
    raw_feature_counts: Counter = Counter()
    preprocess_success_counts: Counter = Counter()
    preprocess_failure_counts: Counter = Counter()
    dominant_type_file_counts: Counter = Counter()
    homogeneous_type_file_counts: Counter = Counter()

    qualified_files = 0
    files_with_errors = 0

    model_probe_by_type: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "model_forward_ok": 0,
            "model_forward_fail": 0,
            "signature_ok": 0,
            "signature_fail": 0,
        }
    )
    perturbation_ber_by_type: Dict[str, List[float]] = defaultdict(list)

    for analysis in analyses:
        if analysis.error:
            files_with_errors += 1

        if analysis.qualifies_for_global_pipeline:
            qualified_files += 1

        if analysis.dominant_type:
            dominant_type_file_counts[analysis.dominant_type] += 1
        if analysis.homogeneous_type:
            homogeneous_type_file_counts[analysis.homogeneous_type] += 1

        raw_feature_counts.update(analysis.raw_type_counts)
        preprocess_success_counts.update(analysis.preprocess_success_counts)
        preprocess_failure_counts.update(analysis.preprocess_failure_counts)

        dominant_type = analysis.dominant_type or "Unknown"
        if analysis.model_forward_ok is not None:
            key = "model_forward_ok" if analysis.model_forward_ok else "model_forward_fail"
            model_probe_by_type[dominant_type][key] += 1
        if analysis.signature_ok is not None:
            key = "signature_ok" if analysis.signature_ok else "signature_fail"
            model_probe_by_type[dominant_type][key] += 1
        if analysis.perturbation_ber is not None:
            perturbation_ber_by_type[dominant_type].append(analysis.perturbation_ber)

    type_support = {}
    all_observed_types = sorted(
        {
            *raw_feature_counts.keys(),
            *preprocess_success_counts.keys(),
            *preprocess_failure_counts.keys(),
        }
    )

    for geom_type in all_observed_types:
        raw_count = raw_feature_counts.get(geom_type, 0)
        success_count = preprocess_success_counts.get(geom_type, 0)
        failure_count = preprocess_failure_counts.get(geom_type, 0)
        preprocess_rate = (success_count / raw_count) if raw_count > 0 else 0.0

        # Support labels are deliberately strict. "Supported" means the current
        # implementation handles the type in preprocessing, not that the final
        # scientific claim is already proven.
        if geom_type in {"None", "Empty"}:
            support_label = "not_applicable"
        elif preprocess_rate == 0.0:
            support_label = "not_supported"
        elif preprocess_rate < 0.8:
            support_label = "partially_supported"
        else:
            support_label = "supported_at_preprocess_level"

        type_support[geom_type] = {
            "raw_feature_count": int(raw_count),
            "preprocess_success_count": int(success_count),
            "preprocess_failure_count": int(failure_count),
            "preprocess_success_rate": round(preprocess_rate, 4),
            "implemented_in_current_normalizer": geom_type in SUPPORTED_BY_NORMALIZER,
            "support_label": support_label,
        }

    model_probe_summary = {}
    for geom_type, counters in model_probe_by_type.items():
        bers = perturbation_ber_by_type.get(geom_type, [])
        model_probe_summary[geom_type] = {
            **counters,
            "mean_small_perturbation_ber": round(float(np.mean(bers)), 6) if bers else None,
            "std_small_perturbation_ber": round(float(np.std(bers)), 6) if bers else None,
        }

    return {
        "overview": {
            "total_files": len(analyses),
            "files_with_errors": files_with_errors,
            "qualified_files_for_current_global_pipeline": qualified_files,
            "qualification_rule": f"valid_features_after_preprocess >= {min_geometries}",
        },
        "type_support": type_support,
        "file_level_distribution": {
            "dominant_type_file_counts": dict(dominant_type_file_counts),
            "homogeneous_type_file_counts": dict(homogeneous_type_file_counts),
        },
        "model_probe": model_probe_summary,
    }


def print_human_readable_report(summary: Dict) -> None:
    """Print the key decisions in a form you can use immediately."""
    print("=" * 80)
    print("Geometry Type Support Report for Current Global Shapefile Pipeline")
    print("=" * 80)

    overview = summary["overview"]
    print(f"Total files scanned: {overview['total_files']}")
    print(f"Files with read/process errors: {overview['files_with_errors']}")
    print(f"Files qualifying for current global pipeline: {overview['qualified_files_for_current_global_pipeline']}")
    print(f"Qualification rule: {overview['qualification_rule']}")
    print()

    print("Per-type preprocessing support:")
    for geom_type, stats in summary["type_support"].items():
        print(
            f"  {geom_type:<18} "
            f"raw={stats['raw_feature_count']:<8} "
            f"success={stats['preprocess_success_count']:<8} "
            f"fail={stats['preprocess_failure_count']:<8} "
            f"rate={stats['preprocess_success_rate']:<8} "
            f"label={stats['support_label']}"
        )

    if summary["model_probe"]:
        print()
        print("Optional model/signature probe:")
        for geom_type, stats in summary["model_probe"].items():
            print(
                f"  {geom_type:<18} "
                f"forward_ok={stats['model_forward_ok']:<4} "
                f"forward_fail={stats['model_forward_fail']:<4} "
                f"signature_ok={stats['signature_ok']:<4} "
                f"signature_fail={stats['signature_fail']:<4} "
                f"mean_small_perturbation_ber={stats['mean_small_perturbation_ber']}"
            )

    print("=" * 80)


def main() -> None:
    """Entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    shp_files = collect_shapefiles(args.data_dir)
    if not shp_files:
        raise ValueError(f"No shapefiles found under: {args.data_dir}")

    print(f"Found {len(shp_files)} shapefiles to analyze.")

    normalizer = AdaptiveGeometryNormalizer(
        max_points=args.max_points,
        min_points=args.min_points,
    )

    analyses: List[FileAnalysis] = []
    start_time = time.time()
    running_raw_type_counts: Counter = Counter()
    running_success_counts: Counter = Counter()
    running_failure_counts: Counter = Counter()
    running_qualified_files = 0

    for index, shp_path in enumerate(shp_files, start=1):
        analysis = analyze_single_file(
            shp_path=shp_path,
            normalizer=normalizer,
            target_crs=args.target_crs,
            min_geometries=args.min_geometries,
        )
        analyses.append(analysis)

        running_raw_type_counts.update(analysis.raw_type_counts)
        running_success_counts.update(analysis.preprocess_success_counts)
        running_failure_counts.update(analysis.preprocess_failure_counts)
        if analysis.qualifies_for_global_pipeline:
            running_qualified_files += 1

        # Print progress frequently enough to reassure the user, but not so
        # often that I/O dominates runtime on large datasets.
        if index == 1 or index % 20 == 0 or index == len(shp_files):
            elapsed = time.time() - start_time
            dominant_seen = running_raw_type_counts.most_common(5)
            dominant_seen_text = ", ".join(
                f"{geom_type}:{count}" for geom_type, count in dominant_seen
            ) or "none"

            print(
                f"[{index}/{len(shp_files)}] "
                f"elapsed={elapsed:.1f}s "
                f"qualified_files={running_qualified_files} "
                f"top_types={dominant_seen_text}"
            )

    if args.checkpoint:
        print("Starting optional model/signature probe...")
        add_model_probe_results(
            analyses=analyses,
            checkpoint_path=args.checkpoint,
            target_crs=args.target_crs,
            max_points=args.max_points,
            min_points=args.min_points,
            max_files_per_type=args.max_files_per_type,
            device=args.device,
        )

    summary = build_summary(analyses, min_geometries=args.min_geometries)
    report = {
        "summary": summary,
        "files": [analysis.__dict__ for analysis in analyses],
    }

    print_human_readable_report(summary)
    total_elapsed = time.time() - start_time
    print(f"Total elapsed time: {total_elapsed:.1f}s")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
