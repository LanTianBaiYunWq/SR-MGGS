from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_figures"
RAW = ROOT / "data" / "raw"
ATTACK_ROOTS = {
    "Clean": RAW,
    "Random delete 30%": ROOT / "reports" / "raw_geometry_attack_random_delete_30",
    "Crop 50%": ROOT / "reports" / "raw_geometry_attack_crop_50",
    "Light simplification": ROOT / "reports" / "raw_geometry_attack_simplify_light",
}

ROBUSTNESS = [
    {
        "attack": "Clean",
        "n": 55,
        "auc": 0.9614,
        "auc_std": 0.0028,
        "eer": 0.1076,
        "eer_std": 0.0019,
        "tar5": 0.6788,
        "tar5_std": 0.0309,
        "top5": 0.8667,
        "mrr": 0.6863,
        "color": "#1f4e79",
    },
    {
        "attack": "Random delete 30%",
        "n": 38,
        "auc": 0.9146,
        "auc_std": 0.0009,
        "eer": 0.1719,
        "eer_std": 0.0119,
        "tar5": 0.4737,
        "tar5_std": 0.0645,
        "top5": 0.7632,
        "mrr": 0.5788,
        "color": "#f28e2b",
    },
    {
        "attack": "Crop 50%",
        "n": 55,
        "auc": 0.8924,
        "auc_std": 0.0029,
        "eer": 0.2032,
        "eer_std": 0.0100,
        "tar5": 0.4000,
        "tar5_std": 0.0297,
        "top5": 0.5818,
        "mrr": 0.4457,
        "color": "#a23b2a",
    },
    {
        "attack": "Light simplification",
        "n": 37,
        "auc": 0.9582,
        "auc_std": 0.0035,
        "eer": 0.1094,
        "eer_std": 0.0058,
        "tar5": 0.6577,
        "tar5_std": 0.0835,
        "top5": 0.9279,
        "mrr": 0.6850,
        "color": "#2a9d8f",
    },
]

ROC_POINTS = {
    "Train-as-calibration": {
        "auc": 0.9710,
        "eer": 0.0693,
        "points": [(0.01, 0.2848), (0.05, 0.8606), (0.10, 0.9758)],
        "color": "#3b6fb6",
    },
    "Strict independent calibration": {
        "auc": 0.9614,
        "eer": 0.1076,
        "points": [(0.01, 0.2182), (0.05, 0.6788), (0.10, 0.8727)],
        "color": "#e67e22",
    },
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#222222",
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}_600dpi.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}_1200dpi.png", dpi=1200, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_fig9() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for item in ROBUSTNESS:
        size = 900 * item["top5"]
        ax.scatter(
            item["auc"],
            item["tar5"],
            s=size,
            color=item["color"],
            alpha=0.72,
            edgecolor="#222222",
            linewidth=0.7,
            zorder=3,
        )
        ax.errorbar(
            item["auc"],
            item["tar5"],
            xerr=item["auc_std"],
            yerr=item["tar5_std"],
            fmt="none",
            ecolor="#555555",
            elinewidth=0.8,
            capsize=2.2,
            zorder=2,
        )

        dx, dy, ha = {
            "Clean": (-0.018, 0.040, "left"),
            "Random delete 30%": (0.006, 0.052, "left"),
            "Crop 50%": (0.006, -0.018, "left"),
            "Light simplification": (-0.050, -0.060, "left"),
        }[item["attack"]]
        ax.text(
            item["auc"] + dx,
            item["tar5"] + dy,
            f"{item['attack']}\nN={item['n']}, Top-5={item['top5']:.3f}",
            ha=ha,
            va="center",
            fontsize=7.7,
            color="#222222",
        )

    ax.set_title("Robustness under Raw-Geometry Perturbations")
    ax.set_xlabel("AUC")
    ax.set_ylabel("TAR@FAR=5%")
    ax.set_xlim(0.870, 0.982)
    ax.set_ylim(0.350, 0.755)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    ax.text(
        0.981,
        0.362,
        "Bubble size: Top-5 attribution",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    save_figure(fig, "fig9_robustness_bubble")


def common_package_names() -> list[str]:
    sets = []
    for root in ATTACK_ROOTS.values():
        if not root.exists():
            return []
        sets.append({p.name for p in root.iterdir() if p.is_dir()})
    return sorted(set.intersection(*sets))


def shapefiles(package_dir: Path) -> list[Path]:
    return sorted(p for p in package_dir.rglob("*.shp") if p.is_file())


def read_package_geoms(
    package_dir: Path,
    max_features_per_file: int = 250,
    max_files: int = 12,
) -> list[gpd.GeoDataFrame]:
    frames: list[gpd.GeoDataFrame] = []
    for shp in shapefiles(package_dir)[:max_files]:
        try:
            gdf = gpd.read_file(shp, rows=max_features_per_file)
        except Exception:
            continue
        if gdf.empty or "geometry" not in gdf:
            continue
        gdf = gdf[gdf.geometry.notna()]
        if gdf.empty:
            continue
        frames.append(gdf[["geometry"]].copy())
    return frames


def total_features(package_name: str) -> int:
    total = 0
    for root in ATTACK_ROOTS.values():
        for shp in shapefiles(root / package_name):
            try:
                total += len(gpd.read_file(shp, rows=20))
            except Exception:
                pass
    return total


def choose_package() -> str | None:
    names = common_package_names()
    if not names:
        return None
    preferred = ["Amsterdam-shp", "Auckland-shp", "Boulder-shp", "Calgary-shp", "Dublin-shp", "安徽", "海南"]
    for name in preferred:
        if name in names:
            return name
    return max(names[:40], key=total_features)


def plot_frames(ax: plt.Axes, frames: Iterable[gpd.GeoDataFrame]) -> None:
    for gdf in frames:
        geom_types = set(gdf.geometry.geom_type)
        try:
            if any("Polygon" in t for t in geom_types):
                gdf.boundary.plot(ax=ax, color="#4178a8", linewidth=0.20, alpha=0.55)
            elif any("Point" in t for t in geom_types):
                gdf.plot(ax=ax, color="#d55e00", markersize=0.35, alpha=0.35)
            else:
                gdf.plot(ax=ax, color="#1f4e79", linewidth=0.22, alpha=0.58)
        except Exception:
            continue


def bounds_from_frames(frames: Iterable[gpd.GeoDataFrame]) -> tuple[float, float, float, float] | None:
    bounds = []
    for gdf in frames:
        try:
            b = gdf.total_bounds
            if np.all(np.isfinite(b)):
                bounds.append(b)
        except Exception:
            continue
    if not bounds:
        return None
    arr = np.vstack(bounds)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())


def make_fig10() -> None:
    package = choose_package()
    if package is None:
        print("Fig.10 skipped: no common package found.")
        return

    loaded = {name: read_package_geoms(root / package) for name, root in ATTACK_ROOTS.items()}
    if not loaded["Clean"]:
        print("Fig.10 skipped: clean package contains no readable geometry.")
        return

    clean_bounds = bounds_from_frames(loaded["Clean"])
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.15), sharex=False, sharey=False)
    panel_labels = ["(a) Clean", "(b) Random delete 30%", "(c) Crop 50%", "(d) Light simplification"]

    for ax, (name, frames), label in zip(axes, loaded.items(), panel_labels):
        ax.set_title(label, fontsize=4.5)
        plot_frames(ax, frames)
        if clean_bounds is not None:
            minx, miny, maxx, maxy = clean_bounds
            dx = (maxx - minx) * 0.03 or 1.0
            dy = (maxy - miny) * 0.03 or 1.0
            ax.set_xlim(minx - dx, maxx + dx)
            ax.set_ylim(miny - dy, maxy + dy)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#333333")

    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.03, top=0.82, wspace=0.14)
    save_figure(fig, "fig10_attack_examples")

    for label, (name, frames) in zip(panel_labels, loaded.items()):
        fig_single, ax_single = plt.subplots(figsize=(4.8, 3.1))
        ax_single.set_title(label, fontsize=7.5, pad=5)
        plot_frames(ax_single, frames)
        if clean_bounds is not None:
            minx, miny, maxx, maxy = clean_bounds
            dx = (maxx - minx) * 0.03 or 1.0
            dy = (maxy - miny) * 0.03 or 1.0
            ax_single.set_xlim(minx - dx, maxx + dx)
            ax_single.set_ylim(miny - dy, maxy + dy)
        ax_single.set_aspect("equal", adjustable="box")
        ax_single.set_xticks([])
        ax_single.set_yticks([])
        for spine in ax_single.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")
        fig_single.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.86)
        single_stem = {
            "Clean": "fig10a_clean",
            "Random delete 30%": "fig10b_random_delete_30",
            "Crop 50%": "fig10c_crop_50",
            "Light simplification": "fig10d_light_simplification",
        }[name]
        save_figure(fig_single, single_stem)


def make_fig11() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    ax = axes[0]
    for name, data in ROC_POINTS.items():
        xs = [p[0] for p in data["points"]]
        ys = [p[1] for p in data["points"]]
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=1.2, color=data["color"], label=f"{name} (AUC={data['auc']:.4f})")
        for x, y in data["points"]:
            ax.text(x + 0.006, y, f"{int(x*100)}%", fontsize=7.5, va="center", color=data["color"])
    ax.plot([0, 0.13], [0, 0.13], linestyle="--", linewidth=0.8, color="#999999")
    ax.set_title("(a) ROC operating points")
    ax.set_xlabel("False acceptance rate")
    ax.set_ylabel("True acceptance rate")
    ax.set_xlim(0, 0.13)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for name, data in ROC_POINTS.items():
        far = np.array([p[0] for p in data["points"]])
        frr = np.array([1.0 - p[1] for p in data["points"]])
        ax.plot(far, frr, marker="o", markersize=4.5, linewidth=1.2, color=data["color"], label=f"{name} (EER={data['eer']:.4f})")
        for x, y in zip(far, frr):
            ax.text(x + 0.006, y, f"{y:.3f}", fontsize=7.5, va="center", color=data["color"])
    ax.set_title("(b) FAR-FRR trade-off")
    ax.set_xlabel("False acceptance rate")
    ax.set_ylabel("False rejection rate")
    ax.set_xlim(0, 0.13)
    ax.set_ylim(0, 0.85)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Closed-loop Authentication Operating Points", y=1.04, fontsize=8)
    save_figure(fig, "fig11_auth_operating_points")


def main() -> None:
    set_style()
    make_fig9()
    make_fig10()
    make_fig11()
    print(f"Saved figures to: {OUT}")


if __name__ == "__main__":
    main()
