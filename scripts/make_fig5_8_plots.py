from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_sr_msgs_zero_watermark_authentication import (  # noqa: E402
    MixedGenericLinePolygonPointEvaluator,
    MixedLinePolygonEvaluator,
    PointHybridDualViewEvaluator,
    QueryAttack,
    build_dataset,
    build_registered_scores,
    parse_scale_specs,
    score_sets,
    set_seed,
)
from train_roads_hier_stage1_minimal import load_config  # noqa: E402


OUT = PROJECT_ROOT / "outputs" / "paper_figures"
REPORTS = PROJECT_ROOT / "reports"
SEEDS = [7, 17, 27]
ACTIVE_SEEDS = SEEDS


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
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_1200(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}_1200dpi.png", dpi=1200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_mean_std(value: str) -> tuple[float, float]:
    mean, std = value.split("±")
    return float(mean), float(std)


def load_report(seed: int) -> dict:
    path = REPORTS / f"sr_msgs_multigranularity_full_strictindcal_zero_watermark_auth_seed{seed}_e3_k5_k5_far0p05_beta0p25.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compute_or_load_scores(seed: int) -> dict[str, np.ndarray | dict]:
    cache_path = OUT / f"fig5_scores_seed{seed}.npz"
    if cache_path.exists():
        print(f"Load cached Fig.5 scores: seed={seed}", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        return {
            "genuine": data["genuine"],
            "impostor": data["impostor"],
            "thresholds": json.loads(str(data["thresholds_json"])),
        }

    report = load_report(seed)
    print(f"Compute Fig.5 score distribution: seed={seed}", flush=True)
    args = SimpleNamespace(**report["arguments"])
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    if args.point_cache_root:
        config["data"]["point_cache_root"] = args.point_cache_root

    config["device"]["seed"] = int(args.seed)
    set_seed(config["device"]["seed"])
    scale_specs = parse_scale_specs(args.scale_specs)

    lp_evaluator = MixedLinePolygonEvaluator(args.lp_checkpoint, device=args.device)
    print(f"  loaded LP evaluator: seed={seed}", flush=True)
    point_evaluator = PointHybridDualViewEvaluator(args.point_checkpoint, device=args.device)
    print(f"  loaded point evaluator: seed={seed}", flush=True)
    generic_evaluator = MixedGenericLinePolygonPointEvaluator(args.generic_checkpoint, device=args.device)
    print(f"  loaded generic evaluator: seed={seed}", flush=True)

    calibration_dataset = build_dataset(config, args, args.calibration_split_name)
    test_dataset = build_dataset(config, args, args.test_split_name)
    print(f"  built datasets: calibration={len(calibration_dataset)}, test={len(test_dataset)}", flush=True)

    print(f"  scoring calibration split: seed={seed}", flush=True)
    calibration_scores, _, _ = build_registered_scores(
        calibration_dataset,
        config=config,
        args=args,
        scale_specs=scale_specs,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
        template_aggregation=args.template_aggregation,
        query_attack=QueryAttack(name="clean"),
    )
    print(f"  scoring test split: seed={seed}", flush=True)
    test_scores, _, _ = build_registered_scores(
        test_dataset,
        config=config,
        args=args,
        scale_specs=scale_specs,
        lp_evaluator=lp_evaluator,
        point_evaluator=point_evaluator,
        generic_evaluator=generic_evaluator,
        template_aggregation=args.template_aggregation,
        query_attack=QueryAttack(name="clean"),
    )

    genuine, impostor = score_sets(test_scores)
    thresholds = {
        "FAR=1%": report["calibration"]["far_sweep"]["0.0100"]["threshold"],
        "FAR=5%": report["calibration"]["far_sweep"]["0.0500"]["threshold"],
        "FAR=10%": report["calibration"]["far_sweep"]["0.1000"]["threshold"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        genuine=genuine,
        impostor=impostor,
        calibration_scores=calibration_scores,
        test_scores=test_scores,
        thresholds_json=json.dumps(thresholds),
    )
    print(f"  cached Fig.5 scores: seed={seed}", flush=True)
    return {"genuine": genuine, "impostor": impostor, "thresholds": thresholds}


def kde(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.zeros_like(x)
    std = np.std(values, ddof=1)
    bw = 1.06 * std * values.size ** (-1 / 5)
    bw = max(float(bw), 1e-3)
    z = (x[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (values.size * bw * math_sqrt_2pi())


def math_sqrt_2pi() -> float:
    return float(np.sqrt(2.0 * np.pi))


def make_fig5(skip_scores: bool = False) -> None:
    if skip_scores:
        print("Fig.5 skipped by argument.")
        return
    bundles = [compute_or_load_scores(seed) for seed in ACTIVE_SEEDS]
    genuine = np.concatenate([b["genuine"] for b in bundles])
    impostor = np.concatenate([b["impostor"] for b in bundles])
    thresholds = {
        key: float(np.mean([b["thresholds"][key] for b in bundles]))
        for key in ["FAR=1%", "FAR=5%", "FAR=10%"]
    }

    low = min(np.percentile(impostor, 0.2), np.percentile(genuine, 0.2))
    high = max(np.percentile(impostor, 99.8), np.percentile(genuine, 99.8))
    x = np.linspace(low, high, 500)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bins = np.linspace(low, high, 45)
    ax.hist(impostor, bins=bins, density=True, alpha=0.22, color="#f28e2b", label="Impostor scores")
    ax.hist(genuine, bins=bins, density=True, alpha=0.26, color="#1f4e79", label="Genuine scores")
    ax.plot(x, kde(impostor, x), color="#f28e2b", linewidth=1.2)
    ax.plot(x, kde(genuine, x), color="#1f4e79", linewidth=1.2)

    line_styles = {"FAR=1%": "--", "FAR=5%": "-.", "FAR=10%": ":"}
    for label, threshold in thresholds.items():
        ax.axvline(threshold, color="#333333", linestyle=line_styles[label], linewidth=0.9)
        ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.82, label, rotation=90, va="top", ha="left", fontsize=7.2)

    ax.set_title("Authentication Score Distribution")
    ax.set_xlabel("Authentication score")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    save_1200(fig, "fig5_score_distribution")


AUTH_ABLATION = {
    "Single low-budget": {"AUC": (0.8920, 0.0118), "EER": (0.1985, 0.0135), "TAR@FAR5": (0.4606, 0.0477), "MRR": (0.4584, 0.0217)},
    "Single high-budget": {"AUC": (0.8972, 0.0121), "EER": (0.1895, 0.0237), "TAR@FAR5": (0.4848, 0.0309), "MRR": (0.4996, 0.0376)},
    "Budget-matched MG": {"AUC": (0.9654, 0.0048), "EER": (0.0905, 0.0163), "TAR@FAR5": (0.7091, 0.0393), "MRR": (0.6658, 0.0092)},
    "Full MG": {"AUC": (0.9614, 0.0028), "EER": (0.1076, 0.0019), "TAR@FAR5": (0.6788, 0.0309), "MRR": (0.6863, 0.0288)},
}


def make_fig6() -> None:
    metrics = ["AUC", "EER", "TAR@FAR5", "MRR"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6))
    colors = {
        "Single high-budget": "#6f7f8f",
        "Budget-matched MG": "#1f4e79",
        "Single low-budget": "#aab4bf",
        "Full MG": "#2a9d8f",
    }
    for ax, metric in zip(axes, metrics):
        y_positions = np.arange(len(AUTH_ABLATION))[::-1]
        vals = [AUTH_ABLATION[m][metric][0] for m in AUTH_ABLATION]
        stds = [AUTH_ABLATION[m][metric][1] for m in AUTH_ABLATION]
        methods = list(AUTH_ABLATION)
        key_y = {m: y for m, y in zip(methods, y_positions)}
        ax.plot(
            [AUTH_ABLATION["Single high-budget"][metric][0], AUTH_ABLATION["Budget-matched MG"][metric][0]],
            [key_y["Single high-budget"], key_y["Budget-matched MG"]],
            color="#666666",
            linewidth=0.8,
            zorder=1,
        )
        for method, y, val, std in zip(methods, y_positions, vals, stds):
            ax.errorbar(
                val,
                y,
                xerr=std,
                fmt="o",
                markersize=4.0,
                color=colors[method],
                ecolor=colors[method],
                elinewidth=0.8,
                capsize=2,
                zorder=3,
            )
            xmin_tmp = min(vals)
            xmax_tmp = max(vals)
            span_tmp = xmax_tmp - xmin_tmp
            if val > xmin_tmp + 0.82 * span_tmp:
                ax.text(val - 0.004, y, f"{val:.3f}", va="center", ha="right", fontsize=6.7)
            else:
                ax.text(val + 0.004, y, f"{val:.3f}", va="center", ha="left", fontsize=6.7)
        ax.set_title(metric)
        if metric == "EER":
            ax.set_xlabel("Score (lower is better)")
        else:
            ax.set_xlabel("Score")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(methods if ax is axes[0] else [])
        ax.grid(True, axis="x", linestyle="--", linewidth=0.35, alpha=0.3)
        pad = 0.060 if metric != "EER" else 0.045
        ax.set_xlim(min(vals) - pad, max(vals) + pad)
    fig.suptitle("Budget-controlled Multi-granularity Ablation", y=1.03, fontsize=8)
    save_1200(fig, "fig6_multigranularity_dumbbell")


LEVEL0 = {
    "Single low-budget": {"AUC": 0.7786, "EER": 0.3017, "TAR@FAR5": 0.3515, "R@1": 0.2303, "R@5": 0.4424, "MRR": 0.3484},
    "Single high-budget": {"AUC": 0.7941, "EER": 0.3079, "TAR@FAR5": 0.3515, "R@1": 0.2606, "R@5": 0.4424, "MRR": 0.3594},
    "Budget-matched MG": {"AUC": 0.9419, "EER": 0.1195, "TAR@FAR5": 0.6303, "R@1": 0.4061, "R@5": 0.7636, "MRR": 0.5564},
    "Full MG": {"AUC": 0.9433, "EER": 0.1191, "TAR@FAR5": 0.6242, "R@1": 0.4303, "R@5": 0.8121, "MRR": 0.5931},
}


def make_fig7() -> None:
    methods = list(LEVEL0)
    metrics = ["AUC", "EER↓", "TAR@FAR5", "R@1", "R@5", "MRR"]
    raw_metrics = ["AUC", "EER", "TAR@FAR5", "R@1", "R@5", "MRR"]
    raw = np.array([[LEVEL0[m][metric] for metric in raw_metrics] for m in methods])
    norm = raw.copy()
    for j, metric in enumerate(raw_metrics):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        score = (col - lo) / (hi - lo if hi > lo else 1.0)
        if metric == "EER":
            score = 1.0 - score
        norm[:, j] = score

    fig, ax = plt.subplots(figsize=(5.7, 2.8))
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Level-0 Multi-granularity Performance Heatmap")
    for i in range(len(methods)):
        for j in range(len(metrics)):
            color = "white" if norm[i, j] > 0.58 else "#222222"
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=6.7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Normalized performance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    save_1200(fig, "fig7_level0_heatmap")


K_TEMPLATE = {
    1: {"AUC": (0.9496, 0.0051), "TAR@FAR5": (0.7273, 0.0148), "MRR": (0.6524, 0.0341)},
    3: {"AUC": (0.9654, 0.0042), "TAR@FAR5": (0.7697, 0.0227), "MRR": (0.7252, 0.0457)},
    5: {"AUC": (0.9712, 0.0050), "TAR@FAR5": (0.8424, 0.0454), "MRR": (0.7758, 0.0752)},
}


def make_fig8() -> None:
    metrics = ["AUC", "TAR@FAR5", "MRR"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.45))
    ks = np.array(list(K_TEMPLATE))
    for ax, metric in zip(axes, metrics):
        vals = np.array([K_TEMPLATE[k][metric][0] for k in ks])
        stds = np.array([K_TEMPLATE[k][metric][1] for k in ks])
        ax.errorbar(ks, vals, yerr=stds, fmt="o-", color="#1f4e79", markersize=4.2, linewidth=1.1, capsize=2.2)
        for k, val in zip(ks, vals):
            if k == ks.min():
                text_x, ha = k + 0.12, "left"
            elif k == ks.max():
                text_x, ha = k - 0.12, "right"
            else:
                text_x, ha = k, "center"
            ax.text(text_x, val + 0.014, f"{val:.3f}", ha=ha, va="bottom", fontsize=6.5, clip_on=False)
        ax.set_title(metric)
        ax.set_xlabel("Number of templates K")
        ax.set_ylabel("")
        ax.set_xticks(ks)
        ax.set_xlim(0.45, 5.55)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.3)
        ax.set_ylim(max(0, vals.min() - stds.max() - 0.05), min(1.02, vals.max() + stds.max() + 0.07))
    fig.suptitle("Effect of Template Number", y=1.04, fontsize=8)
    save_1200(fig, "fig8_k_template_trend")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_scores", action="store_true")
    parser.add_argument("--seeds", type=str, default="7,17,27")
    args = parser.parse_args()
    global ACTIVE_SEEDS
    ACTIVE_SEEDS = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    set_style()
    make_fig5(skip_scores=args.skip_scores)
    make_fig6()
    make_fig7()
    make_fig8()
    print(f"Saved Fig.5-Fig.8 to: {OUT}")


if __name__ == "__main__":
    main()
