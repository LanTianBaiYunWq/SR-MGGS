"""
批量运行 GeometryCore 文件级全局实验的 token budget sweep。

当前阶段的目标很明确：
1. 固定其余训练设置；
2. 仅扫描 file-level token budget（当前实现里对应 max_geometries）；
3. 对每个 budget 自动执行训练与评估；
4. 汇总输出一张 budget sweep 结果表，便于后续判断 sampling 是否是主要瓶颈。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    """兼容命令行和 IDE 的相对路径解析。"""
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = PROJECT_ROOT / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置，便于补充实验配置列。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_command(command: List[str], cwd: Path) -> None:
    """执行子命令，失败时直接中止 sweep。"""
    print("=" * 100)
    print("Running command:")
    print(" ".join(command))
    print("=" * 100)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Run GeometryCore token budget sweep.")
    parser.add_argument("--config", type=str, default="configs/geometrycore_global.yaml",
                        help="基础配置文件路径。")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="原始 shapefile 根目录。")
    parser.add_argument("--file_pattern", type=str, default="*.shp",
                        help="文件过滤模式，例如 *roads*.shp。")
    parser.add_argument("--max_files", type=int, default=None,
                        help="最多使用多少个文件。")
    parser.add_argument("--budgets", type=int, nargs="+", required=True,
                        help="要扫描的 token budget 列表，例如 128 256 512 1024。")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖训练轮数。")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖文件级 batch size。")
    parser.add_argument("--device_python", type=str, default=sys.executable,
                        help="执行训练和评估所用的 Python 解释器。")
    parser.add_argument("--output_dir", type=str, default="reports/budget_sweep",
                        help="sweep 输出目录。")
    parser.add_argument("--no_teacher", action="store_true",
                        help="显式关闭视觉教师。")
    parser.add_argument("--export_tables", action="store_true",
                        help="每个 budget 额外导出 pair/file/detail 三张表。")
    return parser


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(config_path)

    train_script = PROJECT_ROOT / "train_global.py"
    eval_script = PROJECT_ROOT / "scripts" / "evaluate_geometrycore_global.py"
    export_script = PROJECT_ROOT / "scripts" / "export_geometrycore_analysis_tables.py"

    summary_rows: List[Dict[str, Any]] = []

    for budget in args.budgets:
        exp_id = f"geometrycore_budget_{budget}"
        exp_dir = output_dir / exp_id
        ckpt_dir = exp_dir / "checkpoints"
        log_dir = exp_dir / "logs"
        eval_json = exp_dir / "evaluation.json"
        ckpt_path = ckpt_dir / "best_global.pth"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            args.device_python,
            str(train_script),
            "--config", str(config_path),
            "--dataset_backend", "geometrycore",
            "--data_dir", str(data_dir),
            "--file_pattern", args.file_pattern,
            "--max_geometries", str(budget),
            "--save_dir", str(ckpt_dir),
            "--log_dir", str(log_dir),
        ]
        if args.max_files is not None:
            train_cmd += ["--max_files", str(args.max_files)]
        if args.epochs is not None:
            train_cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            train_cmd += ["--batch_size", str(args.batch_size)]
        if args.no_teacher:
            train_cmd += ["--no_teacher"]

        run_command(train_cmd, PROJECT_ROOT)

        eval_cmd = [
            args.device_python,
            str(eval_script),
            "--checkpoint", str(ckpt_path),
            "--config", str(config_path),
            "--data_dir", str(data_dir),
            "--file_pattern", args.file_pattern,
            "--max_geometries", str(budget),
            "--output", str(eval_json),
        ]
        if args.max_files is not None:
            eval_cmd += ["--max_files", str(args.max_files)]

        run_command(eval_cmd, PROJECT_ROOT)

        if args.export_tables:
            analysis_dir = exp_dir / "analysis_tables"
            export_cmd = [
                args.device_python,
                str(export_script),
                "--checkpoint", str(ckpt_path),
                "--config", str(config_path),
                "--data_dir", str(data_dir),
                "--file_pattern", args.file_pattern,
                "--max_geometries", str(budget),
                "--output_dir", str(analysis_dir),
            ]
            if args.max_files is not None:
                export_cmd += ["--max_files", str(args.max_files)]
            run_command(export_cmd, PROJECT_ROOT)

        with eval_json.open("r", encoding="utf-8") as f:
            eval_report = json.load(f)

        row = {
            "exp_id": exp_id,
            "token_budget": budget,
            "aggregation_type": "attention_pooling",
            "batch_size": args.batch_size if args.batch_size is not None else base_config["training"].get("batch_size_files", 4),
            "temperature": base_config["loss"].get("contrastive", {}).get("temperature", base_config["loss"].get("temperature", 0.07)),
            "projection_dim": 128,
            "hash_bits": base_config.get("signature", {}).get("signature_bits", 256),
            "auc": eval_report.get("auc"),
            "eer": eval_report.get("eer"),
            "separability": eval_report.get("separability"),
            "genuine_mean": eval_report.get("genuine_ber_mean"),
            "impostor_mean": eval_report.get("impostor_ber_mean"),
            "checkpoint": str(ckpt_path),
            "evaluation_json": str(eval_json),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "budget_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    manifest = {
        "config": str(config_path),
        "data_dir": str(data_dir),
        "file_pattern": args.file_pattern,
        "max_files": args.max_files,
        "budgets": args.budgets,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "no_teacher": args.no_teacher,
        "export_tables": args.export_tables,
        "summary_csv": str(summary_path),
        "experiments": summary_rows,
    }
    manifest_path = output_dir / "budget_sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("GeometryCore budget sweep finished.")
    print(f"Summary CSV: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
