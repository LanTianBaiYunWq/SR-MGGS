"""
roads_hier_stage1 训练期读取烟雾测试。

目标：
1. 验证 dataset 只读离线 cache，不再在线解析 shapefile
2. 验证 train / eval / all 三种 mode 下的 tile 选择数量
3. 验证 top-M 选择字段是否已经写入并能被正确读取
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Smoke test roads hierarchical cache reader.")
    parser.add_argument("--cache_root", type=str, required=True, help="离线 cache 根目录。")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["train", "eval", "all"],
        help="要测试的 dataset mode，支持 train/eval/all。",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=1,
        help="最多检查多少个文件样本，用于快速验证。",
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=None,
        help="可选的额外 tile 截断上限，默认使用 manifest 中的 train/eval 配置。",
    )
    parser.add_argument(
        "--tile_selector",
        type=str,
        default="manifest_default",
        choices=["manifest_default", "top_selection_score", "all"],
        help="tile 选择策略。",
    )
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def summarize_mode(
    cache_root: str,
    *,
    mode: str,
    max_files: int,
    max_tiles: int | None,
    tile_selector: str,
) -> Dict[str, object]:
    """汇总一个 mode 下 dataset 的读取结果。"""
    dataset = GeometryCoreHierDataset(
        cache_root,
        mode=mode,
        max_tiles=max_tiles,
        tile_selector=tile_selector,
    )

    sample_rows: List[Dict[str, object]] = []
    selected_counts: List[int] = []

    inspect_count = min(max_files, len(dataset))
    for idx in range(inspect_count):
        sample = dataset[idx]
        manifest = sample["manifest"]
        tile_selection = manifest.get("tile_selection", {})
        first_tiles = sample["tiles"][: min(5, len(sample["tiles"]))]

        sample_rows.append(
            {
                "file_id": sample["file_id"],
                "file_path": sample["file_path"],
                "num_selected_tiles": sample["num_selected_tiles"],
                "manifest_num_tiles": manifest.get("num_tiles"),
                "manifest_num_chunks": manifest.get("num_chunks"),
                "score_key": tile_selection.get("score_key"),
                "train_max_tiles": tile_selection.get("train_max_tiles"),
                "eval_max_tiles": tile_selection.get("eval_max_tiles"),
                "first_tiles": [
                    {
                        "tile_id": tile["tile_id"],
                        "tile_rank": tile["tile_rank"],
                        "tile_score": tile["tile_score"],
                        "num_chunks": int(tile["chunk_features"].shape[0]),
                    }
                    for tile in first_tiles
                ],
            }
        )
        selected_counts.append(sample["num_selected_tiles"])

    return {
        "mode": mode,
        "dataset_len": len(dataset),
        "inspected_files": inspect_count,
        "selected_tile_count_stats": {
            "min": min(selected_counts) if selected_counts else 0,
            "max": max(selected_counts) if selected_counts else 0,
            "mean": (sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0,
        },
        "samples": sample_rows,
    }


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    report = {
        "arguments": {
            "cache_root": args.cache_root,
            "modes": args.modes,
            "max_files": args.max_files,
            "max_tiles": args.max_tiles,
            "tile_selector": args.tile_selector,
        },
        "modes": [],
    }

    print("=" * 90)
    print("Roads Hierarchical Dataset Smoke Test")
    print("=" * 90)

    for mode in args.modes:
        mode_summary = summarize_mode(
            args.cache_root,
            mode=mode,
            max_files=args.max_files,
            max_tiles=args.max_tiles,
            tile_selector=args.tile_selector,
        )
        report["modes"].append(mode_summary)

        print(f"Mode: {mode}")
        print(f"  dataset_len: {mode_summary['dataset_len']}")
        print(f"  inspected_files: {mode_summary['inspected_files']}")
        print(f"  selected_tile_count_stats: {mode_summary['selected_tile_count_stats']}")

        if mode_summary["samples"]:
            first = mode_summary["samples"][0]
            print("  first_sample:")
            print(f"    file_id: {first['file_id']}")
            print(f"    num_selected_tiles: {first['num_selected_tiles']}")
            print(f"    manifest_num_tiles: {first['manifest_num_tiles']}")
            print(f"    score_key: {first['score_key']}")
            if first["first_tiles"]:
                tile0 = first["first_tiles"][0]
                print(
                    f"    first_tile: id={tile0['tile_id']} "
                    f"rank={tile0['tile_rank']} "
                    f"score={tile0['tile_score']:.6f} "
                    f"chunks={tile0['num_chunks']}"
                )

    print("=" * 90)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
