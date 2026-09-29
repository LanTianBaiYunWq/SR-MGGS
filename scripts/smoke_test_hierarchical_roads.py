"""
只检查 roads / line-only / hierarchical dataset 入口，不启动训练。

用途：
1. 看单个文件会生成多少个 tile；
2. 看每个 tile 有多少个 chunks；
3. 估计 quadtree/chunk/subgraph 这一层的时间开销；
4. 在真正训练前先判断配置是否过重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from preprocess.hierarchical_roads_dataset import HierarchicalRoadsDataset


def resolve_path(path_str: str) -> Path:
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test hierarchical roads dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="原始数据目录。")
    parser.add_argument("--file_pattern", type=str, default="*roads*.shp", help="文件过滤模式。")
    parser.add_argument("--max_files", type=int, default=4, help="最多加载多少个文件。")
    parser.add_argument("--quadtree_depth", type=int, default=1, help="quadtree depth。")
    parser.add_argument("--tile_overlap_ratio", type=float, default=0.1, help="tile overlap ratio。")
    parser.add_argument("--max_points_per_chunk", type=int, default=32, help="每个 chunk 最大点数。")
    parser.add_argument("--min_chunks_per_tile", type=int, default=2, help="每个 tile 最少 chunks。")
    parser.add_argument("--connect_radius", type=float, default=50.0, help="局部子图连边半径。")
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    dataset = HierarchicalRoadsDataset(
        data_dir=str(resolve_path(args.data_dir)),
        file_pattern=args.file_pattern,
        max_files=args.max_files,
        quadtree_depth=args.quadtree_depth,
        tile_overlap_ratio=args.tile_overlap_ratio,
        max_points_per_chunk=args.max_points_per_chunk,
        min_chunks_per_tile=args.min_chunks_per_tile,
        connect_radius=args.connect_radius,
        verbose=True,
    )

    rows: List[Dict[str, object]] = []
    for sample in dataset.file_samples:
        tile_chunk_counts = [tile.num_chunks for tile in sample.tiles]
        tile_edge_counts = [tile.edge_index.shape[1] for tile in sample.tiles]
        rows.append(
            {
                "file_id": sample.file_id,
                "file_path": sample.file_path,
                "family": sample.family,
                "num_tiles": len(sample.tiles),
                "num_canonical_lines": sample.metadata["num_canonical_lines"],
                "mean_chunks_per_tile": mean(tile_chunk_counts) if tile_chunk_counts else 0.0,
                "max_chunks_per_tile": max(tile_chunk_counts) if tile_chunk_counts else 0,
                "mean_edges_per_tile": mean(tile_edge_counts) if tile_edge_counts else 0.0,
                "max_edges_per_tile": max(tile_edge_counts) if tile_edge_counts else 0,
            }
        )

    report = {
        "num_files": len(rows),
        "rows": rows,
    }

    print("=" * 80)
    print("Hierarchical Roads Dataset Smoke Test")
    print("=" * 80)
    print(f"Loaded files: {len(rows)}")
    for row in rows:
        print(
            f"{Path(row['file_path']).name}: "
            f"tiles={row['num_tiles']}, "
            f"canonical_lines={row['num_canonical_lines']}, "
            f"mean_chunks_per_tile={row['mean_chunks_per_tile']:.2f}, "
            f"max_chunks_per_tile={row['max_chunks_per_tile']}, "
            f"mean_edges_per_tile={row['mean_edges_per_tile']:.2f}"
        )
    print("=" * 80)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
