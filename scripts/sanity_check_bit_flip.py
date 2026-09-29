"""
Sanity Check: 签名比特翻转率验证
===============================

本脚本验证 BitMarginLoss 对签名稳定性的效果。

测试流程：
    1. 加载训练好的模型
    2. 对同一几何生成多个扰动视图
    3. 计算签名的比特翻转率
    4. 对比有/无 L_bit 的版本

预期结果：
    - 使用 L_bit 训练的模型，flip_rate 应显著低于不使用的版本
    - 典型值：有 L_bit 约 5-10%，无 L_bit 约 15-25%

使用方法：
    python scripts/sanity_check_bit_flip.py --checkpoint checkpoints/best_aligned.pt

作者：GSD Team
版本：1.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from preprocess.geometry_dataset import (
    TwoViewGeometryDataset,
    collate_parts,
    augment_parts
)
from preprocess.geometry_types import GeomPart
from models.universal_encoder import create_universal_encoder
from loss.bit_margin import BitMarginLoss


def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    # 创建模型
    encoder = create_universal_encoder(config).to(device)
    encoder.load_state_dict(checkpoint["geo_encoder_state_dict"])
    encoder.eval()
    
    return encoder, config


def compute_signature(
    encoder,
    batch: Dict[str, torch.Tensor],
    loss_bit: BitMarginLoss
) -> torch.Tensor:
    """计算签名"""
    with torch.no_grad():
        E = encoder(batch)
        sig = loss_bit.generate_signature(E)
    return sig


def test_single_geometry(
    encoder,
    geometry_data,
    loss_bit: BitMarginLoss,
    device: torch.device,
    n_views: int = 10
) -> Dict[str, float]:
    """
    对单个几何测试比特翻转率
    
    生成 n_views 个扰动视图，计算两两之间的翻转率
    """
    canonical_parts = geometry_data.parts
    
    # 生成多个扰动视图
    all_sigs = []
    
    for i in range(n_views):
        rng = np.random.default_rng(seed=i * 42)
        augmented = augment_parts(canonical_parts, rng=rng)
        
        # 构造 batch
        batch_item = {"parts": augmented, "container_type": geometry_data.container_type}
        batch = collate_parts([batch_item])
        
        if batch is None:
            continue
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        sig = compute_signature(encoder, batch, loss_bit)
        all_sigs.append(sig.cpu())
    
    if len(all_sigs) < 2:
        return {"flip_rate": 0.0, "n_pairs": 0}
    
    # 计算两两翻转率
    flip_rates = []
    for i in range(len(all_sigs)):
        for j in range(i + 1, len(all_sigs)):
            rate = (all_sigs[i] != all_sigs[j]).float().mean().item()
            flip_rates.append(rate)
    
    return {
        "flip_rate": np.mean(flip_rates),
        "flip_rate_std": np.std(flip_rates),
        "n_pairs": len(flip_rates),
        "min_rate": min(flip_rates),
        "max_rate": max(flip_rates)
    }


def run_sanity_check(
    checkpoint_path: str,
    data_dir: str,
    n_geometries: int = 100,
    n_views: int = 10,
    device: torch.device = None
):
    """运行完整的 sanity check"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Sanity Check: Bit Flip Rate Analysis")
    print("=" * 60)
    
    # 加载模型
    print(f"\nLoading model from {checkpoint_path}...")
    encoder, config = load_model(checkpoint_path, device)
    
    # 创建 BitMarginLoss
    embed_dim = config.get("geometry_encoder", {}).get("embed_dim", 256)
    n_bits = config.get("signature", {}).get("n_bits", 256)
    seed = config.get("device", {}).get("seed", 42)
    
    loss_bit = BitMarginLoss(
        d_model=embed_dim,
        n_bits=n_bits,
        margin=0.5,
        seed=seed
    ).to(device)
    
    # 加载测试数据
    print(f"\nLoading data from {data_dir}...")
    shp_files = list(Path(data_dir).rglob("*.shp"))[:5]  # 只用少量文件
    
    if not shp_files:
        print(f"No shapefiles found in {data_dir}")
        return
    
    dataset = TwoViewGeometryDataset(
        shp_files=[str(f) for f in shp_files],
        max_geometries_per_file=n_geometries // len(shp_files) + 1
    )
    
    n_test = min(n_geometries, len(dataset.geometries))
    print(f"Testing on {n_test} geometries, {n_views} views each...")
    
    # 测试每个几何
    results = []
    
    for i in tqdm(range(n_test), desc="Testing"):
        geom_data = dataset.geometries[i]
        result = test_single_geometry(
            encoder, geom_data, loss_bit, device, n_views
        )
        results.append(result)
    
    # 统计
    flip_rates = [r["flip_rate"] for r in results]
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Mean Flip Rate: {np.mean(flip_rates):.4f} ({np.mean(flip_rates) * 100:.2f}%)")
    print(f"  Std Flip Rate:  {np.std(flip_rates):.4f}")
    print(f"  Min Flip Rate:  {np.min(flip_rates):.4f}")
    print(f"  Max Flip Rate:  {np.max(flip_rates):.4f}")
    
    # 分级统计
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1.0]
    hist, _ = np.histogram(flip_rates, bins=bins)
    
    print("\n  Distribution:")
    for i in range(len(bins) - 1):
        pct = hist[i] / len(flip_rates) * 100
        print(f"    {bins[i]:.2f} - {bins[i+1]:.2f}: {hist[i]:4d} ({pct:5.1f}%)")
    
    # 判断是否通过
    threshold = 0.15  # 15% 翻转率阈值
    passed = np.mean(flip_rates) < threshold
    
    print("\n" + "=" * 60)
    if passed:
        print(f"✓ PASSED: Mean flip rate {np.mean(flip_rates):.4f} < {threshold}")
    else:
        print(f"✗ FAILED: Mean flip rate {np.mean(flip_rates):.4f} >= {threshold}")
    print("=" * 60)
    
    return {
        "mean_flip_rate": np.mean(flip_rates),
        "std_flip_rate": np.std(flip_rates),
        "passed": passed
    }


def main():
    parser = argparse.ArgumentParser(description="Sanity Check: Bit Flip Rate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Data directory")
    parser.add_argument("--n_geometries", type=int, default=100,
                        help="Number of geometries to test")
    parser.add_argument("--n_views", type=int, default=10,
                        help="Number of views per geometry")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    run_sanity_check(
        args.checkpoint,
        args.data_dir,
        args.n_geometries,
        args.n_views,
        device
    )


if __name__ == "__main__":
    main()
