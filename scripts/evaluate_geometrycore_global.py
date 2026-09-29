"""
评估 GeometryCore 文件级全局零水印的 genuine / impostor 识别效果。

当前脚本的 genuine 定义为：
同一个文件在两个增强视图下生成的文件级签名。

当前脚本的 impostor 定义为：
不同文件的文件级签名两两比较。

这个脚本先解决“新主线能否做文件级识别评估”的问题，
后续再扩展到更强的扰动、hard negatives 和 bit-level 训练闭环。
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from models.aggregator import AttentionPooling
from models.geometry_encoder import create_geometry_encoder
from preprocess.global_geometrycore_dataset import (
    GeometryCoreAugmentor,
    GeometryCoreGlobalDataset,
)
from utils.metrics import SignatureEvaluator, compute_ber, compute_normalized_correlation
from utils.signature import SignatureGenerator
from utils.seed import set_seed
from train_global import load_config


def resolve_path(path_str: str) -> Path:
    """兼容命令行和 IDE 的相对路径解析。"""
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
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate GeometryCore global signatures.")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练完成的 checkpoint 路径。")
    parser.add_argument("--config", type=str, default="configs/geometrycore_global.yaml", help="配置文件路径。")
    parser.add_argument("--data_dir", type=str, default=None, help="覆盖数据目录。")
    parser.add_argument("--file_pattern", type=str, default=None, help="文件过滤模式。")
    parser.add_argument("--max_files", type=int, default=None, help="最多加载多少个文件。")
    parser.add_argument("--max_geometries", type=int, default=None, help="每个文件最多保留多少个几何 token。")
    parser.add_argument("--genuine_views", type=int, default=2, help="每个文件生成多少个 genuine 视图。")
    parser.add_argument("--max_impostor_pairs", type=int, default=None, help="最多评估多少个 impostor 对。")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备。")
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


class GeometryCoreGlobalSignatureEvaluator:
    """基于 GeometryCore 前端的文件级签名评估器。"""

    def __init__(self, checkpoint_path: str, config: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.geometry_encoder = create_geometry_encoder(
            config["geometry_encoder"]
        ).to(self.device)
        embed_dim = config["geometry_encoder"]["embed_dim"]
        self.aggregator = AttentionPooling(
            embed_dim=embed_dim,
            num_heads=config["geometry_encoder"].get("num_heads", 8),
            dropout=config["geometry_encoder"].get("dropout", 0.1),
        ).to(self.device)

        self.geometry_encoder.load_state_dict(checkpoint["geometry_encoder"])
        self.aggregator.load_state_dict(checkpoint["aggregator"])

        self.geometry_encoder.eval()
        self.aggregator.eval()

        sig_cfg = config.get("signature", {})
        use_bch = sig_cfg.get("use_bch", True)
        try:
            self.signature_generator = SignatureGenerator(
                embed_dim=embed_dim,
                signature_bits=sig_cfg.get("signature_bits", 256),
                quantization=sig_cfg.get("quantization", "sign"),
                use_bch=use_bch,
                bch_poly=sig_cfg.get("bch_poly", 137),
                bch_bits=sig_cfg.get("bch_bits", 5),
                use_hmac=sig_cfg.get("use_hmac", True),
                hmac_key=sig_cfg.get("hmac_key", "gsd_secret_key_2025"),
                device=str(self.device),
            )
            self.signature_mode = "with_bch" if use_bch else "without_bch"
        except RuntimeError as exc:
            # 某些环境下 bchlib 会因为参数或本地实现差异初始化失败。
            # 评估阶段先自动降级为“无 BCH”模式，避免阻塞 genuine / impostor 分析。
            print(f"Warning: BCH initialization failed, fallback to no-BCH mode. Reason: {exc}")
            self.signature_generator = SignatureGenerator(
                embed_dim=embed_dim,
                signature_bits=sig_cfg.get("signature_bits", 256),
                quantization=sig_cfg.get("quantization", "sign"),
                use_bch=False,
                bch_poly=sig_cfg.get("bch_poly", 137),
                bch_bits=sig_cfg.get("bch_bits", 5),
                use_hmac=sig_cfg.get("use_hmac", True),
                hmac_key=sig_cfg.get("hmac_key", "gsd_secret_key_2025"),
                device=str(self.device),
            )
            self.signature_mode = "without_bch_fallback"

        aug_cfg = config.get("augmentation", {})
        self.augmentor = GeometryCoreAugmentor(
            rotation_angles=aug_cfg.get("rotation", {}).get("angles", [0, 90, 180, 270]),
            noise_std=aug_cfg.get("noise", {}).get("std", 0.001),
        )

    @torch.no_grad()
    def encode_global(self, geometries: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """将一个文件样本编码为全局 embedding。"""
        geometries = geometries.unsqueeze(0).to(self.device)  # [1, N, P, 2]
        mask = mask.unsqueeze(0).to(self.device)              # [1, N]

        batch_size, num_geoms, num_points, _ = geometries.shape
        flat_geoms = geometries.reshape(batch_size * num_geoms, num_points, -1)
        flat_embeddings = self.geometry_encoder(flat_geoms)
        embeddings = flat_embeddings.reshape(batch_size, num_geoms, -1)
        global_feat, _ = self.aggregator(embeddings, mask)
        return global_feat[0]

    @torch.no_grad()
    def signature_bits_from_sample(
        self,
        geometries: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从单个文件样本生成 embedding 与 bit 签名。"""
        global_feat = self.encode_global(geometries, mask)
        _, intermediates = self.signature_generator.generate(global_feat, return_intermediate=True)
        return global_feat.detach().cpu().numpy(), intermediates["bits"][0].astype(np.uint8)

    def augment_geometries(self, geometries: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """对有效 geometry token 做增强，生成同源视图。"""
        geom_np = geometries.detach().cpu().numpy().copy()
        mask_np = mask.detach().cpu().numpy().astype(bool)

        for idx, is_valid in enumerate(mask_np):
            if is_valid:
                geom_np[idx] = self.augmentor.augment(geom_np[idx])

        return torch.from_numpy(geom_np).float()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """构建最终配置。"""
    config = load_config(str(resolve_path(args.config)))

    if args.data_dir:
        config["data"]["raw_dir"] = args.data_dir
    if args.file_pattern:
        config["data"]["file_pattern"] = args.file_pattern
    if args.max_files is not None:
        config["data"]["max_files"] = args.max_files
    if args.max_geometries is not None:
        config["data"]["max_geometries"] = args.max_geometries

    config["data"]["dataset_backend"] = "geometrycore"
    return config


def collect_impostor_pairs(file_items: List[Dict[str, Any]], max_pairs: int | None) -> List[Tuple[int, int]]:
    """构造 impostor 文件对。"""
    all_pairs = list(combinations(range(len(file_items)), 2))
    if max_pairs is None or len(all_pairs) <= max_pairs:
        return all_pairs
    return all_pairs[:max_pairs]


def main() -> None:
    """脚本入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config["device"]["seed"])

    dataset = GeometryCoreGlobalDataset(
        data_dir=config["data"]["raw_dir"],
        file_pattern=config["data"].get("file_pattern", "*.shp"),
        max_files=config["data"].get("max_files"),
        source_crs_if_missing=config["data"].get("source_crs_if_missing"),
        working_crs=config["data"].get("working_crs"),
        max_geometries=config["data"].get("max_geometries", 128),
        min_geometries=config["data"].get("min_geometries", 10),
        fixed_points=config["data"].get("max_points", 128),
        include_holes=config["data"].get("include_holes", True),
        point_token_budget=config["data"].get("point_token_budget", 32),
        line_token_budget=config["data"].get("line_token_budget"),
        polygon_token_budget=config["data"].get("polygon_token_budget"),
        point_enabled=config["data"].get("point_enabled", True),
        line_enabled=config["data"].get("line_enabled", True),
        polygon_enabled=config["data"].get("polygon_enabled", True),
        line_eps_len=config["data"].get("line_eps_len", 1e-6),
        polygon_eps_area=config["data"].get("polygon_eps_area", 1e-8),
        line_densify_max_segment_length=config["data"].get("line_densify_max_segment_length"),
        verbose=config["data"].get("verbose_dataset", True),
    )

    evaluator = GeometryCoreGlobalSignatureEvaluator(
        checkpoint_path=str(resolve_path(args.checkpoint)),
        config=config,
        device=args.device,
    )

    metric_evaluator = SignatureEvaluator(
        signature_bits=config.get("signature", {}).get("signature_bits", 256)
    )

    file_items: List[Dict[str, Any]] = []

    # genuine：同一文件的原始视图 vs 增强视图
    for idx in range(len(dataset)):
        sample = dataset[idx]
        geometries = sample["geometries"]
        mask = sample["mask"]

        embedding_ref, bits_ref = evaluator.signature_bits_from_sample(geometries, mask)
        aug_geometries = evaluator.augment_geometries(geometries, mask)
        embedding_aug, bits_aug = evaluator.signature_bits_from_sample(aug_geometries, mask)

        metric_evaluator.add_genuine_pair(bits_ref, bits_aug)

        file_items.append(
            {
                "file_idx": idx,
                "file_path": sample["file_path"],
                "bits_ref": bits_ref,
                "bits_aug": bits_aug,
                "embedding_ref_norm": float(np.linalg.norm(embedding_ref)),
                "embedding_aug_norm": float(np.linalg.norm(embedding_aug)),
                "genuine_ber": float(compute_ber(bits_ref, bits_aug)),
                "genuine_nc": float(compute_normalized_correlation(bits_ref, bits_aug)),
            }
        )

    # impostor：不同文件之间比较参考签名
    impostor_pairs = collect_impostor_pairs(file_items, args.max_impostor_pairs)
    impostor_results: List[Dict[str, Any]] = []
    for i, j in impostor_pairs:
        bits_i = file_items[i]["bits_ref"]
        bits_j = file_items[j]["bits_ref"]
        metric_evaluator.add_impostor_pair(bits_i, bits_j)
        impostor_results.append(
            {
                "file_idx_i": i,
                "file_idx_j": j,
                "file_path_i": file_items[i]["file_path"],
                "file_path_j": file_items[j]["file_path"],
                "ber": float(compute_ber(bits_i, bits_j)),
                "nc": float(compute_normalized_correlation(bits_i, bits_j)),
            }
        )

    impostor_results.sort(key=lambda x: x["ber"])
    metrics = metric_evaluator.compute_metrics()

    report = {
        "arguments": {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "data_dir": args.data_dir,
            "file_pattern": config["data"].get("file_pattern", "*.shp"),
            "max_files": config["data"].get("max_files"),
            "max_geometries": config["data"].get("max_geometries"),
            "genuine_views": args.genuine_views,
            "max_impostor_pairs": args.max_impostor_pairs,
            "device": args.device,
        },
        "n_files": len(file_items),
        "signature_mode": evaluator.signature_mode,
        "n_genuine_pairs": len(file_items),
        "n_impostor_pairs": len(impostor_results),
        "metrics": metrics,
        "genuine_examples": file_items[:10],
        "hard_impostors": impostor_results[:10],
    }

    print("=" * 90)
    print("GeometryCore Global Signature Evaluation")
    print("=" * 90)
    print(f"Files: {report['n_files']}")
    print(f"Genuine pairs: {report['n_genuine_pairs']}")
    print(f"Impostor pairs: {report['n_impostor_pairs']}")
    if "genuine_ber_mean" in metrics:
        print(f"Genuine BER mean:  {metrics['genuine_ber_mean']:.6f}")
    if "impostor_ber_mean" in metrics:
        print(f"Impostor BER mean: {metrics['impostor_ber_mean']:.6f}")
    if "auc" in metrics:
        print(f"AUC:               {metrics['auc']:.6f}")
    if "eer" in metrics:
        print(f"EER:               {metrics['eer']:.6f}")
    if "separability" in metrics:
        print(f"Separability:      {metrics['separability']:.6f}")
    print("=" * 90)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # JSON 导出时去掉原始 bit 数组，避免输出过大且避免 ndarray 序列化失败。
        export_report = dict(report)
        export_report["genuine_examples"] = [
            {
                key: value
                for key, value in item.items()
                if key not in {"bits_ref", "bits_aug"}
            }
            for item in report["genuine_examples"]
        ]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(export_report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
