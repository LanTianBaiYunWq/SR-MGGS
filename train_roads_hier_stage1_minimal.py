"""
roads_hier_stage1 最小训练脚本。

约束：
1. 只读 Gate 1 cache
2. 不接 graph
3. 不接 tile/file distillation
4. 先验证 line-only / tile-to-file contrastive 是否可训练
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from loss.contrastive import NTXentLoss
from models.projector import SimCLRProjector
from roads_hier_stage1.geometrycore_hier_dataset import GeometryCoreHierDataset
from roads_hier_stage1.minimal_signature_model import MinimalRoadsHierarchicalEncoder
from utils.seed import set_seed


def resolve_path(path_str: str) -> Path:
    """兼容相对路径和绝对路径。"""
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = Path(__file__).resolve().parent / raw
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def load_config(config_path: str) -> Dict:
    """读取 YAML 配置。"""
    with open(resolve_path(config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collate_samples(batch: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """DataLoader 直接返回文件样本列表。"""
    return list(batch)


def split_indices(num_items: int, val_ratio: float, seed: int) -> tuple[List[int], List[int]]:
    """按文件维度划分 train/val。"""
    if num_items < 2:
        raise ValueError("Need at least 2 files for minimal contrastive training.")

    if num_items >= 4:
        val_size = max(2, int(round(num_items * val_ratio)))
        val_size = min(val_size, num_items - 2)
    else:
        val_size = 1

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    return train_indices, val_indices


def subsample_rows(
    features: np.ndarray,
    keep_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """随机保留变长集合中的一部分元素。"""
    if features.shape[0] <= 1:
        return features

    mask = rng.random(features.shape[0]) < keep_prob
    if not np.any(mask):
        mask[rng.integers(0, features.shape[0])] = True
    return features[mask]


def build_augmented_tile_features(
    sample: Dict[str, object],
    *,
    tile_keep_prob: float,
    chunk_keep_prob: float,
    feature_noise_std: float,
    device: torch.device,
) -> List[torch.Tensor]:
    """对一个文件样本构造随机 view。"""
    rng = np.random.default_rng()
    tiles = sample["tiles"]
    if not tiles:
        raise ValueError(f"File {sample['file_id']} has no readable tiles.")

    tile_keep_mask = rng.random(len(tiles)) < tile_keep_prob
    if not np.any(tile_keep_mask):
        tile_keep_mask[rng.integers(0, len(tiles))] = True

    selected_tiles: List[torch.Tensor] = []
    for keep_tile, tile in zip(tile_keep_mask, tiles):
        if not keep_tile:
            continue

        chunk_features = np.asarray(tile["chunk_features"], dtype=np.float32)
        if chunk_features.ndim != 2 or chunk_features.shape[0] == 0:
            continue

        chunk_features = subsample_rows(chunk_features, chunk_keep_prob, rng)
        if feature_noise_std > 0:
            noise = rng.normal(0.0, feature_noise_std, size=chunk_features.shape).astype(np.float32)
            chunk_features = chunk_features + noise

        selected_tiles.append(torch.from_numpy(chunk_features).to(device))

    if not selected_tiles:
        tile = tiles[0]
        fallback = np.asarray(tile["chunk_features"], dtype=np.float32)
        if fallback.shape[0] == 0:
            raise ValueError(f"File {sample['file_id']} has empty first tile.")
        selected_tiles.append(torch.from_numpy(fallback).to(device))

    return selected_tiles


class MinimalRoadsHierTrainer:
    """只读 cache 的最小 roads hierarchical trainer。"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        set_seed(config["device"]["seed"])

        self.save_dir = resolve_path(config["logging"]["save_dir"])
        self.log_dir = resolve_path(config["logging"]["log_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._build_dataloaders()
        self._build_models()
        self._build_optimizer()

        self.criterion = NTXentLoss(
            temperature=config["loss"]["temperature"],
            normalize=config["loss"].get("normalize", True),
        )
        self.best_val = float("inf")

    def _setup_device(self) -> torch.device:
        """设置设备。"""
        if self.config["device"]["cuda"] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['gpu_id']}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            return device
        print("Using CPU")
        return torch.device("cpu")

    def _build_dataloaders(self) -> None:
        """构建只读 cache 的 train/val dataloader。"""
        data_cfg = self.config["data"]
        train_dataset = GeometryCoreHierDataset(
            data_cfg["cache_root"],
            mode=data_cfg.get("train_mode", "train"),
            max_tiles=data_cfg.get("max_tiles"),
            tile_selector=data_cfg.get("tile_selector", "manifest_default"),
        )
        val_dataset = GeometryCoreHierDataset(
            data_cfg["cache_root"],
            mode=data_cfg.get("eval_mode", "eval"),
            max_tiles=data_cfg.get("max_tiles"),
            tile_selector=data_cfg.get("tile_selector", "manifest_default"),
        )

        total_files = len(train_dataset)
        if data_cfg.get("max_files") is not None:
            total_files = min(total_files, int(data_cfg["max_files"]))

        batch_size = self.config["training"]["batch_size_files"]
        if total_files < batch_size:
            raise ValueError(
                f"Need at least {batch_size} files, but only {total_files} are available in cache. "
                "Please rebuild cache with more files, for example --max_files 4 or higher."
            )

        inferred_chunk_feature_dim = self._infer_chunk_feature_dim(train_dataset, total_files)
        configured_dim = self.config["model"].get("chunk_feature_dim")
        if configured_dim != inferred_chunk_feature_dim:
            self.config["model"]["chunk_feature_dim"] = inferred_chunk_feature_dim
            print(f"Inferred chunk_feature_dim from cache: {inferred_chunk_feature_dim}")

        train_indices, val_indices = split_indices(
            total_files,
            val_ratio=self.config["validation"].get("val_ratio", 0.25),
            seed=self.config["device"]["seed"],
        )
        if len(val_indices) < batch_size:
            print("Validation split is too small for contrastive loss; reuse train subset for smoke validation.")
            val_indices = train_indices[:batch_size]

        self.train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config["training"].get("num_workers", 0),
            collate_fn=collate_samples,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config["training"].get("num_workers", 0),
            collate_fn=collate_samples,
            drop_last=False,
        )
        print(f"Train files: {len(train_indices)}, Val files: {len(val_indices)}")

    @staticmethod
    def _infer_chunk_feature_dim(dataset: GeometryCoreHierDataset, total_files: int) -> int:
        """从 cache 中推断 chunk feature 维度。"""
        for idx in range(total_files):
            sample = dataset[idx]
            for tile in sample["tiles"]:
                chunk_features = np.asarray(tile["chunk_features"], dtype=np.float32)
                if chunk_features.ndim == 2 and chunk_features.shape[0] > 0:
                    return int(chunk_features.shape[1])
        raise ValueError("Failed to infer chunk feature dim: no non-empty tile chunk_features found in cache.")

    def _build_models(self) -> None:
        """构建最小 tile-to-file 模型。"""
        model_cfg = self.config["model"]
        self.encoder = MinimalRoadsHierarchicalEncoder(
            chunk_feature_dim=model_cfg.get("chunk_feature_dim", 4),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            file_dim=model_cfg.get("file_dim", 128),
            num_heads=model_cfg.get("num_heads", 4),
            num_set_layers=model_cfg.get("num_set_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(self.device)
        self.projector = SimCLRProjector(
            input_dim=model_cfg.get("file_dim", 128),
            hidden_dim=model_cfg.get("file_dim", 128),
            output_dim=model_cfg.get("projector_dim", 128),
        ).to(self.device)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        projector_params = sum(p.numel() for p in self.projector.parameters())
        print(f"Minimal encoder parameters: {encoder_params / 1e6:.3f}M")
        print(f"Projector parameters: {projector_params / 1e6:.3f}M")

    def _build_optimizer(self) -> None:
        """构建优化器。"""
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=self.config["training"].get("learning_rate", 1e-4),
            weight_decay=self.config["training"].get("weight_decay", 1e-4),
        )

    def encode_file_view(self, sample: Dict[str, object]) -> torch.Tensor:
        """编码一个文件的随机 view。"""
        aug_cfg = self.config["augmentation"]
        tile_feature_list = build_augmented_tile_features(
            sample,
            tile_keep_prob=aug_cfg.get("tile_keep_prob", 0.85),
            chunk_keep_prob=aug_cfg.get("chunk_keep_prob", 0.85),
            feature_noise_std=aug_cfg.get("feature_noise_std", 0.01),
            device=self.device,
        )
        return self.encoder(tile_feature_list)

    def run_epoch(self, loader: DataLoader, training: bool) -> float:
        """运行一个 epoch。"""
        self.encoder.train(training)
        self.projector.train(training)

        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc="Train" if training else "Val")

        for batch in pbar:
            if len(batch) < 2:
                continue

            file_emb_view1 = []
            file_emb_view2 = []
            for sample in batch:
                file_emb_view1.append(self.encode_file_view(sample))
                file_emb_view2.append(self.encode_file_view(sample))

            z1 = self.projector(torch.stack(file_emb_view1, dim=0))
            z2 = self.projector(torch.stack(file_emb_view2, dim=0))
            loss = self.criterion(z1, z2)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = self.config["training"].get("gradient_clip", 0.0)
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.projector.parameters()),
                        grad_clip,
                    )
                self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        """保存检查点。"""
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "encoder": self.encoder.state_dict(),
                "projector": self.projector.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            self.save_dir / filename,
        )

    def train(self) -> None:
        """训练入口。"""
        epochs = self.config["training"].get("epochs", 5)
        print("=" * 60)
        print("Starting Roads Hier Stage1 Minimal Training")
        print("cache -> chunk features -> tile -> file")
        print("=" * 60)

        for epoch in range(epochs):
            train_loss = self.run_epoch(self.train_loader, training=True)
            val_loss = self.run_epoch(self.val_loader, training=False)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint("best_roads_hier_stage1_minimal.pth", epoch, val_loss)
                print(f"New best model: {val_loss:.4f}")

        self.save_checkpoint("final_roads_hier_stage1_minimal.pth", epochs - 1, self.best_val)
        print("Training completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal roads hierarchical stage1 model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/roads_hier_stage1_minimal.yaml",
        help="配置文件路径。",
    )
    parser.add_argument("--cache_root", type=str, default=None, help="覆盖 cache 根目录。")
    parser.add_argument("--max_files", type=int, default=None, help="覆盖最多训练文件数。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数。")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖文件级 batch size。")
    parser.add_argument("--max_tiles", type=int, default=None, help="覆盖读取的 tile 上限。")
    parser.add_argument("--save_dir", type=str, default=None, help="覆盖 checkpoint 目录。")
    parser.add_argument("--log_dir", type=str, default=None, help="覆盖日志目录。")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cache_root:
        config["data"]["cache_root"] = args.cache_root
    if args.max_files is not None:
        config["data"]["max_files"] = args.max_files
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size_files"] = args.batch_size
    if args.max_tiles is not None:
        config["data"]["max_tiles"] = args.max_tiles
    if args.save_dir:
        config["logging"]["save_dir"] = args.save_dir
    if args.log_dir:
        config["logging"]["log_dir"] = args.log_dir

    trainer = MinimalRoadsHierTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
