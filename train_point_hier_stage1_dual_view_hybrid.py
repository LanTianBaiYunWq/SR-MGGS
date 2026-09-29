"""Train point hybrid branch with learned point geometry plus visible statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loss.contrastive import NTXentLoss
from models.projector import SimCLRProjector
from point_hier_stage1.dual_view_protocol import (
    build_fixed_budget_point_views,
    visible_point_hybrid_stats_from_tile_features,
)
from point_hier_stage1.point_hier_dataset import PointHierDataset
from point_hier_stage1.point_signature_model import PointHybridSignatureEncoder
from mixed_hier_stage1.line_polygon_dataset import package_key_from_file_path
from roads_hier_stage1.fold_utils import split_indices_kfold
from train_roads_hier_stage1_minimal import load_config, resolve_path, split_indices
from utils.seed import set_seed


def collate_samples(batch: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return list(batch)


def load_package_subset(split_path: str | None, split_name: str) -> set[str] | None:
    if not split_path:
        return None
    with open(resolve_path(split_path), "r", encoding="utf-8") as f:
        split = json.load(f)
    key_map = {
        "train": "train_packages",
        "heldout": "heldout_packages",
        "eval": "heldout_packages",
    }
    key = key_map.get(split_name, split_name)
    if key not in split:
        raise KeyError(f"Split file does not contain key: {key}")
    return {str(value).lower() for value in split[key]}


class PointHybridDualViewTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        set_seed(config["device"]["seed"])

        self.save_dir = resolve_path(config["logging"]["save_dir"])
        self.log_dir = resolve_path(config["logging"]["log_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.feature_names: List[str] = []
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None
        self.include_local_structure = bool(
            config.get("visible_stats", {}).get("include_local_structure", False)
        )
        self.local_grid_size = int(config.get("visible_stats", {}).get("local_grid_size", 4))
        self.local_radial_bins = int(config.get("visible_stats", {}).get("local_radial_bins", 4))

        self._build_dataloaders()
        self._build_visible_feature_stats()
        self._build_models()
        self._build_optimizer()

        self.criterion = NTXentLoss(
            temperature=config["loss"]["temperature"],
            normalize=config["loss"].get("normalize", True),
        )
        self.best_val = float("inf")

    def _setup_device(self) -> torch.device:
        if self.config["device"]["cuda"] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['gpu_id']}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            return device
        print("Using CPU")
        return torch.device("cpu")

    def _build_dataloaders(self) -> None:
        data_cfg = self.config["data"]
        train_dataset = PointHierDataset(
            data_cfg["cache_root"],
            mode=data_cfg.get("train_mode", "train"),
            max_tiles=data_cfg.get("max_tiles"),
            tile_selector=data_cfg.get("tile_selector", "manifest_default"),
        )
        val_dataset = PointHierDataset(
            data_cfg["cache_root"],
            mode=data_cfg.get("eval_mode", "eval"),
            max_tiles=data_cfg.get("max_tiles"),
            tile_selector=data_cfg.get("tile_selector", "manifest_default"),
        )

        allowed_indices = list(range(len(train_dataset)))
        package_subset = load_package_subset(data_cfg.get("package_split"), data_cfg.get("split_name", "train"))
        if package_subset is not None:
            allowed_indices = [
                idx
                for idx, item in enumerate(train_dataset.items)
                if package_key_from_file_path(str(item["file_path"])).lower() in package_subset
            ]
        total_files = len(allowed_indices)
        if data_cfg.get("max_files") is not None:
            total_files = min(total_files, int(data_cfg["max_files"]))
            allowed_indices = allowed_indices[:total_files]

        batch_size = int(self.config["training"]["batch_size_files"])
        if total_files < batch_size:
            raise ValueError(f"Need at least {batch_size} files, but only {total_files} are available.")

        inferred_dim = self._infer_point_feature_dim(train_dataset, allowed_indices)
        configured_dim = self.config["model"].get("point_feature_dim")
        if configured_dim != inferred_dim:
            self.config["model"]["point_feature_dim"] = inferred_dim
            print(f"Inferred point_feature_dim from cache: {inferred_dim}")

        fold_cfg = self.config.get("fold", {})
        if fold_cfg.get("enabled", False):
            train_indices, val_indices = split_indices_kfold(
                total_files,
                num_folds=int(fold_cfg["num_folds"]),
                fold_index=int(fold_cfg["fold_index"]),
                seed=self.config["device"]["seed"],
            )
            print(
                f"Using fold split: fold_index={fold_cfg['fold_index']}, "
                f"num_folds={fold_cfg['num_folds']}"
            )
        else:
            train_indices, val_indices = split_indices(
                total_files,
                val_ratio=self.config["validation"].get("val_ratio", 0.25),
                seed=self.config["device"]["seed"],
            )
        train_indices = [allowed_indices[idx] for idx in train_indices]
        val_indices = [allowed_indices[idx] for idx in val_indices]
        if len(val_indices) < batch_size:
            print("Validation split too small; reusing train subset for smoke validation.")
            val_indices = train_indices[:batch_size]

        self.train_dataset = train_dataset
        self.train_indices = train_indices
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
    def _infer_point_feature_dim(dataset: PointHierDataset, indices: Sequence[int]) -> int:
        for idx in indices:
            sample = dataset[idx]
            for tile in sample["tiles"]:
                point_features = np.asarray(tile["point_features"], dtype=np.float32)
                if point_features.ndim == 2 and point_features.shape[0] > 0:
                    return int(point_features.shape[1])
        raise ValueError("Failed to infer point feature dim from cache.")

    def _build_visible_feature_stats(self) -> None:
        protocol_cfg = self.config["protocol"]
        n_tiles = int(protocol_cfg["n_tiles"])
        k_points = int(protocol_cfg["k_points_per_tile"])
        depth_template = {int(key): int(value) for key, value in protocol_cfg["depth_template"].items()}
        rows: List[np.ndarray] = []
        seed = int(self.config["device"]["seed"])

        for order_idx, dataset_idx in enumerate(self.train_indices):
            sample = self.train_dataset[dataset_idx]
            dual_views = build_fixed_budget_point_views(
                sample,
                n_tiles=n_tiles,
                k_points=k_points,
                depth_template=depth_template,
                rng_a=np.random.default_rng(seed + order_idx * 2),
                rng_b=np.random.default_rng(seed + order_idx * 2 + 1),
                feature_noise_std=0.0,
                device=None,
            )
            for key in ("view_a", "view_b"):
                current_names, stats = visible_point_hybrid_stats_from_tile_features(
                    dual_views[key].tile_feature_list,
                    include_local_structure=self.include_local_structure,
                    grid_size=self.local_grid_size,
                    radial_bins=self.local_radial_bins,
                )
                if not self.feature_names:
                    self.feature_names = list(current_names)
                elif self.feature_names != list(current_names):
                    raise ValueError("Visible point feature names are inconsistent across train files.")
                rows.append(np.asarray(stats, dtype=np.float32))

        matrix = np.vstack(rows)
        mean = np.mean(matrix, axis=0).astype(np.float32)
        std = np.std(matrix, axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)

        self.feature_mean = torch.from_numpy(mean).to(self.device)
        self.feature_std = torch.from_numpy(std).to(self.device)

        inferred_input_dim = int(matrix.shape[1])
        configured_dim = self.config["model"].get("visible_feature_dim")
        if configured_dim != inferred_input_dim:
            self.config["model"]["visible_feature_dim"] = inferred_input_dim
            print(f"Inferred visible_feature_dim from sampled train views: {inferred_input_dim}")

    def _build_models(self) -> None:
        model_cfg = self.config["model"]
        protocol_cfg = self.config["protocol"]
        max_depth = max(int(key) for key in protocol_cfg["depth_template"].keys())

        self.encoder = PointHybridSignatureEncoder(
            point_feature_dim=int(model_cfg.get("point_feature_dim", 8)),
            visible_feature_dim=int(model_cfg.get("visible_feature_dim", 24)),
            hidden_dim=int(model_cfg.get("hidden_dim", 192)),
            file_dim=int(model_cfg.get("file_dim", 192)),
            visible_hidden_dim=int(model_cfg.get("visible_hidden_dim", 128)),
            visible_dim=int(model_cfg.get("visible_dim", 128)),
            fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
            fusion_dim=int(model_cfg.get("fusion_dim", 192)),
            n_tiles=int(protocol_cfg["n_tiles"]),
            max_depth=max_depth,
            tile_encoder_type=str(model_cfg.get("tile_encoder_type", "set_transformer")),
            tile_num_heads=int(model_cfg.get("tile_num_heads", 4)),
            num_tile_layers=int(model_cfg.get("num_tile_layers", 2)),
            file_num_heads=int(model_cfg.get("file_num_heads", 4)),
            num_file_layers=int(model_cfg.get("num_file_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        ).to(self.device)
        self.projector = SimCLRProjector(
            input_dim=int(model_cfg.get("fusion_dim", 192)),
            hidden_dim=int(model_cfg.get("fusion_dim", 192)),
            output_dim=int(model_cfg.get("projector_dim", 128)),
        ).to(self.device)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        projector_params = sum(p.numel() for p in self.projector.parameters())
        print(f"Point hybrid encoder parameters: {encoder_params / 1e6:.3f}M")
        print(f"Projector parameters: {projector_params / 1e6:.3f}M")

    def _build_optimizer(self) -> None:
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"].get("weight_decay", 1e-4)),
        )

    def _normalize_visible(self, stats: np.ndarray) -> torch.Tensor:
        stats_tensor = torch.as_tensor(stats, dtype=torch.float32, device=self.device)
        return (stats_tensor - self.feature_mean) / self.feature_std

    def _encode_view(self, dual_view: Any, *, training: bool) -> torch.Tensor:
        _, stats = visible_point_hybrid_stats_from_tile_features(
            dual_view.tile_feature_list,
            include_local_structure=self.include_local_structure,
            grid_size=self.local_grid_size,
            radial_bins=self.local_radial_bins,
        )
        visible_stats = self._normalize_visible(np.asarray(stats, dtype=np.float32))
        visible_jitter_std = float(self.config["augmentation"].get("visible_jitter_std", 0.0)) if training else 0.0
        if visible_jitter_std > 0.0:
            visible_stats = visible_stats + torch.randn_like(visible_stats) * visible_jitter_std
        return self.encoder(dual_view.tile_feature_list, dual_view.tile_depths, visible_stats)

    def run_epoch(self, loader: DataLoader, *, training: bool) -> tuple[float, float, float]:
        protocol_cfg = self.config["protocol"]
        n_tiles = int(protocol_cfg["n_tiles"])
        k_points = int(protocol_cfg["k_points_per_tile"])
        depth_template = {int(key): int(value) for key, value in protocol_cfg["depth_template"].items()}
        feature_noise_std = float(self.config["augmentation"].get("feature_noise_std", 0.0)) if training else 0.0

        self.encoder.train(training)
        self.projector.train(training)

        total_loss = 0.0
        total_items = 0
        tile_overlaps: List[float] = []
        point_overlaps: List[float] = []
        iterator = tqdm(loader, desc="Train" if training else "Val")

        for batch in iterator:
            view1_embeddings: List[torch.Tensor] = []
            view2_embeddings: List[torch.Tensor] = []
            for sample in batch:
                dual_views = build_fixed_budget_point_views(
                    sample,
                    n_tiles=n_tiles,
                    k_points=k_points,
                    depth_template=depth_template,
                    rng_a=np.random.default_rng(),
                    rng_b=np.random.default_rng(),
                    feature_noise_std=feature_noise_std,
                    device=self.device,
                )
                tile_overlaps.append(float(dual_views["tile_overlap_ratio"]))
                point_overlaps.append(float(dual_views["point_overlap_ratio"]))
                view1_embeddings.append(self._encode_view(dual_views["view_a"], training=training))
                view2_embeddings.append(self._encode_view(dual_views["view_b"], training=training))

            h1 = torch.stack(view1_embeddings, dim=0)
            h2 = torch.stack(view2_embeddings, dim=0)
            z1 = self.projector(h1)
            z2 = self.projector(h2)
            loss = self.criterion(z1, z2)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_clip = self.config["training"].get("gradient_clip")
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.projector.parameters()),
                        float(gradient_clip),
                    )
                self.optimizer.step()

            total_loss += float(loss.item()) * len(batch)
            total_items += len(batch)
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = total_loss / max(total_items, 1)
        mean_tile_overlap = float(np.mean(tile_overlaps)) if tile_overlaps else 0.0
        mean_point_overlap = float(np.mean(point_overlaps)) if point_overlaps else 0.0
        return mean_loss, mean_tile_overlap, mean_point_overlap

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "config": self.config,
                "feature_names": self.feature_names,
                "feature_mean": self.feature_mean.detach().cpu(),
                "feature_std": self.feature_std.detach().cpu(),
                "encoder": self.encoder.state_dict(),
                "projector": self.projector.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.save_dir / filename,
        )

    def train(self) -> None:
        epochs = int(self.config["training"]["epochs"])
        print("=" * 70)
        print("Starting Point Hier Stage1 Dual-View Hybrid Training")
        print("fixed-budget point dual-view: point encoder + visible stats -> fusion signature")
        print("=" * 70)

        for epoch in range(epochs):
            train_loss, train_tile_overlap, train_point_overlap = self.run_epoch(self.train_loader, training=True)
            val_loss, val_tile_overlap, val_point_overlap = self.run_epoch(self.val_loader, training=False)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"train_tile_overlap={train_tile_overlap:.4f}, train_point_overlap={train_point_overlap:.4f}, "
                f"val_tile_overlap={val_tile_overlap:.4f}, val_point_overlap={val_point_overlap:.4f}"
            )

            self.save_checkpoint("final_point_hier_stage1_dual_view_hybrid.pth", epoch, val_loss)
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint("best_point_hier_stage1_dual_view_hybrid.pth", epoch, val_loss)
                print(f"New best model: {val_loss:.4f}")

        print("Training completed.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train point hybrid branch.")
    parser.add_argument("--config", type=str, default="configs/point_hier_stage1_dual_view_hybrid.yaml")
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--package_split", type=str, default=None)
    parser.add_argument("--split_name", type=str, default="train")
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
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
    if args.seed is not None:
        config["device"]["seed"] = args.seed
    if args.package_split:
        config["data"]["package_split"] = args.package_split
        config["data"]["split_name"] = args.split_name
    if args.save_dir:
        config["logging"]["save_dir"] = args.save_dir
    if args.log_dir:
        config["logging"]["log_dir"] = args.log_dir

    if args.num_folds is not None or args.fold_index is not None:
        if args.num_folds is None or args.fold_index is None:
            raise ValueError("Both --num_folds and --fold_index must be provided together.")
        config["fold"] = {
            "enabled": True,
            "num_folds": int(args.num_folds),
            "fold_index": int(args.fold_index),
        }
    else:
        config["fold"] = {"enabled": False}

    trainer = PointHybridDualViewTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
