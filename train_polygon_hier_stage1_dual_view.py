"""Train polygon branch V1 with fixed-budget dual-view contrastive learning."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from loss.contrastive import NTXentLoss
from models.projector import SimCLRProjector
from polygon_hier_stage1.dual_view_protocol import build_fixed_budget_polygon_views
from polygon_hier_stage1.polygon_hier_dataset import PolygonHierDataset
from polygon_hier_stage1.polygon_signature_model import PolygonHierarchicalEncoder
from roads_hier_stage1.fold_utils import split_indices_kfold
from train_roads_hier_stage1_minimal import load_config, resolve_path, split_indices
from utils.seed import set_seed


def collate_samples(batch: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return list(batch)


class PolygonDualViewTrainer:
    def __init__(self, config: Dict[str, Any]):
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
        if self.config["device"]["cuda"] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['gpu_id']}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            return device
        print("Using CPU")
        return torch.device("cpu")

    def _build_dataloaders(self) -> None:
        data_cfg = self.config["data"]
        train_dataset = PolygonHierDataset(
            data_cfg["cache_root"],
            mode=data_cfg.get("train_mode", "train"),
            max_tiles=data_cfg.get("max_tiles"),
            tile_selector=data_cfg.get("tile_selector", "manifest_default"),
        )
        val_dataset = PolygonHierDataset(
            data_cfg["cache_root"],
            mode=data_cfg.get("eval_mode", "eval"),
            max_tiles=data_cfg.get("max_tiles"),
            tile_selector=data_cfg.get("tile_selector", "manifest_default"),
        )

        total_files = len(train_dataset)
        if data_cfg.get("max_files") is not None:
            total_files = min(total_files, int(data_cfg["max_files"]))

        batch_size = int(self.config["training"]["batch_size_files"])
        if total_files < batch_size:
            raise ValueError(f"Need at least {batch_size} files, but only {total_files} are available.")

        inferred_dim = self._infer_polygon_feature_dim(train_dataset, total_files)
        configured_dim = self.config["model"].get("polygon_feature_dim")
        if configured_dim != inferred_dim:
            self.config["model"]["polygon_feature_dim"] = inferred_dim
            print(f"Inferred polygon_feature_dim from cache: {inferred_dim}")

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
        if len(val_indices) < batch_size:
            print("Validation split too small; reusing train subset for smoke validation.")
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
    def _infer_polygon_feature_dim(dataset: PolygonHierDataset, total_files: int) -> int:
        for idx in range(total_files):
            sample = dataset[idx]
            for tile in sample["tiles"]:
                polygon_features = np.asarray(tile["polygon_features"], dtype=np.float32)
                if polygon_features.ndim == 2 and polygon_features.shape[0] > 0:
                    return int(polygon_features.shape[1])
        raise ValueError("Failed to infer polygon feature dim from cache.")

    def _build_models(self) -> None:
        model_cfg = self.config["model"]
        protocol_cfg = self.config["protocol"]
        max_depth = max(int(key) for key in protocol_cfg["depth_template"].keys())
        self.encoder = PolygonHierarchicalEncoder(
            polygon_feature_dim=int(model_cfg.get("polygon_feature_dim", 10)),
            hidden_dim=int(model_cfg.get("hidden_dim", 192)),
            file_dim=int(model_cfg.get("file_dim", 192)),
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
            input_dim=int(model_cfg.get("file_dim", 192)),
            hidden_dim=int(model_cfg.get("file_dim", 192)),
            output_dim=int(model_cfg.get("projector_dim", 128)),
        ).to(self.device)

    def _build_optimizer(self) -> None:
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"].get("weight_decay", 1e-4)),
        )
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        projector_params = sum(p.numel() for p in self.projector.parameters())
        print(f"Polygon encoder parameters: {encoder_params / 1e6:.3f}M")
        print(f"Projector parameters: {projector_params / 1e6:.3f}M")

    def _encode_view(self, dual_view: Any) -> torch.Tensor:
        return self.encoder(dual_view.tile_feature_list, dual_view.tile_depths)

    def run_epoch(self, loader: DataLoader, *, training: bool) -> tuple[float, float, float]:
        protocol_cfg = self.config["protocol"]
        n_tiles = int(protocol_cfg["n_tiles"])
        k_polygons = int(protocol_cfg["k_polygons_per_tile"])
        depth_template = {int(key): int(value) for key, value in protocol_cfg["depth_template"].items()}
        feature_noise_std = float(self.config["augmentation"].get("feature_noise_std", 0.0)) if training else 0.0

        self.encoder.train(training)
        self.projector.train(training)

        total_loss = 0.0
        total_items = 0
        tile_overlaps: List[float] = []
        polygon_overlaps: List[float] = []
        iterator = tqdm(loader, desc="Train" if training else "Val")

        for batch in iterator:
            view1_embeddings: List[torch.Tensor] = []
            view2_embeddings: List[torch.Tensor] = []
            for sample in batch:
                dual_views = build_fixed_budget_polygon_views(
                    sample,
                    n_tiles=n_tiles,
                    k_polygons=k_polygons,
                    depth_template=depth_template,
                    rng_a=np.random.default_rng(),
                    rng_b=np.random.default_rng(),
                    feature_noise_std=feature_noise_std,
                    device=self.device,
                )
                tile_overlaps.append(float(dual_views["tile_overlap_ratio"]))
                polygon_overlaps.append(float(dual_views["polygon_overlap_ratio"]))
                view1_embeddings.append(self._encode_view(dual_views["view_a"]))
                view2_embeddings.append(self._encode_view(dual_views["view_b"]))

            h1 = torch.stack(view1_embeddings, dim=0)
            h2 = torch.stack(view2_embeddings, dim=0)
            z1 = self.projector(h1)
            z2 = self.projector(h2)
            loss = self.criterion(z1, z2)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            total_loss += float(loss.item()) * len(batch)
            total_items += len(batch)
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = total_loss / max(total_items, 1)
        mean_tile_overlap = float(np.mean(tile_overlaps)) if tile_overlaps else 0.0
        mean_polygon_overlap = float(np.mean(polygon_overlaps)) if polygon_overlaps else 0.0
        return mean_loss, mean_tile_overlap, mean_polygon_overlap

    def train(self) -> None:
        epochs = int(self.config["training"]["epochs"])
        print("=" * 60)
        print("Starting Polygon Hier Stage1 Dual-View Training")
        print("fixed-budget polygon dual-view: tile set encoder -> file transformer")
        print("=" * 60)

        for epoch in range(epochs):
            train_loss, train_tile_overlap, train_polygon_overlap = self.run_epoch(self.train_loader, training=True)
            val_loss, val_tile_overlap, val_polygon_overlap = self.run_epoch(self.val_loader, training=False)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"train_tile_overlap={train_tile_overlap:.4f}, train_polygon_overlap={train_polygon_overlap:.4f}, "
                f"val_tile_overlap={val_tile_overlap:.4f}, val_polygon_overlap={val_polygon_overlap:.4f}"
            )

            state = {
                "epoch": epoch,
                "encoder": self.encoder.state_dict(),
                "projector": self.projector.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            }
            torch.save(state, self.save_dir / "final_polygon_hier_stage1_dual_view.pth")
            if val_loss < self.best_val:
                self.best_val = val_loss
                torch.save(state, self.save_dir / "best_polygon_hier_stage1_dual_view.pth")
                print(f"New best model: {val_loss:.4f}")

        print("Training completed.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train polygon branch V1.")
    parser.add_argument("--config", type=str, default="configs/polygon_hier_stage1_dual_view.yaml")
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    if args.save_dir:
        config["logging"]["save_dir"] = args.save_dir
    if args.log_dir:
        config["logging"]["log_dir"] = args.log_dir

    if args.num_folds is not None or args.fold_index is not None:
        if args.num_folds is None or args.fold_index is None:
            raise ValueError("Both --num_folds and --fold_index must be provided together.")
        config["fold"] = {"enabled": True, "num_folds": int(args.num_folds), "fold_index": int(args.fold_index)}
    else:
        config["fold"] = {"enabled": False}

    trainer = PolygonDualViewTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
