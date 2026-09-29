"""Train generic mixed backbone for line + polygon + optional point."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from loss.contrastive import HardNegativeRankingLoss, NTXentLoss
from mixed_hier_stage1.generic_line_polygon_point_fusion_model import MixedGenericLinePolygonPointFusionEncoder
from mixed_hier_stage1.line_polygon_point_dataset import MixedLinePolygonOptionalPointPackageDataset
from models.projector import SimCLRProjector
from scripts.evaluate_mixed_missing_subtype_tolerant_score_fusion import build_tolerant_views
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


class MixedGenericLinePolygonPointTrainer:
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
        self._load_branch_initializers()
        self._load_init_checkpoint()
        self._build_optimizer()
        self.criterion = NTXentLoss(
            temperature=float(config["loss"]["temperature"]),
            normalize=bool(config["loss"].get("normalize", True)),
        )
        self.hard_negative_weight = float(config["loss"].get("hard_negative_weight", 0.0))
        self.hard_negative_criterion = HardNegativeRankingLoss(
            margin=float(config["loss"].get("hard_negative_margin", 0.05)),
            normalize=bool(config["loss"].get("normalize", True)),
            same_group_only=False,
        )
        self.best_val = float("inf")
        self.branches_frozen = False

    def _setup_device(self) -> torch.device:
        if self.config["device"]["cuda"] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['gpu_id']}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            return device
        print("Using CPU")
        return torch.device("cpu")

    def _build_dataloaders(self) -> None:
        data_cfg = self.config["data"]
        common_kwargs = dict(
            line_max_tiles=data_cfg.get("line_max_tiles"),
            polygon_max_tiles=data_cfg.get("polygon_max_tiles"),
            point_max_tiles=data_cfg.get("point_max_tiles"),
            line_tile_selector=data_cfg.get("line_tile_selector", "manifest_default"),
            polygon_tile_selector=data_cfg.get("polygon_tile_selector", "manifest_default"),
            point_tile_selector=data_cfg.get("point_tile_selector", "manifest_default"),
            line_subtypes=data_cfg.get("line_subtypes", ("roads", "railways", "waterways")),
            polygon_subtypes=data_cfg.get("polygon_subtypes", ("building", "landuse", "natural")),
            point_subtypes=data_cfg.get("point_subtypes", ("pois", "traffic", "transport", "pofw")),
            max_packages=data_cfg.get("max_packages"),
            require_complete_lp=bool(data_cfg.get("require_complete_lp", False)),
        )
        train_dataset = MixedLinePolygonOptionalPointPackageDataset(
            data_cfg["line_cache_root"],
            data_cfg["polygon_cache_root"],
            data_cfg.get("point_cache_root"),
            mode=data_cfg.get("train_mode", "train"),
            **common_kwargs,
        )
        val_dataset = MixedLinePolygonOptionalPointPackageDataset(
            data_cfg["line_cache_root"],
            data_cfg["polygon_cache_root"],
            data_cfg.get("point_cache_root"),
            mode=data_cfg.get("eval_mode", "eval"),
            **common_kwargs,
        )
        package_subset = load_package_subset(data_cfg.get("package_split"), data_cfg.get("split_name", "train"))
        if package_subset is not None:
            train_dataset.package_records = [
                record for record in train_dataset.package_records if str(record["package_id"]).lower() in package_subset
            ]
            val_dataset.package_records = [
                record for record in val_dataset.package_records if str(record["package_id"]).lower() in package_subset
            ]
        total_packages = len(train_dataset)
        batch_size = int(self.config["training"]["batch_size_files"])
        if total_packages < batch_size:
            raise ValueError(f"Need at least {batch_size} packages, but only {total_packages} are available.")

        train_indices, val_indices = split_indices(
            total_packages,
            val_ratio=float(self.config["validation"].get("val_ratio", 0.25)),
            seed=int(self.config["device"]["seed"]),
        )
        if len(val_indices) < batch_size:
            print("Validation split too small; reusing train subset for smoke validation.")
            val_indices = train_indices[:batch_size]

        self.train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(self.config["training"].get("num_workers", 0)),
            collate_fn=collate_samples,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(self.config["training"].get("num_workers", 0)),
            collate_fn=collate_samples,
            drop_last=False,
        )
        print(f"Available LP packages: {total_packages}")
        print(f"Train packages: {len(train_indices)}, Val packages: {len(val_indices)}")

    def _build_models(self) -> None:
        line_protocol = self.config["line_protocol"]
        polygon_protocol = self.config["polygon_protocol"]
        point_protocol = self.config["point_protocol"]
        line_model = self.config["line_model"]
        polygon_model = self.config["polygon_model"]
        point_model = self.config["point_model"]
        fusion_model = self.config["fusion_model"]
        self.encoder = MixedGenericLinePolygonPointFusionEncoder(
            line_chunk_feature_dim=int(line_model.get("chunk_feature_dim", 10)),
            polygon_feature_dim=int(polygon_model.get("polygon_feature_dim", 10)),
            point_feature_dim=int(point_model.get("point_feature_dim", 8)),
            point_visible_feature_dim=int(point_model.get("visible_feature_dim", 103)),
            line_hidden_dim=int(line_model.get("hidden_dim", 192)),
            line_file_dim=int(line_model.get("file_dim", 192)),
            polygon_hidden_dim=int(polygon_model.get("hidden_dim", 192)),
            polygon_file_dim=int(polygon_model.get("file_dim", 192)),
            point_hidden_dim=int(point_model.get("hidden_dim", 192)),
            point_file_dim=int(point_model.get("file_dim", 192)),
            point_visible_hidden_dim=int(point_model.get("visible_hidden_dim", 160)),
            point_visible_dim=int(point_model.get("visible_dim", 128)),
            point_fusion_hidden_dim=int(point_model.get("fusion_hidden_dim", 256)),
            point_fusion_dim=int(point_model.get("fusion_dim", 192)),
            line_n_tiles=int(line_protocol["n_tiles"]),
            polygon_n_tiles=int(polygon_protocol["n_tiles"]),
            point_n_tiles=int(point_protocol["n_tiles"]),
            line_max_depth=max(int(key) for key in line_protocol["depth_template"].keys()),
            polygon_max_depth=max(int(key) for key in polygon_protocol["depth_template"].keys()),
            point_max_depth=max(int(key) for key in point_protocol["depth_template"].keys()),
            polygon_tile_encoder_type=str(polygon_model.get("tile_encoder_type", "set_transformer")),
            point_tile_encoder_type=str(point_model.get("tile_encoder_type", "set_transformer")),
            line_chunk_num_heads=int(line_model.get("chunk_num_heads", 4)),
            line_file_num_heads=int(line_model.get("file_num_heads", 4)),
            line_num_chunk_layers=int(line_model.get("num_chunk_layers", 2)),
            line_num_file_layers=int(line_model.get("num_file_layers", 3)),
            polygon_tile_num_heads=int(polygon_model.get("tile_num_heads", 4)),
            polygon_num_tile_layers=int(polygon_model.get("num_tile_layers", 2)),
            polygon_file_num_heads=int(polygon_model.get("file_num_heads", 4)),
            polygon_num_file_layers=int(polygon_model.get("num_file_layers", 3)),
            point_tile_num_heads=int(point_model.get("tile_num_heads", 4)),
            point_num_tile_layers=int(point_model.get("num_tile_layers", 2)),
            point_file_num_heads=int(point_model.get("file_num_heads", 4)),
            point_num_file_layers=int(point_model.get("num_file_layers", 3)),
            branch_num_heads=int(fusion_model.get("branch_num_heads", 4)),
            branch_num_layers=int(fusion_model.get("branch_num_layers", 2)),
            modality_num_heads=int(fusion_model.get("modality_num_heads", 4)),
            modality_num_layers=int(fusion_model.get("modality_num_layers", 2)),
            fusion_dim=int(fusion_model.get("fusion_dim", 192)),
            dropout=float(fusion_model.get("dropout", 0.1)),
            point_include_local_structure=bool(point_model.get("include_local_structure", True)),
            point_local_grid_size=int(point_model.get("local_grid_size", 4)),
            point_local_radial_bins=int(point_model.get("local_radial_bins", 4)),
        ).to(self.device)
        self.projector = SimCLRProjector(
            input_dim=int(fusion_model.get("fusion_dim", 192)),
            hidden_dim=int(fusion_model.get("fusion_dim", 192)),
            output_dim=int(fusion_model.get("projector_dim", 128)),
        ).to(self.device)
        print(f"Generic mixed encoder parameters: {sum(p.numel() for p in self.encoder.parameters()) / 1e6:.3f}M")
        print(f"Projector parameters: {sum(p.numel() for p in self.projector.parameters()) / 1e6:.3f}M")

    def _load_branch_initializers(self) -> None:
        init_cfg = self.config.get("initialization", {})
        strict = bool(init_cfg.get("strict", True))
        if init_cfg.get("line_checkpoint"):
            checkpoint = torch.load(resolve_path(init_cfg["line_checkpoint"]), map_location=self.device)
            self.encoder.line_encoder.load_state_dict(checkpoint["encoder"], strict=strict)
            print(f"Loaded line branch checkpoint: {resolve_path(init_cfg['line_checkpoint'])}")
        if init_cfg.get("polygon_checkpoint"):
            checkpoint = torch.load(resolve_path(init_cfg["polygon_checkpoint"]), map_location=self.device)
            self.encoder.polygon_encoder.load_state_dict(checkpoint["encoder"], strict=strict)
            print(f"Loaded polygon branch checkpoint: {resolve_path(init_cfg['polygon_checkpoint'])}")
        if init_cfg.get("point_checkpoint"):
            checkpoint = torch.load(resolve_path(init_cfg["point_checkpoint"]), map_location=self.device)
            self.encoder.point_encoder.load_state_dict(checkpoint["encoder"], strict=strict)
            self.encoder.set_point_feature_stats(checkpoint["feature_mean"], checkpoint["feature_std"])
            print(f"Loaded point branch checkpoint: {resolve_path(init_cfg['point_checkpoint'])}")

    def _load_init_checkpoint(self) -> None:
        init_checkpoint = self.config.get("initialization", {}).get("init_checkpoint")
        if not init_checkpoint:
            return
        checkpoint = torch.load(resolve_path(init_checkpoint), map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.projector.load_state_dict(checkpoint["projector"], strict=True)
        print(
            f"Loaded full generic checkpoint: {resolve_path(init_checkpoint)} "
            f"(epoch={checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss')})"
        )

    def _build_optimizer(self) -> None:
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"].get("weight_decay", 1e-4)),
        )

    def _set_branch_trainable(self, trainable: bool) -> None:
        self.branches_frozen = not trainable
        for module in (self.encoder.line_encoder, self.encoder.polygon_encoder, self.encoder.point_encoder):
            for param in module.parameters():
                param.requires_grad = trainable

    def _build_package_views(self, package_sample: Dict[str, object], *, training: bool) -> Dict[str, Any]:
        feature_noise_std = float(self.config["augmentation"].get("feature_noise_std", 0.0)) if training else 0.0
        return build_tolerant_views(
            package_sample,
            config=self.config,
            scenario="natural_missing",
            rng_a=np.random.default_rng(),
            rng_b=np.random.default_rng(),
            device=self.device,
            feature_noise_std=feature_noise_std,
        )

    def run_epoch(self, loader: DataLoader, *, training: bool) -> tuple[float, float, float, float, float, float]:
        self.encoder.train(training)
        self.projector.train(training)
        if training and self.branches_frozen:
            self.encoder.line_encoder.eval()
            self.encoder.polygon_encoder.eval()
            self.encoder.point_encoder.eval()
        total_loss = 0.0
        total_items = 0
        hard_negative_losses: List[float] = []
        hard_negative_valid: List[float] = []
        hard_negative_gaps: List[float] = []
        point_used: List[float] = []
        modality_counts: List[float] = []
        iterator = tqdm(loader, desc="Train" if training else "Val")
        for batch in iterator:
            fused_a: List[torch.Tensor] = []
            fused_b: List[torch.Tensor] = []
            for package_sample in batch:
                views = self._build_package_views(package_sample, training=training)
                emb_a, modalities_a = self.encoder(views["line_view_a"], views["polygon_view_a"], views["point_view_a"])
                emb_b, modalities_b = self.encoder(views["line_view_b"], views["polygon_view_b"], views["point_view_b"])
                fused_a.append(emb_a)
                fused_b.append(emb_b)
                point_used.append(float(bool(views["point_view_a"]) and bool(views["point_view_b"])))
                modality_counts.append(float((modalities_a.shape[0] + modalities_b.shape[0]) * 0.5))
            z1 = self.projector(torch.stack(fused_a, dim=0))
            z2 = self.projector(torch.stack(fused_b, dim=0))
            ntx_loss = self.criterion(z1, z2)
            hn_loss, hn_stats = self.hard_negative_criterion(z1, z2)
            loss = ntx_loss + self.hard_negative_weight * hn_loss
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
            hard_negative_losses.append(float(hn_loss.item()))
            hard_negative_valid.append(float(hn_stats["valid_anchors"]))
            hard_negative_gaps.append(float(hn_stats["mean_positive"] - hn_stats["mean_hard_negative"]))
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                ntx=f"{ntx_loss.item():.4f}",
                hn=f"{hn_loss.item():.4f}",
                point=f"{np.mean(point_used):.3f}",
            )
        return (
            total_loss / max(total_items, 1),
            float(np.mean(point_used)) if point_used else 0.0,
            float(np.mean(modality_counts)) if modality_counts else 0.0,
            float(np.mean(hard_negative_losses)) if hard_negative_losses else 0.0,
            float(np.mean(hard_negative_valid)) if hard_negative_valid else 0.0,
            float(np.mean(hard_negative_gaps)) if hard_negative_gaps else 0.0,
        )

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "config": self.config,
                "encoder": self.encoder.state_dict(),
                "projector": self.projector.state_dict(),
            },
            self.save_dir / filename,
        )

    def train(self) -> None:
        print("=" * 78)
        print("Starting Generic Mixed Hier Stage1 Line+Polygon+Optional-Point Training")
        print("generic family pooling: no subtype embeddings; point is optional")
        print("=" * 78)
        for epoch in range(int(self.config["training"]["epochs"])):
            freeze_epochs = int(self.config["training"].get("freeze_branch_epochs", 0))
            branches_trainable = epoch >= freeze_epochs
            self._set_branch_trainable(branches_trainable)
            if freeze_epochs > 0:
                print(f"Epoch {epoch}: branch_trainable={branches_trainable}")
            train_stats = self.run_epoch(self.train_loader, training=True)
            val_stats = self.run_epoch(self.val_loader, training=False)
            print(
                f"Epoch {epoch}: train_loss={train_stats[0]:.4f}, val_loss={val_stats[0]:.4f}, "
                f"train_point_used={train_stats[1]:.4f}, val_point_used={val_stats[1]:.4f}, "
                f"train_modalities={train_stats[2]:.4f}, val_modalities={val_stats[2]:.4f}, "
                f"train_hn={train_stats[3]:.4f}, val_hn={val_stats[3]:.4f}, "
                f"train_hn_valid={train_stats[4]:.2f}, val_hn_valid={val_stats[4]:.2f}, "
                f"train_hn_gap={train_stats[5]:.4f}, val_hn_gap={val_stats[5]:.4f}"
            )
            self.save_checkpoint("final_mixed_hier_stage1_line_polygon_point_generic.pth", epoch, val_stats[0])
            if val_stats[0] < self.best_val:
                self.best_val = val_stats[0]
                self.save_checkpoint("best_mixed_hier_stage1_line_polygon_point_generic.pth", epoch, val_stats[0])
                print(f"New best model: {val_stats[0]:.4f}")
        print("Training completed.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train generic mixed line+polygon+optional-point backbone.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon_point_generic.yaml")
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--point_cache_root", type=str, default=None)
    parser.add_argument("--max_packages", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--package_split", type=str, default=None)
    parser.add_argument("--split_name", type=str, default="train")
    parser.add_argument("--line_init_checkpoint", type=str, default=None)
    parser.add_argument("--polygon_init_checkpoint", type=str, default=None)
    parser.add_argument("--point_init_checkpoint", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--no_branch_init", action="store_true")
    parser.add_argument("--freeze_branch_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--hard_negative_weight", type=float, default=None)
    parser.add_argument("--hard_negative_margin", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    if args.point_cache_root:
        config["data"]["point_cache_root"] = args.point_cache_root
    if args.max_packages is not None:
        config["data"]["max_packages"] = int(args.max_packages)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size_files"] = int(args.batch_size)
    if args.seed is not None:
        config["device"]["seed"] = int(args.seed)
    if args.package_split:
        config["data"]["package_split"] = args.package_split
        config["data"]["split_name"] = args.split_name
    if args.no_branch_init:
        config["initialization"] = {"strict": True}
    if args.line_init_checkpoint:
        config.setdefault("initialization", {})["line_checkpoint"] = args.line_init_checkpoint
    if args.polygon_init_checkpoint:
        config.setdefault("initialization", {})["polygon_checkpoint"] = args.polygon_init_checkpoint
    if args.point_init_checkpoint:
        config.setdefault("initialization", {})["point_checkpoint"] = args.point_init_checkpoint
    if args.init_checkpoint:
        config.setdefault("initialization", {})["init_checkpoint"] = args.init_checkpoint
    if args.freeze_branch_epochs is not None:
        config["training"]["freeze_branch_epochs"] = int(args.freeze_branch_epochs)
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = float(args.learning_rate)
    if args.hard_negative_weight is not None:
        config.setdefault("loss", {})["hard_negative_weight"] = float(args.hard_negative_weight)
    if args.hard_negative_margin is not None:
        config.setdefault("loss", {})["hard_negative_margin"] = float(args.hard_negative_margin)
    if args.save_dir:
        config["logging"]["save_dir"] = args.save_dir
    if args.log_dir:
        config["logging"]["log_dir"] = args.log_dir
    trainer = MixedGenericLinePolygonPointTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
