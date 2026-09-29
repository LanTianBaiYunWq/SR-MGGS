"""Train SR-MSGS V1 for line+polygon metric signatures.

SR-MSGS V1 keeps the existing LP encoder architecture and changes the training
protocol: each package is represented by several fixed-budget views at multiple
geometry budgets, with a cross-view consistency term and mined hard negatives.
The output remains a cosine metric-space signature, not a pairwise classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mixed_hier_stage1.line_polygon_dataset import MixedLinePolygonPackageDataset
from mixed_hier_stage1.line_polygon_protocol import build_partial_package_views
from scripts.evaluate_mixed_generic_score_fusion_ablation import load_package_subset
from scripts.evaluate_mixed_hier_stage1_line_polygon import MixedLinePolygonEvaluator
from scripts.evaluate_mixed_lpp_score_fusion import score_matrix
from train_roads_hier_stage1_minimal import load_config, resolve_path
from utils.seed import set_seed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SR-MSGS V1 for LP geometry signatures.")
    parser.add_argument("--config", type=str, default="configs/mixed_hier_stage1_line_polygon.yaml")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--line_cache_root", type=str, default=None)
    parser.add_argument("--polygon_cache_root", type=str, default=None)
    parser.add_argument("--package_split", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="train_packages")
    parser.add_argument("--dataset_mode", type=str, default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prehash_space", type=str, default="encoder", choices=["encoder", "projector"])
    parser.add_argument(
        "--scale_specs",
        type=str,
        default="local:8:16,regional:16:32,global:32:64",
        help="Comma-separated scale:line_chunks:polygon_objects specs. n_tiles stays config-compatible.",
    )
    parser.add_argument("--negative_sampling", type=str, default="mined", choices=["mined", "random"])
    parser.add_argument("--top_k_negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--anchors_per_batch", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2.0e-5)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--margin_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--freeze_bottom_encoders", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, required=True)
    return parser


def parse_scale_specs(value: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid scale spec '{item}', expected name:line_chunks:polygon_objects")
        specs.append(
            {
                "name": parts[0],
                "line_k_chunks": int(parts[1]),
                "polygon_k_objects": int(parts[2]),
            }
        )
    if not specs:
        raise ValueError("No scale specs provided")
    return specs


def build_dataset(config: Dict[str, Any], args: argparse.Namespace) -> MixedLinePolygonPackageDataset:
    data_cfg = config["data"]
    dataset = MixedLinePolygonPackageDataset(
        data_cfg["line_cache_root"],
        data_cfg["polygon_cache_root"],
        mode=args.dataset_mode,
        line_max_tiles=data_cfg.get("line_max_tiles"),
        polygon_max_tiles=data_cfg.get("polygon_max_tiles"),
        line_tile_selector=data_cfg.get("line_tile_selector", "manifest_default"),
        polygon_tile_selector=data_cfg.get("polygon_tile_selector", "manifest_default"),
        line_subtypes=data_cfg.get("line_subtypes", ("roads", "railways", "waterways")),
        polygon_subtypes=data_cfg.get("polygon_subtypes", ("building", "landuse", "natural", "water")),
        max_packages=data_cfg.get("max_packages"),
        require_complete_packages=bool(data_cfg.get("require_complete_packages", False)),
    )
    subset = load_package_subset(args.package_split, args.split_name)
    if subset is not None:
        dataset.package_records = [
            record for record in dataset.package_records if str(record["package_id"]).lower() in subset
        ]
    return dataset


def build_multiscale_views(
    dataset: MixedLinePolygonPackageDataset,
    config: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    line_protocol = config["line_protocol"]
    polygon_protocol = config["polygon_protocol"]
    line_depth_template = {int(key): int(value) for key, value in line_protocol["depth_template"].items()}
    polygon_depth_template = {int(key): int(value) for key, value in polygon_protocol["depth_template"].items()}
    scale_specs = parse_scale_specs(args.scale_specs)
    package_ids: List[str] = []
    all_views: List[Dict[str, Any]] = []

    for idx in tqdm(range(len(dataset)), desc="Build SR-MSGS multiscale views"):
        sample = dataset[idx]
        base_seed = int(args.seed) * 1000003 + idx * 1009
        scale_views: Dict[str, Any] = {}
        for scale_idx, spec in enumerate(scale_specs):
            scale_seed = base_seed + scale_idx * 10007
            scale_views[str(spec["name"])] = build_partial_package_views(
                sample,
                line_subtypes=tuple(config["data"].get("line_subtypes", ("roads", "railways", "waterways"))),
                polygon_subtypes=tuple(config["data"].get("polygon_subtypes", ("building", "landuse", "natural", "water"))),
                line_subtypes_per_view=config["data"].get("line_subtypes_per_view"),
                polygon_subtypes_per_view=config["data"].get("polygon_subtypes_per_view"),
                line_n_tiles=int(line_protocol["n_tiles"]),
                line_k_chunks=int(spec["line_k_chunks"]),
                line_depth_template=line_depth_template,
                polygon_n_tiles=int(polygon_protocol["n_tiles"]),
                polygon_k_polygons=int(spec["polygon_k_objects"]),
                polygon_depth_template=polygon_depth_template,
                rng_a=np.random.default_rng(scale_seed + 1),
                rng_b=np.random.default_rng(scale_seed + 2),
                feature_noise_std=0.0,
                device=device,
            )
        package_ids.append(str(sample["package_id"]))
        all_views.append(scale_views)
    return package_ids, all_views, scale_specs


def encode_lp_tensor(
    evaluator: MixedLinePolygonEvaluator,
    view_bundle: Dict[str, Any],
    suffix: str,
    prehash_space: str,
) -> torch.Tensor:
    fused, _ = evaluator.encoder(view_bundle[f"line_view_{suffix}"], view_bundle[f"polygon_view_{suffix}"])
    if prehash_space == "projector":
        fused = evaluator.projector(fused.unsqueeze(0))[0]
    return fused


def encode_multiscale_features(
    evaluator: MixedLinePolygonEvaluator,
    scale_views: Dict[str, Any],
    scale_specs: Sequence[Dict[str, Any]],
    prehash_space: str,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for spec in scale_specs:
        bundle = scale_views[str(spec["name"])]
        features.append(encode_lp_tensor(evaluator, bundle, "a", prehash_space))
        features.append(encode_lp_tensor(evaluator, bundle, "b", prehash_space))
    return F.normalize(torch.stack(features, dim=0), dim=-1)


def mean_signature(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.mean(dim=0), dim=-1)


@torch.no_grad()
def mine_hard_negative_groups(
    evaluator: MixedLinePolygonEvaluator,
    all_views: Sequence[Dict[str, Any]],
    scale_specs: Sequence[Dict[str, Any]],
    prehash_space: str,
    top_k: int,
) -> List[List[int]]:
    evaluator.encoder.eval()
    evaluator.projector.eval()
    embeddings: List[np.ndarray] = []
    for views in tqdm(all_views, desc="Encode SR-MSGS mining signatures"):
        features = encode_multiscale_features(evaluator, views, scale_specs, prehash_space)
        embeddings.append(mean_signature(features).detach().cpu().numpy())
    scores = score_matrix(np.asarray(embeddings), np.asarray(embeddings))
    groups: List[List[int]] = []
    for i in range(scores.shape[0]):
        row = [i]
        order = np.argsort(-scores[i])
        for j in order:
            j_int = int(j)
            if j_int == i:
                continue
            row.append(j_int)
            if len(row) - 1 >= int(top_k):
                break
        groups.append(row)
    return groups


def build_random_negative_groups(num_items: int, top_k: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(int(seed))
    all_indices = np.arange(int(num_items), dtype=np.int64)
    groups: List[List[int]] = []
    for i in range(int(num_items)):
        candidates = all_indices[all_indices != i]
        count = min(int(top_k), len(candidates))
        if count <= 0:
            groups.append([i])
            continue
        negatives = rng.choice(candidates, size=count, replace=False)
        groups.append([i] + [int(value) for value in negatives])
    return groups


def consistency_loss(features: torch.Tensor) -> torch.Tensor:
    if features.size(0) <= 1:
        return features.new_zeros(())
    center = mean_signature(features).detach()
    return (1.0 - torch.matmul(features, center)).mean()


def group_loss(
    evaluator: MixedLinePolygonEvaluator,
    all_views: Sequence[Dict[str, Any]],
    scale_specs: Sequence[Dict[str, Any]],
    anchor_indices: Sequence[int],
    mined_groups: Sequence[Sequence[int]],
    *,
    prehash_space: str,
    temperature: float,
    margin: float,
    margin_weight: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    feature_cache: Dict[int, torch.Tensor] = {}
    mean_cache: Dict[int, torch.Tensor] = {}

    def features(index: int) -> torch.Tensor:
        if index not in feature_cache:
            feature_cache[index] = encode_multiscale_features(
                evaluator,
                all_views[index],
                scale_specs,
                prehash_space,
            )
        return feature_cache[index]

    def signature(index: int) -> torch.Tensor:
        if index not in mean_cache:
            mean_cache[index] = mean_signature(features(index))
        return mean_cache[index]

    losses: List[torch.Tensor] = []
    ce_losses: List[torch.Tensor] = []
    margin_losses: List[torch.Tensor] = []
    consistency_losses: List[torch.Tensor] = []
    gaps: List[float] = []

    for anchor_idx in anchor_indices:
        group = list(mined_groups[int(anchor_idx)])
        anchor_features = features(int(anchor_idx))
        anchor_signature = signature(int(anchor_idx))
        negative_signatures = torch.stack([signature(int(index)) for index in group[1:]], dim=0)
        candidates = torch.cat([anchor_signature.unsqueeze(0), negative_signatures], dim=0)
        logits = torch.matmul(anchor_features, candidates.t()) / float(temperature)
        labels = torch.zeros(anchor_features.size(0), dtype=torch.long, device=anchor_features.device)
        ce = F.cross_entropy(logits, labels)
        positive_scores = torch.matmul(anchor_features, anchor_signature)
        negative_scores = torch.matmul(anchor_features, negative_signatures.t())
        hard_negative = negative_scores.max()
        positive = positive_scores.mean()
        margin_term = F.relu(float(margin) + hard_negative - positive)
        consistency = consistency_loss(anchor_features)
        loss = ce + float(margin_weight) * margin_term + float(consistency_weight) * consistency
        losses.append(loss)
        ce_losses.append(ce.detach())
        margin_losses.append(margin_term.detach())
        consistency_losses.append(consistency.detach())
        gaps.append(float((positive - hard_negative).detach().cpu().item()))

    total = torch.stack(losses).mean()
    return total, {
        "ce": float(torch.stack(ce_losses).mean().item()),
        "margin": float(torch.stack(margin_losses).mean().item()),
        "consistency": float(torch.stack(consistency_losses).mean().item()),
        "gap": float(np.mean(gaps)),
    }


def set_trainable(evaluator: MixedLinePolygonEvaluator, freeze_bottom: bool) -> None:
    for param in evaluator.encoder.parameters():
        param.requires_grad = True
    for param in evaluator.projector.parameters():
        param.requires_grad = True
    if freeze_bottom:
        for module in (evaluator.encoder.line_encoder, evaluator.encoder.polygon_encoder):
            for param in module.parameters():
                param.requires_grad = False


def save_checkpoint(path: Path, evaluator: MixedLinePolygonEvaluator, args: argparse.Namespace, epoch: int, loss: float) -> None:
    base = torch.load(resolve_path(args.init_checkpoint), map_location=evaluator.device)
    base["epoch"] = epoch
    base["val_loss"] = loss
    base["encoder"] = evaluator.encoder.state_dict()
    base["projector"] = evaluator.projector.state_dict()
    base["sr_msgs_v1"] = vars(args)
    torch.save(base, path)


def main() -> None:
    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    if args.line_cache_root:
        config["data"]["line_cache_root"] = args.line_cache_root
    if args.polygon_cache_root:
        config["data"]["polygon_cache_root"] = args.polygon_cache_root
    config["device"]["seed"] = int(args.seed)
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config, args)
    evaluator = MixedLinePolygonEvaluator(args.init_checkpoint, device=args.device)
    package_ids, all_views, scale_specs = build_multiscale_views(dataset, config, args, device)
    if args.negative_sampling == "mined":
        mined_groups = mine_hard_negative_groups(
            evaluator,
            all_views,
            scale_specs,
            args.prehash_space,
            int(args.top_k_negatives),
        )
    else:
        mined_groups = build_random_negative_groups(len(all_views), int(args.top_k_negatives), int(args.seed))
    print(
        f"SR-MSGS LP packages: {len(package_ids)}, scales={len(scale_specs)}, "
        f"groups={len(mined_groups)}, top_k={args.top_k_negatives}, negatives={args.negative_sampling}"
    )

    set_trainable(evaluator, bool(args.freeze_bottom_encoders))
    evaluator.encoder.train()
    evaluator.projector.train()
    if args.freeze_bottom_encoders:
        evaluator.encoder.line_encoder.eval()
        evaluator.encoder.polygon_encoder.eval()
    trainable_params = [
        param
        for param in list(evaluator.encoder.parameters()) + list(evaluator.projector.parameters())
        if param.requires_grad
    ]
    optimizer = optim.AdamW(trainable_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    anchor_order = np.arange(len(mined_groups), dtype=np.int64)
    best_loss = float("inf")

    for epoch in range(int(args.epochs)):
        rng.shuffle(anchor_order)
        losses: List[float] = []
        ces: List[float] = []
        margins: List[float] = []
        consistencies: List[float] = []
        gaps: List[float] = []
        iterator = tqdm(range(0, len(anchor_order), int(args.anchors_per_batch)), desc=f"SR-MSGS epoch {epoch}")
        for start in iterator:
            anchors = anchor_order[start : start + int(args.anchors_per_batch)]
            optimizer.zero_grad(set_to_none=True)
            loss, stats = group_loss(
                evaluator,
                all_views,
                scale_specs,
                anchors,
                mined_groups,
                prehash_space=args.prehash_space,
                temperature=float(args.temperature),
                margin=float(args.margin),
                margin_weight=float(args.margin_weight),
                consistency_weight=float(args.consistency_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            ces.append(stats["ce"])
            margins.append(stats["margin"])
            consistencies.append(stats["consistency"])
            gaps.append(stats["gap"])
            iterator.set_postfix(
                loss=f"{np.mean(losses):.4f}",
                ce=f"{np.mean(ces):.4f}",
                cons=f"{np.mean(consistencies):.4f}",
                gap=f"{np.mean(gaps):.4f}",
            )

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Epoch {epoch}: loss={epoch_loss:.4f}, ce={np.mean(ces):.4f}, "
            f"margin={np.mean(margins):.4f}, consistency={np.mean(consistencies):.4f}, "
            f"gap={np.mean(gaps):.4f}"
        )
        save_checkpoint(save_dir / "final_mixed_hier_stage1_line_polygon.pth", evaluator, args, epoch, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(save_dir / "best_mixed_hier_stage1_line_polygon.pth", evaluator, args, epoch, epoch_loss)
            print(f"New best SR-MSGS LP loss: {best_loss:.4f}")

    with (save_dir / "sr_msgs_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"package_ids": package_ids, "scale_specs": scale_specs, "arguments": vars(args)},
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
