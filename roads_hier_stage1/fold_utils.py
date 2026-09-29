"""
Cross-fold split helpers for roads_hier_stage1 experiments.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def split_indices_kfold(num_items: int, num_folds: int, fold_index: int, seed: int) -> Tuple[List[int], List[int]]:
    """Split file indices into train/val for one fold."""
    if num_items < 2:
        raise ValueError("Need at least 2 files for k-fold splitting.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if num_folds > num_items:
        raise ValueError(f"num_folds={num_folds} exceeds num_items={num_items}.")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator).tolist()

    base = num_items // num_folds
    remainder = num_items % num_folds
    fold_sizes = [base + (1 if idx < remainder else 0) for idx in range(num_folds)]

    starts = [0]
    for size in fold_sizes[:-1]:
        starts.append(starts[-1] + size)

    start = starts[fold_index]
    end = start + fold_sizes[fold_index]
    val_indices = perm[start:end]
    train_indices = perm[:start] + perm[end:]
    return train_indices, val_indices
