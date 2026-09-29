# Reproduction protocol

All commands are run from the repository root. Replace cache paths only; do not
change the source-isolated split when reproducing the paper tables.

## 1. Train the line-polygon base encoder

```bash
python train_mixed_hier_stage1_line_polygon.py \
  --config configs/mixed_hier_stage1_line_polygon.yaml \
  --line_cache_root artifacts/caches/line \
  --polygon_cache_root artifacts/caches/polygon \
  --epochs 8 --batch_size 8 --seed 7 \
  --package_split splits/source_isolated_split.json \
  --split_name model_train_packages \
  --freeze_branch_epochs 0 \
  --save_dir checkpoints/seed7/lp_base \
  --log_dir logs/seed7/lp_base
```

## 2. Train the optional point auxiliary encoder

```bash
python train_point_hier_stage1_dual_view_hybrid.py \
  --config configs/point_hier_stage1_dual_view_hybrid_local.yaml \
  --cache_root artifacts/caches/point \
  --max_files 276 --epochs 8 --batch_size 8 --seed 7 \
  --package_split splits/source_isolated_split.json \
  --split_name model_train_packages \
  --save_dir checkpoints/seed7/point \
  --log_dir logs/seed7/point
```

## 3. Train the generic auxiliary encoder

```bash
python train_mixed_hier_stage1_line_polygon_point_generic.py \
  --config configs/mixed_hier_stage1_line_polygon_point_generic.yaml \
  --line_cache_root artifacts/caches/line \
  --polygon_cache_root artifacts/caches/polygon \
  --point_cache_root artifacts/caches/point \
  --epochs 8 --batch_size 8 --seed 7 \
  --package_split splits/source_isolated_split.json \
  --split_name model_train_packages \
  --no_branch_init --freeze_branch_epochs 0 \
  --save_dir checkpoints/seed7/generic \
  --log_dir logs/seed7/generic
```

## 4. Train SR-MGGS

```bash
python train_sr_msgs_line_polygon.py \
  --config configs/mixed_hier_stage1_line_polygon.yaml \
  --init_checkpoint checkpoints/seed7/lp_base/best_mixed_hier_stage1_line_polygon.pth \
  --line_cache_root artifacts/caches/line \
  --polygon_cache_root artifacts/caches/polygon \
  --package_split splits/source_isolated_split.json \
  --split_name model_train_packages \
  --dataset_mode eval --seed 7 --prehash_space encoder \
  --scale_specs low:8:16,mid:16:32,high:32:64 \
  --negative_sampling random --top_k_negatives 5 \
  --epochs 3 --anchors_per_batch 2 \
  --learning_rate 2.0e-5 --temperature 0.1 \
  --margin 0.05 --margin_weight 0.0 --consistency_weight 0.0 \
  --save_dir checkpoints/seed7/sr_mggs
```

Repeat Sections 1-4 with seeds `17` and `27`.

## 5. Strict independent authentication

Use the command in `README.md`, changing checkpoint paths and seed for each
run. Evaluate target FAR values `0.01`, `0.05`, and `0.10` independently.

## 6. Budget-controlled ablation

Train/evaluate the following `--scale_specs` values under the same split and
calibration protocol:

| Variant | `scale_specs` |
| --- | --- |
| Single low budget | `low:8:16` |
| Single high budget | `high:32:64` |
| Budget-matched multi-granularity | `low:4:8,mid:8:16,high:16:32` |
| Full SR-MGGS | `low:8:16,mid:16:32,high:32:64` |

The budget-matched setting is essential because it distinguishes the effect of
joint observations from the effect of simply processing more objects.

## 7. Raw-geometry perturbations

The canonical perturbations reported in the paper are:

- random deletion of 30% of geometry objects;
- cropping 50% of the spatial extent; and
- light geometric simplification.

Create attacked queries with `scripts/create_attacked_vector_dataset.py`,
rebuild their caches, and evaluate them with clean templates and calibration
thresholds. Do not mix attacked samples into enrollment or threshold
calibration.

The delete and simplify experiments have fewer valid queries than the clean
condition. Their results describe the retained valid subsets and must not be
interpreted as strictly paired degradation estimates.

## 8. Verification against released summaries

Compare generated JSON summaries with `paper_results/`. Small floating-point
differences may occur across GPU models and PyTorch/CUDA versions. Package
counts, split membership, threshold protocol, and metric definitions must
remain unchanged.

