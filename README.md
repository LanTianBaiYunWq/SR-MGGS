# SR-MGGS

Official supplementary code package for **Source-Robust Multi-Granularity
Geometric Signatures for Vector GIS Zero-Watermark Authentication**.

SR-MGGS learns a package-level signature from line and polygon geometry under
three sampling budgets. The physical spatial extent and tile hierarchy are
kept fixed; "multi-granularity" refers to different numbers of sampled line
chunks and polygon objects, not to different physical spatial scales.

The repository supports:

- multi-granularity line-polygon signature training;
- source-isolated train/calibration/test evaluation;
- K-template zero-watermark registration;
- max-over-K matching and independently calibrated FAR thresholds;
- budget-controlled ablation;
- raw-geometry perturbation evaluation; and
- a lightweight, CPU-only authentication smoke test.

## Important naming note

The paper uses the final method name **SR-MGGS**. Some internal filenames retain
the earlier experiment identifier `sr_msgs` or the word `multiscale` to preserve
compatibility with the checkpoints and result files used in the experiments.
Those names refer to the same sampling-budget-based method and do not imply
physical multi-scale spatial windows.

## Repository layout

```text
SR-MGGS-release/
|-- README.md
|-- README_CN.md
|-- REPRODUCIBILITY.md
|-- DATA_AVAILABILITY.md
|-- THIRD_PARTY_DATA.md
|-- SUBMISSION_CHECKLIST.md
|-- environment.yml
|-- requirements.txt
|-- configs/                 # model and protocol configurations
|-- data/                    # user-provided raw data or released sample data
|-- checkpoints/             # pretrained checkpoints for seeds 7, 17, and 27
|-- examples/                # CPU-only minimal authentication demo
|-- paper_results/           # machine-readable reported results
|-- scripts/                 # preprocessing and evaluation entry points
|-- mixed_hier_stage1/       # line-polygon/generic encoders
|-- roads_hier_stage1/       # line geometry encoder and view construction
|-- polygon_hier_stage1/     # polygon geometry encoder
|-- point_hier_stage1/       # optional point auxiliary encoder
|-- preprocess/              # geometry preprocessing utilities
|-- utils/                   # metrics and reproducibility utilities
`-- tests/
```

## 1. Environment

The experiments were developed with Python 3.10, PyTorch 2.1.0 and CUDA 11.8.
NumPy is pinned below version 2 because the original PyTorch binary is not
compatible with NumPy 2.x.

```bash
conda env create -f environment.yml
conda activate sr-mggs
```

For a CPU-only environment, install the packages in `requirements.txt` and use
the CPU build of PyTorch from the official PyTorch installation selector.

## 2. Minimal reproducible example

The minimal example validates the authentication stage without requiring GIS
data, a GPU, or pretrained weights. It uses a small set of deterministic
signature bundles and reproduces:

1. composite LP/generic/optional-point similarity;
2. max-over-K template aggregation;
3. independent threshold calibration at a target FAR; and
4. accept/reject source authentication.

Run:

```bash
python examples/run_minimal_demo.py \
  --input examples/minimal_demo/input.json \
  --output outputs/minimal_demo_result.json
```

Expected final lines:

```text
Predicted source: source_alpha
Decision: Accept
```

Run the smoke test:

```bash
python -m unittest tests/test_minimal_demo.py
```

This example checks the decision logic only. It is not presented as a
reproduction of the paper's learned embeddings or reported metrics.

## 3. Data and checkpoints

The release includes the three compatible checkpoints used by each reported
seed:

```text
checkpoints/
|-- seed7/
|   |-- sr_mggs_lp.pth
|   |-- point_auxiliary.pth
|   `-- generic_auxiliary.pth
|-- seed17/
`-- seed27/
```

Full model evaluation additionally requires the line, polygon, and point cache
roots under `artifacts/caches/`. The OpenStreetMap-derived source data are
publicly obtainable under the ODbL but are not duplicated here because of
their volume. The independently collected source data and all generated caches
are not publicly distributed. The exact source-isolated split is provided in
`splits/source_isolated_split.json`. See `data/README.md`,
`checkpoints/README.md`, `DATA_AVAILABILITY.md`, and `THIRD_PARTY_DATA.md` for
access conditions, attribution, and expected paths.

## 4. Main authentication evaluation

After placing the caches and checkpoints, run one seed as follows:

```bash
python scripts/evaluate_sr_msgs_zero_watermark_authentication.py \
  --config configs/mixed_hier_stage1_line_polygon_point_generic.yaml \
  --lp_checkpoint checkpoints/seed7/sr_mggs_lp.pth \
  --point_checkpoint checkpoints/seed7/point_auxiliary.pth \
  --generic_checkpoint checkpoints/seed7/generic_auxiliary.pth \
  --line_cache_root artifacts/caches/line \
  --polygon_cache_root artifacts/caches/polygon \
  --point_cache_root artifacts/caches/point \
  --dataset_mode eval \
  --package_split splits/source_isolated_split.json \
  --calibration_split_name calibration_packages \
  --test_split_name heldout_packages \
  --seed 7 \
  --prehash_space encoder \
  --scale_specs low:8:16,mid:16:32,high:32:64 \
  --point_alpha 0.5 \
  --generic_beta 0.25 \
  --enrollment_templates 5 \
  --template_aggregation max \
  --target_far 0.05 \
  --device cuda \
  --output outputs/auth_seed7_far005.json
```

Repeat with seeds `7`, `17`, and `27`, using the correspondingly named
checkpoint directory for each run, then summarize:

```bash
python scripts/summarize_zero_watermark_auth_seed_runs.py \
  --reports outputs/auth_seed7_far005.json outputs/auth_seed17_far005.json outputs/auth_seed27_far005.json \
  --output outputs/auth_far005_3seed_summary.json
```

The formal paper results use target FAR values of `0.01`, `0.05`, and `0.10`.

## 5. Training and full reproduction

Training requires precomputed line, polygon, and point caches. Canonical
commands for base-encoder training, SR-MGGS training, independent calibration,
budget-controlled ablation, and perturbation evaluation are provided in
`REPRODUCIBILITY.md`.

## 6. Reported results

`paper_results/` contains JSON/Markdown summaries for:

- strict independent calibration at FAR 1%, 5%, and 10%;
- Level-0 line-polygon matching;
- budget-controlled multi-granularity ablation; and
- clean, random deletion, cropping, and light simplification conditions.

The primary strict-independent-calibration result at FAR 5% is:

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| AUC | 0.9614 | 0.0028 |
| EER | 0.1076 | 0.0019 |
| TAR@FAR5 | 0.6788 | 0.0309 |
| Actual FAR@5 | 0.0444 | 0.0040 |
| R@1 | 0.5515 | 0.0374 |
| R@5 | 0.8667 | 0.0227 |
| MRR | 0.6863 | 0.0288 |

## 7. Reproducibility notes

- The split unit is the raw source package ID.
- A source package appears in exactly one of model training, calibration, or
  held-out testing.
- The main results use five clean enrollment templates per registered source.
- Acceptance thresholds are estimated only from the independent calibration
  split.
- The final score is the LP score plus `0.5 * point score` when point features
  are jointly available, plus `0.25 * generic score`.
- The three geometry budgets are `(8,16)`, `(16,32)`, and `(32,64)` for line
  chunks and polygon objects, respectively.

## 8. Citation

The final bibliographic entry should be inserted after acceptance. For an
anonymous submission, do not add author-identifying metadata to this package.

## 9. License

No public software license is asserted by this review artifact. Add the
license required by the journal or the authors' institution before public
archiving.

