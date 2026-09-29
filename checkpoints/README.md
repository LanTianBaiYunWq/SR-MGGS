# Pretrained checkpoints

This release contains all nine checkpoints used for the three-seed strict
independent-calibration evaluation:

```text
checkpoints/
|-- seed7/
|   |-- sr_mggs_lp.pth
|   |-- point_auxiliary.pth
|   `-- generic_auxiliary.pth
|-- seed17/
|   |-- sr_mggs_lp.pth
|   |-- point_auxiliary.pth
|   `-- generic_auxiliary.pth
`-- seed27/
    |-- sr_mggs_lp.pth
    |-- point_auxiliary.pth
    `-- generic_auxiliary.pth
```

`sr_mggs_lp.pth` is the multi-granularity line-polygon main model.
`point_auxiliary.pth` is the optional point-feature auxiliary model, and
`generic_auxiliary.pth` is the subtype-independent auxiliary model. Files from
different seed directories must not be mixed in one evaluation run.

`checkpoints/MANIFEST.csv` records each file's seed, role, size, selected epoch,
validation loss, and SHA-256 hash. The root-level `SHA256SUMS.txt` records the
integrity hash of every checkpoint and source file in the release. These files
are PyTorch serialized checkpoints and should only be loaded from this trusted
release.

