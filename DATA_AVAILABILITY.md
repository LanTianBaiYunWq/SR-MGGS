# Data availability

## Dataset scope

The experiments use 277 vector GIS source packages. Splitting is performed at
the raw source package ID level:

- 166 packages for model training;
- 56 packages for independent threshold calibration; and
- 55 packages for held-out testing.

The exact package identifiers are recorded in
`splits/source_isolated_split.json`.

## What is included

- the source-isolated split definition;
- preprocessing and cache-building code;
- the pretrained SR-MGGS, point-auxiliary, and generic-auxiliary checkpoints
  for seeds 7, 17, and 27;
- machine-readable result summaries; and
- a deterministic score-level minimal example.

## Dataset access

The experimental dataset contains an OpenStreetMap-derived portion and an
independently collected portion.

The OpenStreetMap-derived source data are publicly available from
https://www.openstreetmap.org and are licensed under the Open Data Commons Open
Database License (ODbL). They are not duplicated in this repository because of
their large volume. The preprocessing code and source-isolated split
definition are provided to document the experimental protocol. OpenStreetMap
data are copyright OpenStreetMap contributors; see `THIRD_PARTY_DATA.md`.

The independently collected source data are not publicly available because of
data-ownership and redistribution restrictions. This repository does not
offer public access to those raw files or their generated geometry caches.
Access, if required for confidential peer-review verification, is subject to
the data owner's authorization and the journal's controlled-review procedure.

Suggested manuscript statement:

> The dataset used in this study consists of two parts. The
> OpenStreetMap-derived data are publicly available from OpenStreetMap and are
> licensed under the Open Data Commons Open Database License (ODbL).
> OpenStreetMap data are copyright OpenStreetMap contributors. Due to their
> large volume, these data are not redistributed with the supplementary
> materials; preprocessing code and experimental split definitions are
> provided to document the reconstruction protocol. The independently
> collected data are not publicly available because of data-ownership and
> redistribution restrictions. The source code, anonymized split definitions,
> trained model weights, and machine-readable experimental results are
> provided as supplementary materials.

This wording must be checked against the selected journal's data-sharing policy
before submission. Do not claim public access to the independently collected
data, and do not promise access on request unless the data owner has explicitly
authorized it.

