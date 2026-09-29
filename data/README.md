# Data preparation

The model consumes precomputed hierarchical caches rather than shapefiles
directly during training and evaluation.

The OpenStreetMap-derived portion can be obtained from OpenStreetMap under the
ODbL and is not duplicated here because of its volume. The independently
collected portion is not publicly available because of data-ownership and
redistribution restrictions. Generated caches are also not distributed. The
source-isolated package IDs are retained in
`splits/source_isolated_split.json`; identifiers associated with restricted
sources must be anonymized before public release if they disclose protected
location or ownership information.

Recommended artifact layout:

```text
artifacts/caches/
|-- line/
|-- polygon/
`-- point/
```

Cache construction entry points are:

```bash
python scripts/build_vector_manifest.py --help
python scripts/build_roads_hier_cache.py --help
python scripts/build_polygon_hier_cache.py --help
python scripts/build_point_hier_cache.py --help
python scripts/build_mixed_source_isolated_split.py --help
```

The exact input paths depend on the authorized raw-data location. Cache
construction commands must preserve the package IDs and counts documented in
`DATA_AVAILABILITY.md` and `REPRODUCIBILITY.md`.

