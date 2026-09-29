# Released result summaries

These files are direct machine-readable summaries of the experiments reported
in the manuscript. Local paths have been replaced with `${PROJECT_ROOT}`.

The filenames retain the historical `sr_msgs` experiment prefix for
traceability. In the manuscript, the final method name is SR-MGGS.

Primary files:

- `...far0p01...summary.json`: strict calibration at target FAR 1%;
- `...far0p05...summary.json`: strict calibration at target FAR 5%;
- `...far0p10...summary.json`: strict calibration at target FAR 10%;
- `...line_polygon_level0...summary.json`: line-polygon signature evaluation;
- `sr_msgs_budget_control_strictindcal_comparison.json`: budget ablation; and
- `sr_msgs_rawgeom_attack_*_summary.json`: geometry perturbation results.

The JSON files are evidence artifacts, not portable runtime configurations.
Use the relative commands in the root README for reproduction.
