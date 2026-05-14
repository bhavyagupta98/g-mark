# G-MARK Table-I/Q9 Row `gmark_table1_q9_full_v3`

This note records the paper-style Table-I planning result for the G-MARK row on
the released V2V-GoT Q9 future-trajectory split.

## Run Metadata

- method: `G-MARK`
- released Q9 file: `/workspace/repos/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json`
- released Q9 sample count: `3446`
- source manifest: `/workspace/repos/kg_coop_drive/outputs/e2e_runs/r5/e2e_model_manifest.json`
- official summary: `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9/official_eval_reports/gmark_table1_q9_full_v3_official_export_manifest_official_qa_eval_summary.json`
- communication accounting: `compact_json(LocalGraphSerializer(prepared_gmark_scene)) average bytes per Q9 sample`

## Table-I Planning Comparison

Reporting provenance:

- Rows `No Fusion` through `V2V-GoT` are borrowed reference numbers from the
  published V2V-GoT Table I. They are included to provide context and should be
  cited as prior-paper results.
- The shared upstream substrate is V2V4Real-derived processed perception
  evidence. Raw synchronized sensor frames are processed by the V2V4Real /
  OpenCOOD-style cooperative perception stack into structured artifacts under
  `official_models/.../npy/co_llm`; V2V-GoT/V2V-LLM and G-MARK both operate
  downstream of that processed representation.
- The `G-MARK` row is produced by our pipeline on the released Table-I/Q9 split,
  exported to official-compatible format, and scored with the official-compatible
  evaluator listed in the metadata above.
- G-MARK does not claim to run a new image/LiDAR perception frontend. Its
  contribution starts at the representation/reasoning layer: converting the
  structured V2V-GoT-QA scene/perception evidence into an explicit cooperative
  knowledge graph and task-specific planning readout.
- Therefore, the table should be described as "V2V-GoT reported rows plus our
  G-MARK row on the same released Q9 split," not as a full rerun of all
  baselines.

| Method | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | L2 Avg ↓ | CR 1s ↓ | CR 2s ↓ | CR 3s ↓ | CR Avg ↓ | Comm MB ↓ | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `No Fusion` | 3.47 | 5.79 | 8.26 | 5.84 | 1.48 | 4.24 | 7.72 | 4.48 | 0.0000 | `V2V-GoT Table I` |
| `Early Fusion` | 3.48 | 5.61 | 7.82 | 5.63 | 1.16 | 3.51 | 5.66 | 3.44 | 1.9208 | `V2V-GoT Table I` |
| `AttFuse` | 3.65 | 6.21 | 8.75 | 6.20 | 1.19 | 4.41 | 6.38 | 3.99 | 0.4008 | `V2V-GoT Table I` |
| `V2X-ViT` | 3.46 | 5.80 | 8.19 | 5.81 | 1.45 | 4.24 | 6.59 | 4.09 | 0.4008 | `V2V-GoT Table I` |
| `CoBEVT` | 3.38 | 5.42 | 7.46 | 5.42 | 1.31 | 4.41 | 5.75 | 3.82 | 0.4008 | `V2V-GoT Table I` |
| `V2V-LLM` | 2.90 | 4.91 | 6.98 | 4.93 | 0.75 | 2.87 | 4.93 | 2.85 | 0.4068 | `V2V-GoT Table I` |
| `V2V-GoT` | 1.65 | 2.63 | 3.59 | 2.62 | 0.12 | 1.92 | 3.45 | 1.83 | 0.4068 | `V2V-GoT Table I` |
| `G-MARK` | 0.95 | 1.35 | 1.33 | 1.21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0159 | `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9/official_eval_reports/gmark_table1_q9_full_v3_official_export_manifest_official_qa_eval_summary.json` |

## Interpretation

The result is strong: on the released Q9 split, G-MARK reports lower L2 error
than the V2V-GoT Table-I row at every horizon, with a substantially lower
average L2 error.

The strongest headline is:

- G-MARK L2 Avg: `1.21 m`
- V2V-GoT L2 Avg: `2.62 m`
- absolute reduction: `1.41 m`
- relative reduction: about `53.8%`

The collision-rate result should be treated as dependency-sensitive:

- G-MARK CR Avg: `0.00%`
- V2V-GoT CR Avg: `1.83%`

The G-MARK CR values are parsed from the official-compatible evaluator output;
they are not computed by a custom G-MARK collision formula. However, this metric
depends on the V2V-GoT/AB3DMOT collision-checking helpers and the `npy` GT-box
assets being available in the evaluation environment. If those helpers are
missing, the patched evaluator can preserve the L2 path while returning
non-collision for every sample, which would make `0.00%` look artificially good.
Before reporting the CR row, verify the evaluator log/dependency status on the
VM run.

The communication number is much smaller than the Table-I LLM fusion rows:

- G-MARK Comm: `0.0159 MB`
- V2V-GoT Comm: `0.4068 MB`

This is consistent with the method design: G-MARK transmits or accounts for a
compact symbolic scene graph rather than dense perception features or
LLM-oriented feature payloads.

## Defensibility Notes

This row is most defensible when described with the exact evaluation protocol:

- The split is the released V2V-GoT Table-I/Q9 JSON file, not the general
  `v2v4real_3d_grounding_qa_dataset_v2vgot.json` validation QA file.
- The sample count is `3446`, matching the released Q9 validation-style split.
- The output was exported into the V2V-GoT official QA format and scored by the
  official-compatible future-trajectory evaluator.
- The G-MARK row uses the `r5` Q9 future-trajectory regressor from the recorded
  E2E model manifest.
- The communication number is computed from the compact serialized local graph
  for each prepared G-MARK scene, averaged across the released Q9 samples.

The main caveat is that communication accounting should be stated explicitly.
The Table-I baseline communication costs refer to their respective perception or
LLM feature payloads, while G-MARK reports the compact serialized KG payload.
That comparison is still meaningful for a communication-efficiency argument,
but the payload definition should be transparent in the paper.

## Paper-Facing Takeaway

For the Q9 planning task, the result supports the claim that an explicit
cooperative scene graph can be an efficient planning substrate. G-MARK improves
trajectory accuracy while using a much smaller reported communication payload
than the V2V-GoT LLM-fusion row. The collision-rate comparison should be
included only after confirming that collision dependencies were active during
the official-compatible evaluation run.
