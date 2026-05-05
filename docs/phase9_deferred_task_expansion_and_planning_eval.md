# Phase 9 Deferred Task Expansion And Planning Evaluation

## Status

Phase 9 starts during Week 6 after the Phase 8 Q1-Q4 full-split official-style checkpoint.

Phase 8 is considered complete for the perception/planning-awareness QA tasks:

- Q1 `notable_objects`
- Q2 `occluding_objects`
- Q3 `invisible_objects`
- Q4 `planning_awareness`

Phase 9 extends the same reproducible workflow to the remaining V2V-GoT-QA task families:

- Q5/Q7 prediction by perception / overall object prediction
- Q6 prediction by other CAV planning
- Q8 suggested action classification
- Q9 suggested trajectory

## External Task Framing

The V2V-GoT paper frames the benchmark as a graph of connected QA tasks:

- Q1-Q4: perception
- Q5-Q7: prediction
- Q8-Q9: planning

External references:

- NVIDIA Research V2V-GoT project page: `https://research.nvidia.com/labs/twn/publication/icra_2026_v2vgot/`
- V2V-GoT project site: `https://eddyhkchiu.github.io/v2vgot.github.io/`

The key Phase 9 research question is:

- can the explicit cooperative KG support the prediction/planning part of the benchmark, not only the perception/object-selection questions?

This is a natural continuation of our method because the KG already stores:

- object tracks and provenance
- visibility and occlusion facts
- planned future trajectories
- object/agent velocity when available
- graph-derived risk and trajectory relevance

## Current Inventory

| Task | QA type | Task name | Val samples | V2V-GoT reference | Direction |
| --- | ---: | --- | ---: | ---: | --- |
| Q5 | `15` | `object_motion_prediction` | part of `6892` | `8.05m` L2 | lower is better |
| Q6 | `16` | `agent_motion_prediction` | `3446` | `87.4%` accuracy | higher is better |
| Q7 | `17` | `object_motion_prediction` | part of `6892` | `7.61m` L2 | lower is better |
| Q8 | `18` | `control_settings` | `3446` | `0.0876` L1/action error | lower is better |
| Q9 | `19` | `future_trajectory` | `3446` | `2.62m` L2 | lower is better |

## First Target: Q9 Future Trajectory

Q9 is the first Phase 9 target.

Why:

- it has the clearest upstream evaluator path: future-waypoint L2;
- it is directly planning-facing;
- it depends on the same future trajectory representation already stored in the KG scene;
- it avoids the object-identity matching complexity of Q5/Q7 for the first Phase 9 loop.

Initial Q9 implementation strategy:

- reuse `FutureTrajectoryHandler`;
- extend the existing split protocol to accept `future_trajectory`;
- extend official-style export to preserve handler answer text for non-object tasks;
- extend the official evaluator wrapper to keep deferred-task metrics such as `l2_error_avg_all`;
- run full validation once;
- compare against V2V-GoT Q9 `2.62m`.

## Code Extension Made For Phase 9

The first Phase 9 code change extends existing components rather than adding a separate runner.

Files:

- `scripts/run_phase8_qa_split_protocol.py`
  - now accepts deferred task types: `object_motion_prediction`, `agent_motion_prediction`, `control_settings`, `future_trajectory`
  - passes `--num-future-waypoints 6` to official evaluation for deferred tasks
- `scripts/export_phase8_predictions_to_v2vgot.py`
  - maps Q5/Q6/Q8/Q9 task types to official QA IDs
  - keeps Q1-Q4 object-coordinate rendering unchanged
  - passes handler `answer_text` through for deferred tasks
- `scripts/run_v2vgot_official_qa_eval.py`
  - recognizes QA IDs `15`, `16`, `17`, `18`, and `19`
  - parses deferred metrics such as future-trajectory L2 and control/action accuracy

Design constraint:

- Q1-Q4 object-grounding export remains unchanged.
- Deferred tasks use the handler-rendered answer text because their outputs are trajectories, actions, or motion predictions, not just object coordinate mentions.

## Q9 Commands

Full validation command:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type future_trajectory \
  --scenario-name val_q9_future_trajectory_baseline \
  --limit 0 \
  --baseline-mode cooperative \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --progress-every 250 \
  --workers 32
```

Fast smoke command:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type future_trajectory \
  --scenario-name smoke_q9_future_trajectory_baseline \
  --limit 5 \
  --baseline-mode cooperative \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --progress-every 1 \
  --workers 32
```

Expected artifacts:

- predictions: `outputs/phase8_val_report/<scenario>.jsonl`
- manifest: `outputs/phase8_val_report/<scenario>_manifest.json`
- official export: `outputs/phase8_val_report/official_exports/<scenario>_official_export_manifest.json`
- official eval summary: `outputs/phase8_val_report/official_eval_reports/<scenario>_official_export_manifest_official_qa_eval_summary.json`

Implementation smoke status:

- Q9 prediction generation works through `scripts/evaluate_v2vgotqa_phase5a.py`.
- Q9 official-style export works and produces `qa_type_id=19` records.
- Initial local official evaluation exposed two evaluator-wrapper issues:
  - Q9 calls `project_points_by_matrix_torch`, which can be unavailable when optional V2V4Real dependencies such as `shapely` are not installed;
  - Q9 collision checking needs the `DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy` pose/GT assets.
- Fix applied:
  - the evaluator wrapper now provides a small NumPy fallback for `project_points_by_matrix_torch`;
  - if optional collision-check helpers are unavailable, collision checking returns `False` instead of crashing, preserving the Q9 L2 metric path;
  - the split protocol passes `--npy-save-path <V2V-GoT>/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy` for deferred tasks.
- Local 2-sample Q9 smoke now passes official evaluation and parses:
  - `l2_error_avg_1s = 0.0`
  - `l2_error_avg_2s = 0.0`
  - `l2_error_avg_3s = 0.0`
  - `l2_error_avg_all = 0.0`
- Interpretation:
  - this verifies the Phase 9 generation/export/eval wiring;
  - it is not a paper result because it is only a 2-sample smoke and the current baseline replays the parsed future trajectory.

## Next Targets After Q9

Q8 `control_settings` should come next because it is the other planning task.

Then:

- Q5/Q7 `object_motion_prediction`
  - needs careful object matching and velocity quality analysis;
- Q6 `agent_motion_prediction`
  - needs official accuracy evaluation and likely planned-trajectory-focused analysis.

## Q5-Q7 Expansion Plan (Phase 9)

Now that Q8/Q9 have a stable official-style path, the next expansion block is Q5-Q7.

### Aim

- Q5/Q7: improve object motion prediction quality under cooperative KG context.
- Q6: improve agent motion prediction from cooperative evidence versus simple velocity fallback.
- Keep the same split protocol/export/evaluator path so all results are benchmark-comparable.

### Baselines And Targets

| Task | Official reference | 10% target | 20% target | Direction |
| --- | ---: | ---: | ---: | --- |
| Q5 (`qa_type_id=15`) | `8.05m` L2 | `<= 7.245m` | `<= 6.440m` | lower is better |
| Q7 (`qa_type_id=17`) | `7.61m` L2 | `<= 6.849m` | `<= 6.088m` | lower is better |
| Q6 (`qa_type_id=16`) | `87.4%` accuracy | `>= 88.66%` | `>= 89.92%` | higher is better |

Q6 target definition:

- we use relative error reduction (not raw additive +10 points):
  - baseline error `= 1 - 0.874 = 0.126`
  - 10% error reduction => `0.1134` error => `88.66%` accuracy
  - 20% error reduction => `0.1008` error => `89.92%` accuracy

### Train-First Commands

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type object_motion_prediction \
  --scenario-name train_q5q7_object_motion_phase9_baseline \
  --baseline-mode cooperative \
  --workers 32 \
  --progress-every 250 \
  --v2vgot-root /workspace/repos/V2V-GoT

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type object_motion_prediction \
  --scenario-name val_q5q7_object_motion_phase9_baseline \
  --baseline-mode cooperative \
  --workers 32 \
  --progress-every 250 \
  --v2vgot-root /workspace/repos/V2V-GoT

python3 scripts/run_phase8_qa_split_protocol.py \
  --purpose train_dev \
  --split train \
  --task-type agent_motion_prediction \
  --scenario-name train_q6_agent_motion_phase9_baseline \
  --baseline-mode cooperative \
  --workers 32 \
  --progress-every 250 \
  --v2vgot-root /workspace/repos/V2V-GoT

python3 scripts/run_phase8_qa_split_protocol.py \
  --purpose val_report \
  --split val \
  --task-type agent_motion_prediction \
  --scenario-name val_q6_agent_motion_phase9_baseline \
  --baseline-mode cooperative \
  --workers 32 \
  --progress-every 250 \
  --v2vgot-root /workspace/repos/V2V-GoT
```

### Interpretation Rule For Official Summaries

- `object_motion_prediction` official summary includes two QA runs:
  - `qa_type_id=15` => Q5 metrics
  - `qa_type_id=17` => Q7 metrics
- `agent_motion_prediction` official summary run:
  - `qa_type_id=16` => Q6 metrics
- Promote checkpoints based on held-out validation first; train is for direction and regression checks.

## Logging Rule

Every Phase 9 result should record:

- task and QA type;
- whether it is smoke, train, or validation;
- exact command;
- exact artifact paths;
- metric values;
- comparison to V2V-GoT reference;
- whether the checkpoint is promoted or rejected;
- what failure mode or next action follows.
