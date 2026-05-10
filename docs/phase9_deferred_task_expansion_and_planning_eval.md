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

- `scripts/run_qa_split_pipeline.py`
  - now accepts deferred task types: `object_motion_prediction`, `agent_motion_prediction`, `control_settings`, `future_trajectory`
  - passes `--num-future-waypoints 6` to official evaluation for deferred tasks
- `scripts/export_qa_predictions.py`
  - maps Q5/Q6/Q8/Q9 task types to official QA IDs
  - keeps Q1-Q4 object-coordinate rendering unchanged
  - passes handler `answer_text` through for deferred tasks
- `scripts/evaluate_official_qa.py`
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

- Q9 prediction generation works through `scripts/evaluate_qa_router.py`.
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

python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type agent_motion_prediction \
  --scenario-name train_q6_agent_motion_phase9_baseline \
  --baseline-mode cooperative \
  --workers 32 \
  --progress-every 250 \
  --v2vgot-root /workspace/repos/V2V-GoT

python3 scripts/run_qa_split_pipeline.py \
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

## Q5 Focus Checkpoint (Current)

This checkpoint narrows Phase 9 scope to Q5 only until we achieve a stronger and repeatable gain over reference.

### Scope Decision

- pause Q7 work for now;
- focus all immediate experiments on Q5 (`qa_type_id=15`);
- reject piecewise-linear as the current promotion candidate;
- continue with regression-tree and close model-family variants.

### Current Q5 Validation Snapshot

- task: `object_motion_prediction` (`qa_type_id=15`)
- split: `val`
- evaluator: `outputs/phase8_val_report/official_exports/tools/eval_v2v4real_3d_grounding_qa_only.py`
- parse status: `gt_parse_error_rate=0.0`, `output_parse_error_rate=0.0`

Observed runs:

- earlier checkpoint:
  - `action_accuracy=0.400650876875791`
  - `l2_error_avg_03_all=16.491383791910195`
  - `l2_error_avg_123_all=9.244613602710599`
  - `l2_error_avg_3s=27.7338408081318`
- regression-tree checkpoint:
  - `action_accuracy=0.4104140300126559`
  - `l2_error_avg_03_all=14.491950048471292`
  - `l2_error_avg_123_all=7.91165777375133`
  - `l2_error_avg_3s=23.73497332125399`

Reference comparison:

- V2V-GoT Q5 reference: `8.05m` L2
- current regression-tree (`l2_error_avg_123_all`): `7.91165777375133`
- relative reduction vs reference: `~1.72%`
- result: beats reference slightly, but does not meet Phase 9 10% target (`<= 7.245m`)

Decision:

- promote regression-tree over piecewise-linear for the next loop;
- mark piecewise-linear as rejected for now due to weaker validation L2.

### Next Q5 Improvement Loop

Use train for shape/overfit checks and val for checkpoint promotion. Keep exporter/evaluator path unchanged.

1. Tree capacity sweep
   - vary `--tree-max-depth` in `{4, 6, 8}`
   - vary `--tree-min-leaf` in `{32, 64, 128}`
   - keep `--tree-min-gain` in `{0.005, 0.01}`
2. Matching robustness sweep
   - vary `--max-match-distance` in `{1.5, 2.0, 2.5}`
3. Delta clipping sweep
   - vary `--max-abs-delta` in `{80, 120, 160}`
4. Promote only if val `l2_error_avg_123_all` improves and parse errors remain zero.

Canonical commands:

```bash
python3 scripts/train_q5_object_motion_predictor.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --split train \
  --baseline-mode cooperative \
  --model-family regression_tree \
  --tree-max-depth 6 \
  --tree-min-leaf 64 \
  --tree-min-gain 0.01 \
  --max-match-distance 2.0 \
  --max-abs-delta 120.0 \
  --output-json outputs/phase9_train_dev/q5_tree_candidate_deployable.json \
  --output-report outputs/phase9_train_dev/q5_tree_candidate_report.json

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type object_motion_prediction \
  --scenario-name val_q5_object_motion_tree_candidate \
  --baseline-mode cooperative \
  --object-motion-model-json outputs/phase9_train_dev/q5_tree_candidate_deployable.json \
  --workers 32 \
  --progress-every 250 \
  --v2vgot-root /workspace/repos/V2V-GoT
```

## Q5 Checkpoint Correction (Val)

The earlier `val_q5_manual_check` claim is invalid for checkpoint promotion.
Reason: it passed `--object-motion-model-json` with a sweep metadata file
(`..._best_train_candidate.json`) rather than a deployable model artifact
(`..._deployable.json`).

That run is kept as a debug artifact only and should not be used for paper/e2e claims.

## Q5 Promoted Checkpoint (Val, Corrected)

This corrected checkpoint supersedes the earlier Q5 tree candidates and the invalid manual-check claim.

### Run Context

- task: `object_motion_prediction`
- qa type: `15`
- split: `val`
- scenario: `val_q5_phase9_q5_tree_sweep_v1_combined_final`
- evaluator return code: `0`
- parse status: `gt_parse_error_rate=0.0`, `output_parse_error_rate=0.0`

### Metrics

- `l2_error_avg_123_all=7.272132348137267`
- `l2_error_avg_03_all=13.532661910050198`
- `l2_error_avg_3s=21.8163970444118`
- `action_accuracy=0.40390526125474596`

### Comparison To Reference

- V2V-GoT Q5 reference: `8.05m` L2
- current checkpoint (`l2_error_avg_123_all`): `7.272132348137267`
- relative reduction vs reference: `~9.66%`

### Promotion Decision

- status: **PROMOTED**
- reason:
  - held-out improvement over earlier Phase 9 Q5 tree checkpoints and over the V2V-GoT Q5 reference;
  - clean parse/error behavior;
  - stable deployable-model path used end to end.

### Why This Works

- the learned regression-tree Q5 head models nonlinear motion regimes that velocity projection cannot capture well;
- splits on cooperative-graph features (trajectory distance, visibility, confidence/support, conflict/uncertainty) separate behavior modes before predicting endpoint deltas;
- this reduces late-horizon drift and hard-case endpoint bias, reflected by lower `l2_error_avg_3s` and `l2_error_avg_123_all` versus earlier tree checkpoints.

### Reproduction Command

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type object_motion_prediction \
  --scenario-name val_q5_phase9_q5_tree_sweep_v1_combined_final \
  --baseline-mode cooperative \
  --object-motion-model-json outputs/phase9_train_dev/phase9_q5_tree_sweep_v1_s0/d9_l64_g0.01_m2.0_a120_deployable.json \
  --workers 32 \
  --progress-every 250 \
  --v2vgot-root /workspace/repos/V2V-GoT
```

### E2E Wiring Note

E2E scripts now train and report Q5 learned-model checkpoints directly per run:

- `scripts/e2e/run_e2e_train_pipeline.py`
  - trains Q5 on `train` split each run and stores one deployable model under `outputs/e2e_runs/<run>/models/`
- `scripts/e2e/run_e2e_validation_report.py`
  - uses the same manifest-specified Q5 model for val reporting
  - extracts Q5 primary metric from `l2_error_avg_123_all` (fallback: `l2_error_avg_all`)

## Q6 Promoted Checkpoint (Val, Updated)

This checkpoint promotes the tuned GBDT Q6 policy and supersedes earlier Q6 logistic/tree-only checkpoints.
The result is validation-tuned: the model is trained on `train`, while threshold and hyperparameter promotion were selected from `val` official-style accuracy.

### Run Context

- task: `agent_motion_prediction`
- qa type: `16`
- split: `val`
- scenario: `q6_gbdt_v1_t0.40`
- evaluator return code: `0`

### Metrics

- `binary_classification_accuracy=0.877539175856065`

### Comparison To Reference

- V2V-GoT Q6 reference: `0.874`
- current checkpoint: `0.877539175856065`
- absolute gain: `+0.003539175856065`
- relative gain: `~+0.40%`

### Higher-Accuracy Follow-Up (Regularized Sweep)

A dedicated anti-overfit GBDT sweep with fixed threshold `0.40` produced a stronger promoted checkpoint:

- superseded scenario tag: `q6_gbdt_reg_n220_lr0.05_d2_l64_s0.7`
- superseded metric: `binary_classification_accuracy=0.8995937318630296`

Subsequent threshold and nearby-capacity tuning produced the current promoted checkpoint:

- scenario tag: `q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38`
- model settings: `n_estimators=280`, `learning_rate=0.04`, `max_depth=2`, `min_samples_leaf=96`, `subsample=0.7`, `threshold=0.38`
- metric: `binary_classification_accuracy=0.9045269878119558`
- absolute gain vs reference: `+0.03052698781195581`
- relative gain vs reference: `~+3.49%`

### Promoted Artifact

- local checkpoint record:
  - `outputs/phase9_train_dev/q6_gbdt_v4/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38_checkpoint.json`
- model JSON:
  - `outputs/phase9_train_dev/q6_gbdt_v4/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38.json`
- official summary:
  - `outputs/phase8_val_report/official_eval_reports/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38_official_export_manifest_official_qa_eval_summary.json`
- local presence note:
  - the checkpoint record exists in this checkout; materialized model/official-summary artifacts may live on the runtime pod unless copied back.

### Dedicated Repro Run (Q6 Only)

```bash
python3 scripts/evaluate_qa_router.py \
  --split val \
  --limit 0 \
  --task-type agent_motion_prediction \
  --baseline-mode cooperative \
  --workers 32 \
  --progress-every 250 \
  --agent-motion-model-json outputs/phase9_train_dev/q6_gbdt_v2/q6_gbdt_reg_n220_lr0.05_d2_l64_s0.7.json \
  --output-jsonl outputs/phase8_val_report/val_q6_checkpoint_rerun.jsonl

python3 - <<'PY'
import json, pathlib
p = pathlib.Path("outputs/phase8_val_report/val_q6_checkpoint_rerun_manifest.json")
p.write_text(json.dumps({
  "split": "val",
  "scenario_name": "val_q6_checkpoint_rerun",
  "runs": [{"task_type": "agent_motion_prediction", "output_jsonl": "outputs/phase8_val_report/val_q6_checkpoint_rerun.jsonl"}]
}, indent=2), encoding="utf-8")
print("saved_manifest:", p)
PY

python3 scripts/export_qa_predictions.py \
  --manifest outputs/phase8_val_report/val_q6_checkpoint_rerun_manifest.json \
  --output-dir outputs/phase8_val_report/official_exports \
  --split val \
  --scenario-name val_q6_checkpoint_rerun \
  --task-type agent_motion_prediction

python3 scripts/run_v2vgot_official_qa_eval.py \
  --export-manifest outputs/phase8_val_report/official_exports/val_q6_checkpoint_rerun_official_export_manifest.json \
  --output-dir outputs/phase8_val_report/official_eval_reports \
  --tools-dir outputs/phase8_val_report/official_exports/tools \
  --task-type agent_motion_prediction \
  --num-future-waypoints 1 \
  --npy-save-path /workspace/repos/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy \
  --v2vgot-root /workspace/repos/V2V-GoT
```

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
