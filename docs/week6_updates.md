# Week 6 Updates

## Starting Point

Week 6 begins from the Week 5 full-split Q1-Q4 checkpoint.

Current paper-facing validation results:

| Task | Current policy | Val F1 @ 0.5m | Precision | Recall | V2V-GoT ref | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Q1 `notable_objects` | visible-object `heuristic` | `0.585836` | `0.674759` | `0.517621` | `0.525000` | clears reference and +10% target |
| Q2 `occluding_objects` | `risk_adaptive` with sparse-evidence fallback | `0.427921` | `0.452542` | `0.405840` | `0.301000` | clears reference and +10% target |
| Q3 `invisible_objects` | broad-pool `logreg_acceptor_t0p33` | `0.493934` | `0.488014` | `0.500000` | `0.440000` | clears reference and +10% target |
| Q4 `planning_awareness` | `relational_importance + trajectory_calibrated_acceptor` | `0.613774` | `0.576685` | `0.655962` | `0.608000` | exceeds reference |

Primary reference document:

- `docs/project_status_summary.md`

Phase 8 archival reference:

- `docs/phase8_scored_evaluation_and_baseline_archival.md`

## Week 6 Goal

The main Week 6 goal is paper consolidation plus one concrete expansion target beyond Q1-Q4.

Priority:

1. turn the current Q1-Q4 checkpoint into clean paper tables;
2. run or document ego-only versus cooperative ablations with the current official-style evaluator;
3. extract graph-quality checks into an ablation/validation subsection;
4. prepare method figures for KG construction, query/retrieval, and Q3/Q4 policy flow;
5. bring one deferred V2V-GoT task family into the same reproducible train/validation/export/eval workflow;
6. only run new model experiments if they answer a paper-critical question.

## V2V-GoT Task Inventory Beyond Q1-Q4

The local V2V-GoT-QA inventory confirms that the benchmark has eight task families in the current `val` split:

| Task | QA type | Task name | Val samples | Paper metric/reference | Current repo status |
| --- | ---: | --- | ---: | --- | --- |
| Q1 | `11` | `notable_objects` | `3446` | F1, V2V-GoT ref `52.5` | full-split official-style checkpoint promoted |
| Q2 | `12` | `occluding_objects` | `3446` | F1, V2V-GoT ref `30.1` | full-split official-style checkpoint promoted |
| Q3 | `13` | `invisible_objects` | `3446` | F1, V2V-GoT ref `44.0` | full-split official-style checkpoint promoted |
| Q4 | `14` | `planning_awareness` | `3446` | F1, V2V-GoT ref `60.8` | full-split official-style checkpoint promoted |
| Q5 | `15` | `object_motion_prediction` | part of `6892` | L2, V2V-GoT ref `8.05m` | router support exists, needs official-style scoring path |
| Q6 | `16` | `agent_motion_prediction` | `3446` | accuracy, V2V-GoT ref `87.4%` | router support exists, needs official-style scoring path |
| Q7 | `17` | `object_motion_prediction` | part of `6892` | L2, V2V-GoT ref `7.61m` | router support exists, needs official-style scoring path |
| Q8 | `18` | `control_settings` | `3446` | L1/action error, V2V-GoT ref `0.0876` | promoted full-split official-style checkpoint (`v7 extended ordinal risk3`) |
| Q9 | `19` | `future_trajectory` | `3446` | L2, V2V-GoT ref `2.62m` | router support exists, best first Week 6 target |

Notes from the scan:

- `object_motion_prediction` has `6892` validation rows because it covers two QA types: Q5 and Q7.
- Phase 7 already added first-pass deterministic handlers for Q5-Q9:
  - Q5/Q7 object motion from object-track velocity when available, with stationary fallback;
  - Q6 agent motion from CAV velocity when available, then planned-trajectory fallback;
  - Q8 control settings from graph risk-ranked objects;
  - Q9 future trajectory from parsed scene future trajectory.
- Week 6 extended the reproducible split protocol/export/eval path to deferred tasks Q5-Q9, including Q9 official-style scoring.

## Phase 9 Start: Deferred Planning/Prediction Tasks

Phase 9 begins inside Week 6.

Primary Phase 9 reference:

- `docs/phase9_deferred_task_expansion_and_planning_eval.md`

Phase 9 boundary:

- Phase 8 remains frozen for Q1-Q4 unless a paper-blocking issue appears.
- Phase 9 owns Q5-Q9 expansion and starts with Q9.

## Week 6 Expansion Target

Primary target: Q9 `future_trajectory`.

Why Q9 first:

- It has the cleanest official metric path in the V2V-GoT evaluator: future-waypoint L2.
- It is most aligned with our existing KG/planning story.
- The handler already has deterministic support through `FutureTrajectoryHandler`.
- It gives a tangible Week 6 deliverable beyond Q1-Q4 without immediately taking on the harder object-motion identity/matching problem.

Week 6 Q9 success criteria:

1. extend the Phase 8 split protocol so `--task-type future_trajectory` can run with the same train/val output convention;
2. extend official-style export so Q9 outputs preserve `qa_type_id=19` and the V2V-GoT record fields required by the upstream evaluator;
3. run full validation once with `--workers 32`;
4. score Q9 with the upstream-compatible evaluator and log:
   - `l2_error_avg_1s`
   - `l2_error_avg_2s`
   - `l2_error_avg_3s`
   - `l2_error_avg_all`
   - parse-error/failure count if available;
5. compare against the V2V-GoT Q9 reference `2.62m`;
6. log exact artifacts and interpretation in `docs/project_status_summary.md` and this Week 6 file.

Initial Q9 generation command once the protocol accepts deferred tasks:

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

Current lower-level fallback command before protocol extension:

```bash
python3 scripts/evaluate_v2vgotqa_phase5a.py \
  --split val \
  --limit 0 \
  --task-type future_trajectory \
  --baseline-mode cooperative \
  --output-jsonl outputs/week6_q9/val_future_trajectory_baseline.jsonl \
  --progress-every 250 \
  --workers 32
```

Secondary target after Q9: Q8 `control_settings`.

Why Q8 second:

- It is also planning-facing and uses graph risk evidence.
- The upstream evaluator reports speed, steering, and combined action accuracy/error.
- It can become a natural bridge from Q4 planning-awareness to control recommendation.

Q8 should wait until Q9 proves the deferred-task export/eval path is working.

## Q9 Method And Outcome (Week 6)

What the Q9 model does:

- Q9 uses `ControlConditionedFutureTrajectoryPlanner` as the trajectory prediction head.
- It predicts 6 future waypoints directly as coordinates:
  - `[(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]`.
- Base predictor is multivariate linear regression (not logistic regression):
  - feature vector `f` has 19 dims
  - output `o` has 12 dims
  - `o = Wf` with frozen train-learned coefficients.
- Tail-residual variant keeps base prediction and applies a second linear residual head only on `(x5,y5,x6,y6)` to reduce long-horizon drift.

How feature vectors are constructed:

- `f = [1, current_x, current_y, asker_is_cav1, speed_onehot(5), steering_onehot(5), dist, sin(angle), cos(angle), dist*sin(angle), dist*cos(angle)]`.
- `current_x,current_y` come from question context (`I am CAV_X at (x,y)`).
- `asker_is_cav1`, `speed_idx`, `steering_idx`, `dist`, `angle` come from Q9 metadata.
- Tail residual appends nonlinear terms (`dist^2`, `sin(2*angle)`, `cos(2*angle)`, and cross terms).

How KG helps Q9:

- KG scene preparation provides consistent coordinate context and asker identity across tasks.
- Deterministic `prepare -> route -> export -> official eval` keeps Q9 reproducible and comparable with Q1-Q4.
- Q9 uses the same benchmark-compliant output and evaluator path as the rest of the system.

Official-style Q9 results vs V2V-GoT reference:

| Checkpoint | Train L2 all | Val L2 all | Val L2 @1s | Val L2 @2s | Val L2 @3s | Ref L2 all | Relative reduction vs ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_metadata_linear_v1` | `0.894774` | `1.320964` | `0.950570` | `1.354670` | `1.657653` | `2.620000` | `49.58%` |
| `control_metadata_linear_tail_residual_v1` | pending full official train rerun | `1.211582` | `0.950570` | `1.354670` | `1.329508` | `2.620000` | `53.76%` |

Interpretation:

- Q9 now shows a large baseline-relative gain in held-out validation.
- The tail-residual head improves long-horizon localization most strongly (`3s` term), while preserving short-horizon behavior.

Key Q9 artifacts:

- `outputs/phase9_train_dev/q9_future_trajectory_control_metadata_linear_v1_deployable.json`
- `outputs/phase9_train_dev/q9_future_trajectory_control_metadata_linear_tail_residual_v1_deployable.json`
- `outputs/phase8_val_report/official_eval_reports/val_q9_future_trajectory_control_linear_tail_residual_v1_official_export_manifest_official_qa_eval_summary.json`

## Q8 Promoted Checkpoint (Week 6)

Final selected Q8 approach:

- policy: `linear_classifier`
- speed head: ordinal thresholds
- threshold policy: risk-conditional (`risk3`)
- feature set: `extended_v1` (trajectory geometry + heading alignment)
- model artifact:
  - `outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json`

Promoted val metrics:

- `speed_accuracy`: `0.684272`
- `steering_accuracy`: `0.890308`
- `action_accuracy`: `0.620720`
- `speed_edit_dist`: `0.476204`
- `steering_edit_dist`: `0.132908`
- `action_edit_dist`: `0.609112`
- normalized action error proxy: `0.609112 / 8 = 0.076139`

Baseline comparison:

- V2V-GoT Q8 reference: `0.087600`
- current normalized error: `0.076139`
- relative reduction: `13.08%` (lower is better)

Why it worked:

- Q8 mismatch analysis showed a speed-dominated residual error, with steering already strong.
- Flat/global threshold tuning plateaued.
- Ordinal speed decoding reduced large class-index errors.
- Risk-conditional thresholds corrected context-dependent speed bias.
- Extended trajectory-aware features improved path-critical vs non-critical separation.

Train/freeze + official evaluation commands (promoted):

```bash
python3 scripts/train_phase9_q8_control_classifier.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --split train \
  --baseline-mode cooperative \
  --feature-set extended_v1 \
  --speed-head-type ordinal \
  --speed-class-weighting sqrt_inverse_freq \
  --steering-class-weighting none \
  --l2-regularization 1e-4 \
  --speed-ordinal-threshold-policy risk3 \
  --speed-risk-split-low 0.2 \
  --speed-risk-split-high 0.5 \
  --output-json outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json \
  --output-report outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_report.json

python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type control_settings \
  --scenario-name train_q8_control_linear_classifier_v7_extended_ordinal_risk3 \
  --baseline-mode cooperative \
  --control-selection-policy linear_classifier \
  --control-model-json outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json \
  --workers 32 \
  --v2vgot-root /workspace/repos/V2V-GoT

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type control_settings \
  --scenario-name val_q8_control_linear_classifier_v7_extended_ordinal_risk3 \
  --baseline-mode cooperative \
  --control-selection-policy linear_classifier \
  --control-model-json outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json \
  --workers 32 \
  --v2vgot-root /workspace/repos/V2V-GoT
```

Later targets:

- Q5/Q7 object motion prediction:
  - harder because scoring combines object localization, action parsing, and future trajectory L2;
  - likely needs better object velocity from temporal tracks.
- Q6 agent motion prediction:
  - useful but may be limited by sparse/static CAV pose deltas;
  - planned-trajectory fallback is already implemented, but needs official evaluation before further tuning.

## Phase 9 Q5-Q7 Plan (Train First)

This is the next concrete Phase 9 block after promoted Q8/Q9.

Baseline references and target band:

| Task | V2V-GoT reference | 10% target | 20% target | Direction |
| --- | ---: | ---: | ---: | --- |
| Q5 `object_motion_prediction` (`qa_type_id=15`) | `8.05m` L2 | `<= 7.245m` | `<= 6.440m` | lower is better |
| Q7 `object_motion_prediction` (`qa_type_id=17`) | `7.61m` L2 | `<= 6.849m` | `<= 6.088m` | lower is better |
| Q6 `agent_motion_prediction` (`qa_type_id=16`) | `87.4%` accuracy | `>= 88.66%` | `>= 89.92%` | higher is better |

Why Q6 targets are written this way:

- for accuracy, a direct `+10%` on `87.4` is not a stable goal definition;
- we use relative error reduction:
  - baseline error = `1 - 0.874 = 0.126`
  - 10% error reduction => `0.1134` error => `88.66%` accuracy
  - 20% error reduction => `0.1008` error => `89.92%` accuracy

Train-first commands (workers fixed at 32):

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

python3 scripts/run_phase8_qa_split_protocol.py \
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

How to read the official summaries:

- `object_motion_prediction` summaries contain two runs in one JSON:
  - `qa_type_id=15` => Q5
  - `qa_type_id=17` => Q7
- `agent_motion_prediction` summary has:
  - `qa_type_id=16` => Q6
- promote only checkpoints that improve on validation, not train only.

## Current Method Story

The paper-facing approach is:

- build an explicit cooperative KG from V2V-GoT/V2V4Real scene assets;
- preserve object tracks, observation evidence, visibility facts, relations, provenance, conflict, uncertainty, and cooperative support;
- query and rank graph facts deterministically;
- use train-frozen learned acceptors only where hand-written rules were insufficient;
- export answers back to V2V-GoT-compatible coordinate text and score with the official-style evaluator.

Current task-specific story:

- Q1: deterministic visible-object grounding.
- Q2: risk-adaptive visible blocker selection with sparse-evidence backfill.
- Q3: broad hidden-object retrieval, then train-frozen logistic acceptance.
- Q4: relational planning-importance scoring, train-frozen logistic acceptance, near-duplicate calibration, and trajectory-based false-positive suppression.

## Week 6 Ablation Plan

### A. Ego-Only Versus Cooperative

Motivation:

- Show whether the cooperative KG changes performance compared with the same pipeline restricted to the asking vehicle.
- Keep the downstream policy fixed and change only `--baseline-mode`.

Current official-style Q3 cooperative command:

```bash
python3 scripts/run_phase8_qa_split_protocol.py \
  --purpose val_report \
  --split val \
  --task-type invisible_objects \
  --scenario-name ablation_val_q3_cooperative_broadpool_logreg \
  --limit 0 \
  --baseline-mode cooperative \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --progress-every 250 \
  --workers 32
```

Current official-style Q3 ego-only command:

```bash
python3 scripts/run_phase8_qa_split_protocol.py \
  --purpose val_report \
  --split val \
  --task-type invisible_objects \
  --scenario-name ablation_val_q3_ego_only_broadpool_logreg \
  --limit 0 \
  --baseline-mode ego_only \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --progress-every 250 \
  --workers 32
```

Current official-style Q4 cooperative command:

```bash
python3 scripts/run_phase8_qa_split_protocol.py \
  --purpose val_report \
  --split val \
  --task-type planning_awareness \
  --scenario-name ablation_val_q4_cooperative_trajcal_v1 \
  --limit 0 \
  --baseline-mode cooperative \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-selection-source orchestrator \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32
```

Current official-style Q4 ego-only command:

```bash
python3 scripts/run_phase8_qa_split_protocol.py \
  --purpose val_report \
  --split val \
  --task-type planning_awareness \
  --scenario-name ablation_val_q4_ego_only_trajcal_v1 \
  --limit 0 \
  --baseline-mode ego_only \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-selection-source orchestrator \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32
```

Logging rule:

- record train/validation metrics, exact artifact paths, and interpretation in `docs/project_status_summary.md`;
- keep ego-only/cooperative ablations separate from the main headline table;
- do not tune on ego-only validation results.

### B. Graph-Quality Checks

Already logged in `docs/project_status_summary.md`:

- Phase 3 local graph sanity checks;
- Phase 4 deterministic query-engine checks;
- Phase 5/Week 4 structural ego-only/cooperative smoke comparisons.

Week 6 task:

- turn these into a concise paper ablation/validation table:
  - local graph construction sanity;
  - query engine safe-failure sanity;
  - ego-only versus cooperative mode switch;
  - current official-style Q3/Q4 ego/cooperative results once run.

### C. Model/Policy Ablations

Promoted or useful:

- Q2 `risk_adaptive` with sparse-evidence backfill.
- Q3 broad-pool logistic acceptor.
- Q4 `relational_importance + trajectory_calibrated_acceptor`.

Rejected or not promoted:

- Q3 narrow trajectory-only legacy policies.
- Q3 MLP acceptor, because it did not beat logistic on the broad-pool feature table.
- Q4 L1/L2/elastic-net variants, because validation did not improve.
- Q4 MLP acceptor, because train-side selection F1 was far below the promoted logistic checkpoint.
- Q4 hard and soft count gates, because precision improved but recall/F1 dropped.

Week 6 task:

- convert these into a compact ablation table with one-line reasons.

## Paper Tables To Prepare

Recommended tables:

1. Main Q1-Q4 validation results versus V2V-GoT references.
2. Q3 improvement path:
   - old precision-oriented `logreg_acceptor_t0p25`;
   - broad-pool `logreg_acceptor_t0p33`;
   - MLP comparison as rejected.
3. Q4 improvement path:
   - relational default/count-adaptive;
   - logreg acceptor;
   - near-duplicate `1.0m`;
   - trajectory-calibrated final.
4. Ego-only versus cooperative ablation.
5. Graph-quality/query-engine sanity checks.

## Paper Figures To Prepare

Recommended figures:

1. End-to-end pipeline:
   - V2V-GoT/V2V4Real assets -> local KGs -> cooperative KG -> deterministic retrieval/ranking -> official export/eval.
2. KG schema:
   - agents, object tracks, observations, visibility facts, relation facts, provenance.
3. Q3 policy:
   - broad retrieval -> train-frozen logistic acceptor -> coordinate answer rendering.
4. Q4 policy:
   - relational candidate scoring -> logistic acceptance -> duplicate/trajectory calibration -> final answer.

## Week 6 Working Rule

Every new result should be logged with:

- what changed;
- why it was tried;
- whether it is train-only, validation, local proxy, or official-style;
- exact command;
- exact artifact path;
- interpretation;
- whether it is promoted or rejected.

This keeps the paper trail clean and avoids mixing exploratory runs with final claims.
