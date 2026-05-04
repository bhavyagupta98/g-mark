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

The main Week 6 goal is paper consolidation, not another broad metric sweep.

Priority:

1. turn the current Q1-Q4 checkpoint into clean paper tables;
2. run or document ego-only versus cooperative ablations with the current official-style evaluator;
3. extract graph-quality checks into an ablation/validation subsection;
4. prepare method figures for KG construction, query/retrieval, and Q3/Q4 policy flow;
5. only run new model experiments if they answer a paper-critical question.

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
