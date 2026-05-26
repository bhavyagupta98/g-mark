# Option 2 Plan: LightGBM Learned Baselines and KG Re-Ranking

## Motivation
Our current G-MARK results are strong, but reviewers may still ask whether performance comes mainly from handcrafted rules and benchmark-specific engineering. We need clear evidence that the cooperative KG is useful as a learning-ready representation, not only as a deterministic pipeline.

## Why This Matters
Adding learned comparisons strengthens the CoRL story in three ways:

1. It adds a clear robot-learning component around G-MARK.
2. It tests whether KG-aware features outperform flat object-only features.
3. It improves claim credibility even if deterministic G-MARK remains the primary method.

## Aim
Build a strong and fair learned comparison suite on top of the existing pipeline, without replacing the core system.

Primary aim:
- show whether learned models over KG-derived features beat learned models over flat features.

Secondary aim:
- show which KG components (for example provenance and visibility) are responsible for gains.

## What This Adds To The Current Pipeline
Current pipeline (today):
- cooperative scene construction,
- KG enrichment (relations, provenance, uncertainty, visibility),
- task-specific decision heads,
- benchmark evaluation.

Extended pipeline (Option 2):
1. Keep the same candidate generation and evaluation protocol.
2. Export candidate-level training features.
3. Train flat learned baselines.
4. Train KG-feature learned re-rankers.
5. Add feature-drop ablations (no provenance, no visibility).
6. Report same benchmark metrics and runtime footprint.

This is an extension layer, not a rewrite.

## Is This Ablation Or Result Strengthening?
It is both.

- Result strengthening:
  - stronger learned baselines,
  - potentially better headline numbers,
  - stronger CoRL positioning.
- Ablation evidence:
  - no-provenance and no-visibility variants isolate what KG information matters.

## Scope And Dataset Strategy
Primary dataset (required):
- V2V-GoT-QA (built on V2V4Real), using the same train/val/test protocol as current results.

Optional extension (after primary):
- OPV2V transfer/generalization check using compatible processed detections.

Recommended order:
1. complete all Option 2 runs on V2V-GoT/V2V4Real first,
2. then port the same feature/export/train/eval flow to OPV2V.

## LightGBM Common-Model Strategy

We use one common training framework with LightGBM and task-specific heads.

This is not one single objective across all tasks. It is one shared pipeline plus three model types:
- `LGBMRanker` for ranking tasks,
- `LGBMClassifier` for classification tasks,
- `LGBMRegressor` for regression tasks.

This keeps training consistent while respecting different target types.

## Task Family Mapping (Current Working Plan)

`Q5` and `Q7` are treated as one family for reporting consistency.

- Q1 notable_objects: rank/classify candidate object relevance.
- Q2 occluding_objects: rank/classify candidate object relevance.
- Q3 invisible_objects: rank/classify candidate object relevance.
- Q4 planning_awareness: rank/classify candidate object relevance.
- Q5/Q7 object_motion_prediction: regression target with optional candidate-ranking view.
- Q6 agent_motion_prediction: regression target with optional candidate-ranking view.
- Q8 control_settings: classification.
- Q9 future_trajectory: regression (non-ranking if no candidate list).

## Feature Sets

### A. Flat Features (Non-KG Baseline)
- object/candidate geometry,
- class/type indicators,
- detection confidence,
- ego-relative distance/angle/lane-side cues,
- temporal deltas from local history where available.

### B. Flat + KG Features (KG Learned Variant)
All flat features plus:
- support-agent/provenance strength,
- visibility and occlusion state features,
- relation-graph counts/statistics,
- uncertainty/disagreement indicators,
- planning relevance or conflict cues.

### C. KG Feature-Drop Ablations
- remove provenance block,
- remove visibility block.

Purpose:
- isolate which KG signals drive gains.

## Training Protocol

1. Use the exact same split discipline as current benchmark runs.
2. Keep identical candidate generation and preprocessing for all model variants.
3. Train paired models per task family:
- flat-only model,
- flat+KG model,
- feature-drop ablations for KG model.
4. Hyperparameter tuning on validation only.
5. Single final test evaluation per selected checkpoint/config.

## Evaluation Protocol

### Ranking tasks
- `NDCG@k`,
- `MRR`,
- `Recall@1/@3/@5`.

### Classification tasks
- `Accuracy`,
- `Precision/Recall/F1`,
- `AUROC` where meaningful.

### Regression tasks
- `MAE`,
- `RMSE`,
- optional `R2`.

### Efficiency
- sample average latency,
- p50/p90 latency,
- throughput for learned module.

## Comparison Rules (Fairness)

1. Same train/val/test split for all variants.
2. Same model family and objective per task type.
3. Same post-processing and threshold policy.
4. Only changed factor is feature set (`flat` vs `flat+KG` or dropped KG blocks).

## Deliverables

1. Per-task results table (Q1-Q9, with Q5/Q7 merged row in summary view).
2. Overall macro summary table across ranking/classification/regression groups.
3. Latency comparison table for flat vs KG learned variants.
4. Short ablation summary on provenance and visibility contributions.

## Initial Results Log (Implemented)

Scope:
- module: `option2_object_motion`
- task scope: `Q5/Q7` merged (`qa_type_id=15,17`)
- train split: `train`
- validation split: `val`
- backend: `sklearn_gbdt` (`n_estimators=280`, `learning_rate=0.04`, `max_depth=2`, `min_samples_leaf=96`, `subsample=0.7`)

### Flat vs Flat+KG (Validation)

| Metric | Flat Only | Flat + KG | Absolute Delta | Relative Delta |
| --- | ---: | ---: | ---: | ---: |
| MAE | 9.5488 | 9.1231 | -0.4257 | -4.46% |
| RMSE | 16.6275 | 15.3594 | -1.2681 | -7.63% |
| endpoint_l2_avg | 18.2476 | 17.3774 | -0.8702 | -4.77% |
| endpoint_l2_p90 | 36.2213 | 34.3768 | -1.8445 | -5.09% |

### Data Summary

- train total samples scanned: `24580` (Q5+Q7 combined)
- train matched examples: `45112`
- val total samples scanned: `6892`
- val matched examples: `11128`
- note: matched examples can exceed sample count because one QA sample can produce multiple GT object rows.

### Guardrail Status

- feature source: `scene_only_no_result_metadata`
- sample overlap guard: logged only (`train_val_sample_overlap_count=1876`), not failing by default.
- optional strict mode remains available: `--fail-on-sample-overlap`

### Current Interpretation

1. KG-derived features improve Q5/Q7 prediction quality over flat-only features across all key validation metrics.
2. Tail error improves (`endpoint_l2_p90`), indicating gains are not limited to easy samples.
3. This is positive evidence for Option 2 claim on temporal object-motion tasks; next step is extending the same protocol to additional task families.

## Expected Outcomes
Best case:
- KG-feature model beats flat model on visibility-sensitive tasks (especially Q2/Q3/Q4).

Moderate case:
- gains on Q1-Q4 but not Q8.

Weak case:
- flat model matches KG model; still useful as negative evidence and for refining KG feature design.

## Minimal Implementation Sequence
1. finalize task-to-head mapping and label schema per question family,
2. export candidate-level features and labels for all Q families on V2V4Real split,
3. train LightGBM flat baselines (rank/classify/regress heads),
4. train LightGBM flat+KG variants,
5. run provenance-drop and visibility-drop ablations,
6. generate task-wise and grouped summary tables,
7. run learned-module latency profiling and merge into docs,
8. optionally extend same protocol to OPV2V.

## Bottom Line
Option 2 is the highest-value low-risk extension because it directly addresses a likely reviewer concern while preserving the current G-MARK system and evaluation flow.
