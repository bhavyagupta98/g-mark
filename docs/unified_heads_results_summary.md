# Unified Heads Experiment (Stages 1-3)

## Scope

This experimental path keeps the existing promoted KG + QA pipeline untouched.

The first half remains the same:
- Load V2V-GoT QA rows.
- Convert rows to `BenchmarkSample`.
- Build cooperative provenance-aware KG using existing scene preparation stack.

The new path starts **after KG construction**.

## Design

One shared feature bank is built per sample, then exposed as three task-family views:

- `object_retrieval_view` for Q1/Q2/Q3/Q4 (`qa_type_id` 11/12/13/14)
- `motion_regression_view` for Q5/Q7/Q9 (`qa_type_id` 15/17/19)
- `scene_action_view` for Q6/Q8 (`qa_type_id` 16/18)

Stage 1 builds features, Stage 2 attaches labels, and Stage 3 trains unified family models.
Final QA inference/evaluation is still out of scope here (next stage).

## Files

- `src/gmark/features/unified_feature_bank.py`
- `src/gmark/features/task_family_views.py`
- `configs/unified_heads/default.yaml`
- `scripts/build_unified_heads_features.py`
- `src/gmark/features/leakage_checks.py`
- `src/gmark/training/unified_label_builders.py`
- `scripts/attach_unified_heads_labels.py`
- `src/gmark/models/unified_trainers.py`
- `scripts/train_unified_heads_models.py`

## Smoke-Test Run

```bash
python scripts/build_unified_heads_features.py \
  --config configs/unified_heads/default.yaml \
  --split train \
  --output-dir outputs/unified_heads/v1/features/train \
  --num-workers 8
```

Quick debug mode:

```bash
python scripts/build_unified_heads_features.py \
  --config configs/unified_heads/default.yaml \
  --split val \
  --output-dir outputs/unified_heads/v1/features/val_debug \
  --num-workers 1 \
  --max-samples 100
```

## Exact Commands Run (Stage 1 + Stage 2)

### Stage 1: Unified Feature Build

```bash
python3 scripts/build_unified_heads_features.py \
  --config configs/unified_heads/default.yaml \
  --split train \
  --output-dir outputs/unified_heads/v1/features/train \
  --num-workers 8
```

```bash
python3 scripts/build_unified_heads_features.py \
  --config configs/unified_heads/default.yaml \
  --split val \
  --output-dir outputs/unified_heads/v1/features/val \
  --num-workers 8
```

### Stage 2: Label Attachment

```bash
python3 scripts/attach_unified_heads_labels.py \
  --config configs/unified_heads/default.yaml \
  --split train \
  --feature-dir outputs/unified_heads/v1/features/train \
  --output-dir outputs/unified_heads/v1/labeled/train \
  --num-workers 8 \
  --log-every 100000
```

```bash
python3 scripts/attach_unified_heads_labels.py \
  --config configs/unified_heads/default.yaml \
  --split val \
  --feature-dir outputs/unified_heads/v1/features/val \
  --output-dir outputs/unified_heads/v1/labeled/val \
  --num-workers 8 \
  --log-every 100000
```

## Output

Under `outputs/unified_heads/<run_name>/features/<split>/` (or custom `--output-dir`):
- `object_retrieval_rows.jsonl`
- `motion_regression_rows.jsonl`
- `scene_action_rows.jsonl`
- `feature_build_summary.json`

Label outputs are written under `outputs/unified_heads/<run_name>/labeled/<split>/`:
- `object_retrieval_labeled.jsonl`
- `motion_regression_labeled.jsonl`
- `scene_action_labeled.jsonl`
- `label_summary.json`

## Label Building

- Q1/Q2/Q3/Q4 (`qa_type_id` 11/12/13/14): candidate-level binary labels using GT coordinate matching at `0.5m`.
- Q5/Q7/Q9 (`qa_type_id` 15/17/19): trajectory regression targets as flattened waypoints `[x1,y1,...,x6,y6]`.
- Q6 (`qa_type_id` 16): binary notable-agent-motion label.
- Q8 (`qa_type_id` 18): speed and steering class labels.

Train/val split separation is preserved by loading labels from the same split source records used for feature build.

## Leakage Checks

- Q9 rows are strictly checked so model input feature names cannot include:
  - `dist`
  - `angle`
  - `suggested_speed_idx`
  - `suggested_steering_idx`
  - `future_trajectory_str_in_ego`
  - `future_trajectory_str_in_self`
- Generic target/reference-like names are also blocked in `model_input` feature names.

## Label Attachment Commands

```bash
python scripts/attach_unified_heads_labels.py \
  --config configs/unified_heads/default.yaml \
  --split train \
  --feature-dir outputs/unified_heads/v1/features/train \
  --output-dir outputs/unified_heads/v1/labeled/train \
  --num-workers 8
```

```bash
python scripts/attach_unified_heads_labels.py \
  --config configs/unified_heads/default.yaml \
  --split val \
  --feature-dir outputs/unified_heads/v1/features/val \
  --output-dir outputs/unified_heads/v1/labeled/val \
  --num-workers 8
```

## Label Summary Note (Post-Fix)

- The label-attachment script was fixed to avoid double-counting per-`qa_type_id` buckets in `label_summary.json`.
- Any label-count totals produced before this fix should be treated as stale.
- Re-run label attachment for train/val and use the regenerated `label_summary.json` as the source of truth.

## Q8 Clean A/B (GBDT vs Linear)

For clean Q8 testing, keep the same data/artifacts path and train only `scene_action`.

- Default (current): `q8_head_model_type: gbdt`
- Promoted-like linear ablation: `q8_head_model_type: logreg`

Suggested config toggles under `models.scene_action`:
- `q8_head_model_type`: `gbdt` or `logreg`
- `q8_speed_class_weighting`: `none` (recommended for clean baseline)
- `q8_steering_class_weighting`: `none`

Train only scene_action:

```bash
python3 scripts/train_unified_heads_models.py \
  --config configs/unified_heads/default.yaml \
  --labeled-dir outputs/unified_heads/v1/labeled/train \
  --output-dir outputs/unified_heads/v1/artifacts \
  --family scene_action \
  --num-workers 4 \
  --log-every 10000 \
  --overwrite
```

Run val eval:

```bash
python3 scripts/run_unified_heads_qa_eval.py \
  --config configs/unified_heads/default.yaml \
  --split val \
  --artifact-dir outputs/unified_heads/v1/artifacts \
  --output-dir outputs/unified_heads/v1/val_eval \
  --feature-dir outputs/unified_heads/v1/features/val \
  --labeled-dir outputs/unified_heads/v1/labeled/val \
  --reuse-feature-rows \
  --num-workers 4 \
  --batch-size 50000 \
  --overwrite
```

## Completed Label Runs

### Train Split (`outputs/unified_heads/v1/labeled/train/label_summary.json`)

- `num_samples_loaded`: `110610`
- `num_workers`: `8`
- `rows_read_per_family`:
  - `object_retrieval`: `4104672`
  - `motion_regression`: `697626`
  - `scene_action`: `24580`
- `rows_written_per_family`:
  - `object_retrieval`: `4104672`
  - `motion_regression`: `697626`
  - `scene_action`: `24580`
- `failed_rows`: `0`
- `q9_leakage_checks_passed_rows`: `232542`

### Validation Split (`outputs/unified_heads/v1/labeled/val/label_summary.json`)

- `num_samples_loaded`: `31014`
- `num_workers`: `8`
- `rows_read_per_family`:
  - `object_retrieval`: `1415712`
  - `motion_regression`: `223962`
  - `scene_action`: `6892`
- `rows_written_per_family`:
  - `object_retrieval`: `1415712`
  - `motion_regression`: `223962`
  - `scene_action`: `6892`
- `failed_rows`: `0`
- `q9_leakage_checks_passed_rows`: `74654`

### Labeling Fix Follow-Up

- Stage 2 was updated to join by `(sample_id, qa_type_id)` and to set object-retrieval `has_label` from `label is not None` (so `label=0` is valid, not missing).
- Q1-Q4 positives/non-positives now populate correctly.
- Q6/Q8 parsing now emits valid labels instead of `label_type=unknown`.

## Safety / Sanity Checks

- Q9 model-input features exclude leakage-risk fields configured in `exclude_leakage_fields`.
- Every row separates `model_input` features from `metadata`.
- Every row includes `sample_id` and `qa_type_id`.
- Model-input features are numeric.
- Empty rows and invalid rows are counted and reported.

## Completed Run Results

### Train Split (`outputs/unified_heads/v1/features/train/feature_build_summary.json`)

- `run_name`: `unified_heads_v1`
- `split`: `train`
- `num_samples_loaded`: `110610`
- `num_workers`: `8`
- `rows_per_family`:
  - `object_retrieval`: `4104672`
  - `motion_regression`: `697626`
  - `scene_action`: `24580`
- `failed_samples`: `0`
- `invalid_rows`: `0`
- `empty_rows`: `0`

### Validation Split (`outputs/unified_heads/v1/features/val/feature_build_summary.json`)

- `run_name`: `unified_heads_v1`
- `split`: `val`
- `num_samples_loaded`: `31014`
- `num_workers`: `8`
- `rows_per_family`:
  - `object_retrieval`: `1415712`
  - `motion_regression`: `223962`
  - `scene_action`: `6892`
- `failed_samples`: `0`
- `invalid_rows`: `0`
- `empty_rows`: `0`

Status: feature-build implementation and smoke validation for this phase are complete.

## Stage 3: Unified Model Training

Stage 3 trains models only from labeled `train` rows and keeps the promoted pipeline untouched.

- Q1/Q2/Q3/Q4 use one shared logistic-regression retrieval model.
- The shared retrieval model adds task one-hot features:
  - `task_onehot::is_q1_notable`
  - `task_onehot::is_q2_occluding`
  - `task_onehot::is_q3_invisible`
  - `task_onehot::is_q4_planning`
- Q5/Q7 use a shared ElasticNet motion head when compatible.
- Q9 uses a separate clean ElasticNet head with strict leakage checks.
- Q6 uses a binary GBDT classifier.
- Q8 uses two GBDT classifier heads: speed and steering.

Val is not used for fitting or threshold tuning in this stage.

### Stage 3 Train Command

```bash
python scripts/train_unified_heads_models.py \
  --config configs/unified_heads/default.yaml \
  --labeled-dir outputs/unified_heads/v1/labeled/train \
  --output-dir outputs/unified_heads/v1/artifacts \
  --num-workers 4 \
  --log-every 10000 \
  --overwrite
```

### Family-Specific Stage 3 Commands

```bash
python scripts/train_unified_heads_models.py \
  --config configs/unified_heads/default.yaml \
  --labeled-dir outputs/unified_heads/v1/labeled/train \
  --output-dir outputs/unified_heads/v1/artifacts \
  --family object_retrieval \
  --num-workers 4 \
  --log-every 10000 \
  --overwrite
```

```bash
python scripts/train_unified_heads_models.py \
  --config configs/unified_heads/default.yaml \
  --labeled-dir outputs/unified_heads/v1/labeled/train \
  --output-dir outputs/unified_heads/v1/artifacts \
  --family motion_regression \
  --num-workers 4 \
  --log-every 10000 \
  --overwrite
```

```bash
python scripts/train_unified_heads_models.py \
  --config configs/unified_heads/default.yaml \
  --labeled-dir outputs/unified_heads/v1/labeled/train \
  --output-dir outputs/unified_heads/v1/artifacts \
  --family scene_action \
  --num-workers 4 \
  --log-every 10000 \
  --overwrite
```

### Stage 3 Reproducibility Notes

- Full-family Stage 3 execution is sequential (`object_retrieval` -> `motion_regression` -> `scene_action`) to reduce peak memory.
- Object-retrieval training uses two-pass streaming over JSONL to avoid loading all rows into memory at once.
- Feature matrices use `float32` to reduce memory footprint.

### Stage 3 Completed Run Snapshot

From `outputs/unified_heads/v1/artifacts/train_unified_heads_summary.json`:
- `families`: `["object_retrieval", "motion_regression", "scene_action"]`
- `execution_mode`: `sequential`
- `num_workers`: `4`
- `failures`: `[]`
- `runtime_sec`: `3314.061`
- Artifacts produced:
  - `object_retrieval/object_retrieval_logreg_shared_q1_q4.json`
  - `motion_regression/motion_elasticnet_q5_q7.json`
  - `motion_regression/motion_elasticnet_q9_clean.json`
  - `scene_action/scene_action_gbdt_q6.json`
  - `scene_action/scene_action_gbdt_q8_speed.json`
  - `scene_action/scene_action_gbdt_q8_steering.json`

Q9 safety confirmation (from `motion_elasticnet_q9_clean.json`):
- `q9_clean_only: true`
- `leakage_check_passed: true`
- leakage exclusions include:
  - `dist`, `angle`, `suggested_speed_idx`, `suggested_steering_idx`
  - `future_trajectory_str_in_ego`, `future_trajectory_str_in_self`

## Stage 4: Unified Inference and Evaluation

Stage 4 runs inference-only on saved Stage 3 artifacts. It does not retrain and does not tune thresholds on val.

### Files Added

- `scripts/run_unified_heads_qa_eval.py`
- `src/gmark/models/unified_artifacts.py`
- `src/gmark/output/unified_prediction_formatters.py`

### Stage 4 Full Command

```bash
python scripts/run_unified_heads_qa_eval.py \
  --config configs/unified_heads/default.yaml \
  --split val \
  --artifact-dir outputs/unified_heads/v1/artifacts \
  --output-dir outputs/unified_heads/v1/val_eval \
  --feature-dir outputs/unified_heads/v1/features/val \
  --labeled-dir outputs/unified_heads/v1/labeled/val \
  --reuse-feature-rows \
  --num-workers 8 \
  --batch-size 50000
```

### Stage 4 Smoke Command

```bash
python scripts/run_unified_heads_qa_eval.py \
  --config configs/unified_heads/default.yaml \
  --split val \
  --artifact-dir outputs/unified_heads/v1/artifacts \
  --output-dir outputs/unified_heads/v1/val_eval_smoke \
  --feature-dir outputs/unified_heads/v1/features/val \
  --labeled-dir outputs/unified_heads/v1/labeled/val \
  --reuse-feature-rows \
  --num-workers 4 \
  --batch-size 10000 \
  --max-rows 100000 \
  --skip-official-eval
```

### Stage 4 Behavior

- Loads all unified artifacts and validates schema.
- Reuses feature rows (`--reuse-feature-rows`) for memory-safe inference.
- Runs:
  - Q1-Q4 with shared logreg + saved per-task thresholds/max-results
  - Q5/Q7 with shared ElasticNet
  - Q9 with clean ElasticNet and leakage check revalidation
  - Q6 with GBDT binary head
  - Q8 with GBDT speed + steering heads
- Writes raw predictions first, then best-effort official export/eval.
- If export/eval fails, raw predictions and summaries are still saved.

### OOM Safety and Parallelism

- Execution mode is sequential for family inference to avoid peak memory spikes.
- Object retrieval is streamed/grouped by `(sample_id, qa_type_id)` and avoids loading the full val table at once.
- Motion/scene families aggregate compact per-sample structures.
- Batch size and progress logging are configurable (`--batch-size`, `--log-every`, `--memory-log-every`).
- Output summaries are written atomically.

### Stage 4 Outputs

Under `outputs/unified_heads/v1/val_eval/`:
- `raw_predictions.jsonl`
- `prediction_summary.json`
- `artifact_load_summary.json`
- `inference_metrics_diagnostic.json` (if `--labeled-dir` provided)
- `official_exports/*` and `official_eval/*` (if export/eval succeed)
