# Q9 Clean Sweep Progress Checkpoint

Date: 2026-05-14

This file captures the current state before session restart so we do not lose context.

## Frozen Table-I Result (Best ElasticNet-26)

Date captured: 2026-05-14

Final row to carry forward:

| Method | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | L2 Avg ↓ | CR 1s ↓ | CR 2s ↓ | CR 3s ↓ | CR Avg ↓ | Comm MB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `G-MARK (ElasticNet-26)` | 1.65 | 2.71 | 3.77 | 2.71 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0159 |

Reference comparator:

- `V2V-GoT`: L2 Avg `2.62`, Comm MB `0.4068`

Gap summary:

- L2 Avg gap to V2V-GoT: `+0.09` (about `+3.4%`)
- Best parity: 1s horizon (`1.65` vs `1.65`)
- Remaining error is mostly long-horizon (3s)
- Communication is much lower (`0.0159` vs `0.4068`, about `25.6x` lower)

Primary artifact paths:

- official summary JSON: `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/official_eval_reports/gmark_q9_table1_elasticnet26_best_v1_official_export_manifest_official_qa_eval_summary.json`
- table row JSON/MD from runner:
  - `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/gmark_q9_table1_elasticnet26_best_v1_table1_row.json`
  - `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/gmark_q9_table1_elasticnet26_best_v1_table1_row.md`

## Approach We Took (Condensed)

1. Moved from leakage-risk legacy Q9 metadata to clean split-safe inputs.
2. Established context-only Q8->Q9 feature path from Q9 prompt context (26 features total).
3. Benchmarked model families on identical feature protocol.
4. Used official export + official evaluator as source of truth.
5. Added communication accounting via serialized local graph size (Table-I comm metric style).

## Meaningful Findings From Runs

- Clean no-Q8 ridge baseline was very weak (`~11.85` official L2 avg).
- Adding Q8 context features was the key jump to near-SOTA range.
- Best stable family in this setup: ElasticNet (better than RF/HGB/XGB here).
- Q8 float/probability-heavy variants did not help in current form:
  - float-jsonl variant degraded to around `~4.08`.
  - probs14 variant degraded strongly (`~7.27`).
- Feature-importance runs indicated:
  - core geometry + Q8 speed signal contributed most;
  - several trajectory-shape fields had near-zero incremental utility under this model.

## Current Freeze Decision

- Freeze this row for reporting:
  - `G-MARK (ElasticNet-26)` with L2 Avg `2.71`, Comm MB `0.0159`.
- Use it as the baseline for next iteration targeting only the 3s horizon gap.

## Reproduce Exact Frozen Row

1) Create a minimal manifest that points to the frozen Q9 model JSON:

```bash
python3 - <<'PY'
import json, pathlib
manifest = {
  "v2vgot_root": "/workspace/repos/V2V-GoT",
  "model_paths": {
    "q9_model_json": "/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_sweep/gmark_q9_v2_context_only_elasticnet/gmark_q9_v2_context_only_elasticnet_elasticnet_model.json"
  }
}
out = pathlib.Path("/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/gmark_q9_elasticnet26_manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print(out)
PY
```

2) Run official Table-I style Q9 evaluation + Comm(MB):

```bash
python3 -u scripts/run_gmark_table1_q9_eval.py \
  --run-name gmark_q9_table1_elasticnet26_best_v1 \
  --manifest-json /workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/gmark_q9_elasticnet26_manifest.json \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_table1 \
  --method-name "G-MARK (ElasticNet-26)" \
  --progress-every 250
```

Expected key outputs:

- official summary JSON:
  - `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/official_eval_reports/gmark_q9_table1_elasticnet26_best_v1_official_export_manifest_official_qa_eval_summary.json`
- table row report:
  - `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/gmark_q9_table1_elasticnet26_best_v1_table1_row.md`
  - `/workspace/repos/kg_coop_drive/outputs/v2vgot_table1_reproduction/gmark_q9_table1/gmark_q9_table1_elasticnet26_best_v1_table1_row.json`

## Why We Paused

We identified that earlier strong Q9 results were likely inflated by target-adjacent metadata usage in the legacy Q9 path (`suggested_*`, `dist`, `angle`).

The goal now is a defensible, clean Q9 evaluation path and a model sweep to recover performance without leakage.

## Leakage Diagnosis (Current Understanding)

- Not intentional cheating.
- Methodology issue: earlier Q9 model family used benchmark-side control metadata that can act as oracle-like hints.
- Clean policy now treats these fields as excluded from direct model inputs:
  - `dist`
  - `angle`
  - `suggested_speed_idx`
  - `suggested_steering_idx`
  - `future_trajectory_str_in_ego`
  - `future_trajectory_str_in_self`

## New Work Added (No Regression to Old Pipeline)

We added a **new standalone script** and did not replace legacy scripts:

- `scripts/run_gmark_q9_model_sweep.py`
- `scripts/run_gmark_q9_sweep_wrapper.py`

This script:

1. Builds enriched but clean Q9 features.
2. Sweeps multiple model families (`ridge`, `elasticnet`, `rf`).
3. Writes per-model artifacts:
   - model JSON
   - prediction JSONL
   - prediction manifest JSON
4. Writes one consolidated manifest with all model results.
5. Optionally runs official export and official eval per model.

Wrapper behavior (`run_gmark_q9_sweep_wrapper.py`):

1. Runs a no-Q8-feature Q9 sweep branch.
2. Runs a Q9 sweep branch that includes Q8-predicted-control features.
3. Default Q8 feature source is `--q8-feature-source question_context`, which parses the Q8 speed/steering context already present in the Q9 prompt.
4. Optional diagnostic source `--q8-feature-source q8_model` still exists, but it rebuilds KG-prepared scenes and can block in native I/O on some rows.
5. Writes a wrapper manifest linking both branches and the Q8 feature source.

## Important Constraint: Old Code Intact

- Existing/legacy scripts remain intact.
- This is additive only; old workflows are still available.

## Q8-to-Q9 Feature Plan

Accepted approach:

- Use **Q8 staged context/predictions** as optional Q9 features.
- Do **not** use raw dataset `suggested_speed_idx` / `suggested_steering_idx` directly.

This preserves clean causal direction:

Q8 control answer/context -> Q9 trajectory model.

Implementation details:

- Q8-derived features are included only when `--include-q8-pred-features` is enabled in `run_gmark_q9_model_sweep.py`.
- Preferred source: `--q8-feature-source question_context`.
- With `question_context`, Q8 features are parsed from the existing V2V-GoT Q9 prompt text:
  - `The suggested speed setting is: ...`
  - `The suggested steering setting is: ...`
  - generated Q8 feature JSONLs are written under `<run_root>/q8_question_context_features/`.
- This avoids rerunning KG/Q8 inference and matches the staged V2V-GoT NQ8 -> NQ9 prompt construction.
- Diagnostic source: `--q8-feature-source q8_model --q8-model-json <path>`.
  - This rebuilds KG-prepared scenes and reruns Q8 inference per Q9 row.
  - It can get stuck in uninterruptible I/O (`D` process state), where Python timeouts do not fire.
- Feature vocabulary now matches V2V-GoT Q8:
  - speed: `fast`, `moderate`, `slow`, `very slow`, `stop`
  - steering: `left`, `slightly left`, `straight`, `slightly right`, `right`
- Q8 features now include both:
  - one-hot predicted speed/steering labels,
  - numeric control scalars:
    - `q8_pred_speed_control_value`: `fast=1.0`, `moderate=0.65`, `slow=0.35`, `very slow=0.15`, `stop=0.0`
    - `q8_pred_steering_control_value`: `left=-1.0`, `slightly left=-0.5`, `straight=0.0`, `slightly right=0.5`, `right=1.0`
- Legacy source still exists: `--q8-feature-source legacy_jsonl --q8-predictions-jsonl <path>`, but it is now a compatibility path only.

Q8 model selection in wrapper:

- Preferred wrapper source: `--q8-feature-source question_context` with no Q8 model path.
- If using diagnostic `--q8-feature-source q8_model`, choose the model with `--q8-model-json` or `--e2e-manifest-json` -> `model_paths.q8_model_json`.

## Commands to Resume

### A) Clean Q9 sweep (no Q8-pred features)

```bash
python3 scripts/run_gmark_q9_model_sweep.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --run-name gmark_q9_sweep_v1 \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_sweep \
  --models ridge elasticnet rf \
  --progress-every 250
```

### B) Clean Q9 sweep with Q8-predicted-control features

```bash
python3 scripts/run_gmark_q9_model_sweep.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --run-name gmark_q9_sweep_v1_q8feat \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_sweep \
  --models ridge elasticnet rf \
  --include-q8-pred-features \
  --q8-feature-source question_context \
  --progress-every 250
```

### C) Enable official export/eval inside sweep

Add:

```bash
--run-official-eval
```

### D) One-command wrapper (both branches + Q8 feature branch)

```bash
python3 scripts/run_gmark_q9_sweep_wrapper.py \
  --run-name gmark_q9_clean_combo_v1 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_sweep \
  --q8-feature-source question_context \
  --models ridge elasticnet rf \
  --progress-every 250
```

If you explicitly want the diagnostic KG/Q8-model rerun path:

```bash
python3 scripts/run_gmark_q9_sweep_wrapper.py \
  --run-name gmark_q9_clean_combo_v1 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_sweep \
  --q8-feature-source q8_model \
  --q8-model-json /absolute/path/to/q8_model.json \
  --models ridge elasticnet rf \
  --progress-every 250
```

## Expected Outputs

Run folder pattern:

- `outputs/v2vgot_table1_reproduction/gmark_q9_sweep/<run_name>/`

Key files:

- `<run_name>_consolidated_manifest.json`
- `<run_name>_<model>_model.json`
- `<run_name>_<model>_predictions.jsonl`
- `<run_name>_<model>_manifest.json`

If `--run-official-eval`:

- `<run_name>_<model>_official_exports/...`
- `<run_name>_<model>_official_eval_reports/...`

## Open Follow-Ups

1. Compare additional model families (`elasticnet`, `rf`) against the ridge clean-sweep result.
2. Decide whether paper-facing Q9 should report `Q9 + provided Q8 prompt context` or require a strict predicted-Q8 context path.
3. Keep older control-metadata Q9 checkpoints clearly marked as leakage-risk/historical.
4. If strict end-to-end Q8 -> Q9 is needed, redesign Q8 feature generation to avoid the current KG/Q8 native-I/O block.

## Recorded VM Result: Q8 Prompt-Context Ridge

Run identity:

- run date: 2026-05-14
- run name: `gmark_q9_q8context_ridge_v1_withq8feat`
- model: `ridge`
- Q8 feature source: `question_context`
- feature matrix: `x_train=(11925, 26)`, `y_train=(11925, 12)`, `x_val=(3446, 26)`, `y_val=(3446, 12)`
- protocol label: **Q9 + provided Q8 prompt context**

Command:

```bash
python3 scripts/run_gmark_q9_model_sweep.py \
  --run-name gmark_q9_q8context_ridge_v1_withq8feat \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_sweep \
  --val-file-name v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json \
  --models ridge \
  --include-q8-pred-features \
  --q8-feature-source question_context \
  --progress-every 250 \
  --run-official-eval
```

Local metrics:

- `train_l2_avg=3.457594`
- `val_l2_avg=3.919460`

Official validation metrics:

- `l2_error_avg_1s=1.8233988941198525`
- `l2_error_avg_2s=2.912197374602219`
- `l2_error_avg_3s=3.92484623731654`
- `l2_error_avg_all=2.886814168679537`
- `PDMS=0.9532377083160599`
- `collision_rate_avg_all=0.0`
- `C=1.0`, `NC=1.0`, `TTC=1.0`

Comparison:

- no-Q8 clean ridge official `l2_error_avg_all=11.853387290683997`
- Q8 prompt-context ridge official `l2_error_avg_all=2.886814168679537`
- V2V-GoT Q9 remembered/reference `l2_error_avg_all ~= 2.62`

Metric naming note:

- Local `val_l2_avg=3.919460` tracks official `l2_error_avg_3s=3.924846`.
- Table-style Q9 headline should use official `l2_error_avg_all=2.886814`.

Artifacts:

- run root: `outputs/v2vgot_table1_reproduction/gmark_q9_sweep/gmark_q9_q8context_ridge_v1_withq8feat`
- consolidated manifest: `outputs/v2vgot_table1_reproduction/gmark_q9_sweep/gmark_q9_q8context_ridge_v1_withq8feat/gmark_q9_q8context_ridge_v1_withq8feat_consolidated_manifest.json`
- official eval summary JSON: `outputs/v2vgot_table1_reproduction/gmark_q9_sweep/gmark_q9_q8context_ridge_v1_withq8feat/gmark_q9_q8context_ridge_v1_withq8feat_ridge_official_eval_reports/gmark_q9_q8context_ridge_v1_withq8feat_ridge_official_export_manifest_official_qa_eval_summary.json`

## Recorded VM Result: Q8 Prompt-Context ElasticNet

Run identity:

- run date: 2026-05-14
- run name: `gmark_q9_q8context_elasticnet_v1_withq8feat`
- model: `elasticnet`
- Q8 feature source: `question_context`
- protocol label: **Q9 + provided Q8 prompt context**

Official validation metrics:

- `l2_error_avg_1s=1.823044394702993`
- `l2_error_avg_2s=2.9093408591957775`
- `l2_error_avg_3s=3.9207553504543937`
- `l2_error_avg_all=2.8843802014510547`
- `PDMS=0.9489262913522926`
- `collision_rate_avg_all=0.0`
- `C=1.0`, `NC=1.0`, `TTC=1.0`

Comparison:

- ridge prompt-context official `l2_error_avg_all=2.886814168679537`
- elasticnet prompt-context official `l2_error_avg_all=2.8843802014510547`
- absolute improvement over ridge: `0.00243396722848233m`
- practical interpretation: effectively tied with ridge, but ElasticNet is the best clean-sweep Q8 prompt-context number recorded so far.
- V2V-GoT reported/reference Q9 `l2_error_avg_all=2.620000`
- gap to V2V-GoT reference: `+0.2643802014510547m`, or about `10.09%` worse
- RF follow-up underperformed the linear baselines locally (`val_l2_avg=5.2500`; `1s=3.7633`, `2s=4.5215`, `3s=5.2500`), so the next improvement path should prioritize richer KG-derived features and/or smoother nonlinear models rather than plain RF.

Artifacts:

- official export manifest: `outputs/v2vgot_table1_reproduction/gmark_q9_sweep/gmark_q9_q8context_elasticnet_v1_withq8feat/gmark_q9_q8context_elasticnet_v1_withq8feat_elasticnet_official_exports/gmark_q9_q8context_elasticnet_v1_withq8feat_elasticnet_official_export_manifest.json`
- official eval log: `outputs/v2vgot_table1_reproduction/gmark_q9_sweep/gmark_q9_q8context_elasticnet_v1_withq8feat/gmark_q9_q8context_elasticnet_v1_withq8feat_elasticnet_official_eval_reports/future_trajectory_qa_type_19_official_eval.log`

## Freeze Decision: ElasticNet Context-Only Baseline

Status:

- Freeze `Q9 + Q8 prompt-context ElasticNet` as the active clean baseline.
- Official metric to track: `l2_error_avg_all=2.8843802014510547`.

v2 ablation summary (official):

- `context_only_elasticnet`: `l2_error_avg_all=2.8843802014510547` (best)
- `kgsubset_only_elasticnet`: `l2_error_avg_all=6.181112923881373`
- `context_plus_kgsubset_elasticnet`: `l2_error_avg_all=3.3317519194973264`

Interpretation:

- Current KG-control feature block (as appended tabular inputs) hurts Q9 performance in both KG-only and combined settings.
- Keep KG feature exploration optional, but do not replace the context-only baseline with KG-augmented variants at this stage.

Next model family:

- `scripts/run_gmark_q9_model_sweep_v2.py` now supports `--models hgb` (HistGradientBoosting via `MultiOutputRegressor`) for the same clean workflow.

## Model-Family Sweep Closure

Status:

- Keep `context_only_elasticnet` as the frozen clean baseline (`l2_error_avg_all=2.8843802014510547`).
- Tree/deeper model-family follow-ups (`rf`, `hgb`, `mlp`, `xgb`) did not beat this baseline in current runs.

Decision:

- Close broad model-family sweep for this phase.
- Shift effort to feature-diagnostics and feature-quality improvements.

## Latest Q8-Output Ablations

Goal:

- Keep the same clean Q9 setup, but replace/augment the Q8-derived feature block with richer Q8 model outputs from split-correct JSONLs.

Runs and outcomes (official):

- `gmark_q9_v2_context_elasticnet_q8float_r5_v1`:
  - protocol: keep first 10 Q8 one-hot columns, replace last 2 with Q8 float values from JSONL
  - result: `l2_error_avg_all=2.9858321342385437` (worse than baseline)
- `gmark_q9_v2_context_elasticnet_q8probs14_r5_v1`:
  - protocol: replace Q8 block with model probability/margin features (`model_probs14`)
  - result: `l2_error_avg_all=5.208098302540332` (much worse than baseline)

Comparison to frozen clean baseline:

- baseline (`context_only_elasticnet`): `2.8843802014510547`
- q8-float delta vs baseline: `+0.1014519327874890m` (worse)
- q8-probs14 delta vs baseline: `+2.323718101089277m` (much worse)

Decision:

- Keep `context_only_elasticnet` as the active clean Q9 baseline.
- Mark `q8float` and `q8probs14` as negative ablations for this phase.

## Latest Local Smoke Test

Command shape tested locally with one model and small limits:

```bash
python3 scripts/run_gmark_q9_sweep_wrapper.py \
  --run-name gmark_q9_debug_q8feat_ridge_smoke \
  --v2vgot-root /Users/bhavya/Desktop/ms_projects/V2V-GoT \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_sweep \
  --q8-model-json outputs/phase9_train_dev/q8_control_linear_classifier_smoke500_deployable.json \
  --models ridge \
  --progress-every 20 \
  --limit-train 80 \
  --limit-val 40
```

Result:

- No-Q8 branch: `x_train=(40, 14)`, `x_val=(40, 14)`, ridge `val_l2_avg=2.987099`.
- Q8-feature branch before numeric scalars: `x_train=(40, 24)`, `x_val=(40, 24)`, ridge `val_l2_avg=2.987099`.
- Q8 numeric scalar follow-up smoke: `x_train=(10, 26)`, `x_val=(10, 26)` with `--allow-train-val-overlap` only for local plumbing validation.
- The smoke Q8 model predicted `fast`/`straight` for every row, so this validates plumbing only, not final performance.
- Q8 question-context smoke: `x_train=(10, 26)`, `x_val=(10, 26)`, parsed Q8 context directly from Q9 prompts and avoided KG/Q8 rerun.

## Restart Resume Template

Before restarting session, fill this block so the next session can resume without ambiguity.

```text
run_date_utc:
run_name:
output_root:
wrapper_script: scripts/run_gmark_q9_sweep_wrapper.py

wrapper_command:

run_official_eval: true|false
skip_q8_feature_branch: true|false

q8_model_source_type: q8_model_json|e2e_manifest_json
q8_model_source_value:

q8_file_name: v2v4real_3d_grounding_qa_dataset_v2vgot.json
q9_val_file_name: v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json

models: ridge elasticnet rf
progress_every:
limit_train:
limit_val:
allow_train_val_overlap: true|false

notes:
```

Recommended one-command wrapper template:

```bash
python3 scripts/run_gmark_q9_sweep_wrapper.py \
  --run-name <RUN_NAME> \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --output-root outputs/v2vgot_table1_reproduction/gmark_q9_sweep \
  --e2e-manifest-json outputs/e2e_runs/r5/e2e_model_manifest.json \
  --models ridge elasticnet rf \
  --progress-every 250
```
