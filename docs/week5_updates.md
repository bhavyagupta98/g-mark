# Week 5 Updates

## Starting Point

Week 5 begins from the current Phase 8 QA-best checkpoint rather than from implementation plumbing.

What is already stable:

- Phase 7 deferred task coverage is complete.
- All eight V2V-GoT-QA task families run end to end through the benchmark path.
- Deterministic baseline outputs have been archived for the first `100` validation samples.

## Current Phase 8 QA-Best Checkpoint

Using the current-code QA-best v2 manifest on the first `100` validation samples:

- `notable_objects`
  - local benchmark-style score:
    - `F1 = 0.990`
    - `P = 1.000`
    - `R = 0.980`
    - exact = `99/100`
- `planning_awareness`
  - local benchmark-style score:
    - `F1 = 0.982`
    - `P = 1.000`
    - `R = 0.965`
    - exact = `98/100`
- `invisible_objects`
  - local benchmark-style score:
    - `F1 = 0.923`
    - `P = 1.000`
    - `R = 0.857`
    - exact = `99/100`
- `occluding_objects`
  - local benchmark-style score:
    - `F1 = 0.661`
    - `P = 0.725`
    - `R = 0.607`
    - exact = `39/100`

Current QA-best v2 artifacts:

- `outputs/phase8_baselines/phase8_qa_best_v2_current_code_manifest.json`
- `outputs/phase8_baselines/phase8_qa_best_v2_current_code_scored_report.md`
- `outputs/phase8_baselines/phase8_qa_best_v2_current_code_proxy_report.md`

Official-style V2V-GoT export/evaluation artifacts:

- `outputs/phase8_official_exports/phase8_qa_best_v2_current_code_official_export_manifest.json`
- `outputs/phase8_official_exports/notable_objects_phase8_qa_best_v2_current_code_official.jsonl`
- `outputs/phase8_official_exports/occluding_objects_phase8_qa_best_v2_current_code_official.jsonl`
- `outputs/phase8_official_exports/invisible_objects_phase8_qa_best_v2_current_code_official.jsonl`
- `outputs/phase8_official_exports/planning_awareness_phase8_qa_best_v2_current_code_official.jsonl`

Official-style upstream evaluation on the exported files:

| Task | V2V-GoT QA type | Localization F1 @ 0.5m | Precision | Recall | Binary F1 | Parse errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `notable_objects` | Q1 / `qa_type_id=11` | `0.989899` | `1.000000` | `0.980000` | `0.989899` | `0.0` |
| `occluding_objects` | Q2 / `qa_type_id=12` | `0.660907` | `0.725118` | `0.607143` | `1.000000` | `0.0` |
| `invisible_objects` | Q3 / `qa_type_id=13` | `0.923077` | `1.000000` | `0.857143` | `0.923077` | `0.0` |
| `planning_awareness` | Q4 / `qa_type_id=14` | `0.990991` | `1.000000` | `0.982143` | `0.989899` | `0.0` |

Paper-facing comparison against the published V2V-GoT QA references:

| Task | V2V-GoT paper F1 | +10% target | Our official-style F1 | Absolute gain | Relative gain | Target status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q1 `notable_objects` | `52.5` | `57.75` | `98.99` | `+46.49` pts | `+88.55%` | exceeds |
| Q2 `occluding_objects` | `30.1` | `33.11` | `66.09` | `+35.99` pts | `+119.57%` | exceeds |
| Q3 `invisible_objects` | `44.0` | `48.40` | `92.31` | `+48.31` pts | `+109.79%` | exceeds |
| Q4 `planning_awareness` | `60.8` | `66.88` | `99.10` | `+38.30` pts | `+62.99%` | exceeds |

Official-style notes:

- The upstream evaluator was run through a QA-only temporary copy of `LLaVA/scripts/eval_v2v4real_3d_grounding.py` because the VM image was missing import-time-only dependencies such as `cv2`, `numba`, `PIL`, and `scipy`.
- The metric logic for the simplified Q1-Q4 path was unchanged; only unused heavy imports were guarded.
- The official-style scores align with the local benchmark-style scorer, so the local Phase 8 QA loop is now calibrated against the upstream V2V-GoT answer format for Q1-Q4.
- The temporary `/tmp` workaround has been replaced by `scripts/run_v2vgot_official_qa_eval.py`, which writes the QA-only evaluator copy into the mounted repo at `outputs/phase8_official_exports/tools/`.
- This comparison is official-evaluator-compatible but currently limited to the first `100` validation samples; final claims should use the same flow on the full validation split.

## What Improved Before Week 5

### `notable_objects`

What was breaking:

- under-recall on the 100-sample slice
- many rows incorrectly returned no visible notable object

What we changed:

- fixed processed-scene BEV projection from `x,z` to `x,y`
- widened the visible-notable trajectory gate
- preferred grounded visible objects over candidate-only visible objects

Why it likely worked:

- geometry and coordinate interpretation were wrong before
- candidate leakage was polluting visible-object selection

### `planning_awareness`

What was breaking:

- severe over-firing
- proxy precision was low even though recall was already high

What we changed:

- aligned the answer logic to the benchmark structure
- planning-awareness now combines:
  - at most one hidden relevant object
  - at most one visible notable object
- deduplicates and renders the merged result directly

Why it likely worked:

- the benchmark question is narrower than the generic orchestrator policy we had been using
- composing the stronger `notable_objects` and `invisible_objects` paths matched the benchmark shape much better

## Current Interpretation

- `notable_objects` is no longer a bottleneck
- `planning_awareness` is no longer a bottleneck
- `invisible_objects` is in good shape
- `occluding_objects` improved substantially but remains the clearest remaining QA weakness

The key clue for `occluding_objects` is:

- proxy presence score is very high
- benchmark-style identity-aligned score is much lower

This suggests the remaining issue is likely blocker identity selection or object alignment, not simply detecting whether some blocker exists.

## Week 5 Primary Goal

Focus on `occluding_objects` improvement.

Recommended order:

1. inspect representative mismatch samples
2. identify whether the failure is blocker identity, pairing, or ranking
3. make one focused selector/scoring change
4. rerun the same 100-sample evaluation and scorer
5. checkpoint results before moving to the next task

### Occluding Mismatch Inspection Checkpoint

The first Week 5 occluding inspection pass is now complete.

New artifact:

- `outputs/phase8_baselines/phase8_occluding_mismatch_report.md`

What it shows on the archived 100-sample occluding slice:

- reference coordinate mentions: `252`
- predicted object mentions: `186`
- exact reference/predicted count matches: `39/100`
- under-predicted counts: `61/100`
- over-predicted counts: `0/100`
- empty predictions despite a positive reference: `4/100`

Interpretation:

- the dominant archived failure is count/recall under-selection, not over-selection
- reference answers contain either `2` or `3` occluding objects on this slice
- archived predictions contain `2` objects for `90/100` samples, `1` object for `6/100`, and `0` objects for `4/100`
- the next selector experiment should expand occluding recall, especially from top-2 to a confidence-gated top-3 path, while preserving the current no-overprediction behavior

Local environment note:

- this checkout's adjacent `../V2V-GoT` tree currently has no `.npy` processed assets
- live evaluator reruns therefore produce empty prepared scenes locally, so selector tuning should be validated in an asset-complete VM/pod or after restoring processed assets locally
- the new mismatch report intentionally uses archived Phase 8 predictions plus raw QA references, so it remains useful without processed assets

### Occluding Top-3 Open Checkpoint

The asset-complete VM confirmed that a permissive top-3 occluding selector improves the current bottleneck.

VM result on the same 100-sample occluding slice:

- previous live VM score:
  - `F1 = 0.564`
  - `P = 0.660`
  - `R = 0.492`
- `top3_open` score:
  - `F1 = 0.597`
  - `P = 0.657`
  - `R = 0.548`

Mismatch-shape change:

- predicted object mentions increased from `188` to `210`
- exact count matches improved from `39/100` to `47/100`
- under-predicted counts dropped from `61/100` to `46/100`
- over-predicted counts rose from `0/100` to `7/100`

Interpretation:

- this is a useful recall gain with only a small precision cost
- `top3_open` is now the current Phase 8 occluding-best selector
- the next loop should inspect the `46` remaining under-predicted samples and the `7` over-predicted samples to decide whether a third-candidate confidence gate can keep most of the recall gain while trimming the new over-predictions

### Occluding Risk-Adaptive Checkpoint

After inspecting the top-3 overfit risk, the occluding selector was moved toward a generic risk-adaptive policy rather than fixed scenario-distance thresholds.

Rationale:

- occlusion-aware driving literature frames this problem as risk assessment under limited visibility, not fixed-distance candidate counting
- useful signals should be relative to the scene/candidate distribution:
  - trajectory proximity
  - line-of-sight alignment
  - hidden-object relevance
  - provenance/support
  - model score
  - candidate uncertainty
- this lets future weather/traffic/situation settings adjust the policy without hardcoding one validation slice's geometry

VM result on the same 100-sample occluding slice:

- previous heuristic score:
  - `F1 = 0.564`
  - `P = 0.660`
  - `R = 0.492`
- `top3_open` score:
  - `F1 = 0.597`
  - `P = 0.657`
  - `R = 0.548`
- `top3_far_supported` score:
  - `F1 = 0.592`
  - `P = 0.675`
  - `R = 0.528`
- `risk_adaptive` score:
  - `F1 = 0.596`
  - `P = 0.667`
  - `R = 0.540`

Mismatch-shape comparison:

- `top3_open`: `210` predicted mentions, `46` under-predicted rows, `7` over-predicted rows
- `risk_adaptive`: `204` predicted mentions, `50` under-predicted rows, `5` over-predicted rows

Interpretation:

- `risk_adaptive` is within `0.001` F1 of `top3_open`
- it improves precision and reduces over-prediction
- it is preferred as the current Phase 8 occluding-best selector because it is configurable and less scenario-specific

Reference context:

- Yu, Vasudevan, and Johnson-Roberson, "Occlusion-Aware Risk Assessment for Autonomous Driving in Urban Environments", IEEE Robotics and Automation Letters, 2019
  - project page: `https://www.ri.cmu.edu/publications/occlusion-aware-risk-assessment-for-autonomous-driving-in-urban-environments/`
- Mobileye Responsibility-Sensitive Safety
  - limited-visibility and safe-distance framing: `https://www.mobileye.com/technology/responsibility-sensitive-safety/`
- "Occlusion-aware on-road autonomous driving: A trajectory planning method considering occlusions of Lidars", Optik, 2021
  - DOI page: `https://doi.org/10.1016/j.ijleo.2021.167347`
- Wang et al., "Potential risk assessment for safe driving of autonomous vehicles under occluded vision", Scientific Reports, 2022
  - article: `https://www.nature.com/articles/s41598-022-08810-z`

### Occluding Sparse-Evidence Backfill Checkpoint

The next inspection showed that several remaining under-predicted examples were not final-selector failures:

- some samples had only one blocker-role candidate
- some samples had zero blocker-role candidates
- therefore no top-3 policy could recover the missing objects unless the candidate set was broadened

Generic improvement:

- `risk_adaptive` now backfills sparse occluding evidence from visible objects when fewer than `2` blocker candidates are available
- the backfill uses normalized visible-risk features rather than reference labels:
  - trajectory proximity
  - asker proximity
  - support/provenance
  - confidence
  - conflict and uncertainty penalties

VM result on the same 100-sample occluding slice:

- `risk_adaptive` before backfill:
  - `F1 = 0.596`
  - `P = 0.667`
  - `R = 0.540`
- `risk_adaptive` with sparse-evidence backfill:
  - `F1 = 0.661`
  - `P = 0.725`
  - `R = 0.607`

Mismatch-shape change:

- predicted object mentions: `204 -> 211`
- exact count matches: `45/100 -> 49/100`
- under-predicted counts: `50/100 -> 46/100`
- over-predicted counts: `5/100 -> 5/100`
- empty predictions with positive references: `2/100 -> 0/100`

Interpretation:

- this is the strongest occluding result so far
- it improves precision and recall together
- the improvement comes from a generic sparse-evidence fallback, not from scenario-specific reference tuning
- this is the current Phase 8 occluding-best checkpoint

## Secondary Goals

After `occluding_objects`:

1. revisit `invisible_objects` recall only if a clear hypothesis appears
2. keep the current `notable_objects` and `planning_awareness` outputs frozen unless a later full upstream run exposes a mismatch
3. move from Q1-Q4 QA calibration toward the remaining official task families: control, trajectory, and motion prediction

### Invisible Objects Broad-Pool Train Checkpoint

Q3 `invisible_objects` was revisited after the full-split checkpoint showed that the selected `logreg_acceptor_t0p25` policy was precision-oriented but recall-limited.

Hypothesis:

- the main remaining Q3 bottleneck is candidate-pool recall, not MLP/logistic expressiveness
- if the true invisible object is absent from the shortlist, no acceptor can recover it

Shortlist coverage diagnosis:

- narrow `legacy_traj6` shortlist:
  - `fn_gt_absent_from_shortlist = 1889`
  - `fn_gt_present_in_shortlist = 334`
- broad `legacy_traj8_short64` shortlist:
  - `fn_gt_absent_from_shortlist = 1417`
  - `fn_gt_present_in_shortlist = 806`
  - `fn_gt_present_rank_le_3 = 781`
  - `fn_gt_present_rank_1 = 446`

Broad candidate feature export:

- split: `train`
- samples: `12290`
- candidate rows: `4994`
- unmatched GT rows: `1537`
- artifact: `outputs/phase8_train_dev/invisible_candidate_features_legacy_traj8_short64_train.jsonl`

Model comparison on the broad-pool feature table:

- `logreg best_f2p0`:
  - `F1 = 0.521224`
  - `P = 0.595937`
  - `R = 0.463158`
  - threshold `0.28`
- `mlp best_f2p0`:
  - `F1 = 0.519685`
  - `P = 0.591928`
  - `R = 0.463158`
  - threshold `0.28`

Interpretation:

- MLP did not beat the transparent logistic acceptor
- the broad-pool work should continue with logistic regression unless a stronger MLP hypothesis appears

Official-style train result for the current broad-pool checkpoint:

- policy: `broadpool_logreg_p50_t0p33`
- localization F1: `0.464406`
- precision: `0.527863`
- recall: `0.414568`
- binary F1: `0.538892`

Comparison to previous train best:

- old `logreg_acceptor_t0p25` train F1: `0.453823`
- new broad-pool train F1: `0.464406`
- absolute gain: `+0.010583`
- recall improved from `0.333633` to `0.414568`
- precision dropped from `0.709369` to `0.527863`

Status:

- this is the current train-selected Q3 candidate-pool recovery checkpoint
- it is eligible for one held-out validation run
- a validation attempt failed because `--invisible-acceptor-model-json` pointed to the optimization directory rather than the specific deployable JSON file
- rerun validation with the exact `logreg max_recall_p0p5` threshold `0.33` deployable model JSON

## Key Phase 8 Reference

- `docs/phase8_scored_evaluation_and_baseline_archival.md`

## Week 5 Closeout

Week 5 started as a Q2 recovery week and ended as the full Q1-Q4 paper-facing checkpoint week.

Final promoted validation checkpoint:

| Task | Current policy | Val F1 @ 0.5m | Precision | Recall | V2V-GoT ref | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Q1 `notable_objects` | visible-object `heuristic` | `0.585836` | `0.674759` | `0.517621` | `0.525000` | clears reference and +10% target |
| Q2 `occluding_objects` | `risk_adaptive` with sparse-evidence fallback | `0.427921` | `0.452542` | `0.405840` | `0.301000` | clears reference and +10% target |
| Q3 `invisible_objects` | broad-pool `logreg_acceptor_t0p33`, `shortlist_size=64`, trajectory window `8.0m` | `0.493934` | `0.488014` | `0.500000` | `0.440000` | clears reference and +10% target |
| Q4 `planning_awareness` | `relational_importance + trajectory_calibrated_acceptor`, duplicate radius `1.0m` | `0.613774` | `0.576685` | `0.655962` | `0.608000` | exceeds reference |

What was resolved:

- Q2 moved from a brittle blocker-count heuristic toward a general `risk_adaptive` occlusion selector with sparse-evidence backfill.
- Q3 moved from an eligible train checkpoint to a validated held-out checkpoint:
  - validation F1 improved over `legacy_traj6` from `0.395674` to `0.493934`
  - validation recall reached `0.500000`
  - the final story is broad retrieval plus selective train-frozen logistic acceptance.
- Q4 became the final Week 5 improvement cycle:
  - `relational_importance` made the pluggable orchestrator useful for planning-awareness
  - the train-frozen logistic acceptor improved over hand-written count policies
  - residual attribution showed retrieval was not the main blocker
  - reducing near-duplicate suppression to `1.0m` and adding trajectory calibration produced the promoted validation F1 `0.613774`
- `docs/project_status_summary.md` was created as the compact paper-facing checkpoint:
  - current Q1-Q4 metrics
  - KG construction details
  - query/retrieval details
  - Q3/Q4 reasoning and rejected alternatives
  - graph-quality and ego-only/cooperative ablation logging

Final Q3 artifacts:

- train summary: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_invisible_objects_broadpool_logreg_p50_t0p33_verify_official_export_manifest_official_qa_eval_summary.json`
- validation summary: `outputs/phase8_val_report/official_eval_reports/phase8_val_report_val_invisible_objects_broadpool_logreg_p50_t0p33_verify_official_export_manifest_official_qa_eval_summary.json`
- deployable model: `outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json`

Final Q4 artifacts:

- train summary: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1_official_export_manifest_official_qa_eval_summary.json`
- validation summary: `outputs/phase8_val_report/official_eval_reports/phase8_val_report_val_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1_official_export_manifest_official_qa_eval_summary.json`
- deployable model: `outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json`
- residual attribution report: `outputs/phase8_val_report/phase8_q4_planning_orch_rel_logreg_acceptor_residual_attribution.md`

Week 5 final interpretation:

- The current paper-facing result is no longer a 100-sample smoke claim; the selected Q1-Q4 numbers are full-split, official-style validation results.
- The most defensible method story is not one monolithic model. Each question family uses the KG differently:
  - Q1: visible-object grounding
  - Q2: occlusion/blocker risk
  - Q3: broad hidden-object retrieval plus logistic acceptance
  - Q4: relational planning importance plus trajectory-calibrated acceptance
- Week 6 should shift from metric chasing to paper consolidation:
  - produce final ablation tables
  - run or log current ego-only versus cooperative official-style comparisons
  - tighten the methods narrative
  - prepare figures/tables from `docs/project_status_summary.md`
