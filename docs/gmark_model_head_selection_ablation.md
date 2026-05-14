# G-MARK Model-Head Selection Ablation Log

This note collects the model-head and policy-selection experiments that were
used while choosing the current G-MARK task heads. It is separate from the graph
component ablations: these rows answer a different question, namely which
task-specific head or policy was selected after train/validation development.

Where exact rejected-run metrics were documented, they are included. Where a
variant is only described qualitatively in the existing notes, it is marked as a
qualitative rejection rather than assigned a number.

Reporting provenance:

- The rows in this note are our internal G-MARK model-head or policy-selection
  runs unless explicitly labeled as a V2V-GoT reference.
- V2V-GoT reference values are borrowed from the prior V2V-GoT paper/results and
  are used only as comparison anchors.
- The internal rows should not be described as reruns of the V2V-GoT baselines;
  they are G-MARK task-head choices evaluated with our official-compatible
  export/evaluation pipeline.

## Summary

| Task | Current selected head/policy | Main metric | Current value | Selection takeaway |
| --- | --- | --- | ---: | --- |
| Q3 `invisible_objects` | broad-pool logistic acceptor | F1@0.5m ↑ | 0.493934 | Broad retrieval plus selective acceptance beat narrow legacy and precision-only variants. |
| Q4 `planning_awareness` | relational importance + trajectory-calibrated logistic acceptor | F1@0.5m ↑ | 0.613774 | Residual-guided trajectory calibration beat plain logistic, count gates, and MLP. |
| Q5 `object_motion_prediction` | tuned regression-tree motion head | L2 Avg 123 ↓ | 7.272132 | Tree head reduced late-horizon drift versus earlier motion heads. |
| Q6 `agent_motion_prediction` | tuned GBDT classifier | Accuracy ↑ | 0.904527 | Tuned GBDT improved over earlier GBDT and the V2V-GoT reference. |
| Q8 `control_settings` | linear classifier with ordinal speed + risk-conditional thresholds | Action L1/8 ↓ | 0.076139 | Ordinal/risk-aware decoding reduced speed-dominated residual error. |
| Q9 `future_trajectory` | control-metadata linear + tail residual regressor | L2 Avg ↓ | 1.211582 | Tail residual improved long-horizon trajectory error. |

## Q3: Invisible Objects

Current selected configuration:

- ranker: `logreg_acceptor`
- selected model: `outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json`
- max results: `1`
- shortlist size: `64`
- max distance to trajectory: `8.0m`
- acceptor threshold: `0.33`

| Configuration | Val F1@0.5m ↑ | Precision ↑ | Recall ↑ | Status |
| --- | ---: | ---: | ---: | --- |
| `legacy_traj6` | 0.395674 | 0.310379 | 0.545614 | Superseded; high recall but low precision. |
| `logreg_acceptor_t0p25` | 0.403756 | 0.609929 | 0.301754 | Superseded; improved precision but recall remained too low. |
| `logreg_acceptor` high-precision variant | 0.372881 | 0.725888 | 0.250877 | Rejected; precision improved but recall collapsed. |
| `road_region_strict_traj8` | 0.372727 | 0.275630 | 0.575439 | Rejected; recall improved but precision collapsed. |
| broad-pool `logreg_acceptor_t0p33` | 0.493934 | 0.488014 | 0.500000 | Current selected. |

Selection rationale:

- Narrow legacy retrieval missed too many valid invisible candidates.
- Pure precision-oriented logistic acceptance produced too few positives.
- Broad retrieval made candidates reachable; the train-frozen logistic acceptor
  then filtered the broad pool without using validation labels.
- This gives the most balanced held-out validation result among documented Q3
  variants.

## Q4: Planning Awareness

Current selected configuration:

- ranker: `relational_importance`
- selection policy: `trajectory_calibrated_acceptor`
- base acceptor: train-frozen logistic acceptor
- deployable model: `outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json`
- duplicate suppression radius: `1.0m`

| Configuration | Val F1@0.5m ↑ | Precision ↑ | Recall ↑ | Status |
| --- | ---: | ---: | ---: | --- |
| `count_adaptive` | 0.500711 | 0.446938 | 0.569195 | Superseded; reduced over-selection slightly but remained weak. |
| `logreg_acceptor` | 0.607062 | 0.564947 | 0.655962 | Strong baseline. |
| `near_duplicate=1.0m` logistic | 0.607578 | 0.565305 | 0.656684 | Superseded; small consistent gain. |
| hard `count_gated_acceptor` | 0.605134 | 0.583075 | 0.628927 | Rejected; precision up, recall down too much. |
| soft `count_gated_acceptor` | 0.601722 | 0.569263 | 0.638106 | Rejected; did not recover enough recall/F1. |
| MLP acceptor | train F1 0.648767 | train P 0.601554 | train R 0.704022 | Rejected before validation; far below promoted train checkpoint. |
| `trajectory_calibrated_acceptor` | 0.613774 | 0.576685 | 0.655962 | Current selected. |

Additional train confirmation for the selected Q4 model:

- train F1/P/R at `0.5m`: `0.729672 / 0.711258 / 0.749064`
- validation F1 at looser thresholds:
  - `1.0m`: `0.661040`
  - `2.0m`: `0.661209`
  - `4.0m`: `0.733909`

Selection rationale:

- The plain logistic acceptor nearly matched the V2V-GoT Q4 reference but still
  over-predicted far/lateral objects.
- MLP capacity and regularization variants did not solve the residual error
  shape.
- Count gates improved precision but dropped recall.
- Residual attribution pointed to far/lateral false positives and duplicate
  suppression around close two-object answers; trajectory calibration directly
  targeted that error shape.

## Q5: Object Motion Prediction

Current selected configuration:

- task: `object_motion_prediction`
- QA type: `15`
- model family: regression tree
- promoted model: `outputs/phase9_train_dev/phase9_q5_tree_sweep_v1_s0/d9_l64_g0.01_m2.0_a120_deployable.json`
- metric: `l2_error_avg_123_all`, lower is better

| Configuration | Val L2 Avg 123 ↓ | Action Accuracy ↑ | Status |
| --- | ---: | ---: | --- |
| earlier checkpoint | 9.244614 | 0.400651 | Superseded. |
| initial regression-tree checkpoint | 7.911658 | 0.410414 | Superseded; first clear improvement over reference. |
| tuned regression-tree checkpoint | 7.272132 | 0.403905 | Current selected. |
| piecewise-linear variant | exact metric not documented in summary | exact metric not documented in summary | Rejected qualitatively; weaker validation L2 than tree family. |

Reference comparison:

- V2V-GoT Q5 reference: `8.05m`
- selected Q5 L2 Avg 123: `7.272132`
- relative reduction: about `9.66%`

Selection rationale:

- The tree head captured nonlinear motion regimes that a simpler projection or
  piecewise-linear head did not handle as well.
- Improvements were strongest in late-horizon drift, reflected by lower
  `l2_error_avg_3s` and lower overall `l2_error_avg_123_all`.

## Q6: Agent Motion Prediction

Current selected configuration:

- task: `agent_motion_prediction`
- QA type: `16`
- model family: tuned GBDT classifier
- selected scenario: `q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38`
- model settings: `n_estimators=280`, `learning_rate=0.04`, `max_depth=2`,
  `min_samples_leaf=96`, `subsample=0.7`, `threshold=0.38`
- metric: binary classification accuracy, higher is better

| Configuration | Val Accuracy ↑ | Status |
| --- | ---: | --- |
| first GBDT `q6_gbdt_v1_t0.40` | 0.877539 | Superseded. |
| regularized GBDT `n220/lr0.05/d2/l64/s0.7` | 0.899594 | Superseded. |
| tuned GBDT `n280/lr0.04/d2/l96/s0.7/t0.38` | 0.904527 | Current selected. |

Reference comparison:

- V2V-GoT Q6 reference: `0.874`
- selected Q6 accuracy: `0.904527`
- absolute gain: `+0.030527`
- relative gain: about `+3.49%`

Artifacts:

- checkpoint record: `outputs/phase9_train_dev/q6_gbdt_v4/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38_checkpoint.json`
- model JSON: `outputs/phase9_train_dev/q6_gbdt_v4/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38.json`
- official summary: `outputs/phase8_val_report/official_eval_reports/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38_official_export_manifest_official_qa_eval_summary.json`

Selection rationale:

- Logistic/tree-only predecessors were superseded by the GBDT path.
- The documented progression shows that both regularization and nearby capacity
  tuning improved held-out accuracy.
- This is useful for completeness, although Q6 is less central to the graph
  component ablation story because it is mostly a learned binary prediction
  head.

## Q8: Control Settings

Current selected configuration:

- handler: `ControlSettingsHandler(selection_policy=linear_classifier)`
- model family: train-frozen linear heads with ordinal speed decoding
- selected model: `outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json`
- feature set: `extended_v1`
- speed head: ordinal thresholds
- threshold policy: risk-conditional, `risk3`
- metric: normalized action edit distance, `action_edit_dist / 8`, lower is better

| Configuration | Speed Acc ↑ | Steering Acc ↑ | Action Acc ↑ | Action Edit Dist ↓ | Normalized Action ↓ | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train selected checkpoint | 0.655411 | 0.908950 | 0.607648 | 0.516599 | 0.064575 | Train fit/freeze result. |
| validation selected checkpoint | 0.684272 | 0.890308 | 0.620720 | 0.609112 | 0.076139 | Current selected. |

Reference comparison:

- V2V-GoT Q8 reference: `0.087600`
- selected Q8 normalized action error: `0.076139`
- absolute reduction: `0.011461`
- relative error reduction: about `13.08%`

Documented rejected or superseded design directions:

- Flat/global threshold tuning plateaued.
- Non-ordinal speed decoding left large class-index errors.
- Earlier Q8 mismatch analysis showed the residual was speed-dominated while
  steering was already strong.
- The selected model improved by adding ordinal speed decoding, risk-conditional
  thresholds, and extended trajectory/heading features.

Note:

- The scanned markdown contains complete promoted train/validation metrics for
  Q8, but it does not contain a compact numeric table for every rejected Q8
  variant. Those can be added later if the underlying JSON reports are copied
  into the repo.

## Q9: Future Trajectory

Current selected internal Q9 configuration:

- task: `future_trajectory`
- QA type: `19`
- model family: control-conditioned waypoint regression
- selected model: `outputs/phase9_train_dev/q9_future_trajectory_control_metadata_linear_tail_residual_v1_deployable.json`
- metric: L2 average, lower is better

| Configuration | Train L2 All ↓ | Val L2 All ↓ | Val L2 1s ↓ | Val L2 2s ↓ | Val L2 3s ↓ | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `control_metadata_linear_v1` | 0.894774 | 1.320964 | 0.950570 | 1.354670 | 1.657653 | Superseded. |
| `control_metadata_linear_tail_residual_v1` | pending full official train rerun | 1.211582 | 0.950570 | 1.354670 | 1.329508 | Current internal Q9. |
| Table-I/Q9 released split `gmark_table1_q9_full_v3` | not reported here | 1.21 avg | 0.95 | 1.35 | 1.33 | Paper-facing Table-I row. |

Reference comparison:

- V2V-GoT Q9 reference: `2.62m`
- selected internal validation Q9 L2: `1.211582`
- relative reduction: about `53.76%`
- released Table-I/Q9 row: L2 avg `1.21`, CR avg `0.00`, Comm `0.0159 MB`

Selection rationale:

- The control-metadata linear model already aligned well with the benchmark
  answer structure.
- The tail-residual variant specifically reduced late-horizon error without
  changing the short-horizon terms, improving the `3s` component and overall
  L2 average.

## How To Use This In The Paper

Recommended framing:

```text
In addition to graph-component ablations, we report task-head selection
ablations for the learned decision heads. These experiments show that the final
performance is not due to a single arbitrary head choice: invisible-object
reasoning benefits from broad retrieval plus selective logistic acceptance,
planning awareness benefits from residual-guided trajectory calibration, motion
prediction benefits from nonlinear tree heads, and planning/control tasks
benefit from structured control metadata and ordinal/risk-aware decoding.
```

What to avoid:

- Do not present Q6 or Q8 as pure KG-component ablations; they are primarily
  learned-head selection results.
- Do not claim that larger neural heads always helped. In Q4, the MLP acceptor
  underperformed the simpler logistic path.
- Do not report qualitative Q8 rejected variants as numeric ablations unless
  the exact JSON reports are recovered.
