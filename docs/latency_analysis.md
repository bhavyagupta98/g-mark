# Latency Analysis (Q1-Q9, Clean Q9 Mode)

## Why We Ran This

We ran this analysis to quantify the end-to-end latency added by our cooperative KG pipeline, broken down by component, across the full train split.

The goal was to answer:

1. Where does runtime go in the pipeline?
2. Which tasks are most expensive and why?
3. What is the incremental cost of temporal enrichment (Q5/Q6/Q7)?
4. Are we reporting Q9 under clean (no-leak) settings?

This is a system-latency study, not an accuracy study.

## What We Measured

Per sample, we captured stage-wise wall-clock timings (milliseconds), then aggregated by task:

- `avg_ms`: mean latency across all samples in that task bucket.
- `p50_ms`: median latency.
- `p90_ms`: 90th percentile latency.

Core stages:

- `processed_asset_load_ms`: load processed frame assets.
- `local_graph_enrichment_ms`: enrich scene from processed assets.
- `observation_association_ms`: match observations to tracks.
- `cross_agent_association_ms`: cross-agent object association.
- `track_support_enrichment_ms`, `cross_agent_support_enrichment_ms`: support/provenance attachment.
- `candidate_promotion_ms`, `candidate_resolution_ms`, `track_merge_ms`: candidate lifecycle and merge.
- `uncertainty_conflict_assessment_ms`: uncertainty/conflict scoring.
- `visibility_inference_ms`, `relation_build_ms`: graph semantic construction.
- `temporal_*`: previous-frame load + temporal update + temporal post-processing (only for temporal tasks).
- `task_solver_answer_ms`: task head inference/decision latency.
- `sample_total_ms`: end-to-end per-sample runtime.

## Clean Q9 Policy

Q9 in this analysis is run in clean mode (no oracle metadata leakage), using the clean Q9 model path integrated into the latency runner.

## Latest Run Command

```bash
python3 scripts/run_full_latency_analysis.py \
  --run-name latency_full_e2e_parallel_prefetch_20260519 \
  --purpose train_dev \
  --split train \
  --workers 1 \
  --progress-every 500 \
  --temporal-execution-mode parallel_prefetch
```

## Summary Table (Key End-to-End Signals, Cleaned)

Q5 and Q7 are merged into one row for consistent reporting.

| task_type | qa_type_id | samples | sample_total_avg_ms | task_solver_avg_ms | processed_asset_load_avg_ms | temporal_prev_load_avg_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| notable_objects | 11 | 12290 | 14.214 | 0.329 | 7.649 | - |
| occluding_objects | 12 | 12290 | 12.324 | 1.313 | 6.465 | - |
| invisible_objects | 13 | 12290 | 10.332 | 0.111 | 5.934 | - |
| planning_awareness | 14 | 12290 | 12.853 | 0.433 | 7.383 | - |
| object_motion_prediction (Q5/Q7 combined) | 15/17 | 12290 each | 33.384 | 0.812 | 9.178 | 10.233 |
| agent_motion_prediction | 16 | 12290 | 32.374 | 0.294 | 9.141 | 9.443 |
| control_settings | 18 | 12290 | 12.451 | 0.851 | 6.764 | - |
| future_trajectory (clean) | 19 | 12290 | 13.336 | 0.240 | 7.816 | - |

## Interpretation

1. Asset I/O is the dominant shared baseline across tasks.
   Most tasks spend a large fraction in `processed_asset_load_ms` (roughly 6-9 ms average), setting a common latency floor.

2. Temporal tasks are materially more expensive.
   Q5/Q7 and especially Q6 include temporal loading/update stages, which add significant overhead beyond single-frame tasks.

3. Q6 is no longer inference-dominated in this run.
   For Q6, `task_solver_answer_ms` is low (`0.294 ms`), while temporal stages and KG preparation dominate.

4. Clean Q9 is relatively light at inference time.
   Q9 solver cost is low (`0.240 ms` avg); most Q9 runtime is scene prep/loading rather than the prediction head.

5. Tail behavior matters.
   For many components, p90 is much higher than p50, indicating bursty/heavy-tail samples (complex scenes or runtime contention).

## Q6 Non-Spatial Latency Analysis (Added)

To isolate Q6 latency without spatial graph reasoning, run Q6 with graph ablations:

- `no_graph_relations`: keeps most graph pipeline, skips relation construction.
- `flat_non_graph_readout`: removes relation build plus other graph-heavy structure (candidates/provenance/uncertainty), giving a stronger non-spatial lower bound.

Suggested reproducible runs:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type agent_motion_prediction \
  --qa-type-id 16 \
  --scenario-name latency_q6_full \
  --baseline-mode cooperative \
  --graph-ablation-mode full \
  --latency-jsonl outputs/latency_analysis/latency_q6_full.jsonl \
  --skip-official-eval

python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type agent_motion_prediction \
  --qa-type-id 16 \
  --scenario-name latency_q6_no_graph_relations \
  --baseline-mode cooperative \
  --graph-ablation-mode no_graph_relations \
  --latency-jsonl outputs/latency_analysis/latency_q6_no_graph_relations.jsonl \
  --skip-official-eval

python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type agent_motion_prediction \
  --qa-type-id 16 \
  --scenario-name latency_q6_flat_non_graph_readout \
  --baseline-mode cooperative \
  --graph-ablation-mode flat_non_graph_readout \
  --latency-jsonl outputs/latency_analysis/latency_q6_flat_non_graph_readout.jsonl \
  --skip-official-eval
```

Then summarize each JSONL with:

```bash
python3 scripts/report_latency_breakdown.py \
  --latency-jsonl outputs/latency_analysis/<scenario>.jsonl \
  --output-markdown outputs/latency_analysis/<scenario>_summary.md
```

Interpretation for Q6:

- Current full Q6 baseline from this document: `32.374 ms` total, `0.294 ms` inference.
- The non-spatial runs quantify how much of the remaining time is relation/graph overhead versus the Q6 model head itself.
- If `flat_non_graph_readout` is only modestly faster than full mode, temporal loading/update is likely the main Q6 lever.
- If `flat_non_graph_readout` is much faster, graph preparation is still a major Q6 latency lever.

## How to Reduce Q6 Latency

Given the measured split for Q6 in this run (inference is very small; most time is KG + temporal), optimize in this order:

1. Cut duplicate temporal work first.
   - Cache previous-frame processed assets in-memory for consecutive timestamps.
   - Reuse temporal relations where track state changes are below a threshold.
   - Run temporal update at a lower frequency for low-dynamics scenes.

2. Reduce graph-build overhead that does not strongly affect Q6.
   - Skip or thin relation construction for relation types not used by Q6 features.
   - Use `no_graph_relations` as an ablation guardrail to validate minimal impact on Q6 accuracy.

3. Reduce asset-load overhead.
   - Keep processed assets on faster local storage/cache.
   - Prefetch adjacent-frame assets (current + previous) and avoid repeated deserialize work.

4. Keep model-head optimization as a secondary lever.
   - Distill/prune Q6 only after temporal + I/O wins plateau, since current inference share is small.

5. Control tail latency (`p90`) explicitly.
   - Add per-stage timeouts/budgets with fallback policy for heavy scenes.
   - Track p90 by component weekly; optimize whichever component has the largest p90-p50 gap.

## Where the Full Exhaustive Table Lives

This file gives a readable summary. The complete per-component exhaustive table (`avg_ms`, `p50_ms`, `p90_ms` for every task/component pair) is produced by the latency pipeline output:

- `<run_root>/<run_name>_latency_summary.md`
- `<run_root>/<run_name>_latency_summary.json`

These are the canonical artifacts for reporting.
