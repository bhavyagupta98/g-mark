# Reasoning And Planning Benchmark Plan

This note captures the planned SOTA-facing benchmark for comparing G-MARK
against V2V-LLM and V2V-GoT as reasoning/planning systems. It also records how
to interpret the perception-fusion rows in V2V-GoT Table I so that any G-MARK
extension of that table is defensible.

## Goal

Build a reproduced benchmark table where V2V-LLM, V2V-GoT, and G-MARK are
evaluated on the same V2V-GoT-QA / V2V4Real-derived held-out tasks using the
same task definitions and evaluator outputs.

The intended claim is:

> G-MARK is a compact, explicit graph-based reasoning alternative to MLLM-based
> V2V reasoning. On the same cooperative driving QA/planning tasks, it can be
> compared directly against V2V-LLM and V2V-GoT while exposing a more
> interpretable and cheaper reasoning stack.

## Scope

This is a reasoning/planning benchmark, not a raw perception-backbone benchmark.

| Method | Level | Metrics |
| --- | --- | --- |
| V2V-LLM | MLLM reasoning/planning | Q1-Q9 QA metrics, Q9 L2, collision if available |
| V2V-GoT | MLLM + graph-of-thought reasoning/planning | same |
| G-MARK | explicit cooperative graph + task heads/planners | same |

CoBEVT, V2X-ViT, AttFuse, early fusion, and no-fusion rows can appear in a
V2V-GoT-Table-I-style planning table only if they are labeled correctly: they
are perception/evidence-source conditions feeding a downstream planner, not
standalone reasoning systems. Perception-backbone AP/detection comparisons
belong in a separate object-detection or object-fusion benchmark.

## Interpreting V2V-GoT Table I

V2V-GoT Table I should be read as a planning benchmark under different evidence
and reasoning stacks. The top rows are not raw perception methods directly
emitting planning trajectories. They are perception or fusion feature sources
that feed a V2V-LLM-style planning head.

The effective pipeline for the perception-source rows is:

```text
V2V4Real sensor/perception data
        -> perception/fusion backbone or feature source
        -> saved object/scene/perception features
        -> V2V-LLM-style planning-answer generator
        -> text trajectory/control answer
        -> common L2 / collision evaluator
```

Therefore, a row such as `CoBEVT` is best described as:

```text
CoBEVT-derived perception features + V2V-LLM-style planner
```

It should not be described as:

```text
CoBEVT alone solves the planning/reasoning task
```

The lower rows then compare reasoning/planning layers more directly:

- `V2V-LLM`: MLLM planner using the benchmark's cooperative scene/perception
  evidence.
- `V2V-GoT`: MLLM planner augmented with graph-of-thought reasoning.
- `G-MARK`: explicit cooperative KG/object-evidence graph with compact
  task-specific readout and planning heads.

This interpretation matters because it allows us to add a G-MARK row to the
same style of planning table without claiming that G-MARK is a raw perception
backbone or that it directly competes with CoBEVT at detection AP.

## Table-I-Style Extension

The paper-facing planning table can mirror V2V-GoT Table I, provided the row
labels and caption make the stack boundaries explicit.

Recommended row semantics:

| Method Row | Evidence Source | Reasoning / Planning Layer |
| --- | --- | --- |
| No Fusion | no-fusion V2V4Real features | V2V-LLM-style planning head |
| Early Fusion | early-fusion V2V4Real features | V2V-LLM-style planning head |
| AttFuse | AttFuse features | V2V-LLM-style planning head |
| V2X-ViT | V2X-ViT features | V2V-LLM-style planning head |
| CoBEVT | CoBEVT features | V2V-LLM-style planning head |
| V2V-LLM | cooperative benchmark evidence | MLLM planning |
| V2V-GoT | cooperative benchmark evidence | MLLM + graph-of-thought planning |
| G-MARK | structured V2V-GoT-QA scene/object evidence | explicit KG + task heads/planner |

Recommended caption language:

> Planning performance on the V2V-GoT-QA planning task. Perception-fusion rows
> use the corresponding V2V4Real feature source with a V2V-LLM-style planning
> head, following the V2V-GoT evaluation protocol. G-MARK uses structured
> V2V-GoT-QA scene/object evidence with an explicit cooperative KG and
> task-specific readout/planning heads. All rows are scored by the same
> planning-output evaluator; perception rows should not be interpreted as raw
> detector AP comparisons.

For now, G-MARK should be labeled as `G-MARK (structured V2V-GoT-QA evidence)`.
Do not label it as `G-MARK + CoBEVT` or `G-MARK from raw sensors` unless we
actually implement a CoBEVT/V2X-ViT-to-G-MARK adapter.

## Required Inputs

1. Held-out V2V-GoT-QA task files under the V2V-GoT repository, typically:

```text
/workspace/repos/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/
```

2. V2V-LLM released or trained checkpoint.

3. V2V-GoT released checkpoint.

4. G-MARK trained artifacts from the e2e train split flow.

5. A common evaluator path for each task, with parsed JSON summaries.

6. A run manifest recording the exact split, checkpoint, command, prediction
path, export manifest path, and evaluator log for every method/task pair.

## Output Artifact Layout

Use a method/task artifact layout so the benchmark can be resumed, audited, and
extended without touching existing e2e outputs:

```text
outputs/reasoning_planning_benchmark/<run_id>/
  method_manifest.json
  report.md
  summary.json
  v2v_llm/
    q1_notable_objects/
      predictions.jsonl
      export_manifest.json
      eval_summary.json
      eval.log
      command.txt
    ...
  v2v_got/
    q1_notable_objects/
      ...
  gmark/
    q1_notable_objects/
      ...
```

## Metrics Table

The main table should use one row per task:

| Task | Metric | V2V-LLM | V2V-GoT | G-MARK |
| --- | --- | ---: | ---: | ---: |
| Q1 notable objects | F1@0.5m | TBD | TBD | TBD |
| Q2 occluding objects | F1@0.5m | TBD | TBD | TBD |
| Q3 invisible objects | F1@0.5m | TBD | TBD | TBD |
| Q4 planning awareness | F1@0.5m | TBD | TBD | TBD |
| Q5 object motion prediction | L2 Avg 123 | TBD | TBD | TBD |
| Q6 agent motion prediction | Binary Accuracy | TBD | TBD | TBD |
| Q7 object motion prediction | L2 Avg 123 | TBD | TBD | TBD |
| Q8 control settings | Action L1 | TBD | TBD | TBD |
| Q9 future trajectory | L2 Avg All | TBD | TBD | TBD |

Use reproduced numbers when available. If a number is taken from a paper rather
than rerun locally, label it as reported, not reproduced.

## Planning-Focused Table

If the evaluator exposes per-horizon trajectory and collision metrics, produce a
second planning table:

| Method | Category | Q9 L2 1s | Q9 L2 2s | Q9 L2 3s | Avg L2 | Collision Rate | Comm Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Fusion | perception-source + planner | TBD | TBD | TBD | TBD | TBD | TBD |
| Early Fusion | perception-source + planner | TBD | TBD | TBD | TBD | TBD | TBD |
| AttFuse | perception-source + planner | TBD | TBD | TBD | TBD | TBD | TBD |
| V2X-ViT | perception-source + planner | TBD | TBD | TBD | TBD | TBD | TBD |
| CoBEVT | perception-source + planner | TBD | TBD | TBD | TBD | TBD | TBD |
| V2V-LLM | MLLM planner | TBD | TBD | TBD | TBD | TBD | TBD |
| V2V-GoT | MLLM + graph-of-thought planner | TBD | TBD | TBD | TBD | TBD | TBD |
| G-MARK | explicit KG + task heads/planner | TBD | TBD | TBD | TBD | TBD | TBD |

Only include collision rate and communication cost if they are computed under a
consistent protocol for all methods. Otherwise mark them unavailable.

## Recommended Execution Phases

### Phase 1: Asset Audit

Check that each method has the required checkpoint, QA files, evaluator scripts,
and feature roots. This phase should not run GPU-heavy inference.

For V2V-GoT / V2V-LLM audit, use:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/build_v2vgot_table1_reproduction.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --audit-only
```

### Phase 2: Smoke Evaluation

Run a small reproduced evaluation using only checkpoints that already exist.
This confirms the V2V-LLM and V2V-GoT inference/evaluator path before launching
full runs.

Current smoke command:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/build_v2vgot_table1_reproduction.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --run-v2v-baselines \
  --run-v2vgot \
  --only-method V2V-LLM \
  --only-method V2V-GoT \
  --skip-missing \
  --output-dir outputs/v2vgot_table1_reproduction
```

### Phase 3: Full V2V-LLM / V2V-GoT Evaluation

Run full held-out inference and evaluation for the V2V-LLM and V2V-GoT rows.
This likely needs GPU because both methods use LLaVA-style inference.

This phase should write method/task outputs into
`outputs/reasoning_planning_benchmark/<run_id>/`.

### Phase 4: G-MARK Held-Out Evaluation

Run the normal G-MARK e2e validation report using train-split trained artifacts
and held-out validation tasks. Do not retrain on validation data.

G-MARK evaluation should be imported into the same benchmark report format rather
than replacing the existing e2e outputs.

### Phase 5: Unified Report

Build a final markdown/JSON report that:

- lists all method checkpoints and artifact paths;
- states whether each number is reproduced or reported;
- shows per-task Q1-Q9 metrics;
- shows Q9 planning metrics if available;
- records commands used for every method/task pair;
- preserves raw evaluator logs for audit.

## General Runner Design

The benchmark runner should be method-agnostic:

```text
MethodRunner
  V2VLLMRunner
  V2VGoTRunner
  GMarkRunner

TaskEvaluator
  Q1-Q4 localization evaluator
  Q5/Q7 motion evaluator
  Q6 binary evaluator
  Q8 control/action evaluator
  Q9 trajectory/planning evaluator

ReportBuilder
  Reads eval summaries
  Emits markdown/json/csv
```

This keeps the benchmark reusable if later we add new reasoning systems or
additional G-MARK ablations.

## Defensibility Rules

- Use the same held-out task files for all methods.
- Use the same evaluator output format for all methods.
- Keep train-split fitting and held-out evaluation separate.
- Do not mix reported and reproduced numbers without labels.
- Do not claim G-MARK directly beats a perception backbone such as CoBEVT at raw
  perception; the planning table compares final planning stacks.
- Label perception rows as feature-source-plus-planner rows.
- Label the G-MARK row by its actual input evidence. Current row:
  `G-MARK (structured V2V-GoT-QA evidence)`.
- Only use `G-MARK + CoBEVT` or `G-MARK + V2X-ViT` labels after implementing a
  real adapter from those perception outputs into the G-MARK graph.
- Store all commands, logs, manifests, and checkpoints used for every number.

## What This Adds

The current G-MARK results show strong task-specific QA metrics. This benchmark
adds the SOTA-facing comparison needed to defend the architecture against the
closest prior reasoning systems:

- V2V-LLM: MLLM-based V2V reasoning.
- V2V-GoT: MLLM reasoning with graph-of-thought structure.
- G-MARK: explicit cooperative scene graph plus compact task-aware heads.

This gives a clearer paper story than only reporting internal ablations.
