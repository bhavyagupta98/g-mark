# Phase 8: Scored Evaluation, Metric Alignment, And Baseline Archival

## Purpose

Phase 8 starts from the implementation-complete Phase 7 checkpoint.

Phase 7 proved that all V2V-GoT-QA task families now run through the benchmark path:

- `notable_objects`
- `occluding_objects`
- `invisible_objects`
- `planning_awareness`
- `future_trajectory`
- `control_settings`
- `object_motion_prediction`
- `agent_motion_prediction`

Phase 8 is where we move from "the tasks run" to "we understand how well they perform, where they fail, and which improvements are worth making."

This phase is expected to take time. It is the crucial bridge between implementation-complete task coverage and paper-facing evaluation.

## Phase 8 Goal

By the end of Phase 8, we want:

- scored deterministic baselines for every supported task family
- task-appropriate metrics for QA, control, trajectory, and motion-prediction outputs
- archived baseline outputs that can be compared against future changes
- qualitative failure buckets for each task
- a prioritized improvement plan based on measured weaknesses rather than blind tuning
- evidence that performance improvements do not destabilize the frozen Phase 6 QA checkpoint
- paper-facing improvements measured against the published V2V-GoT references where our local scorer is compatible

## Core Principle

Phase 8 should proceed in this order:

1. measure the current deterministic baselines
2. archive the outputs and scores
3. inspect failures qualitatively
4. choose one improvement hypothesis at a time
5. rerun the same scoring path after each change

The aim is not simply to add more heuristics. The aim is to improve performance with respect to the published V2V-GoT baseline wherever the metric is compatible.

The archived local deterministic baseline is still important, but it is a regression anchor rather than the primary success target. A change is useful only if it moves us toward, matches, or exceeds the V2V-GoT reference metric without breaking the stable Phase 6/7 behavior.

## Paper-Facing Targets

Phase 8 should use the V2V-GoT paper numbers as the primary target baseline. For higher-is-better metrics, the stretch target is at least `10%` relative improvement over V2V-GoT. For lower-is-better metrics, the stretch target is at least `10%` relative error reduction.

If repeated measured iterations cannot reach the `10%` improvement target, the fallback target is to match or exceed V2V-GoT in at least the most compatible local scoring view and to document exactly where our method is stronger or weaker.

| Task | V2V-GoT reference | Phase 8 +10% target | Direction |
| --- | --- | --- | --- |
| `notable_objects` | Q1 F1 = `52.5` | F1 >= `57.75` | higher is better |
| `occluding_objects` | Q2 F1 = `30.1` | F1 >= `33.11` | higher is better |
| `invisible_objects` | Q3 F1 = `44.0` | F1 >= `48.40` | higher is better |
| `planning_awareness` | Q4 F1 = `60.8` | F1 >= `66.88` | higher is better |
| `object_motion_prediction` | Q5 L2 = `8.05 m` | L2 <= `7.245 m` | lower is better |
| `agent_motion_prediction` | Q6 Accuracy = `87.4` | Accuracy >= `96.14` | higher is better |
| `object_motion_prediction` | Q7 L2 = `7.61 m` | L2 <= `6.849 m` | lower is better |
| `control_settings` | Q8 L1 = `0.0876` | L1 <= `0.07884` | lower is better |
| `future_trajectory` | Q9 L2 = `2.62 m` | L2 <= `2.358 m` | lower is better |

Notes:

- F1 and accuracy are stored in the paper on a percentage-style scale; local code often reports F1 on a `0-1` scale, so reports must show both or clearly state the conversion.
- The `10%` target is relative to the V2V-GoT reference, not relative to our local deterministic baseline.
- The local deterministic baseline remains useful for detecting regressions and measuring incremental change during experiments.
- Any metric that is only an approximation of the paper's scorer must be labeled as local/proxy until the scoring contract is verified.

## Workstreams

### 1. Baseline Archive

#### Goal

Freeze the current Phase 7 deterministic outputs before making further tuning changes.

#### Tasks

- export JSONL predictions for all task families
- store the exact command/config used for each run
- record sample count, supported count, unsupported count, and timestamp/date
- keep separate outputs for:
  - frozen QA tasks
  - deferred Phase 7 tasks

#### Success criteria

- every task has a reproducible baseline output file
- future experiments can be compared against this exact checkpoint

### 2. Metric Alignment

#### Goal

Define the right metric for each task family.

#### Task groups

- QA object-selection tasks:
  - `notable_objects`
  - `occluding_objects`
  - `invisible_objects`
  - `planning_awareness`
  - likely metrics: precision, recall, F1, exact match, support/unsupported counts

- Trajectory task:
  - `future_trajectory`
  - likely metrics: endpoint L2, average waypoint L2, collision-rate proxy if available

- Control task:
  - `control_settings`
  - likely metrics: action label agreement, speed/steering classification agreement, L1-style action error if labels are numeric or recoverable

- Motion-prediction tasks:
  - `object_motion_prediction`
  - `agent_motion_prediction`
  - likely metrics: direction agreement, endpoint L2, average displacement error, stationary-vs-moving classification accuracy

#### Success criteria

- each task has an explicit scoring contract
- unsupported or weakly defined official metrics are documented honestly
- local metrics are clearly distinguished from official benchmark metrics

### 3. Full Scored Runs

#### Goal

Move beyond 25-sample smoke runs into larger scored validation slices.

#### Recommended order

1. rerun the frozen Phase 6 QA tasks on a scored slice
2. score `future_trajectory`
3. score `control_settings`
4. score `object_motion_prediction`
5. score `agent_motion_prediction`

#### Success criteria

- every task has a first scored result
- results are saved with enough context to reproduce them
- scores are compared against Phase 6/Phase 7 checkpoint expectations where applicable

### 4. Failure Analysis

#### Goal

Turn numeric results into actionable improvement hypotheses.

#### Questions

- are failures caused by parsing/output format?
- are failures caused by object matching or ID resolution?
- are failures caused by missing velocity/trajectory context?
- are failures caused by overly conservative ranking?
- are failures task-specific or shared across the graph pipeline?

#### Deliverables

- failure buckets per task
- representative examples per bucket
- recommended first improvement for each task

### 5. Targeted Improvement Loops

#### Goal

Improve performance with respect to the published V2V-GoT reference metric.

#### Rules

- choose targets from the V2V-GoT reference table, not only from local baseline deltas
- change one task or shared component at a time
- rerun the relevant score after each change
- report both V2V-GoT-relative progress and local-baseline regression status
- rerun frozen QA smoke/regression if shared infrastructure changes
- keep failed experiments documented if they explain why an approach was abandoned

#### Likely early candidates

- `object_motion_prediction`:
  - improve temporal matching stability
  - tune moving/stationary thresholds
  - improve candidate-track identity continuity

- `agent_motion_prediction`:
  - refine trajectory-derived direction labels
  - compare pose-delta vs planned-trajectory signals
  - check whether `future_trajectory_str_in_self` needs frame transformation before direct use

- `control_settings`:
  - validate speed/steering labels against benchmark answers
  - tune risk thresholds
  - separate object-risk ranking from action rendering

- `future_trajectory`:
  - compare rendered trajectory against reference format and scoring expectations
  - add waypoint-level metric support

### 6. Paper-Facing Comparison

#### Goal

Relate local results to the V2V-GoT paper references without overstating compatibility.

#### Tasks

- map local metrics to paper metrics where possible
- document unit conversions, such as F1 `0-1` vs percentage scale
- flag metrics that are only local approximations
- prepare a clean summary table for advisor/supervisor updates

#### Success criteria

- paper-facing comparisons are clear, cautious, and reproducible
- any unofficial scorer limitations are stated directly

## Recommended Phase 8 Order

1. create/export baseline prediction files for all task families
2. define scoring contracts for deferred tasks
3. run full or larger scored slices for the frozen QA tasks
4. run first scored slices for deferred tasks
5. inspect failures and build task-specific failure buckets
6. pick the first improvement target based on measured weakness
7. rerun scores and update the V2V-GoT target table plus the local regression table
8. repeat targeted improvement loops
9. archive final Phase 8 deterministic baselines

## Immediate Next Step

Start Phase 8 by archiving deterministic baseline outputs.

Recommended first run:

- export JSONL predictions for all eight task families
- use `cooperative` mode
- use deterministic rankers only
- record commands and output paths in this document

This gives Phase 8 a stable baseline before any scoring or improvement work changes behavior.

## Exit Criteria

Phase 8 is complete when:

- [ ] every task family has an archived deterministic baseline output
- [ ] every task family has an explicit scoring contract
- [ ] frozen QA tasks have refreshed scored validation results
- [ ] deferred tasks have first scored validation results
- [ ] at least one measured weakness has been improved against the V2V-GoT paper reference target, or a documented failed-improvement loop has shown why matching V2V-GoT is the defensible fallback target
- [ ] all changes after improvement loops pass frozen QA regression checks
- [ ] final deterministic baselines are documented for paper-facing follow-up

## Status

Phase 8 has started.

### Baseline Archive: Initial 100-Sample Cooperative Outputs

Initial deterministic baseline outputs were copied locally under:

- `outputs/phase8_baselines/`

Files present:

- `notable_objects_cooperative_limit100.jsonl`
- `occluding_objects_cooperative_limit100.jsonl`
- `invisible_objects_cooperative_limit100.jsonl`
- `planning_awareness_cooperative_limit100.jsonl`
- `future_trajectory_cooperative_limit100.jsonl`
- `control_settings_cooperative_limit100.jsonl`
- `object_motion_prediction_cooperative_limit100.jsonl`
- `agent_motion_prediction_cooperative_limit100.jsonl`

Archive completeness:

- all 8 task files are present
- each file has 100 JSONL rows
- total archived predictions: 800
- all rows are supported
- unsupported predictions: 0 across all task families

### Baseline Shape Summary

#### Frozen QA Tasks

- `notable_objects`
  - 100/100 supported
  - object-count distribution: 67 empty, 9 one-object, 24 two-object
  - candidate-object rows: 18
  - early signal: still a strong first improvement target because many outputs are empty

- `occluding_objects`
  - 100/100 supported
  - object-count distribution: 4 empty, 6 one-object, 90 two-object
  - candidate-object rows: 19
  - early signal: broad coverage, but candidate usage should be checked during scoring

- `invisible_objects`
  - 100/100 supported
  - object-count distribution: 94 empty, 6 one-object
  - candidate-object rows: 0
  - early signal: conservative outputs remain expected from the frozen Phase 6 baseline

- `planning_awareness`
  - 100/100 supported
  - object-count distribution: 9 empty, 39 one-object, 26 two-object, 26 three-object
  - candidate-object rows: 0
  - early signal: stable output shape and no candidate-heavy noise in this archived slice

#### Deferred Phase 7 Tasks

- `future_trajectory`
  - 100/100 supported
  - all outputs render 6 trajectory points
  - object IDs are empty by design

- `control_settings`
  - 100/100 supported
  - object-count distribution: 76 one-object, 24 two-object
  - speed labels:
    - `reduce speed sharply`: 83
    - `slow down`: 10
    - `maintain current speed`: 7
  - steering labels:
    - `steer left`: 81
    - `steer right`: 19
  - candidate-object rows: 51
  - early signal: needs scoring against benchmark action answers before threshold tuning

- `object_motion_prediction`
  - 100/100 supported
  - object-count distribution: 7 one-object, 25 two-object, 68 three-object
  - motion labels:
    - `stationary`: 96 object predictions
    - `moving forward`: 73 object predictions
    - `moving backward`: 68 object predictions
    - `moving right`: 15 object predictions
    - `moving left`: 9 object predictions
  - rows with all-stationary outputs: 15
  - rows with at least one moving output: 85
  - candidate-object rows: 84
  - early signal: temporal velocity enrichment is active, but identity/candidate stability is likely the key scoring risk

- `agent_motion_prediction`
  - 100/100 supported
  - one predicted CAV per row
  - queried agents:
    - `CAV_1`: 50
    - `CAV_EGO`: 50
  - motion labels:
    - `move forward`: 50
    - `move backward`: 50
  - early signal: planned-trajectory fallback is active and produces directional outputs consistently

### Immediate Analysis Takeaway

The baseline archive is complete enough to proceed to scoring.

Recommended next step:

1. score the frozen QA tasks first, beginning with `notable_objects`
2. use the V2V-GoT `Q1 F1 = 52.5` reference as the first Phase 8 improvement target
3. delay heuristic tuning until the scored baseline and failure buckets are documented

### Notable Objects Improvement Loop 1

The first notable-objects improvement loop produced a strong local proxy gain on the archived 100-sample QA slice.

Key changes:

- corrected processed-scene BEV projection from `x,z` to `x,y` for GT tracks and detector observations
- widened the visible-notable trajectory gate to keep late-path visible objects
- filtered notable-object outputs to prefer grounded visible objects over visible candidates when both were present

Measured result under the local proxy scorer:

- previous `notable_objects` proxy presence F1: `0.795`
- updated `notable_objects` proxy presence F1: `0.990`
- previous precision / recall: `1.000 / 0.660`
- updated precision / recall: `1.000 / 0.980`
- reference-positive rows: `50`
- predicted-positive rows: `49`

Interpretation:

- this does **not** yet claim official benchmark parity
- it does show that `notable_objects` is no longer the weakest QA task under the local proxy view
- the dominant remaining QA bottleneck is now `planning_awareness`, whose proxy presence precision remains low despite perfect recall on the same slice

### Updated QA Priority

Current QA priority order for the next Phase 8 loop:

1. `planning_awareness` precision cleanup
2. `occluding_objects` tail-case cleanup
3. `invisible_objects` recall cleanup
4. `notable_objects` identity-level stabilization only if official-compatible scoring still shows a gap

### Official Evaluation Integration Note

We are **not** avoiding paper metrics. The current sequencing is local-first because the upstream V2V-GoT evaluation path is not yet a drop-in scorer for our normalized JSONL outputs.

What is currently blocking direct upstream evaluation:

- the upstream V2V-GoT evaluation scripts expect result files in the original repo-specific format, with model-style `outputs` fields embedded alongside raw dataset records
- our current Phase 8 artifacts are normalized JSONL prediction files with `sample_id`, `task_type`, `object_ids`, and `answer_text`
- the upstream evaluation flow is organized per QA node / `qa_type_id` and shell-script entrypoints, not as one generic scorer that accepts arbitrary exported predictions
- several task families require task-specific parsing and task-specific assets:
  - Q1-Q4/Q5/Q7 parse predicted locations from free-form text
  - Q6 is classification-style
  - Q8/Q9 require action / trajectory evaluation logic and future-scene assets

Why local-first was still the right choice:

- the local benchmark loop was fast enough to expose real pipeline bugs
- the `notable_objects` recovery came from exactly that loop:
  - first the BEV projection bug was identified
  - then visible-object gating and candidate filtering were refined
- doing upstream-only evaluation before fixing geometry would have been slower and much less informative

Current working plan:

1. continue improving local benchmark performance until the major task bottlenecks are reduced
2. freeze the strongest local deterministic outputs per task
3. implement one export bridge from our normalized JSONL predictions into the upstream V2V-GoT evaluation format
4. run the upstream evaluator task-by-task, then report the final official-style comparison in one pass
