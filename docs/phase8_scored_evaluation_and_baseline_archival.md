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

## Core Principle

Phase 8 should proceed in this order:

1. measure the current deterministic baselines
2. archive the outputs and scores
3. inspect failures qualitatively
4. choose one improvement hypothesis at a time
5. rerun the same scoring path after each change

The aim is not simply to add more heuristics. The aim is to improve performance with respect to a documented baseline.

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

Improve performance with respect to the archived deterministic baseline.

#### Rules

- change one task or shared component at a time
- rerun the relevant score after each change
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
7. rerun scores and update the baseline table
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
- [ ] at least one measured weakness has been improved against the archived baseline
- [ ] all changes after improvement loops pass frozen QA regression checks
- [ ] final deterministic baselines are documented for paper-facing follow-up

## Status

Phase 8 has not started yet.

Planned start: tomorrow.

Starting point: Phase 7 implementation-complete checkpoint.
