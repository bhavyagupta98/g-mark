# Phase 7: Deferred Task Expansion Checklist

## Purpose

Phase 7 starts from the frozen Phase 6 QA checkpoint and expands benchmark support into the deferred V2V-GoT task families.

Phase 6 established:

- a stable scored checkpoint for the supported QA subset
- stronger benchmark-facing local scoring
- frozen best baselines for:
  - `notable_objects`
  - `occluding_objects`
  - `invisible_objects`
  - `planning_awareness`

Phase 7 should answer:

- can we support the first deferred planning and prediction tasks end to end?
- can we do that without destabilizing the frozen QA baselines?
- can we produce the same kind of benchmark-facing evaluation and inspection workflow for those deferred tasks?

## Phase 7 Goal

By the end of Phase 7, we want to have:

- at least one deferred task fully supported end to end
- a clear task-to-output design for the remaining deferred tasks
- benchmark runners, inspection tools, and scoring or comparison outputs for the new task(s)
- a stable expansion path that reuses the Phase 6 checkpoint rather than reopening it

## Why Phase 7 Exists

The next most valuable work is no longer more blind tuning on the supported QA subset.

Phase 6 showed:

- `occluding_objects` improved strongly with relational scoring
- `notable_objects` improved and then stabilized at a direct baseline
- `invisible_objects` is strong enough to freeze
- `planning_awareness` has a stable best baseline even though it remains below the paper-facing target

So the best way to keep momentum is to move into the deferred tasks that were intentionally left out of Phase 5 and Phase 6:

- `future_trajectory`
- `control_settings`
- `object_motion_prediction`
- `agent_motion_prediction`

## Workstreams

### 1. Task Mapping And Output Design

#### Goal

Define exactly how each deferred benchmark task should map onto the graph pipeline and what output format it should produce.

#### Questions to answer

- what is the prediction target for each task?
- which existing scene features already support that target?
- where do we need new scoring, decoding, or rendering logic?
- which tasks should be framed as:
  - ranking
  - classification
  - regression
  - structured prediction

#### Deliverables

- a task-to-graph mapping note for each deferred task
- one explicit output contract per task
- one short note on likely scoring/evaluation format for each task

#### Success criteria

- each deferred task has an explicit implementation shape before coding starts
- unsupported tasks remain explicitly unsupported until their mapping is clear

### 2. First Deferred Task: `future_trajectory`

#### Goal

Add the first end-to-end deferred task using the most natural extension of the current planning-aware graph pipeline.

#### Why start here

- it aligns best with the current `planning_awareness` direction
- the repo already contains planning-aware signals and future-trajectory context
- it is easier to interpret than the broader motion-prediction tasks

#### Likely implementation steps

- define the benchmark answer contract for `future_trajectory`
- map the task to a trajectory proposal / ranking output
- implement a task-specific handler
- add evaluation script support
- add sample inspection support
- add comparison and reporting support

#### Success criteria

- `future_trajectory` runs end to end through the benchmark path
- at least one qualitative inspection workflow is available
- the output is stable enough for repeatable evaluation runs

### 3. Second Deferred Task: `control_settings`

#### Goal

Support the action/control recommendation task after `future_trajectory` is stable.

#### Why second

- it is planning-adjacent
- it can likely reuse trajectory/planning-aware features
- it is narrower than the motion-prediction tasks

#### Deliverables

- one control-settings handler
- one evaluation path
- one inspection workflow

#### Success criteria

- `control_settings` is runnable end to end
- the task remains clearly separated from `future_trajectory` even if they share upstream signals

### 4. Motion-Prediction Task Planning

#### Goal

Prepare `object_motion_prediction` and `agent_motion_prediction` without forcing implementation too early.

#### Why this is separate

- these tasks need a stronger predictive output layer
- they may require different evaluation logic than the QA-like tasks
- they are more likely to need new hypotheses rather than direct reuse of the current handlers

#### Deliverables

- one design note for `object_motion_prediction`
- one design note for `agent_motion_prediction`
- a recommendation on implementation order and required new abstractions

#### Success criteria

- both tasks have clear design plans even if only one is implemented in Phase 7

### 5. Checkpoint Integrity

#### Goal

Protect the frozen Phase 6 QA baselines while new task support is added.

#### Why this matters

Phase 7 should build on the checkpoint, not destabilize it.

#### Deliverables

- one regression run over the frozen QA subset after major changes
- one short note explaining whether any shared infrastructure changes altered old task behavior

#### Success criteria

- the frozen QA checkpoint remains reproducible
- any intentional changes to old tasks are documented explicitly

## Recommended Phase 7 Order

1. define task mappings for all deferred tasks
2. implement `future_trajectory`
3. add evaluation and inspection support for `future_trajectory`
4. implement `control_settings`
5. run QA regression check against the frozen Phase 6 checkpoint
6. plan motion-prediction tasks
7. decide whether one motion-prediction task should be pulled into late Phase 7

## Immediate Next Step

Start Phase 7 with `future_trajectory`.

Why:

- it is the most natural continuation of the current planning-aware graph pipeline
- it should benefit the most from the reasoning work already completed in Phases 5 and 6
- it gives the cleanest first deferred-task story for a supervisor-facing update

## Exit Criteria

Phase 7 is complete when:

- at least one deferred task is supported end to end
- `future_trajectory` is runnable through the benchmark workflow
- `control_settings` is either supported or explicitly designed and queued next
- the QA checkpoint from Phase 6 remains reproducible
- motion-prediction tasks have clear implementation plans, even if one or both remain for a later phase

## What Comes After Phase 7

If Phase 7 succeeds, the next phase can focus on:

- broader motion-prediction support
- stronger official-style benchmark evaluation
- cross-task hybrid reasoning strategies
- paper-facing ablation and final benchmarking passes
