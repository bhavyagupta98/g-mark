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

## Progress Tracking

### ✅ Completed (24 April 2026)

#### Phase 7 Final Status
- [x] Status: **IMPLEMENTATION COMPLETE**
- [x] Focused VM test sweep passed after Phase 7 motion-fidelity changes
- [x] Final deferred-task smoke sweep passed:
  - `future_trajectory`: 25/25 supported, 0 unsupported
  - `control_settings`: 25/25 supported, 0 unsupported
  - `object_motion_prediction`: 25/25 supported, 0 unsupported
  - `agent_motion_prediction`: 25/25 supported, 0 unsupported
- [x] Final frozen-QA smoke sweep passed:
  - `notable_objects`: 25/25 supported, 0 unsupported
  - `occluding_objects`: 25/25 supported, 0 unsupported
  - `invisible_objects`: 25/25 supported, 0 unsupported
  - `planning_awareness`: 25/25 supported, 0 unsupported
- [x] Motion fidelity upgrades confirmed:
  - object motion uses adjacent-frame object-track velocity when available
  - agent motion uses nonzero CAV pose velocity when available, otherwise planned trajectory direction

#### Task Mapping and Output Design (In Progress)
- [x] `future_trajectory` output contract: render scene.future_trajectory.points as "[(x, y), (x, y), ...]" string
- [x] `control_settings` output contract: render as "speed=<instruction>; steering=<instruction>; key objects: [...]"
- [x] `object_motion_prediction` output contract: render as "object_id=<motion label> from (x, y) to (x, y)" entries
- [x] `agent_motion_prediction` output contract: render other CAVs as held-position predictions, with asker trajectory fallback

#### First Deferred Task: `future_trajectory` (Complete)
- [x] Handler implementation: `FutureTrajectoryHandler` in `v2vgotqa_router.py`
- [x] Benchmark adapter classification: already maps qa_type_id 9/19 → FUTURE_TRAJECTORY
- [x] Router integration: registered in `V2VGoTQARouter.default_handlers`
- [x] Unit tests: direct handler + router integration coverage
- [x] VM validation: ✅ Tests passed on user's VM
- [x] VM benchmark smoke run: 25/25 supported, 0 unsupported
- [x] Status: **FULLY SUPPORTED END-TO-END**

#### Second Deferred Task: `control_settings` (Complete)
- [x] Handler implementation: `ControlSettingsHandler` in `v2vgotqa_router.py` with:
  - Risk-based object ranking: combines distance_to_trajectory, distance_to_asker, visibility bonuses, status penalties, provenance bonuses
  - Speed instruction logic: distance ≤ 4.0 or occluded → "reduce speed sharply"; ≤ 8.0 or uncertain → "slow down"; else → "maintain current speed"
  - Steering instruction logic: lateral offset > 0.1 → "steer right"; < -0.1 → "steer left"; else → "keep steering centered"
- [x] Benchmark adapter classification: already maps qa_type_id 18 → CONTROL_SETTINGS
- [x] Router integration: registered in `V2VGoTQARouter.default_handlers`
- [x] Unit tests: direct handler test + router integration test
- [x] VM validation: ✅ Tests passed on user's VM
- [x] VM benchmark smoke run: 25/25 supported, 0 unsupported
- [x] Status: **FULLY SUPPORTED END-TO-END**

#### Motion-Prediction Tasks (Complete)
- [x] `object_motion_prediction` handler implementation: `ObjectMotionPredictionHandler` in `v2vgotqa_router.py`
- [x] Object prediction logic:
  - ranks relevant object tracks by distance to the planned trajectory, status, confidence, and object id
  - projects one-step future positions from `ObjectTrack.velocity` when available
  - falls back to stationary predictions when velocity is unavailable
  - supports qa_type_id 15 and 17 through existing benchmark adapter classification
- [x] `agent_motion_prediction` handler implementation: `AgentMotionPredictionHandler` in `v2vgotqa_router.py`
- [x] Agent prediction logic:
  - renders non-asker CAVs as held-position predictions because `AgentContext` has pose but no velocity/future trajectory field
  - falls back to the asker's future trajectory direction when no other CAVs are present
  - supports qa_type_id 16 through existing benchmark adapter classification
- [x] Router integration: both handlers registered in `V2VGoTQARouter.default_handlers`
- [x] Unit tests added: direct handler + router integration tests for both tasks
- [x] Local focused test run: `21 passed` for `tests/test_v2vgotqa_router.py` and `tests/test_v2vgot_benchmark_adapter.py`
- [x] VM validation: focused router and adapter tests passed on user's VM
- [x] VM benchmark smoke run:
  - `object_motion_prediction`: 25/25 supported, 0 unsupported
  - `agent_motion_prediction`: 25/25 supported, 0 unsupported
- [x] Qualitative note: first-pass motion outputs are mostly stationary because many benchmark/enriched tracks currently lack velocity, and `AgentContext` exposes pose but not per-agent velocity/future trajectory
- [x] Status: **SUPPORTED END-TO-END IN ROUTER**

#### Deferred Task Benchmark Smoke Results (Complete)
- [x] `object_motion_prediction`: 25/25 supported, 0 unsupported
- [x] `agent_motion_prediction`: 25/25 supported, 0 unsupported
- [x] `control_settings`: 25/25 supported, 0 unsupported
- [x] `future_trajectory`: 25/25 supported, 0 unsupported
- [x] Status: **ALL PHASE 7 DEFERRED TASKS RUN THROUGH THE BENCHMARK PATH**

#### Motion-Fidelity Follow-Up (In Progress)
- [x] Research finding: processed GT tracks and detector observations currently load positions/confidences but no velocity fields
- [x] Research finding: the repo already has `TemporalTrackManager` for adjacent-frame identity carry-forward
- [x] Implementation direction: use adjacent processed frames to infer per-track velocity only for motion-prediction tasks, preserving frozen QA behavior
- [x] First implementation:
  - `TemporalTrackManager` now computes `ObjectTrack.velocity` when a current track persists from a previous frame
  - `V2VGoTQAPhase5AEvaluator` now performs temporal enrichment for `object_motion_prediction` and `agent_motion_prediction` samples only
  - temporal enrichment uses previous timestamp data when available and falls back to single-frame behavior when it is not
- [x] Local focused test run: `26 passed` for temporal manager, evaluator, router, and benchmark adapter tests
- [x] VM benchmark smoke rerun for `object_motion_prediction`:
  - 25/25 supported, 0 unsupported
  - adjacent-frame matches now produce non-stationary outputs such as "moving forward", "moving backward", and "moving right"
  - fallback stationary outputs remain for timestamp 0 and tracks without stable adjacent-frame matches
- [x] VM benchmark smoke rerun for `agent_motion_prediction`:
  - 25/25 supported, 0 unsupported
  - outputs remain held-position predictions because current temporal enrichment estimates object-track velocity, while `AgentContext` still exposes only per-sample pose
- [x] Agent-motion fidelity implementation:
  - `AgentContext` now carries optional velocity
  - `V2VGoTSceneAdapter` can derive CAV velocity from a previous same-scenario QA record
  - `V2VGoTQABenchmarkAdapter` indexes adjacent records and passes previous-record context while loading samples
  - `AgentMotionPredictionHandler` projects CAV pose using derived velocity when available, with hold-position fallback
- [x] Local focused test run: `32 passed` for scene adapter, benchmark adapter, router, evaluator, and temporal manager tests
- [x] VM benchmark smoke rerun for `agent_motion_prediction`:
  - 25/25 supported, 0 unsupported
  - adjacent-record velocity path is active for later samples, shown by "from ... to ..." output format
  - first 25 validation samples still produce hold-position labels because adjacent CAV pose deltas are zero or below the motion threshold in this slice
- [x] Raw QA pose diagnostic:
  - adjacent pose pairs checked: 30,852 for each CAV
  - nonzero pose deltas: `CAV_EGO` 216, `CAV_1` 0
  - conclusion: CAV pose deltas are too sparse/static to be the primary agent-motion signal
- [x] Raw QA trajectory diagnostic:
  - `future_trajectory_str_in_ego`: 31,014/31,014 present and nonempty
  - `future_trajectory_str_in_self`: 31,014/31,014 present and nonempty
  - conclusion: planned trajectories are the stronger agent-motion signal
- [x] Planned-trajectory agent-motion implementation:
  - `AgentContext` now carries optional `planned_trajectory`
  - `V2VGoTSceneAdapter` attaches `future_trajectory_str_in_ego` to `CAV_EGO` and `future_trajectory_str_in_self` to `CAV_1`
  - `AgentMotionPredictionHandler` uses nonzero pose velocity first, then planned trajectory direction, then hold-position fallback
- [x] Local focused test run: `33 passed` for scene adapter, benchmark adapter, router, evaluator, and temporal manager tests
- [x] VM benchmark smoke rerun for `agent_motion_prediction` after planned-trajectory fallback:
  - 25/25 supported, 0 unsupported
  - outputs now produce directional movement such as `move forward` for `CAV_1` and `move backward` for `CAV_EGO`
  - planned trajectory fallback is now the primary useful signal when adjacent CAV poses are static

#### Checkpoint Integrity (Smoke Regression Complete)
- [x] Phase 6 QA smoke regression run completed on user's VM for:
  - `notable_objects`: 25/25 supported, 0 unsupported
  - `occluding_objects`: 25/25 supported, 0 unsupported
  - `invisible_objects`: 25/25 supported, 0 unsupported
  - `planning_awareness`: 25/25 supported, 0 unsupported
- [x] Regression documentation: no routing/support regression observed in the 25-sample CPU smoke run
- [x] Final post-motion-fidelity QA smoke regression completed with the same support outcome
- [ ] Full scored regression over the frozen Phase 6 validation slices: optional before final paper-facing claims

### Implementation Details

#### Code Changes Made

**File: `src/kg_coop_drive/application/v2vgotqa_router.py`**
- Added `FutureTrajectoryHandler` class: reads scene.future_trajectory.points, renders as trajectory string
- Added `ControlSettingsHandler` class: derives speed and steering from risk-ranked objects
- Added `ControlSettingsDecision` dataclass: encapsulates speed_instruction, steering_instruction, object_ids
- Added `ObjectMotionPredictionHandler` class: projects relevant object tracks from velocity or stationary fallback
- Added `AgentMotionPredictionHandler` class: renders non-asker CAV position predictions with asker trajectory fallback
- Added `MotionPrediction` dataclass: encapsulates projected entity motion
- Registered Phase 7 handlers in `V2VGoTQARouter.default_handlers` tuple

**File: `src/kg_coop_drive/application/temporal_track_manager.py`**
- Added velocity computation for persisted tracks using adjacent-frame position deltas

**File: `src/kg_coop_drive/application/v2vgotqa_evaluator.py`**
- Added motion-only temporal enrichment from the previous processed timestamp
- Kept non-motion QA tasks on the existing single-frame path to protect Phase 6 baselines

**File: `src/kg_coop_drive/domain/scene.py`**
- Added optional `AgentContext.velocity` for adjacent-record CAV motion.
- Added optional `AgentContext.planned_trajectory` for trajectory-derived CAV motion.

**File: `src/kg_coop_drive/infrastructure/v2vgot_scene_adapter.py`**
- Added optional previous-record pose comparison for deriving `CAV_EGO` and `CAV_1` velocity.
- Added per-agent planned trajectory parsing from `future_trajectory_str_in_ego` and `future_trajectory_str_in_self`.

**File: `src/kg_coop_drive/infrastructure/v2vgot_benchmark_adapter.py`**
- Added same-scenario adjacent-record indexing so loaded samples can carry agent velocity.

**File: `tests/test_v2vgotqa_router.py`**
- Added test_v2vgotqa_router_answers_future_trajectory(): verifies answer_text matches trajectory format
- Added test_future_trajectory_handler_renders_points_directly(): direct handler unit test
- Added test_v2vgotqa_router_answers_control_settings(): expects risk-ranked objects with speed/steering
- Added test_control_settings_handler_renders_speed_and_steering(): direct handler unit test
- Added object-motion direct handler and router integration tests
- Added agent-motion direct handler and router integration tests
- Updated test_v2vgotqa_router_marks_unsupported_tasks_explicitly(): changed deferred tasks → UNKNOWN since Phase 7 task families are now supported

**File: `tests/test_temporal_track_manager.py`**
- Added coverage that persisted tracks receive velocity from adjacent-frame displacement

**File: `tests/test_v2vgotqa_evaluator.py`**
- Added coverage that motion-prediction evaluation uses temporal velocity enrichment

**File: `tests/test_v2vgot_scene_adapter.py`**
- Added coverage for adjacent-record CAV velocity derivation

**File: `tests/test_v2vgot_benchmark_adapter.py`**
- Added coverage that sample loading attaches adjacent-record agent velocity

### Handler Design Pattern (Reusable)

All Phase 7 handlers follow this pattern:
1. Inherit from `_BaseQueryHandler` for utility methods (distance, visibility, relation lookups)
2. Implement task_type attribute and answer(sample) → BenchmarkAnswer protocol
3. Use deterministic logic (no LLM) for first-pass implementation
4. Include focused unit tests (direct handler + router integration)

## Exit Criteria

Phase 7 is complete when:

- [x] at least one deferred task is supported end to end → **DONE: future_trajectory + control_settings**
- [x] `future_trajectory` is runnable through the benchmark workflow → **DONE**
- [x] `control_settings` is either supported or explicitly designed and queued next → **DONE: FULLY SUPPORTED**
- [x] motion-prediction tasks have clear implementation plans, even if one or both remain for a later phase → **DONE: FIRST-PASS HANDLERS IMPLEMENTED**
- [x] the QA checkpoint from Phase 6 remains reproducible → **SMOKE REGRESSION PASSED: full scored regression still recommended before final paper-facing claims**

### Phase 7 Closeout Note

Phase 7 should now be treated as implementation-complete for benchmark-path task coverage and smoke-level regression.

Remaining work is paper-facing evaluation depth rather than implementation plumbing:

- full scored validation slices for old and new tasks
- official-style trajectory/motion metrics where available
- optional LLM/reranker experiments after the deterministic baselines are archived

## What Comes After Phase 7

The next phase should be:

**Phase 8: Scored Evaluation, Metric Alignment, And Baseline Archival**

Phase 8 should focus on:

- full scored validation slices for old and new tasks
- official-style or paper-aligned metrics for trajectory, motion, and control outputs
- deterministic baseline archival before further tuning
- comparison to V2V-GoT references where the local reproduction scorer is compatible
- optional LLM/reranker experiments after deterministic baselines are frozen

Phase 8 planning note:

- `docs/phase8_scored_evaluation_and_baseline_archival.md`
