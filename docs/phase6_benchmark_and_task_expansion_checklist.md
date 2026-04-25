# Phase 6: Benchmark Scoring And Task Expansion Checklist

## Purpose

Phase 6 turns the stabilized Phase 5 benchmark path into a stronger paper-facing evaluation and broader task-coverage phase.

Phase 5 answered:

- can the pipeline run reliably on a supported QA subset?
- can we compare cooperative vs `ego_only`?
- can we compare classical vs LLM decision logic?
- can we produce first object-level scoring?

Phase 6 should answer:

- can we evaluate more rigorously and more directly against benchmark-style targets?
- can we improve the supported QA tasks as a group rather than only stabilizing one strong task?
- can we extend beyond the first supported QA subset into the deferred planning/prediction tasks?

## Phase 6 Goal

By the end of Phase 6, we want to have:

- a stronger benchmark-style scoring layer
- improved QA-task performance across the supported Phase 5 subset
- at least partial support for the Phase 5 deferred task families
- clearer paper-facing result tables that can be compared to V2V-GoT more directly

## Why Phase 6 Exists

Phase 5 showed that:

- `planning_awareness` is now stable and meaningfully improved by orchestration
- `invisible_objects` is already reasonably strong
- `notable_objects` improved with a direct precision-oriented visible scorer, but later follow-on experiments did not beat that baseline
- `occluding_objects` required a more relational approach than simple visibility heuristics
- the local LLM ranker works, but does not outperform the best classical planning baseline
- our current scoring layer is informative, but still a local reproduction rather than an official benchmark reproduction

So the next work is no longer about plumbing. It is about:

1. improving evaluation fidelity
2. improving weak task behavior
3. expanding task coverage

## Workstreams

### 1. Official-Style Scoring Alignment

#### Goal

Reduce the gap between our current local object-level scorer and a paper-comparable benchmark evaluation.

#### Why this matters

Right now we can say:

- how scenarios differ
- how many object-level true positives / false positives / false negatives we have
- approximate F1-style performance on the QA subset

But we still cannot say with full confidence:

- how closely our numbers match the official V2V-GoT evaluation protocol

#### Deliverables

- a documented benchmark-scoring interpretation note
- a stricter QA-task scoring layer for Q1-Q4-style tasks
- a small validation study comparing:
  - current local scorer
  - improved scorer
  - expected paper metric framing

#### Success criteria

- the scoring layer is explicit enough to explain to a supervisor or reviewer
- the remaining differences from paper evaluation are clearly documented

### 2. Supported QA Task Improvement

#### Goal

Improve all supported QA tasks, while still prioritizing the ones with the largest current gaps.

#### Improvement order

1. `notable_objects`
2. `planning_awareness`
3. `invisible_objects`
4. deferred-task expansion after the supported QA block is strong enough

#### Why this matters

Phase 6 progress so far shows:

- `occluding_objects` improved substantially after switching from hard visibility heuristics to pairwise visible-blocker / hidden-target scoring
- the cooperative `occluding_objects` scorer reached local `F1 = 0.305`, which corresponds to `30.5%` on the paper's percentage-style F1 scale and is close to the V2V-GoT `Q2 F1 = 30.1`
- the same occluding result did not improve further with the current local LLM reranker, so the main gain came from the relational scorer rather than the model backend
- `notable_objects` has a best current cooperative baseline at local `F1 = 0.430`, and later additive-relation, energy-style, and local LLM rerank experiments did not improve over it
- `planning_awareness` is already relatively mature compared to the others
- `invisible_objects` is comparatively strong, but still has recall headroom

#### Likely improvement directions

- add task-specific routing/orchestration for `notable_objects`
- add occlusion-specific reasoning rather than only visibility-state filtering for `occluding_objects`
- revisit object-selection criteria for visible-object tasks where cooperative mode currently underperforms
- test whether `planning_awareness` can improve beyond the current `risk_aware + top2/diverse_top2` plateau
- improve recall for `invisible_objects` without sacrificing precision

#### Deliverables

- one concrete router/scoring improvement for `notable_objects`
- one concrete review pass on `planning_awareness` to decide whether additional scoring changes are justified
- one concrete review pass on `invisible_objects` recall behavior
- ablation runs before/after the change

#### Success criteria

- `notable_objects` cooperative mode is no longer clearly worse than `ego_only` on the validation slice
- `planning_awareness` is either improved further or explicitly frozen as a mature baseline
- `invisible_objects` recall behavior is documented and, if possible, improved

#### Confirmed result: `occluding_objects`

- best current method:
  - pairwise visible-blocker / hidden-target scoring
- best current cooperative result under the local reproduction scorer:
  - `F1 = 0.305`
  - `P = 0.201`
  - `R = 0.633`
- percentage-style interpretation for comparison with the paper:
  - `0.305 -> 30.5%`
- V2V-GoT target reference:
  - `Q2 F1 = 30.1`

Working interpretation:

- the unit conversion is important:
  - our scorer reports F1 on a `0-1` scale
  - the paper reports F1 on a `0-100` percentage-style scale
- after conversion, the current cooperative `occluding_objects` result is close to the paper target under the local scorer
- this should still be described as a local reproduction result rather than an official benchmark reproduction

#### Current checkpoint: `notable_objects`

- best current cooperative result under the local reproduction scorer:
  - `F1 = 0.430`
  - `P = 0.404`
  - `R = 0.460`
- best current ego-only result under the same scorer:
  - `F1 = 0.417`
  - `P = 0.435`
  - `R = 0.400`
- V2V-GoT target reference:
  - `Q1 F1 = 52.5`

Working interpretation:

- the direct visible-object scorer is the best notable baseline so far
- the cooperative path is slightly better than ego-only under that scorer
- later experiments:
  - additive relation features
  - an energy-inspired scorer
  - a local LLM reranker
  did not improve over the direct baseline
- for now, the direct scorer should remain the frozen baseline and the other approaches should be kept only as optional future experiments

### 3. Deferred Task Expansion

#### Goal

Extend benchmark support beyond the Phase 5A QA subset.

#### Deferred tasks from Phase 5

- `object_motion_prediction`
- `agent_motion_prediction`
- `control_settings`
- `future_trajectory`

#### Why these were deferred

Phase 5 intentionally avoided them because they require:

- prediction-oriented outputs
- action or trajectory rendering
- stronger planning semantics than the first QA subset

#### Recommended order

1. `future_trajectory`
2. `control_settings`
3. `object_motion_prediction`
4. `agent_motion_prediction`

Rationale:

- `future_trajectory` and `control_settings` align best with the current planning-aware graph direction
- motion-prediction tasks require a stronger predictive output layer

#### Deliverables

- explicit task-to-graph/task-to-output mappings
- first benchmark runner support for at least one deferred task
- initial qualitative evaluation for that task

#### Success criteria

- at least one deferred task is runnable end to end
- unsupported deferred tasks remain explicitly unsupported rather than silently mishandled

### 4. Classical vs LLM Role Clarification

#### Goal

Clarify where LLM-based reasoning actually helps and where classical logic remains stronger or sufficient.

#### Why this matters

Phase 5 showed:

- local AWQ LLM ranking is operational
- it matches the best planning-awareness classical baseline
- it does not yet surpass it

That means the LLM should now be treated as:

- a comparison axis
- a possible complementary component
- not an automatic improvement mechanism

#### Deliverables

- one short note on the role of LLM ranking in the current architecture
- one decision on whether to:
  - keep it as a baseline only
  - try hybrid scoring
  - postpone further LLM tuning until weak classical tasks are fixed

#### Success criteria

- there is a documented rationale for the LLM's role in the system
- no time is wasted on LLM tuning without a clear hypothesis

## Recommended Phase 6 Order

1. tighten scoring fidelity
2. log and freeze the current `occluding_objects` pairwise baseline
3. keep the current `notable_objects` direct baseline unless a stronger new hypothesis emerges
4. review and refine `planning_awareness` only if there is a strong hypothesis for improvement
5. review and refine `invisible_objects` recall behavior
6. add first deferred-task support
7. revisit whether hybrid classical + LLM ranking is worth testing

## Immediate Next Step

Start Phase 6 with a QA-improvement pass across the supported subset, beginning with:

1. log and freeze the current `notable_objects` direct baseline
2. review `planning_awareness`
3. review `invisible_objects`

Why:

- this keeps Phase 6 focused on improving the full supported QA subset rather than overfitting to only one task
- it locks in the strongest new Phase 6 result first
- it keeps the deferred trajectory/control tasks as a second-wave expansion after the QA subset has been strengthened

## Exit Criteria

Phase 6 is complete when:

- benchmark-style scoring is more defensible and clearly documented
- all supported QA tasks have been revisited with explicit improvement attempts or explicit freeze decisions
- `occluding_objects` pairwise scorer is documented and frozen as the current baseline
- `notable_objects` direct baseline and failed follow-on experiments are documented clearly enough to justify freezing it for now
- `planning_awareness` and `invisible_objects` are either improved or explicitly justified as stable baselines
- at least one deferred task is supported end to end
- the role of LLM reranking is explicitly documented based on evidence

## Phase 6 Checkpoint Status

Phase 6 is complete for now as a QA-benchmark strengthening phase.

What is now frozen as the current checkpoint:

- `notable_objects`
  - best current cooperative baseline:
    - direct visible-object scorer
    - local `F1 = 0.430`
- `occluding_objects`
  - best current cooperative baseline:
    - pairwise visible-blocker / hidden-target scorer
    - local `F1 = 0.309`
    - `30.9%` after unit conversion, close to the V2V-GoT `Q2 F1 = 30.1`
- `invisible_objects`
  - current heuristic path remains very strong:
    - local `F1 = 0.923`
- `planning_awareness`
  - best current baseline remains:
    - `risk_aware + top2`
    - `risk_aware + diverse_top2`
    - `llm + top2`
  - all tied at:
    - local `F1 = 0.440`

What Phase 6 established:

- a stronger local object-level scoring layer for the supported QA subset
- a clear improvement path for `occluding_objects`
- a better direct baseline for `notable_objects`
- stronger evidence that the local LLM path is useful as a comparison axis, but not automatically better than the strongest classical methods
- a stable four-task QA checkpoint that can be used as the handoff point for the next phase

What is intentionally deferred:

- official benchmark-script reproduction, beyond the current local scorer
- support for deferred benchmark tasks:
  - `future_trajectory`
  - `control_settings`
  - `object_motion_prediction`
  - `agent_motion_prediction`

Working conclusion:

- Phase 6 should be treated as complete for now
- the next phase should start from the frozen QA checkpoint rather than continuing to iterate blindly on the current four supported QA tasks

## What Comes After Phase 6

If Phase 6 succeeds, the next phase can branch into:

- broader planner and prediction experiments
- V2V-QA adaptation
- stronger synthetic data generation for hard occlusion cases
- broader paper-facing ablations and final benchmarking
