# Week 4 Updates

## Phase 5 Progress Log

### What was added

- A modular benchmark adapter layer for V2V-GoT-QA.
- A task inventory script that classifies the benchmark into stable task families.
- A Phase 5A benchmark router for:
  - `notable_objects`
  - `occluding_objects`
  - `invisible_objects`
  - `planning_awareness`
- A Phase 5A evaluation runner with:
  - `cooperative` mode
  - `ego_only` mode
  - JSONL prediction export
- A comparison script for cooperative vs ego-only predictions.

### What was confirmed

From the real `val` benchmark file:

- total samples: `31,014`
- task families:
  - `notable_objects`
  - `occluding_objects`
  - `invisible_objects`
  - `planning_awareness`
  - `object_motion_prediction`
  - `agent_motion_prediction`
  - `control_settings`
  - `future_trajectory`

Phase 5A is intentionally scoped to the first four task families above.

### Early pilot findings

#### `occluding_objects`

Comparing `cooperative` vs `ego_only` over the first `100` validation samples:

- exact-answer matches: `100/100`
- object-id matches: `100/100`
- differing samples: `0`

Interpretation:

- this task path is stable
- but it currently shows no cooperative advantage on this slice

#### `planning_awareness`

Comparing `cooperative` vs `ego_only` over the first `100` validation samples:

- exact-answer matches: `85/100`
- object-id matches: `85/100`
- differing samples: `15/100`

Observed pattern:

- cooperative mode often adds `pred_candidate_*` objects
- ego-only often produces fewer objects or empty outputs

Interpretation:

- this is the first task where cooperative and ego-only diverge in a measurable way
- however, the divergence is currently driven largely by candidate tracks
- so the result is promising, but not yet strong evidence of improved correctness

### Best next step

The next thing to do is inspect the `planning_awareness` divergence cases qualitatively, especially:

- `sample_id=43`
- `sample_id=65`
- `sample_id=69`
- `sample_id=95`

These are the best early cases for deciding whether cooperative-only candidate objects are useful signal or extra noise.

### Ranker Expansion

To avoid jumping to LLM reranking too early, the planning-awareness module was refactored into a ranker family with paper-cited implementations:

- `heuristic`
  - transparent confidence/support/provenance baseline
- `relational_importance`
  - inspired by Li et al., *Important Object Identification with Semi-Supervised Learning for Autonomous Driving* (ICRA 2022, arXiv:2203.02634)
- `risk_aware`
  - inspired by Nyberg et al., *Risk-aware Motion Planning for Autonomous Vehicles with Safety Specifications* (IV 2021, DOI:10.1109/IV48863.2021.9575928)
- `energy_based`
  - inspired by Tian et al., *KLDrive: Fine-Grained 3D Scene Reasoning for Autonomous Driving based on Knowledge Graph* (arXiv:2603.21029)
- `llm`
  - interface scaffold only for now; not connected to a real model yet

Implementation notes:

- the code citations are embedded directly in `src/kg_coop_drive/application/planning_awareness.py`
- `energy_based` uses an evidence-derived unary score plus a pairwise redundancy-aware decision policy
- evaluation and inspection scripts now support `--planning-ranker`

Current status:

- local smoke tests passed for `heuristic`, `relational_importance`, `risk_aware`, and `energy_based`
- local `pytest` still could not be run in this shell because `pytest` is not installed here

### Current Best Classical Baseline

The planning-awareness work is now in a much cleaner state than the early pilot phase.

#### Why the orchestrator refactor mattered

We moved from a brittle task-specific rule flow to an orchestrator because the original logic mixed together:

- candidate gathering
- object ranking
- final answer selection

That made it hard to:

- debug failures cleanly
- compare alternative ranking methods
- separate real semantic changes from output-format noise

The orchestrator fixed this by splitting the problem into:

- candidate collection
- scoring
- decision policy
- rendering

#### What changed after stabilization

1. Candidate-heavy noise was reduced.
2. Output ordering was canonicalized, so comparisons now measure real content differences.
3. Multiple planning-awareness rankers became directly comparable.

#### Current ranking conclusion

Among the non-LLM rankers:

- `risk_aware` is the strongest current baseline
- `energy_based` is useful but not stronger on the pilot slice
- `relational_importance` is semantically motivated but still noisier
- `heuristic` remains the simplest transparent reference point

#### Current best selection policy

For the current `planning_awareness` pilot slice, the best answer behavior comes from:

- `risk_aware + top2`
or
- `risk_aware + diverse_top2`

These policies reduce the weaker third object that often appeared in older top-3 outputs.

#### Most important qualitative example

For `sample_id=95`, the benchmark reference answer contains:

- one invisible/occluded car
- one visible car

With `risk_aware + diverse_top2`, the cooperative answer becomes:

- `1, 107`

This is the cleanest benchmark-aligned result seen so far for that example.

#### Current state

- cooperative vs ego-only with `risk_aware` differs on only `3/100` planning-awareness pilot samples
- heuristic vs risk-aware differs semantically on `10/100`
- ordering-only noise has been reduced to `0`

This means the pipeline is now stable enough to support the next step:

- keep `risk_aware + top2` or `risk_aware + diverse_top2` as the classical baseline
- move next to an LLM reranker comparison

### Phase 5 Closeout

Phase 5 has now been pushed beyond structural diff summaries into a scored closeout pass.

#### What was added after the initial planning-awareness stabilization

- a local AWQ LLM serving path
- a full Phase 5 closeout runner across:
  - `notable_objects`
  - `occluding_objects`
  - `invisible_objects`
  - `planning_awareness`
- a VM-side object-level scorer that resolves benchmark reference coordinates back to scene object IDs and computes:
  - precision
  - recall
  - F1
  - exact-match counts

#### Final scored outcomes on the first 100 validation samples

- `planning_awareness`
  - best methods:
    - `risk_aware + top2`
    - `risk_aware + diverse_top2`
    - `llm + top2`
  - tied at:
    - `F1 = 0.440`
    - `P = 0.308`
    - `R = 0.772`
- `notable_objects`
  - best current result: `ego_only` variants at `F1 = 0.380`
- `occluding_objects`
  - current result remains weak: `F1 = 0.048`
- `invisible_objects`
  - current result is strong: `F1 = 0.727`

#### Final interpretation

- the main Phase 5 gain came from the orchestrator refactor plus `risk_aware + top2/diverse_top2`
- the local AWQ LLM reranker now works end to end and matches the best planning-awareness classical baseline
- the LLM path does not currently surpass the strongest classical planning baseline
- `occluding_objects` is the clearest remaining weakness

### Phase 6 Progress: Occluding Objects Recovery

Phase 6 has now materially improved `occluding_objects`.

What changed:

- we replaced brittle hard blocker heuristics with pairwise visible-blocker / hidden-target scoring
- the scorer uses soft geometric evidence rather than requiring a strict pre-filtered blocker shortlist
- the local LLM reranker was tested on top of this path, but the main gain came from the pairwise relational scorer itself

Best current result on the first 100 validation samples:

- `occluding_objects` cooperative:
  - `F1 = 0.305`
  - `P = 0.201`
  - `R = 0.633`

Comparison to V2V-GoT:

- our scorer reports F1 on a `0-1` scale
- the V2V-GoT paper reports F1 on a `0-100` percentage-style scale
- so:
  - local `0.305` corresponds to `30.5%`
  - V2V-GoT `Q2 F1` target is `30.1`

Working conclusion:

- under the local reproduction scorer, the new cooperative `occluding_objects` baseline is now close to the paper reference
- this should still be presented as a local reproduced comparison, not as an official benchmark-script reproduction
- with `occluding_objects` now much healthier, the next Phase 6 priority shifts to `notable_objects`

### Phase 6 Progress: Notable Objects Checkpoint

We also revisited `notable_objects` in Phase 6.

Best current local result on the first 100 validation samples:

- cooperative direct visible-object scorer:
  - `F1 = 0.430`
  - `P = 0.404`
  - `R = 0.460`
- best ego-only comparison:
  - `F1 = 0.417`
  - `P = 0.435`
  - `R = 0.400`

Comparison to V2V-GoT:

- local `0.430` corresponds to `43.0%`
- V2V-GoT `Q1 F1` target is `52.5`

What was tried after that baseline:

- additive ego-centric relation features
- an energy-inspired notable-object scorer
- a local LLM reranker over the visible shortlist

Current conclusion:

- none of those follow-on experiments improved over the direct visible-object baseline
- for now, the direct scorer should remain the frozen `notable_objects` baseline
- the other approaches should be kept only as optional future experiments until a stronger hypothesis emerges

### Phase 6 Checkpoint Review

We ran a refreshed four-task scored checkpoint over:

- `notable_objects`
- `occluding_objects`
- `invisible_objects`
- `planning_awareness`

Current best checkpoint summary:

- `notable_objects`
  - cooperative direct scorer:
    - `F1 = 0.430`
- `occluding_objects`
  - cooperative pairwise scorer:
    - `F1 = 0.309`
    - `30.9%` after unit conversion, close to V2V-GoT `Q2 F1 = 30.1`
- `invisible_objects`
  - heuristic baseline:
    - `F1 = 0.923`
- `planning_awareness`
  - best tied methods:
    - `risk_aware + top2`
    - `risk_aware + diverse_top2`
    - `llm + top2`
  - all at:
    - `F1 = 0.440`

Checkpoint interpretation:

- the supported QA subset is now much healthier than at the end of Phase 5
- `occluding_objects` is the strongest Phase 6 improvement story
- `notable_objects` improved, but its simpler direct scorer remains best
- `invisible_objects` is strong enough to freeze
- `planning_awareness` is stable with a clear best baseline, but still below the paper-facing reference

Status:

- Phase 6 is complete for now as a QA-benchmark strengthening phase
- the current four-task scored report should be treated as the frozen handoff checkpoint for the next build phase

#### Relative to V2V-GoT target references

Using the local object-level scoring layer:

- `notable_objects`: below paper target
- `occluding_objects`: now close to the paper target under the local scorer after unit conversion
- `planning_awareness`: below paper target
- `invisible_objects`: appears competitive or above under the local scorer, but this should be treated cautiously until an official-style scorer is in place

#### Phase 5 status

Phase 5 is now closed from a pipeline and evaluation-stability perspective.

What remains is no longer Phase 5 plumbing. The next work should focus on:

- official benchmark-style scoring reproduction
- task-specific improvement only where there is still a strong hypothesis for gain
- expansion to deferred tasks:
  - `object_motion_prediction`
  - `agent_motion_prediction`
  - `control_settings`
  - `future_trajectory`
