# Week 5 Updates

## Starting Point

Week 5 begins from the current Phase 8 QA-best checkpoint rather than from implementation plumbing.

What is already stable:

- Phase 7 deferred task coverage is complete.
- All eight V2V-GoT-QA task families run end to end through the benchmark path.
- Deterministic baseline outputs have been archived for the first `100` validation samples.

## Current Phase 8 QA-Best Checkpoint

Using the current QA-best manifest on the first `100` validation samples:

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
    - `F1 = 0.566`
    - `P = 0.667`
    - `R = 0.492`
    - exact = `36/100`

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
- `occluding_objects` is now the clearest remaining QA weakness

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

## Secondary Goals

After `occluding_objects`:

1. revisit `invisible_objects` recall only if a clear hypothesis appears
2. keep the current `notable_objects` and `planning_awareness` outputs frozen unless official-style integration later exposes a mismatch
3. continue preparing for final upstream V2V-GoT evaluation integration once local QA improvements are stabilized

## Key Phase 8 Reference

- `docs/phase8_scored_evaluation_and_baseline_archival.md`
