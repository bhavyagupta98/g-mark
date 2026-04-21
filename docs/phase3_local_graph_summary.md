# Phase 3 Local Graph Summary

## Status

Phase 3 has begun successfully and already has a strong single-agent baseline.

At this stage, we can:

- build a local graph for one agent
- serialize it deterministically into JSON
- run deterministic local queries over it
- validate the local graph over multiple timestamps

This gives us a clean single-agent control condition before relying on cooperative fusion in later phases.

## Phase 3 Objective

The objective of Phase 3 is:

- prove that we can construct a grounded **local graph** correctly from one agent's evidence before relying on cross-agent fusion

In practical terms, that means:

- one agent
- one local scene
- local objects
- local relations
- local visibility
- local deterministic querying

## What We Implemented

### 1. Local graph builder

Implemented in [local_graph_builder.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive/application/local_graph_builder.py:1).

This module:

- projects a cooperative scene seed down to one agent
- filters processed observations to that agent only
- filters visibility facts to that agent only
- reuses the existing modules for:
  - observation association
  - support attachment
  - candidate creation
  - candidate pruning
  - conservative merge
  - uncertainty/conflict scoring
  - visibility reasoning
  - relation derivation

So Phase 3 does not duplicate logic unnecessarily; it reuses the proven building blocks from earlier work while enforcing a single-agent scope.

### 2. Local graph serializer

Implemented in [local_graph_serializer.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive/application/local_graph_serializer.py:1).

This gives us a deterministic JSON rendering of the local graph for:

- inspection
- debugging
- saved artifacts
- advisor/demo review

### 3. Local graph demo

Implemented in [demo_phase3_local_graph.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/scripts/demo_phase3_local_graph.py:1).

This demo shows:

- local graph summary
- local objects
- serialized JSON
- local deterministic query walkthrough

### 4. Multi-frame local validation

Implemented in [validate_phase3_local_graphs.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/scripts/validate_phase3_local_graphs.py:1).

This validator:

- builds local graphs across multiple timestamps
- prints per-frame summaries
- reports aggregate averages such as:
  - objects per frame
  - relations per frame
  - visibility facts per frame
  - supported tracks per frame
  - candidate tracks per frame
  - visible objects per frame
- can optionally export JSON snapshots

## Example Single-Agent Local Graph

For the real validation example at:

- `scene_id = 0`
- `global_timestamp_index = 0`
- `agent_id = CAV_EGO`

the local graph contains:

- one strong supported object:
  - `object_id = 1`
  - provenance: `GT` + `CAV_EGO`
  - status: `supported`
- one weaker local hypothesis:
  - `pred_candidate_0_0`
  - provenance: `CAV_EGO`
  - status: `candidate`

Local relations:

- `1 behind CAV_EGO`
- `1 right_of CAV_EGO`
- `pred_candidate_0_0 behind CAV_EGO`
- `pred_candidate_0_0 right_of CAV_EGO`

Local visibility facts:

- `visible(CAV_EGO, 1)`

This demonstrates an important local-graph property:

- supported and candidate beliefs remain distinguishable
- geometric relations are derived locally
- visibility is not overclaimed for weak candidates

## Example Local Query Behavior

The local query walkthrough for the same example shows:

- object selection returns `['1', 'pred_candidate_0_0']`
- visibility filtering returns only `['1']`
- near-trajectory filtering returns none
- relation filtering with `behind` returns both objects

So the local graph is already rich enough to answer:

- what exists locally
- what is visible locally
- what is near the local trajectory
- what is behind / left / right / front relative to ego

## Multi-Frame Local Behavior

On the observed validation slice, the local graph behaves sensibly across time.

Observed five-frame validation summary for `CAV_EGO`:

- validated frames: `5`
- average objects per frame: `2.00`
- average relations per frame: `4.00`
- average visibility facts per frame: `1.20`
- average supported tracks per frame: `1.20`
- average candidate tracks per frame: `0.40`
- average visible objects per frame: `1.20`

Representative patterns:

- timestamp `0`
  - `1` supported object
  - `1` candidate object
- timestamp `1`
  - `1` supported object persists
  - weak candidate disappears
- timestamp `2`
  - `1` persists as supported
  - `2` appears as a new supported object

This is a good Phase 3 signal because it shows:

- local graph construction is not a single-frame accident
- supported objects remain stable
- weak candidate clutter does not accumulate indefinitely
- local visibility stays selective rather than flooding the graph
- relation counts scale sensibly with the number of local objects

## Why Phase 3 Matters Research-Wise

Phase 3 gives us the local baseline needed for later claims.

Without it, later cooperative phases would be difficult to interpret because we would not know whether errors come from:

- local graph construction
- relation extraction
- visibility reasoning
- or cooperative fusion itself

So Phase 3 is the control condition that allows us to say:

- “This is what one agent alone can infer.”
- “Later cooperative gains must improve over this baseline.”

## What We Can Now Claim

At this point, we can honestly claim that the repository supports:

- deterministic single-agent graph construction
- deterministic local relation extraction
- conservative local visibility reasoning
- deterministic local querying
- deterministic JSON export of local graph artifacts
- multi-frame local validation over a small timestamp slice

## What Still Remains Within Phase 3

Phase 3 is not fully complete yet, but the core baseline is in place.

Useful next steps inside Phase 3 would be:

1. save and review a small set of exported local graph JSON artifacts
2. add a compact local validation report artifact in `docs/` or `artifacts/`
3. run the local validator over a slightly broader slice and inspect stability

These are now refinement and validation-strengthening steps rather than missing core functionality.

## Suggested One-Paragraph Summary

"In Phase 3, we isolated a single-agent local knowledge graph pipeline on top of the canonical schema built earlier. For one agent, we now load local evidence, construct object tracks and local hypotheses, derive ego-relative relations, infer conservative visibility, serialize the resulting local graph deterministically, and run deterministic local queries over it. We also validated this local graph behavior across multiple timestamps, which gives us a stable single-agent baseline before relying on later cooperative fusion phases."
