# Phase 2 Completion Summary

## Status

Phase 2 is implementation-complete for the currently available processed asset slice.

The only remaining gap is **real-data multi-agent validation** on a frame where both `CAV_EGO` and `CAV_1` expose predictions at the same timestamp. The cooperative path is implemented and unit-tested, but the available processed roots do not contain such a frame.

Observed scanner result:

- repository root: `/workspace/repos/V2V-GoT`
- scanned processed roots:
  - `/workspace/repos/V2V-GoT/cobevt/npy`
  - `/workspace/repos/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy`
  - `/workspace/repos/V2V-GoT/DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy`
- outcome: no candidate processed root currently exposes predictions for both `CAV_EGO` and `CAV_1`

So the correct closeout statement is:

- Phase 2 is complete from an implementation and validation-tooling perspective.
- Full cooperative validation is blocked by processed asset availability rather than missing code.

## What Phase 2 Built

Phase 2 turned raw V2V-GoT benchmark records plus processed perception assets into a **canonical scene-level knowledge graph representation** that can already support deterministic reasoning.

Concretely, we implemented:

1. A canonical scene schema
- scene metadata
- agent poses
- future trajectory
- object tracks
- observation evidence
- relation facts
- visibility facts
- provenance and uncertainty-related fields

2. Dataset-to-scene adaptation
- parsing one V2V-GoT QA record into a `CooperativeScene`
- parsing the future trajectory string into typed 2D points
- normalizing raw agent ids into `CAV_EGO` / `CAV_1`

3. Processed asset loading
- GT object boxes and IDs
- detector outputs from processed prediction files
- visibility arrays when present

4. Single-frame graph construction
- GT-backed bootstrap tracks
- detector-backed observation evidences
- observation-to-track association
- matched support attachment
- unmatched-observation promotion into candidate tracks
- conservative candidate pruning
- conservative candidate-to-track merge

5. Temporal maintenance
- frame-to-frame identity persistence
- stale-track retention with miss counts
- track lifecycle updates

6. Graph semantics
- relation derivation
- conservative visibility reasoning
- provenance and uncertainty/conflict scoring

7. Deterministic querying
- selection
- visibility filtering
- relation filtering
- trajectory-nearness filtering

8. Cooperative scaffolding
- cross-agent observation association
- cross-agent support attachment onto existing tracks

## Source Data We Start From

The Phase 2 pipeline begins from two data families.

### 1. Benchmark QA JSON

Loaded through [v2vgot_scene_adapter.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive/infrastructure/v2vgot_scene_adapter.py:1).

Example fields used:

- `scenario_index`
- `local_timestamp_index`
- `global_timestamp_index`
- `asker_cav_id`
- `cav_ego_lidar_pose`
- `cav_1_lidar_pose`
- `future_trajectory_str_in_ego`
- `conversations`

These give us the **scene seed**:

- who is asking
- when the question applies
- where the cooperative agents are
- what future path the ego refers to
- what the original human question and benchmark answer were

### 2. Processed scene assets

Loaded through [v2vgot_processed_assets.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive/infrastructure/v2vgot_processed_assets.py:1).

Example processed inputs:

- `0000_gt.npy`
- `0000_gt_object_id.npy`
- `ego/0000_pred.npy`
- `ego/0000_pred_score.npy`
- visibility arrays when available

These give us:

- GT-backed object tracks
- detector-backed observations
- optional explicit visibility labels

## How We Form the Knowledge Graph

The important design choice is:

- we are **not** loading a pre-made graph from disk
- we are **constructing** a graph-like scene representation from heterogeneous inputs

The KG is currently represented as a typed in-memory scene structure, not as a graph database.

### Nodes

In the current schema, the main node-like entities are:

- agents
- object tracks
- observations

### Edges / facts

We represent graph edges as typed facts:

- `RelationFact`
  Example: `object 1 behind CAV_EGO`

- `VisibilityFact`
  Example: `object 1 visible to CAV_EGO`

- provenance inside `ObjectTrack`
  Example: track `1` came from `GT` and was supported by `CAV_EGO`

So the graph is best understood as:

- **entity records** plus
- **typed relational facts** plus
- **provenance/evidence links**

This is a deliberate research choice because it keeps every step inspectable and testable.

## Worked Example From the Logs

We use the first validation example from the real run:

- `scene_id = 0`
- `local_timestamp_index = 0`
- `global_timestamp_index = 0`
- `asker_agent_id = CAV_EGO`

Parsed agent poses:

- `CAV_EGO`: position `(-0.86, -0.51)`, yaw `0.510`
- `CAV_1`: position `(-0.68, -0.73)`, yaw `0.735`

Parsed future trajectory:

- `(8.6, 0.2)`
- `(17.2, 0.5)`
- `(26.0, 0.7)`
- `(34.7, 0.8)`
- `(43.6, 0.8)`
- `(52.6, 0.6)`

Original QA:

- Question: "I am CAV_EGO at (0.0,0.0). What are the notable objects visible to me near my planned future trajectory ... ?"
- Answer: "There is no notable object visible to you."

### Step 1: Build the scene seed

From the JSON record we first form:

- the scene context
- agent contexts
- the future trajectory
- the original QA text

At this moment, there are no object tracks yet.

### Step 2: Load GT and predictions

For `timestamp 0`, the processed loader found:

- GT boxes: yes
- GT IDs: yes
- ego predictions: yes
- `CAV_1` predictions: no
- explicit visibility arrays: no

Loaded observations:

- `CAV_EGO` observation near `(-20.65, -1.19)`, confidence `0.584`
- `CAV_EGO` observation near `(-75.06, -1.44)`, confidence `0.329`
- `CAV_EGO` observation near `(-5.92, -1.83)`, confidence `0.213`

Loaded GT-backed track:

- `object_id = 1`, position `(-20.50, -0.98)`

### Step 3: Associate observations to tracks

The association module compares each observation to each current track using a `3.0m` threshold.

Result:

- one observation matched `track 1`
- two observations remained unmatched

Concrete matched pair:

- `track_id = 1`
- observation from `CAV_EGO`
- distance `0.26m`
- confidence `0.584`

Research interpretation:

- this is the first bridge from raw detector evidence to structured world state
- it shows the detector and GT scaffold agree geometrically for that object

### Step 4: Attach support to the matched track

After support enrichment, track `1` becomes:

- status: `supported`
- provenance agents: `['GT', 'CAV_EGO']`
- support count: `1`

So we now have a graph object that is not merely "GT says it exists," but:

- GT-backed
- detector-supported
- provenance-aware

### Step 5: Promote unmatched detections into candidate tracks

The unmatched detections are not discarded immediately.

Instead:

- stronger unmatched detections become `candidate` tracks
- weak ones can be pruned later

In the working example:

- one candidate survives:
  - `pred_candidate_0_0`
  - position `(-75.06, -1.44)`
  - confidence `0.33`
- one weaker candidate is pruned

Research interpretation:

- the KG distinguishes between a strong object belief and a weak object hypothesis
- this is important because a cooperative perception graph must not collapse all evidence into a single certainty level

### Step 6: Derive relations

The relation builder derives geometric facts relative to the asker.

For this example:

- `1 behind CAV_EGO`
- `1 right_of CAV_EGO`
- `pred_candidate_0_0 behind CAV_EGO`
- `pred_candidate_0_0 right_of CAV_EGO`

This means the graph already supports relational reasoning such as:

- which objects are behind ego
- which objects lie to ego's right

### Step 7: Infer visibility conservatively

Because processed visibility arrays are absent for this frame, visibility is reasoned conservatively.

In the final refined version:

- `CAV_EGO:1` is inferred `VISIBLE`
- `CAV_1:1` is inferred `UNCERTAIN`
- the weak candidate is **not** inferred `VISIBLE`

Why?

- `track 1` is supported by an ego-side observation
- the candidate is low-confidence and prediction-only
- we do not want visibility inference to overclaim support for weak hypotheses

### Step 8: Query the graph

The deterministic query path then performs:

1. select all objects
2. filter by visibility to `CAV_EGO`
3. filter near trajectory
4. filter by relation if needed

In this example:

- after visibility filtering, only `['1']` remains
- after trajectory-nearness filtering, nothing remains

So the final query answer is consistent with the benchmark answer:

- there is no object that is both visible and near the planned trajectory

## Example KG Snapshot

An approximate conceptual KG for the example looks like this:

```text
Agent: CAV_EGO
Agent: CAV_1

Track: 1
  type: car
  position: (-20.50, -0.98)
  status: supported
  provenance_agents: [GT, CAV_EGO]
  support_count: 1

Track: pred_candidate_0_0
  type: car
  position: (-75.06, -1.44)
  status: candidate
  provenance_agents: [CAV_EGO]
  support_count: 1

Facts:
  1 behind CAV_EGO
  1 right_of CAV_EGO
  pred_candidate_0_0 behind CAV_EGO
  pred_candidate_0_0 right_of CAV_EGO
  visible(CAV_EGO, 1)
  uncertain(CAV_1, 1)
```

This is already a knowledge graph in the practical research sense:

- entities
- typed relations
- typed visibility facts
- provenance
- uncertainty

## What the Logs Show We Have Implemented

From the Phase 2 logs and tests, the repository now supports:

- QA record parsing into canonical scenes
- GT-backed track loading
- detector observation loading
- observation-track nearest-neighbor association
- support enrichment
- candidate generation and pruning
- conservative candidate merge
- temporal persistence and stale retention
- lifecycle updates
- uncertainty/conflict scoring
- relation derivation
- visibility reasoning
- cross-agent matching scaffolding
- cross-agent support attachment scaffolding
- deterministic query execution
- progress metrics and temporal metrics
- validation utilities for processed asset scanning

Observed unit-test count at the end of this stage:

- `31 passed`

## What Is Still Missing After Phase 2

Implementation-wise, Phase 2 is done.

What is still missing is **data-backed cooperative validation** on a real frame where:

- `CAV_EGO` predictions exist
- `CAV_1` predictions exist
- both are timestamp-aligned

The scanner showed that no currently available processed root exposes such a frame.

So the open item is:

- not "write more Phase 2 code"
- but "obtain richer processed assets for multi-agent cooperative validation"

## Why We May Need To Generate Similar Data

The current Phase 2 pipeline consumes two kinds of inputs:

- benchmark-level scene and QA context from the V2V-GoT JSON files
- processed perception outputs and GT-aligned assets from upstream `.npy` files

That is enough to build and validate the Phase 2 representation layer, but it is not enough to fully stress later phases under controlled conditions.

In particular, later work will need richer coverage of:

- simultaneous predictions from multiple agents at the same timestamp
- disagreement between agents about the same object
- false positives and missed detections
- visibility and occlusion edge cases
- temporal persistence and disappearance patterns

The currently available processed roots do not expose all of those cases, especially not real multi-agent prediction frames for the same timestamp.

So, eventually, we will likely need one of the following:

1. Re-run the upstream cooperative perception pipeline
- start from raw V2V-style inputs
- run the perception models again
- save processed per-agent predictions and scores
- recover richer multi-agent evidence for later KG phases

2. Build a synthetic cooperative scene generator
- generate GT objects, agent poses, and per-agent detections
- simulate misses, false positives, disagreement, and occlusion
- produce controlled test data for association, fusion, visibility, and uncertainty experiments

3. Use a hybrid strategy
- real processed assets for benchmark-facing evaluation
- synthetic scenes for controlled debugging and ablation testing

### Is this required immediately?

No.

It is not required to close Phase 2 from an implementation perspective.

### Will it eventually be required?

Very likely yes.

It becomes important once we want to rigorously validate:

- cross-agent association
- cooperative fusion
- conflict resolution
- visibility/occlusion handling
- uncertainty-aware querying

### Practical conclusion

This should be treated as a real future problem, but not as a blocker for declaring Phase 2 implementation-complete.

The most practical next solution is probably a **synthetic cooperative scene generator** that can produce:

- multiple agents
- GT objects
- per-agent noisy detections
- visibility labels
- temporal continuity across a few frames

That would give us a controlled test harness for later phases while still keeping real V2V-GoT assets as the benchmark-facing evaluation path.

## Research-Level Questions Your Advisor Might Ask

### 1. Why call this a knowledge graph if it is not stored in Neo4j or RDF?

Because the key property is not the storage engine but the **representation and reasoning structure**.

We explicitly model:

- entities
- typed relations
- typed visibility facts
- provenance
- uncertainty

and then reason over them through graph-style queries.

So this is a KG-style structured representation, even though it is currently implemented as typed Python dataclasses rather than a graph database.

### 2. What is the difference between an observation and an object track in your system?

- An `ObservationEvidence` is a local, agent-specific, timestamped detector report.
- An `ObjectTrack` is a scene-level world hypothesis.

In other words:

- observations are **evidence**
- tracks are **beliefs about world entities**

Association is the step that links evidence to beliefs.

### 3. Why do you bootstrap from GT tracks instead of detector-only tracks?

Because this phase is about stabilizing the schema and reasoning pipeline before moving to full detector-driven fusion.

GT-backed tracks give us:

- a stable world scaffold
- easier debugging
- interpretable provenance tests

Then we progressively replace "GT-only world state" with "detector-supported and later fused world state."

### 4. How do you avoid overclaiming certainty?

We explicitly separate:

- `supported` tracks
- `candidate` tracks
- `visible`
- `uncertain`

We also maintain:

- confidence
- support count
- conflict score
- uncertainty score
- miss count

So the graph does not collapse all evidence into a single hard truth label.

### 5. Why is visibility modeled separately from geometric relations?

Because geometry alone does not imply perception.

An object may be:

- geometrically behind ego
- still visible
- occluded
- or simply unknown because no direct evidence exists

So `behind` and `visible` answer different semantic questions and should not be conflated.

### 6. What does cooperative reasoning mean here if you only had ego predictions in the worked example?

It means the **architecture is cooperative-ready**:

- multiple agents are represented
- cross-agent association is implemented
- cross-agent support enrichment is implemented

But the currently available processed assets did not contain a real frame with simultaneous predictions from both agents.

So cooperative reasoning is implemented structurally, but not fully validated on real two-agent perception data yet.

### 7. Why do you keep candidate tracks instead of discarding unmatched detections?

Because unmatched detections may represent:

- true objects not yet represented in the scaffold
- false positives
- temporary one-frame evidence

If we discard them immediately, we lose potentially useful information.

If we keep them forever, we create graph clutter.

So the current design keeps them as `candidate` tracks and then prunes weak ones conservatively.

### 8. What exactly is the query answering over?

It is answering over the **current scene graph state**, which includes:

- object tracks
- relation facts
- visibility facts
- trajectory

The deterministic query engine does not read raw arrays directly.

Instead, raw arrays are first transformed into KG-friendly intermediate structure, and only then queried.

This separation is important because it makes the reasoning layer inspectable and testable.

### 9. How would you defend the scientific contribution of this phase?

The contribution of Phase 2 is not state-of-the-art performance.

Its contribution is:

- a canonical cooperative scene fact schema
- a deterministic graph-construction pipeline from V2V-GoT style assets
- explicit provenance, uncertainty, and visibility handling
- a bridge from raw perception outputs to queryable graph structure

This de-risks later phases where true cooperative fusion and LLM-constrained reasoning are introduced.

## Suggested Oral Summary

If you need to explain Phase 2 quickly in a meeting:

"In Phase 2, we built the canonical scene knowledge representation and the deterministic pipeline that converts V2V-GoT benchmark records plus processed GT and detector outputs into a queryable cooperative scene graph. We now represent agents, tracks, observations, relations, visibility, provenance, and uncertainty in one consistent schema. On a real example frame, we can parse the QA record, load GT and detector evidence, associate observations to tracks, create candidate objects, infer conservative visibility, derive spatial relations, and answer deterministic visible-object queries over the resulting graph. The only thing still missing is real-data validation on a frame where two agents both have predictions at the same timestamp." 
