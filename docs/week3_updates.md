# Week 3 Updates

## Progress This Week

- Revisited the available V2V-GoT assets and benchmark structure to confirm what is usable for knowledge-graph-based scene reasoning.
- Completed the core scene-graph pipeline that converts benchmark records and processed perception outputs into structured scene facts.
- Built and validated a single-agent local graph baseline so we have a clean control condition before stronger cooperative experiments.
- Added a deterministic query layer over the graph for structured selection, filtering, attribute access, comparison, and provenance tracing.
- Confirmed that the main remaining bottleneck for real cooperative validation is missing synchronized multi-agent processed predictions, not missing core graph logic.

## Concrete Outputs Now Working

### 1. Dataset and benchmark grounding

From the inspected V2V-GoT benchmark files:

- `val/v2v4real_3d_grounding_qa_dataset_v2vgot.json`: `31,014` records
- `train/v2v4real_3d_grounding_qa_dataset_v2vgot.json`: `110,610` records
- stable fields include:
  - `scenario_index`
  - `local_timestamp_index`
  - `global_timestamp_index`
  - `asker_cav_id`
  - `future_trajectory_str_in_ego`
  - `cav_ego_lidar_pose`
  - `cav_1_lidar_pose`

Example benchmark question pattern:

> I am CAV_EGO at (0.0,0.0). What are the notable objects visible to me near my planned future trajectory ... ?

This gave us a reliable benchmark entry point for existence, visibility, relation, and trajectory-based reasoning.

### 2. Canonical scene graph pipeline

The current pipeline now builds a structured scene representation containing:

- agent poses
- future trajectory
- object tracks
- observation evidence
- relation facts
- visibility facts
- provenance
- confidence / uncertainty information

This means we are no longer working directly with raw detections only. We now have a typed, inspectable scene state.

### 3. Real local-graph example

For the observed example:

- `scene_id: 0`
- `agent_id: CAV_EGO`
- `local_timestamp_index: 0`
- `objects: 2`
- `relations: 4`
- `visibility_facts: 1`

Local objects:

- `object_id=1`, status=`supported`, position=`(-20.50, -0.98)`, confidence=`1.00`
- `object_id=pred_candidate_0_0`, status=`candidate`, position=`(-75.06, -1.44)`, confidence=`0.33`

Derived relations:

- `1 behind CAV_EGO`
- `1 right_of CAV_EGO`
- `pred_candidate_0_0 behind CAV_EGO`
- `pred_candidate_0_0 right_of CAV_EGO`

Visibility:

- `visible(CAV_EGO, 1)`

This is a useful milestone because the system is now distinguishing strong supported objects from weaker hypotheses instead of flattening everything into one confidence layer.

### 4. Deterministic local query behavior

For the same example, the local query walkthrough currently gives:

- object selection: `['1', 'pred_candidate_0_0']`
- visibility filtering: `['1']`
- near-trajectory filtering: `[]`
- relation filtering with `behind`: `['1', 'pred_candidate_0_0']`

Interpretation:

- the graph contains two objects
- only one is considered visible to ego
- none of the visible objects are close enough to the planned future trajectory

So the system can already answer a meaningful benchmark-style question in a step-by-step and inspectable way.

### 5. Multi-frame local validation

Observed five-frame validation summary for `CAV_EGO`:

- validated frames: `5`
- average objects per frame: `2.00`
- average relations per frame: `4.00`
- average visibility facts per frame: `1.20`
- average supported tracks per frame: `1.20`
- average candidate tracks per frame: `0.40`
- average visible objects per frame: `1.20`

This suggests the local graph behavior is stable enough to use as a baseline before stronger cooperative fusion experiments.

In simple terms, this check was asking:

- if I build the local graph over several consecutive frames, does it still behave sensibly?
- do strong objects remain stable across time?
- do weak candidate detections disappear when they are not supported?

Concrete example in words:

- at timestamp `0`, the local graph contained one strong supported object and one weaker candidate
- at timestamp `1`, the strong supported object remained, but the weak candidate disappeared
- at timestamp `2`, the original supported object still remained and a new supported object appeared

This is useful because it shows the graph is not only correct on one isolated frame. It is also behaving reasonably over time: stable objects persist, while weaker clutter does not keep accumulating.

### 6. Deterministic query engine

The query layer now supports:

- object selection
- filtering by type
- filtering by relation
- filtering by visibility
- filtering by provenance source agent
- count and existence
- attribute lookup
- pairwise comparison
- provenance tracing

Observed example outputs from the query demo:

- `visible_count: 1`
- `visible_exists: True`
- `visible_ids: ['track-1']`
- `confidence_comparisons: [('track-1', 'greater_than', 'track-2')]`
- `object_type_comparisons: [('track-1', 'not_comparable', 'track-2')]`
- `empty_count: 0`
- `empty_exists: False`

This is important because unsupported or empty cases fail safely instead of inventing graph facts.

## Main Blocker Identified

The main blocker is still the lack of real synchronized multi-agent processed predictions.

Specifically:

- the cooperative path is implemented
- cooperative logic is unit-tested
- but the available processed roots do not currently expose a frame where both `CAV_EGO` and `CAV_1` provide predictions at the same timestamp

Practical consequence:

- we can validate the single-agent and reasoning backbone now
- but stronger real-data cooperative fusion experiments are still limited by asset availability

## Baseline Clarification

When I refer to the current single-agent graph as a baseline, I mean it in the experimental sense: it is the reference method that later cooperative reasoning will be compared against.

More specifically:

- the current baseline is single-agent deterministic graph construction and reasoning
- the final benchmark for evaluation is still the V2V-GoT QA task
- so later the comparison will be between:
  - single-agent graph reasoning on V2V-GoT QA
  - cooperative graph reasoning on V2V-GoT QA

In short:

- V2V-GoT is the benchmark
- the single-agent graph pipeline is the current baseline method evaluated on that benchmark

## Why This Week Matters

This week moved the project from setup into a functioning reasoning pipeline.

What is now in place:

- grounded dataset understanding
- canonical scene representation
- local graph baseline
- deterministic query engine

So the project now has the representation and reasoning backbone needed for the more novel next step: explicit cooperative graph fusion and evaluation under partial observability.

## Plan For Next Week

- Generate or curate the missing multi-agent data needed to run controlled cooperative simulations.
- Start multi-agent experiments where different vehicles observe only partial views of the same scene.
- Extend cooperative graph fusion so cross-agent object matching and evidence merging can be tested more directly.
- Evaluate single-agent versus cooperative reasoning on QA-style examples under limited visibility and occlusion.

## One-Paragraph Lab Meeting Summary

This week I completed the main backbone for the project: a structured scene-graph pipeline built on top of V2V-GoT benchmark and processed perception data, along with a local single-agent baseline and a deterministic query layer. On real examples, the system can now construct grounded object tracks, derive relations and visibility, serialize local graphs, and answer benchmark-style queries step by step. The main remaining limitation is not missing core implementation, but the lack of synchronized multi-agent processed predictions needed for stronger real cooperative validation. Next, I plan to generate or curate that missing cooperative setup and start controlled fusion experiments under partial observability.
