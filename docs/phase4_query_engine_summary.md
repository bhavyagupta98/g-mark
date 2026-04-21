# Phase 4 Deterministic Query Engine Summary

## Status

Phase 4 is well underway and already has a strong deterministic query baseline.

At this stage, the repository supports a read-only, structured, reproducible query interface over the current scene graph state.

The query layer is designed to be:

- side-effect free
- deterministic
- structured
- inspectable
- safe on unsupported or empty cases

This makes it the correct reasoning substrate to build on before any LLM layer is introduced.

## Phase 4 Objective

The objective of Phase 4 is:

- build the deterministic reasoning substrate before any LLM is added

In practice, that means:

- typed read-only graph operations
- fixed tool semantics
- reproducible answers from graph state
- safe failure on invalid or unsupported requests

## What We Implemented

Implemented in [query_engine.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive/application/query_engine.py:1).

The current tool set includes:

1. `select_objects`
- returns all current object tracks from a scene

2. `filter_by_type`
- keeps only objects of a requested semantic type

3. `filter_by_relation`
- keeps only objects linked by a requested relation to a reference entity

4. `filter_by_visibility`
- keeps only objects with a requested visibility state for one agent

5. `filter_by_source_agent`
- keeps only objects whose provenance includes a requested source agent

6. `count`
- returns the size of the current selection

7. `exists`
- returns whether the current selection is non-empty

8. `get_attribute`
- returns structured attribute values for selected objects

9. `compare`
- compares one supported numeric attribute across all selected object pairs

10. `trace_provenance`
- returns structured provenance traces for selected objects

## Structured Outputs

Phase 4 introduced structured result records for query-style outputs:

- `QueryAttributeValue`
- `QueryComparison`
- `ProvenanceTrace`

This is important because later reasoning layers should consume typed outputs rather than free-form strings.

## Supported Attribute Reads

The current `get_attribute` tool supports:

- `confidence`
- `object_type`
- `status`
- `position_x`
- `position_y`
- `support_count`
- `uncertainty_score`
- `conflict_score`

Unsupported attributes currently return:

- `None` per object

This is deliberate safe-failure behavior.

The current `compare` tool is intentionally narrower than `get_attribute`.

Comparable v1 attributes are:

- `confidence`
- `position_x`
- `position_y`
- `support_count`
- `uncertainty_score`
- `conflict_score`

Non-comparable attributes such as `object_type` currently return:

- `relation = not_comparable`

## Safe-Failure Semantics

The current deterministic query engine fails conservatively:

- empty selections return:
  - `count = 0`
  - `exists = False`
  - empty tuples for attribute lookup, comparisons, and provenance traces

- unsupported attributes return:
  - structured attribute rows with `value = None`
  - no comparisons rather than an exception or hallucinated value

- non-comparable compare requests return:
  - structured comparison rows with `relation = not_comparable`

This is an important research property because it prevents the reasoning layer from silently inventing graph facts.

## Hand-Authored Phase 4 Example

Implemented in [demo_phase4_query_examples.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/scripts/demo_phase4_query_examples.py:1).

The example scene contains:

- `track-1`
  - car
  - confidence `0.9`
  - provenance `GT + CAV_EGO`
  - status `supported`

- `track-2`
  - car
  - confidence `0.4`
  - provenance `CAV_EGO`
  - status `candidate`

Relations:

- both are `front_of CAV_EGO`
- `track-1` is `left_of CAV_EGO`
- `track-2` is `right_of CAV_EGO`

Visibility:

- `track-1` is `visible`
- `track-2` is `uncertain`

## Observed Example Output

Observed output from the real script run:

- `visible_count: 1`
- `visible_exists: True`
- `visible_ids: ['track-1']`
- `car_ids: ['track-1', 'track-2']`
- `front_of_ids: ['track-1', 'track-2']`
- `source_agent_ids: ['track-1', 'track-2']`
- `confidence_values: [('track-1', 0.9), ('track-2', 0.4)]`
- `confidence_comparisons: [('track-1', 'greater_than', 'track-2')]`
- `object_type_comparisons: [('track-1', 'not_comparable', 'track-2')]`
- `unknown_attribute_values: [('track-1', None), ('track-2', None)]`
- `provenance_traces: [('track-1', ('GT', 'CAV_EGO'), ('gt_track-1_0', 'ego-obs-1')), ('track-2', ('CAV_EGO',), ('ego-obs-2',))]`
- `empty_count: 0`
- `empty_exists: False`

This is a good deterministic query result because:

- normal cases work
- provenance is inspectable
- unsupported attributes do not crash
- empty queries do not hallucinate answers

## Why Phase 4 Matters

Phase 4 matters because it freezes the reasoning interface before any language model is introduced.

Without a deterministic query layer, later LLM-based reasoning would mix:

- graph access
- hidden assumptions
- free-form interpretation

That would make failures difficult to localize.

With Phase 4 in place, we can say:

- the graph is queried through fixed tools
- the tools are read-only
- the outputs are structured
- the semantics are inspectable

This is exactly the right setup for a later constrained reasoning layer.

## What We Can Now Claim

At this point, we can claim that the repository supports:

- deterministic scene-object selection
- deterministic filtering by type, relation, visibility, and provenance source
- deterministic count and existence operations
- structured attribute access
- structured pairwise comparison
- structured provenance tracing
- safe handling of empty selections
- safe handling of unsupported attributes

## What Still Remains Within Phase 4

Phase 4 is not fully complete yet, but the core query engine is already strong.

Useful next steps would be:

1. add a few more hand-authored QA-style query examples
- ideally around the initial supported QA subset

2. formalize a tiny v1 tool specification
- especially supported attributes and compare semantics

3. connect some of these tools into the existing real-scene/local-scene demos more systematically

These are now hardening and documentation steps rather than missing foundational functionality.

## Suggested One-Paragraph Summary

"In Phase 4, we built a deterministic read-only query layer over the scene graph so that graph reasoning becomes reproducible before any LLM is introduced. The current engine supports selection, filtering by type, relation, visibility, and provenance source, along with count, existence, structured attribute lookup, pairwise comparison, and provenance tracing. Tool outputs are structured and deterministic, and unsupported or empty cases fail safely rather than hallucinating values. This gives us the fixed reasoning substrate needed for later tool-constrained LLM reasoning."
