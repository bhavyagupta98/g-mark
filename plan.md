# Cooperative KG Driving Plan

## Purpose

This document is the working implementation plan for the cooperative knowledge graph driving project.

The main goal is to build the system iteratively, understand each component before stacking new complexity on top, and define concrete checkpoints before moving to the next phase.

The project idea is:

- Build KLDrive-style grounded scene facts locally on each CAV.
- Exchange structured facts across vehicles.
- Fuse them into a provenance-aware cooperative knowledge graph.
- Run constrained reasoning over the graph instead of free-form neural reasoning.
- Evaluate whether this improves reliability, grounding, and robustness on cooperative driving QA tasks.

This plan is designed to prevent us from skipping directly to the full system before the foundations are stable.

## Guiding Principles

- Build one layer at a time.
- Test every module independently before integrating it.
- Prefer deterministic baselines before adding LLM reasoning.
- Keep all intermediate representations inspectable.
- Treat object association and graph fusion as first-class problems.
- Do not claim improvements from the reasoning layer if the graph layer is not validated first.

## Project Structure

The work naturally breaks into these components:

1. Dataset and task definition
2. Canonical scene fact schema
3. Local fact construction per CAV
4. Structured V2V packet representation
5. Coordinate and temporal alignment
6. Cross-agent object association
7. Graph fusion and conflict resolution
8. Visibility and occlusion reasoning
9. Provenance and uncertainty tracking
10. Deterministic graph query engine
11. Tool-constrained LLM reasoning
12. QA benchmark integration
13. Evaluation, baselines, and ablations
14. Visualization, debugging, and experiment tracking

## Existing Assets We Can Reuse

### From V2V-GoT

- V2V4Real data plumbing and multi-agent setup
- QA generation pipeline
- Existing QA task structure and graph-style chained question setup
- Evaluation scripts for answer parsing and grounding-style metrics
- Tracking and perception infrastructure in `DMSTrack` and `OpenCOOD`

Useful references:

- [V2V-GoT README](/Users/bhavya/Desktop/ms_projects/V2V-GoT/README.md:1)
- [temp_qa_generation.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/temp_qa_generation.py:25)
- [eval_v2v4real_3d_grounding.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/eval_v2v4real_3d_grounding.py:20)

### From KLDrive-style code in `auto_drive_copy`

- Tool-constrained reasoning pattern
- Scene binding and typed tool invocation flow
- Selector, relation, count, existence, and status/type query patterns
- Safety-oriented execution where the model can only reason through tools

Useful references:

- [agent.py](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/code/agent.py:36)
- [agent_readme.md](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/code/agent_readme.md:1)
- [tools.py](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/tools.py:1)

## High-Level Strategy

We should not start with the full cooperative KG plus LLM loop.

The safest order is:

1. Define the canonical data schema.
2. Build a single-agent graph first.
3. Build a deterministic query engine first.
4. Build cooperative fusion without any LLM.
5. Add provenance and visibility reasoning.
6. Integrate QA evaluation.
7. Only then add the LLM planner over typed tools.

This order limits unknowns and makes failures localizable.

---

## Phase 0: Setup and Scoping

### Goal

Create a stable research and engineering starting point so the rest of the work is reproducible and understandable.

### Scope

- Confirm available datasets, code, and checkpoints.
- Decide the first benchmark slice to use.
- Decide what will be simulated versus what will be treated as real communication.
- Set up repository structure for the new project.

### Deliverables

- A short inventory of reusable assets from V2V-GoT and `auto_drive_copy`
- A first-pass repository layout
- A written definition of the first narrow prototype task

### Recommended first prototype task

"Given two cooperating agents in a single frame, build a fused graph and answer a small set of deterministic QA queries about object existence, count, and relative position."

### Evaluation before moving on

- We can state exactly which data files and benchmark slice we are using.
- We can explain the input and output of the prototype in one paragraph.
- The repo has an agreed folder structure for data, KG, reasoning, eval, and tests.

### Exit criteria

Do not move to Phase 1 until:

- the prototype task is narrow and written down
- the input data source is identified
- the expected output format is identified

### Risks

- Starting too broadly
- Mixing dataset exploration with method design
- Treating V2V communication as networking instead of structured data exchange

---

## Phase 1: Dataset and Task Definition

### Goal

Understand exactly what supervision, annotations, perception outputs, and QA tasks are available, and define what subset we can support first.

### Scope

- Inspect V2V-GoT / V2V4Real assets
- Identify what object-level and relation-level data are available
- Identify which QA tasks can be answered from structured graphs
- Define train/val/test protocol for our early experiments

### Questions to answer

- What raw inputs do we have per agent and per frame?
- What detections/tracks/GT boxes are available?
- What agent IDs, poses, timestamps, and transforms are available?
- Which QA tasks are object grounding, counting, relation, prediction, or planning oriented?
- Which tasks can be mapped cleanly to graph queries?

### Deliverables

- Dataset note in markdown
- Mapping from benchmark QA types to reasoning categories
- A shortlist of initial supported task types

### What we should support first

Start with:

- object existence
- object count
- relative position
- basic visibility-aware questions

Defer:

- full planning support
- complex long-horizon prediction
- free-form explanation generation

### Evaluation before moving on

- We can parse one sample from the dataset and inspect all fields we need.
- We can map at least 10 QA examples into structured reasoning targets.
- We know which QA tasks are in-scope for the first prototype and which are not.

### Exit criteria

Do not move to Phase 2 until:

- at least one sample frame can be loaded successfully
- the initial QA subset is clearly chosen
- task-to-tool mapping is plausible for the chosen subset

### Risks

- Choosing tasks that require prediction/planning before graph basics are working
- Assuming the dataset contains relations explicitly when they may need to be derived

---

## Phase 2: Canonical Scene Fact Schema

### Goal

Define one canonical structured representation that every module will use.

### Scope

- Design typed object, relation, visibility, provenance, and uncertainty records
- Define how a local graph differs from a fused cooperative graph
- Define object IDs and source observation IDs

### Core schema candidates

#### FrameContext

- `scene_id`
- `frame_id`
- `timestamp`
- `ego_agent_id`
- `global_reference_frame`
- `active_agent_ids`

#### ObjectFact

- `object_id`
- `category`
- `bbox_3d`
- `position`
- `velocity`
- `yaw`
- `confidence`
- `timestamp`
- `source_agent_ids`
- `source_observation_ids`
- `is_fused`

#### RelationFact

- `subject_id`
- `relation_type`
- `object_id`
- `confidence`
- `source_agent_ids`
- `derivation_method`

#### VisibilityFact

- `agent_id`
- `object_id`
- `visibility_status`
- `confidence`
- `evidence`

#### ProvenanceFact

- `fused_object_id`
- `contributing_observations`
- `contributing_agents`
- `fusion_method`
- `last_updated`

### Design requirements

- Each fact must be inspectable.
- Every fused object must retain provenance.
- Every confidence value must have meaning.
- Time and frame conventions must be explicit.

### Deliverables

- Schema spec in code and markdown
- Example serialized scene file
- Validation tests for schema compliance

### Evaluation before moving on

- We can represent one local agent scene and one fused scene using the same schema family.
- We can serialize and deserialize without information loss.
- Provenance and confidence are carried through the representation.

### Exit criteria

Do not move to Phase 3 until:

- schema fields are frozen for the first prototype
- sample JSON examples exist
- validation tests pass

### Risks

- Overcomplicating the schema too early
- Making the schema dependent on one specific dataset quirk
- Forgetting time alignment and provenance at schema level

---

## Phase 3: Single-Agent Local Graph Construction

### Goal

Prove that we can build a grounded local graph correctly from one agent before adding cooperative fusion.

### Scope

- Convert one agent's detections or GT into object nodes
- Derive basic relations from geometry
- Optionally derive simple visibility labels
- Store everything in the canonical schema

### What to implement

- Local scene loader
- Local fact builder
- Relation extraction module
- Graph serializer
- Visualization helper for inspecting a local scene graph

### Start simple

Initial relations can be:

- `front_of`
- `left_of`
- `right_of`
- `behind`
- `near`

Initial object states can be:

- category
- position
- velocity if available
- confidence

### Deliverables

- Single-agent graph builder
- Unit tests on relation correctness
- Visual inspection outputs for a few frames

### Evaluation before moving on

- For a few sample frames, local graph objects match the source annotations or detections.
- Derived relations are geometrically sensible.
- We can answer simple deterministic queries from the local graph.

### Suggested tests

- object count matches source input
- relation extraction works for hand-checked examples
- ego-relative position bins are stable
- graph export is deterministic

### Exit criteria

Do not move to Phase 4 until:

- local graph construction is stable on a small validation slice
- simple local queries are correct
- relation extraction errors are understood

### Risks

- Relation definitions are inconsistent
- Ego frame and global frame get mixed
- Query logic depends on hidden assumptions not encoded in the graph

---

## Phase 4: Deterministic Query Engine

### Goal

Build the reasoning substrate before any LLM is added.

### Scope

- Implement typed read-only graph operations
- Reuse the KLDrive-style idea of constrained access through tools
- Ensure every answer is reproducible from the graph state

### Initial tool set

- `select_objects`
- `filter_by_type`
- `filter_by_relation`
- `filter_by_visibility`
- `filter_by_source_agent`
- `count`
- `exists`
- `get_attribute`
- `compare`
- `trace_provenance`

### Design rules

- Tools must be side-effect free.
- Tools must fail safely.
- Tools must return structured outputs.
- Tool semantics must be fixed and documented.

### Deliverables

- Query engine module
- Tool schemas
- Unit tests for each tool
- A few hand-authored graph query examples

### Evaluation before moving on

- We can answer at least 10 hand-picked QA examples without any LLM.
- Tool outputs are stable and interpretable.
- Invalid queries fail safely rather than hallucinating.

### Exit criteria

Do not move to Phase 5 until:

- the deterministic query engine can handle the initial QA subset
- query traces are easy to inspect
- tool definitions are frozen for v1

### Risks

- Tool semantics are too vague
- Query engine silently infers facts not present in the graph
- Tools are designed around language instead of graph structure

---

## Phase 5: Structured V2V Packet Representation

### Goal

Define what each CAV shares with others in the cooperative setting.

### Scope

- Package local facts into a communication-ready scene packet
- Decide what is transmitted and what is recomputed locally
- Keep it structured and compact

### Packet contents

- agent ID
- timestamp
- ego pose
- local object facts
- local relation facts
- visibility facts if available
- uncertainty and confidence values

### Design rule

For the first prototype, V2V communication means serialized structured fact exchange, not real networking.

### Deliverables

- Packet schema
- packet encoder/decoder
- tests for packet consistency

### Evaluation before moving on

- A local graph can be turned into a packet and reconstructed without losing critical fields.
- Two agents' packets can be loaded together for the same frame.

### Exit criteria

Do not move to Phase 6 until:

- packet schema is stable
- packet serialization is tested
- multi-agent sample loading works

### Risks

- Sending too much raw data too early
- Tying packet structure too tightly to one model or one benchmark

---

## Phase 6: Coordinate and Temporal Alignment

### Goal

Make sure that facts from different agents live in a comparable shared frame before fusion.

### Scope

- agent-to-global transforms
- frame-to-frame timestamp alignment
- stale packet handling
- alignment confidence or validity flags

### Why this matters

If alignment is wrong, object association and fusion will fail even if everything else is correct.

### Deliverables

- transform utilities
- temporal alignment utilities
- tests with known correspondences

### Evaluation before moving on

- The same physical object observed from two agents appears in roughly the same global location after transform.
- Time offsets are explicitly handled or bounded.
- Misaligned packets are detectable.

### Suggested tests

- transformed boxes visually align
- cross-agent distance between matched GT objects is within threshold
- stale updates lower confidence or get filtered

### Exit criteria

Do not move to Phase 7 until:

- geometric alignment works on a small sample set
- timestamp policy is defined
- failure cases are visible in logs or debug views

### Risks

- Hidden coordinate-frame mistakes
- Mixing ego-relative and world-relative coordinates
- Ignoring asynchronous observations

---

## Phase 7: Cross-Agent Object Association

### Goal

Match observations from different agents that refer to the same real-world entity.

### Scope

- spatial matching
- motion consistency
- category consistency
- association confidence
- unmatched and ambiguous cases

### Suggested v1 association policy

- spatial distance threshold
- category agreement
- velocity consistency if available
- optional Hungarian matching for one-to-one assignment

### Deliverables

- association module
- association debug outputs
- tests with hand-checked examples

### Evaluation before moving on

- Same-object matches are mostly correct on a small benchmark slice.
- Ambiguous cases are marked uncertain instead of forced into wrong matches.
- Unmatched objects remain separate.

### Metrics to inspect

- association precision
- association recall
- duplicate rate after fusion
- over-merge rate
- under-merge rate

### Exit criteria

Do not move to Phase 8 until:

- association behavior is understandable on visualized examples
- basic metrics are reported
- common failure modes are known

### Risks

- This is one of the highest-risk modules in the project.
- Fusion quality will be capped by association quality.
- Wrong identity merges can poison everything downstream.

---

## Phase 8: Cooperative Graph Fusion and Conflict Resolution

### Goal

Construct a locally maintained cooperative graph from multiple agents' packets.

### Scope

- merge associated objects
- retain unmatched objects
- combine attributes with uncertainty
- resolve conflicting observations
- track provenance for every fused node

### Suggested fusion policy for v1

- confidence-weighted averaging for numeric attributes
- keep contributing observations in provenance
- maintain uncertainty for conflicting attributes
- avoid silent overwrites

### Optional later extension

- KLDrive-inspired consistency or energy-based scoring for scene coherence

### Deliverables

- fusion module
- fused KG output format
- provenance tracking module
- fused scene visualizer

### Evaluation before moving on

- Fused objects are inspectable and list contributing agents.
- Conflicts are represented rather than hidden.
- Fused graph improves scene completeness compared to single-agent view on controlled examples.

### Metrics to inspect

- node count before and after fusion
- number of duplicates remaining
- number of conflicting attributes
- provenance coverage

### Exit criteria

Do not move to Phase 9 until:

- fused graph is stable on a small multi-agent slice
- provenance is preserved end to end
- fusion errors are understandable

### Risks

- Over-averaging conflicting observations
- Losing uncertainty after fusion
- Building a graph that looks clean but is semantically wrong

---

## Phase 9: Visibility and Occlusion Modeling

### Goal

Represent what each agent can or cannot see, since cooperative reasoning depends on asymmetric visibility.

### Scope

- visible vs occluded vs uncertain labels
- per-agent visibility state
- optional reasons or evidence
- visibility-aware filters in queries

### Why this matters

This is one of the main differences between single-agent KLDrive-style reasoning and cooperative driving. Different agents observe different subsets of the world.

### Deliverables

- visibility fact builder
- visibility-aware query support
- debug examples showing asymmetric observation

### Evaluation before moving on

- We can represent that one agent saw an object and another did not.
- Visibility filters change query results in sensible ways.
- Occlusion-specific QA examples can be answered from the graph.

### Exit criteria

Do not move to Phase 10 until:

- visibility facts exist in the graph
- at least a few occlusion examples are validated manually

### Risks

- Treating absence of observation as absence of object
- Overstating confidence in invisibility labels

---

## Phase 10: Provenance and Uncertainty-Aware Querying

### Goal

Make the graph not only queryable, but trustworthy and auditable.

### Scope

- provenance tracing
- confidence-aware answers
- support for uncertain or ambiguous outputs
- explicit reporting when the graph does not support a confident answer

### Deliverables

- provenance query tools
- confidence-aware response utilities
- audit trail for graph answers

### Evaluation before moving on

- We can answer "which agent observed this object?"
- We can answer "is this answer uncertain and why?"
- Query traces show which nodes and edges were used.

### Exit criteria

Do not move to Phase 11 until:

- provenance is available in reasoning results
- uncertainty is exposed rather than hidden

### Risks

- Presenting fused facts as certain truth
- Not distinguishing unsupported answers from negative answers

---

## Phase 11: QA Adapter and Benchmark Integration

### Goal

Connect the cooperative KG and deterministic reasoning engine to actual benchmark questions.

### Scope

- map benchmark QA formats into graph queries
- convert query outputs back into benchmark answer format
- support a first subset of V2V-GoT-style tasks

### Deliverables

- QA adapter
- benchmark-specific parser
- small benchmark runner for the initial subset

### Evaluation before moving on

- At least a small subset of benchmark questions can be answered automatically.
- Answers are derived from tool traces, not free-form guessing.
- Failure cases are classifiable as graph failure, query failure, or unsupported task.

### Exit criteria

Do not move to Phase 12 until:

- deterministic benchmark evaluation works on the selected subset
- unsupported questions are clearly flagged

### Risks

- Trying to support all QA types too early
- Forcing graph methods onto questions that need prediction or planning not yet implemented

---

## Phase 12: Tool-Constrained LLM Reasoning Layer

### Goal

Add the LLM only as a planner and interface over graph tools, not as the source of truth.

### Scope

- define tool schemas
- connect planner to graph executor
- capture full reasoning traces
- keep execution bounded and auditable

### Design rules

- the LLM may plan tool use
- the LLM may summarize results
- the LLM may not invent scene facts
- the final answer must be grounded in tool outputs

### Deliverables

- tool-calling reasoning loop
- prompt design
- trace logger
- evaluation comparing deterministic direct execution vs LLM-planned execution

### Evaluation before moving on

- Tool execution success rate is high.
- Invalid tool use is bounded and inspectable.
- LLM-added reasoning does not degrade correctness on supported tasks.

### Metrics to inspect

- tool call validity rate
- tool execution success rate
- unsupported query rate
- hallucination or ungrounded answer rate

### Exit criteria

Do not move to large-scale experiments until:

- tool loop is stable
- answers remain grounded
- planner errors are understood

### Risks

- The LLM introduces unnecessary instability
- Tool schemas are underspecified
- The system appears grounded but still leaks unsupported inferences

---

## Phase 13: Full Evaluation

### Goal

Evaluate whether the cooperative KG plus constrained reasoning improves correctness, grounding, and robustness.

### Core evaluation axes

- QA accuracy
- grounding correctness
- hallucination or ungrounded-answer rate
- tool execution success
- robustness under occlusion
- robustness under missing or conflicting observations
- interpretability and auditability of intermediate states

### Suggested evaluation groups

#### Group A: Graph quality

- local graph quality
- association quality
- fusion quality
- provenance completeness
- visibility labeling quality

#### Group B: Reasoning quality

- deterministic tool-query correctness
- constrained LLM correctness
- unsupported-query handling
- reasoning trace validity

#### Group C: Robustness

- partial observability
- occlusion-heavy cases
- dropped-agent cases
- stale or noisy observation cases

### Deliverables

- evaluation scripts
- result tables
- per-task breakdowns
- error analysis notes

### Exit criteria

This phase is ongoing, but do not write strong claims until:

- multiple baselines are run
- results are broken down by scenario
- error analysis has been done

---

## Phase 14: Baselines and Ablations

### Goal

Isolate which parts of the system are actually responsible for performance and reliability.

### Required baselines

- single-agent local graph only
- naive multi-agent union without proper fusion
- fused graph without provenance
- fused graph without visibility reasoning
- deterministic graph queries
- unconstrained LLM or free-form answering baseline if feasible
- V2V-GoT or V2V-LLM baseline if runnable

### Required ablations

- no graph refinement vs with refinement
- no provenance vs provenance-aware graph
- no visibility facts vs visibility-aware reasoning
- single-agent vs multi-agent
- deterministic executor vs LLM planner over same tools
- GT or oracle association vs learned or heuristic association

### Evaluation before claiming novelty

- We can show that gains do not come only from stronger perception inputs.
- We can show whether provenance and visibility actually matter.
- We can show whether the LLM helps or only adds overhead.

---

## Phase 15: Visualization, Debugging, and Analysis

### Goal

Make the system understandable during development.

### What to build

- local graph visualizer
- fused graph visualizer
- cross-agent object association viewer
- provenance trace viewer
- QA execution trace viewer

### Evaluation before relying on the system

- For any wrong answer, we can inspect the graph and locate the failure.
- We can tell whether the error came from perception, fusion, tool logic, or planning.

### Risks

- Debugging without visibility into intermediate states
- Mistaking benchmark score changes for real system understanding

---

## Phase 16: Experiment Tracking and Reproducibility

### Goal

Ensure results can be reproduced and compared fairly.

### What to track

- dataset split
- model version
- graph schema version
- tool version
- fusion parameters
- association thresholds
- prompt version
- evaluation protocol

### Deliverables

- config files
- run logs
- saved result artifacts
- experiment manifest

### Evaluation before final reporting

- We can rerun the same experiment and obtain consistent results.
- We can compare ablations fairly under the same setup.

---

## Recommended First 2-Week Milestone

### Objective

Build the smallest end-to-end version that validates the main idea without full complexity.

### Target prototype

- Load a small slice of V2V-GoT/V2V4Real data
- Build local graphs for two agents
- Exchange structured packets
- Fuse them into a cooperative graph
- Run deterministic queries for a tiny QA subset
- Compare single-agent and fused-graph answers on a few occlusion examples

### Must-have outputs by the end of this milestone

- schema implementation
- one local graph builder
- one fusion module
- one deterministic query engine
- one mini evaluation script
- one visualization or debug script

### Success criteria

- We can demonstrate at least a few cases where fused graph reasoning recovers information unavailable to a single agent.
- We can inspect why the system succeeds or fails.
- We have not yet added an LLM unless deterministic querying is already working.

---

## Recommended Folder Layout

```text
kg_coop_drive/
  plan.md
  README.md
  configs/
  data/
  notebooks/
  scripts/
  src/
    kg_coop_drive/
      data/
      schema/
      local_graph/
      packets/
      alignment/
      association/
      fusion/
      visibility/
      reasoning/
      qa/
      eval/
      viz/
  tests/
```

## Suggested Initial File List

```text
src/kg_coop_drive/schema/types.py
src/kg_coop_drive/schema/validate.py
src/kg_coop_drive/data/v2vgot_adapter.py
src/kg_coop_drive/local_graph/build_local_graph.py
src/kg_coop_drive/packets/scene_packet.py
src/kg_coop_drive/alignment/transforms.py
src/kg_coop_drive/association/match_objects.py
src/kg_coop_drive/fusion/fuse_graph.py
src/kg_coop_drive/reasoning/query_engine.py
src/kg_coop_drive/reasoning/tools.py
src/kg_coop_drive/qa/benchmark_adapter.py
src/kg_coop_drive/eval/run_eval.py
src/kg_coop_drive/viz/debug_scene.py
tests/
```

## What Not To Do Too Early

- Do not start with full LLM reasoning.
- Do not start with all QA types.
- Do not treat missing observation as object absence.
- Do not hide uncertainty in the fused graph.
- Do not skip visualization and debug tooling.
- Do not claim safety benefits before measuring failure modes.

## Immediate Next Step

The next concrete step should be:

1. Create the project folder layout.
2. Define the canonical schema in code.
3. Load one V2V-GoT sample and convert it into that schema.
4. Implement 3 to 5 deterministic graph queries.
5. Validate them on a few hand-picked examples before building fusion.

That will give us the first real foundation for the rest of the project.
