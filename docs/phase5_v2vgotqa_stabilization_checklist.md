# Phase 5: V2V-GoT-QA Stabilization Checklist

## Purpose

This note logs the concrete execution plan for Phase 5 before implementation begins.

The goal of Phase 5 is to turn the current V2V-GoT-QA-oriented prototype into a stable, benchmark-evaluable pipeline.

This phase is intentionally focused on **V2V-GoT-QA first**, not V2V-QA, because:

- the current end-to-end pipeline is already wired to V2V-GoT-QA-style inputs
- the cooperative processed asset path is now working
- the graph construction pipeline is already producing multi-agent scene graphs
- we need a reliable benchmark path before expanding to additional QA schemas

## Phase 5 Goal

By the end of Phase 5, we want to have:

- a stable V2V-GoT-QA evaluation path
- a clearly defined subset of supported question types
- reproducible prediction generation
- benchmark-ready answer formatting
- ego-only vs cooperative KG comparisons
- first ablation results
- documented failure cases and limitations

## Core Principle

Phase 5 is not about building a new graph pipeline from scratch.

It is about:

1. restricting benchmark scope intelligently
2. making the current cooperative graph path reliable
3. evaluating it in a controlled and repeatable way

## Architectural Shift: Why We Moved To An Orchestrator

The first version of `planning_awareness` used a mostly hard-coded priority flow:

- visible objects first
- then occluded objects
- then candidate fallbacks

That was useful for getting an initial benchmark path running, but it became too brittle once we started inspecting real divergence cases.

### Why the old routing was not enough

Qualitative inspection showed three concrete failure modes:

- a visible but weak candidate could outrank a more benchmark-aligned supported object
- cooperative and ego-only differences were hard to interpret because ranking logic was entangled with answer rendering
- it was difficult to test new decision methods without rewriting task logic

### Why the orchestrator was introduced

We therefore refactored `planning_awareness` into an orchestrator with separate stages:

1. candidate collection from the cooperative scene graph
2. ranker-specific scoring
3. decision-policy-specific selection
4. benchmark answer rendering

This gives us:

- modularity: scorers and selection policies can be swapped independently
- clean ablations: same graph, different decision logic
- lower risk of regressions: task routing no longer needs to change whenever scoring changes
- paper clarity: we can compare heuristic, risk-aware, energy-based, and later LLM ranking under the same evaluation path

### Current orchestrator structure

The planning-awareness stack now has three layers:

- `PlanningAwarenessOrchestrator`
  - owns candidate gathering and calls into the scorer and decision policy
- `PlanningAwarenessScorer`
  - assigns a score and rationale to each candidate object
- `PlanningAwarenessDecisionPolicy`
  - turns scored candidates into the final selected object set

This was the key architectural move that made the later ranker comparisons possible.

## Workstreams

### 1. Scope Lock

Decide which V2V-GoT-QA question types are in scope for Phase 5.

#### Deliverables

- a list of supported question types
- a list of deferred question types
- a short justification for each deferred type

#### Observed V2V-GoT-QA task inventory

Using the current `val` benchmark file (`v2v4real_3d_grounding_qa_dataset_v2vgot.json`), we observed:

- `qa_type_id=11`: `notable_objects`
- `qa_type_id=12`: `occluding_objects`
- `qa_type_id=13`: `invisible_objects`
- `qa_type_id=14`: `planning_awareness`
- `qa_type_id=15`: `object_motion_prediction`
- `qa_type_id=16`: `agent_motion_prediction`
- `qa_type_id=18`: `control_settings`
- `qa_type_id=19`: `future_trajectory`

Observed `val` counts:

- `notable_objects`: 3446
- `occluding_objects`: 3446
- `invisible_objects`: 3446
- `planning_awareness`: 3446
- `object_motion_prediction`: 6892
- `agent_motion_prediction`: 3446
- `control_settings`: 3446
- `future_trajectory`: 3446

#### Phase 5A supported first

- `notable_objects`
- `occluding_objects`
- `invisible_objects`
- `planning_awareness`

These are intentionally first because they align best with the current cooperative graph substrate:

- object tracks
- visibility facts
- occlusion reasoning
- near-trajectory filtering
- provenance-aware support

#### Phase 5B deferred initially

- `object_motion_prediction`
- `agent_motion_prediction`
- `control_settings`
- `future_trajectory`

These are deferred because they require stronger prediction or planning-oriented output layers than the current graph reasoning path provides.

#### Success criteria

- every supported question type has a clear graph-query interpretation
- no unsupported question type is silently handled

### 2. Question-to-Graph Mapping

Define exactly how each supported V2V-GoT-QA question type is answered from the graph.

#### Deliverables

- a question-type to query-template mapping
- answer schema for each supported question type
- normalization rules for output formatting

#### Example mapping pattern

- parse asking agent
- parse timestamp and future trajectory
- build cooperative scene
- select relevant objects
- apply visibility filter
- apply relation / trajectory / type filters
- render benchmark-compatible answer

#### Success criteria

- each supported question type maps to a deterministic graph pipeline
- answer rendering is predictable and testable

### 3. Scene Construction Stabilization

Harden the cooperative scene builder so benchmark runs behave consistently.

#### Deliverables

- stable cooperative root usage
- verified two-agent observation loading
- stable candidate pruning behavior
- stable visibility fact generation
- stable cross-agent support attachment behavior

#### Checks

- both-agent prediction loading remains active
- both-agent visibility files remain active
- graph density is reasonable across sampled frames
- no benchmark run silently falls back to non-cooperative roots

#### Success criteria

- repeated runs produce consistent graph-level outputs
- scene builder behavior is deterministic for the same sample

### 4. Evaluation Wrapper

Build the scriptable benchmark evaluation entrypoint.

#### Deliverables

- a V2V-GoT-QA evaluation driver
- per-sample prediction logging
- output serialization in a reproducible format
- aggregated metric reporting

#### Expected behavior

- iterate over benchmark samples
- load sample and processed assets
- build graph
- route to benchmark-specific graph query
- render answer
- save predictions
- compute summary metrics

#### Success criteria

- one command can run a selected split or subset
- outputs are saved in a repeatable structure

### 5. Internal Baselines

Establish the first controlled experimental comparisons.

#### Required baselines

- ego-only KG
- cooperative KG

#### Initial ablations

- cooperative KG without provenance-aware fusion
- cooperative KG without uncertainty-aware handling
- cooperative KG without cross-agent support attachment
- cooperative KG with simplified candidate retention or pruning
- planning-awareness ranker comparison:
  - `heuristic`
  - `relational_importance`
  - `risk_aware`
  - `energy_based`
  - later `llm`
- planning-awareness selection-policy comparison:
  - `default`
  - `top2`
  - `diverse_top2`

#### Success criteria

- every baseline uses the same evaluator and output format
- differences reflect method behavior, not evaluation mismatches

### Planning-Awareness Rankers

The current non-LLM planning-awareness rankers are:

#### `heuristic`

Purpose:

- transparent baseline

How it works:

- starts from object confidence
- adds bonuses for:
  - occlusion
  - GT backing
  - supported/confirmed tracks
  - cooperative provenance
- subtracts penalties for:
  - candidate status
  - uncertainty
  - conflict
  - trajectory distance

Interpretation:

- a general-purpose confidence-plus-support ranker
- easiest baseline to understand
- useful as a first benchmark point, but not explicitly safety-aware

#### `relational_importance`

Purpose:

- explicit “important or not” object scoring inspired by important-object-identification literature

Paper inspiration:

- Li et al., *Important Object Identification with Semi-Supervised Learning for Autonomous Driving* (ICRA 2022, arXiv:2203.02634)

How it works in this project:

- scores objects as explicit planning-awareness candidates rather than relying only on raw confidence
- emphasizes:
  - proximity to the future trajectory
  - visibility/occlusion state
  - whether the asking agent observed the object
  - support and cooperative provenance
- still penalizes:
  - candidate-only tracks
  - uncertainty
  - conflict

Interpretation:

- closer to “importance classification” than pure ranking-by-confidence
- currently more semantically motivated than the plain heuristic
- but in our pilot slice it still tended to leak extra `pred_candidate_*` objects into answers

#### `risk_aware`

Purpose:

- rank objects by an object-centric risk proxy for the current planned trajectory

Paper inspiration:

- Nyberg et al., *Risk-aware Motion Planning for Autonomous Vehicles with Safety Specifications* (IV 2021, DOI:10.1109/IV48863.2021.9575928)

How it works in this project:

- builds an evidence term from object confidence
- builds a violation-probability proxy from:
  - occlusion
  - uncertainty
  - conflict
- builds a severity proxy from:
  - trajectory proximity
  - GT/support evidence
- combines them into a risk term
- adds small support/GT bonuses
- penalizes candidate-only tracks

Interpretation:

- this is the best current non-LLM approximation of “what objects matter for safe driving right now?”
- it is the strongest ranker so far on the inspected planning-awareness cases

#### `energy_based`

Purpose:

- select a more coherent awareness set by discouraging redundant objects

Paper inspiration:

- Tian et al., *KLDrive: Fine-Grained 3D Scene Reasoning for Autonomous Driving based on Knowledge Graph* (arXiv:2603.21029)

How it works in this project:

- uses an energy-style unary score where lower energy corresponds to:
  - GT-backed evidence
  - supported tracks
  - cooperative support
  - occlusion relevance
- penalizes:
  - candidates
  - uncertainty
  - conflict
  - distance
- the paired decision policy then discourages selecting redundant nearby same-type objects

Interpretation:

- this is an adaptation of KLDrive’s energy-based reasoning idea, not a literal reproduction of their full scene-fact-construction module
- it improved set coherence, but it was not stronger than `risk_aware` on the current benchmark slice

#### `llm`

Purpose:

- future ranker for candidate relevance and ordering only

Current state:

- interface scaffold exists
- no real model call is wired yet

Intended role:

- rerank graph-selected candidates
- not replace the graph or the whole QA pipeline

### Planning-Awareness Selection Policies

The ranker and selection policy are now separate on purpose.

#### `default`

How it works:

- keep all candidates above threshold
- return up to the configured maximum, usually top-3

Behavior:

- highest recall
- but often keeps an extra third object that is not benchmark-aligned

#### `top2`

How it works:

- same ranking
- but only keep the top two selected objects

Behavior:

- improves precision by removing weaker third objects
- helpful when benchmark answers naturally focus on one invisible object plus one visible object

#### `diverse_top2`

How it works:

- explicitly prefers:
  - one best occluded object
  - one best visible object
- only fills from the global ranking if needed

Behavior:

- most aligned with the observed structure of many `planning_awareness` benchmark answers
- especially useful when benchmark references contain:
  - one occluded/invisible hazard
  - one visible nearby hazard

### 6. Pilot Evaluation

Run a small but representative subset before attempting full benchmark execution.

#### Deliverables

- pilot run over a small sample set
- error breakdown by question type
- notes on brittle parsing or reasoning paths

#### Suggested first run

- 20 to 50 samples
- mixed supported question types
- both ego-only and cooperative KG
- compare at least the non-LLM planning-awareness rankers on the same pilot slice

#### Success criteria

- identify whether failures come from parsing, graph construction, query logic, or answer rendering
- fix major issues before scaling up

### 7. Fuller Evaluation Pass

After pilot stabilization, run a larger subset or benchmark split.

#### Deliverables

- aggregated results for supported question types
- ego-only vs cooperative comparison
- first ablation comparison

#### Success criteria

- results are reproducible
- the evaluation can be re-run without manual intervention

## Current State

### What is now stable

- V2V-GoT-QA benchmark loading
- Phase 5A task routing
- cooperative and ego-only evaluation modes
- JSONL prediction export
- comparison tooling with semantic vs ordering difference separation
- planning-awareness ranker swapping
- planning-awareness selection-policy swapping

### What was improved during Phase 5

#### 1. Candidate-driven noise was reduced

Early `planning_awareness` outputs were often driven by `pred_candidate_*` tracks.

After the orchestrator refactor and scorer redesign:

- supported GT-backed tracks dominate much more often
- ranker behavior is inspectable via per-object scores and rationales

#### 2. Ordering noise was removed

Earlier comparisons showed many apparent differences that were only due to output order.

We fixed this by canonicalizing final answer order, so current comparison results now reflect real content differences rather than serialization noise.

#### 3. Current best non-LLM baseline was identified

From the current pilot experiments on the first `100` `planning_awareness` samples:

- `risk_aware` is the strongest non-LLM ranker so far
- `relational_importance` was weaker because it still allowed more candidate-style noise
- `energy_based` was coherent but not stronger than `risk_aware`
- `heuristic` remains the simplest transparent baseline

### Most important current quantitative observations

#### `heuristic` vs `risk_aware` after removing ordering noise

On the first `100` cooperative `planning_awareness` samples:

- exact-answer matches: `90/100`
- semantic differences: `10/100`
- ordering-only differences: `0`

Interpretation:

- the two methods are mostly similar
- the remaining differences are now meaningful ranking differences

#### `risk_aware` cooperative vs `risk_aware` ego-only

On the first `100` `planning_awareness` samples:

- exact-answer matches: `97/100`
- semantic differences: `3/100`
- ordering-only differences: `0`

Interpretation:

- under the best current non-LLM ranker, cooperation changes only a small number of answers on this slice
- the pipeline is now stable enough that those few differences are worth inspecting closely

#### `risk_aware` with `top3` vs `top2` / `diverse_top2`

On the same slice:

- `top2` and `diverse_top2` removed many weaker third-object selections
- this improved precision on key inspected samples such as `sample_id=95`

### Key qualitative result

For `sample_id=95`, the benchmark reference answer contains:

- one invisible/occluded car near `(-21.3, -0.5)`
- one visible car near `(-25.0, 0.2)`

With `risk_aware + diverse_top2`:

- cooperative prediction becomes `['1', '107']`

This is the cleanest planning-awareness answer we have produced so far for that case, and it is substantially more benchmark-aligned than the earlier top-3 outputs.

### Current best configuration

The strongest current classical configuration is:

- `planning_ranker = risk_aware`
- `planning_selection_policy = top2` or `diverse_top2`

At the moment:

- `top2` and `diverse_top2` behaved equivalently on the current `100`-sample pilot slice
- both were better aligned than the older default top-3 selection on key examples

### Current limitations

- we still do not compute official benchmark accuracy yet; current analysis is based on structured output comparison and qualitative reference alignment
- cooperative gain under the best current non-LLM setup is still modest on the current slice
- `planning_awareness` still relies on hand-designed object-level scoring rather than a learned or LLM-assisted relevance model
- only Phase 5A tasks are supported end-to-end

## Immediate Next Step

The next step is to keep the current best classical baseline fixed and compare it against an LLM reranker:

- classical baseline:
  - `risk_aware + top2` or `risk_aware + diverse_top2`
- next experiment:
  - `llm + top2`

This is now well motivated because:

- the pipeline is stable
- the ordering noise is gone
- the classical baseline is strong enough to be a meaningful comparison target

### 8. Paper-Facing Outputs

Prepare the artifacts that will feed directly into the paper or proposal.

#### Deliverables

- result tables
- ablation tables
- qualitative examples where cooperation helps
- failure case examples
- concise explanation of what is and is not supported yet

#### Success criteria

- at least one strong cooperative example is documented
- at least one failure mode is documented honestly

## Phase 5 Checklist

### Scope

- [x] Enumerate V2V-GoT-QA question types in the available split
- [x] Mark supported and deferred question types
- [x] Log scope decision in docs

### Mapping

- [ ] Create question-type to graph-query mapping
- [ ] Define answer format for each supported type
- [ ] Define output normalization rules

### Stabilization

- [ ] Confirm cooperative root is always selected
- [ ] Confirm both-agent observations load consistently
- [ ] Confirm both-agent visibility facts load consistently
- [ ] Verify deterministic scene construction on repeated runs
- [ ] Verify candidate pruning behavior is stable
- [ ] Verify cross-agent support attachment is stable

### Evaluation

- [x] Build evaluation driver for V2V-GoT-QA
- [ ] Save per-sample predictions
- [ ] Save per-sample debug traces when needed
- [ ] Aggregate metric outputs

### Baselines

- [x] Add ego-only KG mode
- [x] Add cooperative KG mode
- [ ] Add no-provenance ablation
- [ ] Add no-uncertainty ablation
- [ ] Add no-cross-agent-support ablation

### Pilot

- [ ] Run pilot subset
- [ ] Inspect successes and failures manually
- [ ] Fix parsing and answer-format issues
- [ ] Fix graph-query mismatches

### Benchmark Pass

- [ ] Run larger evaluation subset or split
- [ ] Produce ego-only vs cooperative comparison
- [ ] Produce first ablation comparison

### Reporting

- [ ] Write result summary
- [ ] Write qualitative example summary
- [ ] Write failure-mode summary
- [ ] Update paper-facing notes with actual findings

## Current Status Snapshot

This section captures the exact point reached so work can resume cleanly later.

### Implemented today

- Added a benchmark-level domain model for typed benchmark samples and predictions.
- Added a modular `V2VGoTQABenchmarkAdapter` so benchmark parsing and task classification stay separate from the reusable graph core.
- Added `scripts/inspect_v2vgotqa_tasks.py` to inventory the available V2V-GoT-QA task families.
- Confirmed the current V2V-GoT-QA `val` file has `31,014` samples and a stable `qa_type_id 11-19` family.
- Added a Phase 5A benchmark router for:
  - `notable_objects`
  - `occluding_objects`
  - `invisible_objects`
  - `planning_awareness`
- Added a Phase 5A evaluation service and CLI runner.
- Added a comparison script for cooperative vs ego-only JSONL prediction outputs.

### Pilot results observed so far

#### 1. `notable_objects`

- First `10` samples in both `cooperative` and `ego_only` mode produced the same empty answer:
  - `There is no notable object visible to you near your planned future trajectory.`
- This was useful as a smoke test, but not yet a meaningful differentiation result.

#### 2. `occluding_objects`

- First `10` samples produced non-trivial outputs in both modes.
- Over the first `100` validation samples:
  - `cooperative` and `ego_only` were identical
  - exact-answer matches: `100/100`
  - object-id matches: `100/100`
  - differing samples: `0`

Interpretation:

- the current `occluding_objects` routing is stable
- but it does not yet show cooperative gain on this slice

#### 3. `planning_awareness`

- Over the first `100` validation samples:
  - exact-answer matches: `85/100`
  - object-id matches: `85/100`
  - differing samples: `15/100`

Observed pattern:

- cooperative mode often adds `pred_candidate_*` object hypotheses
- ego-only often returns either fewer objects or no object
- some differences are additive
- some differences are substitutions between candidate IDs

Interpretation:

- this is the first Phase 5A task where cooperative and ego-only produce measurably different outputs
- however, many differences currently arise from candidate tracks rather than clearly confirmed objects
- the result is promising, but still noisy

### Most important takeaway

The Phase 5 benchmark execution path is now stable enough to compare cooperative vs ego-only behavior.

The first concrete divergence appears in `planning_awareness`, not `occluding_objects`.

So the next work should focus on:

- diagnosing whether cooperative-only candidate objects are useful or spurious
- inspecting differing sample IDs qualitatively
- tightening candidate handling before claiming benchmark improvement

### Recommended next step

Resume with qualitative inspection of the `planning_awareness` divergence cases, especially:

- `sample_id=43`
- `sample_id=65`
- `sample_id=69`
- `sample_id=95`

These are currently the most informative examples for understanding whether cooperative graph reasoning is helping in a meaningful way.

## What Is Out of Scope for Phase 5

To keep the phase focused, the following are not primary goals of Phase 5:

- adapting the full pipeline to V2V-QA
- creating a new synthetic dataset
- scaling beyond the two-agent cooperative setting
- introducing a new planner from scratch
- optimizing every benchmark task before basic stability is reached

## Exit Criteria

Phase 5 is complete when:

- V2V-GoT-QA support scope is explicit
- the cooperative graph pipeline runs reliably on supported question types
- ego-only and cooperative KG baselines are both runnable
- at least one ablation is runnable
- metrics can be reproduced on repeated runs
- we have benchmark-ready result summaries and qualitative examples

## Next Phase After Phase 5

If Phase 5 succeeds, the next phase can branch into one or more of:

- V2V-QA adaptation
- stronger graph fusion and uncertainty handling
- planning/task expansion
- controlled synthetic data generation for occlusion-heavy cases
- broader paper-facing experiments and comparisons

## Final Closeout

Phase 5 is now closed from an engineering and evaluation-stability perspective.

### What was completed

- A stable Phase 5A benchmark path for:
  - `notable_objects`
  - `occluding_objects`
  - `invisible_objects`
  - `planning_awareness`
- Cooperative vs `ego_only` evaluation support.
- Planning-awareness orchestration with swappable:
  - rankers
  - selection policies
- Local LLM integration with a self-hosted AWQ model.
- Full classical + LLM scenario sweeps across supported Phase 5 tasks.
- VM-side object-level scoring against benchmark reference answers.

### Final best-performing configurations

#### `planning_awareness`

Best object-level F1 on the first `100` validation samples:

- `risk_aware + top2`
- `risk_aware + diverse_top2`
- `llm + top2`

All three tied at:

- `F1 = 0.440`
- `Precision = 0.308`
- `Recall = 0.772`
- `Exact = 15/100`

Interpretation:

- the orchestrator plus `risk_aware` ranking and top-2 style selection produced the main gain
- the local AWQ LLM reranker is competitive and operational, but it does not exceed the best classical baseline on this slice

#### `notable_objects`

Best object-level F1 on the first `100` validation samples:

- `ego_only` variants: `F1 = 0.380`
- `cooperative` variants: `F1 = 0.326`

Interpretation:

- this task still uses deterministic router logic rather than task-specific orchestration
- the current cooperative preparation does not outperform `ego_only` on this slice

#### `occluding_objects`

Best object-level F1 on the first `100` validation samples:

- all variants tied at `F1 = 0.048`

Interpretation:

- this is the weakest current Phase 5 task
- occlusion-specific answer selection remains poorly aligned with the benchmark reference answers

#### `invisible_objects`

Best object-level F1 on the first `100` validation samples:

- all variants tied at:
  - `F1 = 0.727`
  - `Precision = 1.000`
  - `Recall = 0.571`
  - `Exact = 97/100`

Interpretation:

- this task is already strong
- recall is the main remaining limitation

### Relative position vs V2V-GoT target references

Using the current local object-level scoring layer:

- `notable_objects`
  - ours best: `0.380`
  - V2V-GoT `Q1 F1`: `52.5`
  - gap: about `14.5` F1 points
- `occluding_objects`
  - ours best: `0.048`
  - V2V-GoT `Q2 F1`: `30.1`
  - gap: about `25.3` F1 points
- `invisible_objects`
  - ours best: `0.727`
  - V2V-GoT `Q3 F1`: `44.0`
  - local score appears higher, but this should be interpreted cautiously because our scorer is still a local reproduction layer rather than the official benchmark script
- `planning_awareness`
  - ours best: `0.440`
  - V2V-GoT `Q4 F1`: `60.8`
  - gap: about `16.8` F1 points

Important caveat:

- these are not official paper-script reproductions yet
- they are local object-level scores derived by resolving reference answer coordinates back to graph object IDs on prepared cooperative scenes

### Commands used for Phase 5 closeout

#### 1. Run the full closeout sweep on the VM

```bash
cd /workspace/repos/kg_coop_drive

export KG_LOCAL_LLM_BASE_URL="http://127.0.0.1:8000"
export KG_LOCAL_LLM_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
export KG_LOCAL_LLM_API_KEY="local-token"
export KG_LOCAL_LLM_TIMEOUT_SECONDS="180"
export KG_LOCAL_LLM_MAX_TOKENS="128"

python3 scripts/run_phase5_closeout.py \
  --split val \
  --limit 100 \
  --full-sweep-all-tasks \
  --output-dir outputs/phase5_closeout_awq_full_sweep
```

#### 2. Score the closeout sweep against benchmark reference answers

```bash
python3 scripts/score_phase5_closeout.py \
  --manifest outputs/phase5_closeout_awq_full_sweep/phase5_closeout_manifest.json
```

#### 3. Read the scored summary on the VM

```bash
cat outputs/phase5_closeout_awq_full_sweep/phase5_scored_report.md
```

#### 4. Copy the output directory from the pod to local

```bash
bash scripts/copy_phase5_outputs_from_pod.sh
```

### Final Phase 5 conclusion

Phase 5 is now complete with respect to:

- pipeline stabilization
- benchmark routing for the supported QA subset
- cooperative vs `ego_only` comparison support
- planning-awareness ablations
- local LLM integration
- full-sweep reporting
- first object-level precision / recall / F1 scoring

The strongest Phase 5 result is:

- `planning_awareness` improved materially through the orchestrator refactor and `risk_aware + top2/diverse_top2`
- the local AWQ LLM ranker matched that best planning baseline but did not surpass it

The clearest remaining weakness is:

- `occluding_objects`

So Phase 5 should be considered closed, with `occluding_objects` and direct official-score reproduction noted as the main residual gaps.
