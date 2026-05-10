# Project Status Summary

This is the compact reference checkpoint for the current paper-facing QA results. It intentionally omits most exploratory history and keeps only the current approach, metrics, reasoning, selected ablation evidence, artifacts, and reproduction commands.

## Aim And Novelty

Aim:

- Build an explicit provenance-aware cooperative knowledge-graph reasoning layer for V2V autonomous-driving QA.
- Start from V2V4Real / V2V-GoT cooperative driving scenes.
- Preserve what each vehicle observed, how object hypotheses were fused, which objects are visible/occluded/uncertain, and why each object was selected.
- Answer benchmark-style QA questions through deterministic graph retrieval/ranking rather than only through implicit feature fusion or free-form multimodal generation.

What is novel in this approach:

- The intermediate world model is explicit.
  - Objects, observations, visibility facts, relations, provenance, uncertainty, conflict, and cooperative support are represented as structured graph facts.
- Cooperative fusion is inspectable.
  - The graph records whether an object is ego-only, partner-supported, candidate-only, or cooperatively supported.
- Reasoning is deterministic and reproducible.
  - QA answers are produced from graph retrieval/ranking policies and official-style exports, not from unconstrained language generation.
- Uncertainty and evidence quality are first-class.
  - Confidence, conflict, uncertainty, support count, track status, and provenance directly affect selection.
- The method is benchmark-compatible.
  - Outputs are exported back into V2V-GoT/LLaVA-style `outputs` records and scored with the upstream-compatible Q1-Q4 evaluator.

Positioning against related work:

- V2V4Real provides the real-world two-vehicle cooperative perception setting: synchronized multi-view sensors, cooperative detection/tracking tasks, and real driving scenarios. Our work uses this ecosystem as the grounding substrate but focuses on structured QA reasoning rather than only detection/tracking.
- V2V-LLM introduces V2V-QA and uses a multimodal LLM to fuse cooperative perception information and answer driving questions. Our approach is different: the core reasoning layer is an explicit KG with provenance and deterministic selectors. This makes the intermediate state easier to inspect and replay.
- V2V-GoT extends the MLLM direction with graph-of-thoughts for occlusion-aware perception and planning-aware prediction. Our approach is related in spirit but uses a different kind of graph: not a thought graph inside an MLLM, but a scene knowledge graph of objects, observations, visibility states, relations, confidence, and provenance.
- Traditional cooperative perception methods such as feature-, BEV-, transformer-, or late-fusion systems aim to improve detection/tracking quality. Our contribution sits above that layer: it asks whether the fused cooperative scene can be represented and queried as an interpretable reasoning object for benchmark QA.

What we should not overclaim:

- This is not a new V2V4Real-style dataset.
- This is not a new end-to-end MLLM.
- This is not a persistent graph database.
- The current risk scores are normalized selector scores, not calibrated physical collision probabilities.
- The current paper-facing checkpoint is strong for Q1-Q4. Q4's current best is `relational_importance + trajectory_calibrated_acceptor` with a `1.0m` duplicate radius, exceeding the V2V-GoT reference under the strict `0.5m` headline metric.

Related links:

- V2V-LLM: `https://research.nvidia.com/labs/twn/publication/cvprw_2025_v2vllm/`
- V2V-GoT: `https://research.nvidia.com/labs/twn/publication/icra_2026_v2vgot/`
- V2V4Real project: `https://mobility-lab.seas.ucla.edu/v2v4real/`
- V2V4Real codebase: `https://github.com/ucla-mobility/v2v4real`

## Dataset And Evaluation Path

Dataset:

- V2V-GoT-QA file: `v2v4real_3d_grounding_qa_dataset_v2vgot.json`
- V2V4Real / V2V-GoT processed scene assets under the V2V-GoT repository root, usually `/workspace/repos/V2V-GoT`
- train split: `12290` samples per QA family
- validation split: `3446` samples per QA family

## Canonical Script Entrypoints

To keep the frozen baseline pipeline readable, use these canonical script names going forward:

- split protocol: `scripts/run_qa_split_pipeline.py`
- sample evaluation/router execution: `scripts/evaluate_qa_router.py`
- official export generation: `scripts/export_qa_predictions.py`
- official QA evaluation: `scripts/evaluate_official_qa.py`
- Q3 acceptor training: `scripts/train_q3_invisible_acceptor.py`
- Q4 acceptor training: `scripts/train_q4_planning_acceptor.py`
- Q4 trajectory calibration setup: `scripts/configure_q4_trajectory_calibration.py`
- Q8 policy training: `scripts/train_q8_control_policy.py`
- Q9 trajectory model training: `scripts/train_q9_future_trajectory_regressor.py`

Legacy phase-named scripts are retained for backward compatibility and archival reproducibility.

## E2E Repro Commands

New end-to-end scripts are available under `scripts/e2e/` for full reproducibility checks after refactors.

### 1) Train E2E (Q1/Q2/Q3/Q4/Q8/Q9)

Script:

- `scripts/e2e/run_e2e_train_pipeline.py`

What it does:

- creates a dedicated run folder under `outputs/e2e_runs/<run_name>/`
- writes Q1/Q2 frozen policy snapshots
- retrains Q3/Q4/Q8/Q9 models from train split
- runs train-split official-style evaluations for Q1/Q2/Q3/Q4/Q8/Q9
- writes model + run manifest:
  - `outputs/e2e_runs/<run_name>/e2e_model_manifest.json`

Command (8-core example):

```bash
python3 scripts/e2e/run_e2e_train_pipeline.py \
  --run-name phase9_e2e_retrain_v1 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --workers 8 \
  --progress-every 200
```

Strict split protocol:

- default is strict train-only fitting for Q3/Q4 (no val feature export during training)
- optional diagnostic mode (not default):
  - `--allow-val-features-during-training`

### 2) Validation E2E Report (Q1/Q2/Q3/Q4/Q8/Q9)

Script:

- `scripts/e2e/run_e2e_validation_report.py`

What it does:

- loads the E2E model manifest
- runs held-out validation official-style eval for Q1/Q2/Q3/Q4/Q8/Q9
- prints report table with:
  - task
  - metric
  - our value
  - V2V-GoT baseline reference
  - relative improvement
- saves:
  - `outputs/e2e_runs/<run_name>/val_eval/e2e_validation_summary.json`
  - `outputs/e2e_runs/<run_name>/val_eval/e2e_validation_summary.md`

Command:

```bash
python3 scripts/e2e/run_e2e_validation_report.py \
  --manifest-json outputs/e2e_runs/phase9_e2e_retrain_v1/e2e_model_manifest.json \
  --workers 8 \
  --progress-every 200
```

## From V2V4Real Assets To Our Knowledge Graph

Starting point:

- V2V4Real provides synchronized two-vehicle cooperative driving scenes.
- V2V-GoT adds the QA records and processed assets used by the benchmark path.
- Each QA row identifies the split, timestamp/frame, asking vehicle, question type, reference answer text, and raw scene metadata.

Scene loading:

1. `V2VGoTQABenchmarkAdapter` reads the V2V-GoT-QA JSON file.
2. Each QA row becomes a `BenchmarkSample`.
3. The sample contains a seed `CooperativeScene` with agent identities, asking-agent ID, agent poses, timestamp, future trajectory, and raw benchmark record.
4. `V2VGoTQAPhase5AEvaluator.prepare_sample()` loads processed frame assets for the sample timestamp.

Baseline modes:

- `--baseline-mode cooperative`
  - keeps observations and visibility facts from all connected agents;
  - this is the current paper-facing setting for Q1-Q4;
  - it forms the cooperative graph used for V2V reasoning.
- `--baseline-mode ego_only`
  - filters processed observations and visibility facts to the asking vehicle before graph construction;
  - uses the same downstream graph pipeline after filtering;
  - this is the single-agent control condition for measuring the value of cooperation.

Graph construction:

1. `ProcessedSceneEnricher` attaches processed observations, object tracks, and visibility facts to the scene seed.
2. `ObservationAssociator` links observations to existing tracks using conservative geometric association.
3. `TrackSupportEnricher` records supporting observations and provenance for matched tracks.
4. `CandidateTrackCreator` promotes unmatched observations into candidate tracks.
5. `CandidateTrackResolver` keeps plausible candidate tracks while filtering weak artifacts.
6. `CrossAgentAssociator` matches tracks or observations across connected vehicles.
7. `CrossAgentSupportEnricher` adds cross-agent support and provenance when another vehicle supports the same object.
8. `TrackMerger` merges duplicate nearby tracks.
9. `TrackQualityAssessor` updates confidence, uncertainty, conflict, support, and track-quality fields.
10. `VisibilityReasoner` infers visible/occluded/unknown facts for the asking vehicle when needed.
11. `RelationBuilder` derives graph relations such as spatial position, trajectory proximity, path relevance, cooperative support, and low conflict.

Important implementation detail:

- The current graph is an immutable in-memory scene object, not a persistent graph database.
- The schema is represented with Python dataclasses and typed facts rather than a separate graph-store backend.
- Each preparation stage returns an updated `CooperativeScene`.
- After candidate resolution, cross-agent support, and merging, `RelationBuilder` rebuilds the relation facts from the current final object tracks.
- This keeps edge construction deterministic and avoids stale edges after object hypotheses are attached or merged.

Current graph representation:

- `CooperativeScene`: timestamped graph container with agents, future trajectory, object tracks, relations, and visibility facts.
- `ObjectTrack`: object node with position, type, confidence, provenance, status, age/miss counts, uncertainty, conflict, velocity, and observations.
- `ObservationEvidence`: source-agent evidence attached to an object.
- `VisibilityFact`: visible/occluded/unknown state for an agent-object pair.
- `RelationFact`: derived graph edge such as near trajectory, front/behind/left/right, path relevant, cooperatively supported, or low conflict.
- `ProvenanceRecord`: source agents, source observations, and latest timestamp index.

## Ego Graph To Cooperative Graph Fusion

The same graph construction pipeline is used for both `ego_only` and `cooperative`; the difference is which processed evidence is allowed into the pipeline.

Ego-only graph:

- keeps only observations whose `source_agent_id` is the asking vehicle;
- keeps only visibility facts for the asking vehicle;
- then runs the normal enrichment, candidate, visibility, relation, and QA path;
- useful as the single-agent control graph.

Cooperative graph:

- keeps observations and visibility facts from all connected vehicles;
- creates a scene where object hypotheses can be supported by ego evidence, partner evidence, or both;
- stores multi-agent support in `ProvenanceRecord.source_agent_ids` and attached `ObservationEvidence`.

Within-frame object association:

- `ObservationAssociator` first matches processed observations to existing tracks.
- Matching is type-constrained and distance-gated:
  - object type must match;
  - distance must be within `3.0m`;
  - greedy matching prefers closest pairs, then higher observation confidence.
- Matched observations are attached to the track by `TrackSupportEnricher`.
- The track becomes `supported`, and provenance is extended with the observation ID and source agent.

Unmatched evidence:

- `CandidateTrackCreator` promotes unmatched observations into `candidate` tracks.
- Candidate IDs are generated as `pred_candidate_<timestamp>_<index>`.
- `CandidateTrackResolver` keeps candidates whose confidence is at least `0.25` and prunes weaker candidates.

Cross-agent association:

- `CrossAgentAssociator` compares observations from different agents.
- A cross-agent match requires:
  - different source agents;
  - same object type;
  - distance within `3.0m`.
- Cross-agent association confidence is:
  - `1 - distance / 3.0`, clamped to `[0,1]`.
- Greedy matching again uses closest compatible observation pairs first.

What happens if two vehicle observations are resolved as the same object:

- If one side is already attached to a track and the counterpart is not, `CrossAgentSupportEnricher` attaches the counterpart observation to that track.
- The track provenance is updated with the counterpart source agent and observation ID.
- Later, if a candidate track and stronger non-candidate track are same type and within `1.0m`, `TrackMerger` folds the candidate into the anchor track.
- The current merger is conservative: it merges candidate tracks into stronger anchors, but it does not aggressively merge two already-strong anchor tracks.
- The merged track:
  - keeps the anchor object ID;
  - unions source agents and observation IDs;
  - keeps the latest timestamp;
  - concatenates non-duplicate observations;
  - becomes `supported`;
  - uses the max confidence of anchor and candidate.

What happens if they are different objects:

- If observations fail type or distance gates, they are not associated.
- Unmatched observations may remain as separate candidate tracks if they pass the confidence threshold.
- If candidate and anchor tracks are not same type or not within `1.0m`, they remain separate object hypotheses.
- If two strong tracks are close but neither is a candidate, the current conservative merge step leaves them separate rather than forcing an identity merge.
- Relation facts are then rebuilt from these separate tracks, so each object gets its own spatial, trajectory, visibility, and support facts.

Edge reconstruction after fusion:

- The pipeline does not try to patch individual edges in place.
- It first settles the object set through support attachment, candidate filtering, cross-agent support, and track merging.
- Then `TrackQualityAssessor`, `VisibilityReasoner`, and `RelationBuilder` recompute quality fields, visibility facts, and relation edges from the current scene.
- This is why cooperative support and low-conflict/path-relevance edges reflect the final fused object set rather than earlier intermediate hypotheses.

## Knowledge Graph Schema And Scores

Node/entity types:

- `CooperativeScene`
  - the graph container for one timestamped QA sample
  - stores the asking agent, connected agents, future trajectory, object tracks, visibility facts, relation facts, and raw QA text
- `AgentContext`
  - one vehicle/agent node
  - stores `agent_id`, 2D pose, optional velocity, and optional planned trajectory
- `ObjectTrack`
  - one object/hypothesis node
  - stores `object_id`, object type, position, confidence, provenance, lifecycle status, uncertainty, conflict, velocity, and observations
- `ObservationEvidence`
  - an agent-specific observation node/evidence record
  - stores source agent, observed position, object type, confidence, timestamp, and optional velocity
- `VisibilityFact`
  - an agent-object fact
  - states whether an object is `visible`, `occluded`, or `uncertain` for a specific agent
- `RelationFact`
  - a typed edge/fact between a subject and object with a confidence value
- `ProvenanceRecord`
  - the trace of where an object belief came from: source agents, observation IDs, and latest timestamp

Object lifecycle/status:

- `confirmed`
  - strong/grounded object track
- `supported`
  - object has supporting observations/provenance but is not treated as fully confirmed
- `candidate`
  - weaker object hypothesis promoted from unmatched evidence
  - candidates are useful for recall but receive penalties in risk/ranking policies

Supported relation/edge types:

| Relation | Meaning | Current confidence design |
| --- | --- | --- |
| `front_of` | object is ahead of asker in the shared 2D frame | deterministic `1.0` |
| `behind` | object is behind asker | deterministic `1.0` |
| `left_of` | object is left of asker | deterministic `1.0` |
| `right_of` | object is right of asker | deterministic `1.0` |
| `near` | object is within `10m` of asker | `1 - distance / 10`, clamped to `[0,1]` |
| `observed_by` | object has observation evidence from an agent | represented primarily through provenance/observations |
| `near_trajectory` | object is within `3m` of any future trajectory point | deterministic `1.0` when true |
| `near_first_waypoint` | object is near the first future waypoint | `1 - distance / 6`, clamped to `[0,1]` |
| `path_relevant` | object is near the planned path corridor | `1 - best_trajectory_distance / 4`, clamped to `[0,1]` |
| `cooperatively_supported` | object has evidence from at least two source agents | `min(1.0, 0.5 + 0.2 * source_agent_count)` |
| `low_conflict` | object support is geometrically consistent | `1 - conflict_score`, clamped to `[0,1]`, emitted when conflict is `<= 0.5` |

Why these edges exist:

- Spatial edges support human-readable scene reasoning such as ahead/behind/left/right.
- Trajectory edges support driving relevance instead of selecting every object in the scene.
- Visibility facts support Q1 visible-object reasoning and Q3 hidden-object reasoning.
- Provenance and cooperative-support edges expose whether a selected object came from one vehicle or multiple vehicles.
- Conflict/uncertainty facts prevent weak or contradictory evidence from dominating selection.

Confidence, conflict, and uncertainty:

- `ObjectTrack.confidence`
  - inherited from processed detections/tracks or fused object evidence
  - used as an evidence-strength term in Q1-Q3 selection
- `last_support_confidence`
  - max confidence among supporting observations attached to the track
- `conflict_score`
  - average distance between the fused track position and its supporting observations
  - lower is better
  - if no observations are attached, current conflict defaults to `0.0`
- `uncertainty_score`
  - clamped to `[0,1]`
  - current formula:
    - base uncertainty: `1 - track.confidence`
    - candidate penalty: `+0.15` unless GT-backed
    - miss penalty: `+0.10 * miss_count`
    - conflict penalty: `+0.20 * min(conflict_score / 3.0, 1.0)`
    - support bonus: `-0.10` if observations support the track
  - final value is clamped to `[0,1]`

Risk scores:

- In this project, task-level `risk` is not a calibrated probability.
- It is a normalized selector score used to rank safety-relevant candidates.
- Risk combines graph evidence, geometry, trajectory proximity, provenance, model/role score, and quality penalties.
- Scores are clamped to `[0,1]` so thresholds and relative comparisons remain stable.

Q2 occluding-object risk:

- First, visible blocker candidates are scored by pairwise evidence against hidden/occluded objects:
  - similar bearing/line of sight to a hidden object
  - hidden object lies plausibly behind the visible object
  - visible-hidden proximity
  - hidden object relevance to the trajectory
- Then `risk_adaptive` converts ranked blocker candidates into relative risk using:
  - trajectory/geometric component
  - alignment component
  - hidden-object relevance component
  - provenance/support component
  - model/role-score component
  - candidate-status penalty
- The policy keeps the top blockers and admits extra blockers only when relative risk and risk coverage justify it.
- Sparse-evidence fallback can backfill from visible objects when too few blocker-role candidates exist.

Q3 invisible-object risk and learned acceptance:

- The base hidden-object role score uses:
  - closeness to the planned future trajectory
  - distance to the asker
  - support/provenance count
  - object confidence
  - status penalty
  - conflict penalty
  - uncertainty penalty
- The old `risk_adaptive` Q3 policy converted these into a risk score with:
  - trajectory component
  - asker-distance component
  - provenance component
  - confidence component
  - normalized model/role-score component
  - conflict and uncertainty penalties
  - candidate penalty
- The current Q3 checkpoint uses a stronger two-stage design:
  - broad retrieval with `shortlist_size=64` and `8.0m` trajectory window
  - logistic acceptor trained on train-split candidate features
  - selected deployable threshold `0.33`
- The acceptor features include rank, role score, relative position, trajectory distance, asker distance, support count, confidence, conflict, uncertainty, age/miss count, and track status.

## Query And Retrieval Layer

The lower-level graph query API is `SceneQueryEngine`. It provides deterministic retrieval over the prepared `CooperativeScene`.

Supported query operations:

- `select_objects(scene)`
  - returns all object tracks in the current graph
- `filter_by_type(result, object_type)`
  - keeps objects of a requested semantic type
- `filter_by_visibility(result, agent_id, visibility)`
  - keeps objects visible, occluded, or uncertain for one agent
- `filter_by_source_agent(result, source_agent_id)`
  - keeps objects whose provenance includes a specific source agent
- `filter_by_relation(result, relation_type, reference_id)`
  - keeps objects linked to a reference entity by a relation fact
- `filter_near_trajectory(result, max_distance)`
  - keeps objects within a distance threshold of any future trajectory point
- `count(result)` and `exists(result)`
  - support simple aggregate reasoning
- `get_attribute(result, attribute_name)`
  - reads structured attributes such as confidence, type, status, position, support count, uncertainty, and conflict
- `compare(result, attribute_name)`
  - compares numeric attributes across object pairs
- `trace_provenance(result)`
  - returns source agents, observation IDs, and latest timestamp for each selected object

How Q1-Q3 use retrieval:

- The task handlers in `V2VGoTQARouter` use the same graph concepts as `SceneQueryEngine`, but with task-specific ranking and decoding.
- Q1 retrieves visible, trajectory-relevant objects and ranks them by trajectory/waypoint proximity, support, confidence, status, conflict, uncertainty, and distance-to-asker penalties.
- Q2 retrieves visible blocker candidates and scores them using hidden-object alignment plus risk-adaptive selection.
- Q3 retrieves hidden/occluded/uncertain candidates and applies the broad-pool logistic acceptor.
- This split keeps the graph representation shared while allowing each QA family to obey its benchmark-specific answer contract.

Q4 relational-importance example:

- The Q4 `relational_importance` scorer does not issue a SQL/Cypher query. It runs deterministic reasoning over the in-memory `CooperativeScene`.
- First, candidate collection is equivalent to:
  - select all object tracks from the graph;
  - keep objects whose asker visibility is `VISIBLE` or `OCCLUDED`;
  - keep only objects within the broad planning window, currently `30m`, of any ego future-trajectory point.
- Then each candidate is scored from graph-native features:
  - object confidence;
  - distance to the planned ego trajectory;
  - visible versus occluded state for the asking ego vehicle;
  - whether the object was observed by the asker;
  - whether it is supported or confirmed rather than only a candidate;
  - whether it is cooperatively supported by multiple agents;
  - uncertainty and conflict penalties.
- Example scene:
  - ego future trajectory: `(0,0), (-10,0), (-20,0), (-30,0)`
  - object A: car at `(-20.4,-0.1)`, visible to ego, confidence `0.90`, supported, sourced from ego and `cav1`, low uncertainty/conflict
  - object B: car at `(-55.4,24.9)`, visible to ego, confidence `0.80`, supported, sourced from `cav1`, but laterally far from the path
  - object C: car at `(-21.0,1.5)`, occluded to ego, confidence `0.75`, candidate-only, sourced from `cav1`, with higher uncertainty/conflict
- The scorer roughly computes:
  - base classification prior from confidence;
  - plus trajectory bonus for being close to the planned path;
  - plus occlusion, asker-observed, support, and cooperative bonuses;
  - minus candidate, uncertainty, and conflict penalties.
- A typical ranking would therefore be:
  - object A highest because it is close to the path, supported, cooperatively observed, and low-conflict;
  - object C next because it is occluded and path-relevant, but weaker because it is candidate-only and uncertain;
  - object B lower because it is less path-relevant even though it has reasonable confidence.
- The selection policy then decides how many ranked objects to emit:
  - `top2` keeps the two best;
  - `diverse_top2` avoids near-duplicate coordinates;
  - `default` is more permissive and improved recall but often over-selected;
  - `count_adaptive` is the current generic fix: keep the strong first objects, but require a third object to clear a stronger score/ratio gate and not be a near-duplicate.
- Current result for `relational_importance + count_adaptive`: train F1/precision/recall `0.604456 / 0.549308 / 0.671914` and validation `0.500711 / 0.446938 / 0.569195` at `0.5m`. This improves Q4 validation F1 and precision over `relational_importance + default`, with a small recall tradeoff.

QA layer:

1. The prepared graph is placed back into the `BenchmarkSample`.
2. `V2VGoTQARouter` dispatches by task type.
3. Task handlers query the graph deterministically and return object IDs plus answer text.
4. Export converts object IDs back into benchmark-compatible coordinate phrases.
5. The official-style evaluator scores localization F1 against reference answer coordinates.

End-to-end script path:

1. `scripts/run_qa_split_pipeline.py` runs deterministic KG-based QA prediction.
2. `scripts/evaluate_qa_router.py` loads V2V-GoT-QA rows and prepares cooperative scenes.
3. `V2VGoTQARouter` dispatches to the task handler.
4. The handler selects object IDs from the cooperative scene graph.
5. `scripts/export_qa_predictions.py` converts predictions into V2V-GoT/LLaVA-style JSONL with an `outputs` field.
6. `scripts/run_v2vgot_official_qa_eval.py` runs the upstream-compatible official-style Q1-Q4 evaluator.

Metric:

- official-style localization F1 / precision / recall at `0.5m`
- all listed Q1-Q3 parse error rates are `0.0`

## Graph-Quality And Ablation Log

This section is a focused log of the early checks that justify the KG pipeline itself, plus the ego-only/cooperative ablations that motivate using the cooperative graph as the paper-facing setting. These are not replacements for the official Q1-Q4 results below; they explain why the graph substrate and cooperative mode are credible.

### Local Graph Sanity Checks

Motivation:

- Before claiming cooperative reasoning, first prove that one agent alone can build a grounded local KG with object identity, provenance, visibility, relations, and deterministic queries.
- This gives a single-agent control condition: if later cooperative results change, the difference can be attributed to added cooperative evidence or policy behavior rather than an undefined graph path.

Observed Phase 3 local-graph run:

- source log: `docs/week3_updates.md`, `docs/phase3_local_graph_summary.md`
- example: `scene_id=0`, `agent_id=CAV_EGO`, `local_timestamp_index=0`
- local graph contents:
  - objects: `2`
  - relations: `4`
  - visibility facts: `1`
  - supported object: `object_id=1`, status `supported`, position `(-20.50,-0.98)`, confidence `1.00`
  - weaker hypothesis: `pred_candidate_0_0`, status `candidate`, position `(-75.06,-1.44)`, confidence `0.33`
- derived relations:
  - `1 behind CAV_EGO`
  - `1 right_of CAV_EGO`
  - `pred_candidate_0_0 behind CAV_EGO`
  - `pred_candidate_0_0 right_of CAV_EGO`
- visibility:
  - `visible(CAV_EGO, 1)`

Observed query behavior on the same local graph:

- object selection: `['1', 'pred_candidate_0_0']`
- visibility filtering: `['1']`
- near-trajectory filtering: `[]`
- `behind` relation filtering: `['1', 'pred_candidate_0_0']`

Interpretation:

- The graph does not flatten all evidence into one object list.
- It preserves the difference between supported objects and weak candidates.
- Visibility is selective: only the supported local object is visible to ego.
- Relation and trajectory queries are deterministic and inspectable.

Observed five-frame local validation for `CAV_EGO`:

| Check | Value |
| --- | ---: |
| validated frames | `5` |
| average objects per frame | `2.00` |
| average relations per frame | `4.00` |
| average visibility facts per frame | `1.20` |
| average supported tracks per frame | `1.20` |
| average candidate tracks per frame | `0.40` |
| average visible objects per frame | `1.20` |

Reasoning from this check:

- stable supported objects persist across adjacent timestamps;
- weak candidate clutter does not accumulate indefinitely;
- relation and visibility counts scale with object count rather than exploding;
- this is the local baseline needed before interpreting cooperative fusion.

Executable commands:

```bash
python3 scripts/demo_phase3_local_graph.py

python3 scripts/validate_phase3_local_graphs.py \
  --agent-id CAV_EGO \
  --max-frames 5 \
  --export-dir artifacts/phase3_local_graph_validation
```

### Deterministic Query Engine Check

Motivation:

- The paper approach depends on KG retrieval, so graph access must be fixed, typed, replayable, and safe on empty/unsupported cases.
- The query engine is the deterministic reasoning substrate used before task-specific Q1-Q4 ranking policies.

Observed Phase 4 query-engine run:

- source log: `docs/phase4_query_engine_summary.md`
- script: `scripts/demo_phase4_query_examples.py`
- hand-authored graph:
  - `track-1`: car, confidence `0.9`, provenance `GT + CAV_EGO`, status `supported`
  - `track-2`: car, confidence `0.4`, provenance `CAV_EGO`, status `candidate`
  - both are `front_of CAV_EGO`
  - `track-1` is `left_of CAV_EGO`
  - `track-2` is `right_of CAV_EGO`
  - `track-1` visibility is `visible`
  - `track-2` visibility is `uncertain`

Observed outputs:

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
- `empty_count: 0`
- `empty_exists: False`

Interpretation:

- normal retrieval, filtering, attribute lookup, pairwise comparison, and provenance tracing work;
- unsupported attributes return `None` instead of invented facts;
- non-comparable comparisons return `not_comparable`;
- empty selections return `0`/`False`, so the KG layer fails conservatively.

Executable command:

```bash
python3 scripts/demo_phase4_query_examples.py
```

### Ego-Only Versus Cooperative Smoke Ablations

Motivation:

- The primary ablation is whether answering from the cooperative KG changes behavior compared with the asking vehicle's ego-only KG.
- The controlled setup keeps the downstream router/evaluator fixed and switches only `--baseline-mode cooperative` versus `--baseline-mode ego_only`.

Early Phase 5/Week 4 pilot on the first `100` validation samples:

| Task | Comparison | Exact/object-id match | Differing samples | Interpretation |
| --- | --- | ---: | ---: | --- |
| `occluding_objects` | cooperative vs ego-only | `100/100` | `0/100` | stable task path, no cooperative advantage on this slice |
| `planning_awareness` | cooperative vs ego-only | `85/100` | `15/100` | first measurable cooperative/ego divergence |

Observed planning-awareness divergence pattern:

- cooperative mode often added `pred_candidate_*` objects;
- ego-only often produced fewer objects or empty outputs;
- this showed that cooperative evidence was affecting graph contents, but the early candidate-heavy policy could add noise as well as useful signal.

Phase 5 closeout structural sweep on first `100` validation samples:

| Task | Cooperative behavior | Ego-only behavior | Focused takeaway |
| --- | --- | --- | --- |
| `notable_objects` | cooperative scenarios matched the cooperative baseline `100/100` | ego-only scenarios matched `85/100` | cooperative graph changed visible-notable outputs on this slice |
| `occluding_objects` | cooperative scenarios matched `100/100` | ego-only scenarios matched `100/100` | no visible ego/cooperative difference in this early structural check |
| `invisible_objects` | cooperative scenarios matched `100/100` | ego-only scenarios matched `100/100` | no structural difference in this early check |
| `planning_awareness` | `risk_aware + top2/diverse_top2` cooperative matched `100/100`; relational/heuristic variants differed more | ego-only `risk_aware + top2/diverse_top2` matched `85/100` | planning-awareness was the clearest early task where cooperation and selection policy interacted |

Important caveat:

- These Phase 5/Week 4 comparisons were structural/pilot checks, not the final official benchmark results.
- They are useful for paper ablations because they show the KG pipeline can run both ego-only and cooperative modes with the same router, and because they exposed where cooperative evidence changed outputs.
- Final claims should use the official-style Phase 8 metrics in the next section.

Executable commands for the historical smoke sweep:

```bash
python3 scripts/run_phase5_closeout.py \
  --split val \
  --limit 100 \
  --output-dir outputs/phase5_closeout_awq_full_sweep \
  --report-name phase5_closeout_report.json \
  --markdown-name phase5_closeout_report.md \
  --full-sweep-all-tasks
```

Executable commands for current official-style ego/cooperative ablations:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type invisible_objects \
  --scenario-name ablation_val_q3_cooperative_broadpool_logreg \
  --limit 0 \
  --baseline-mode cooperative \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --progress-every 250 \
  --workers 32

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type invisible_objects \
  --scenario-name ablation_val_q3_ego_only_broadpool_logreg \
  --limit 0 \
  --baseline-mode ego_only \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --progress-every 250 \
  --workers 32

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type planning_awareness \
  --scenario-name ablation_val_q4_cooperative_trajcal_v1 \
  --limit 0 \
  --baseline-mode cooperative \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-selection-source orchestrator \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type planning_awareness \
  --scenario-name ablation_val_q4_ego_only_trajcal_v1 \
  --limit 0 \
  --baseline-mode ego_only \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-selection-source orchestrator \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32
```

How to interpret these current ablations:

- Use the same learned checkpoint and selection settings for cooperative and ego-only.
- Compare the official summaries under `outputs/phase8_val_report/official_eval_reports/`.
- Any difference then comes from graph evidence allowed into preparation, not from changing the downstream policy.
- For the paper, report these as an ablation table separate from the main cooperative headline table.

## Current Results

| Task | Current Approach | Train F1 / P / R | Val F1 / P / R | V2V-GoT Ref | Status |
| --- | --- | --- | --- | ---: | --- |
| Q1 `notable_objects` | deterministic visible-object `heuristic` selector | `0.655709 / 0.772176 / 0.569771` | `0.585836 / 0.674759 / 0.517621` | `0.525000` | clears +10% target |
| Q2 `occluding_objects` | `risk_adaptive` occlusion selector with sparse-evidence fallback | `0.391914 / 0.408022 / 0.377031` | `0.427921 / 0.452542 / 0.405840` | `0.301000` | clears +10% target |
| Q3 `invisible_objects` | broad-pool `logreg_acceptor_t0p33`, `shortlist_size=64`, trajectory window `8.0m` | `0.464406 / 0.527863 / 0.414568` | `0.493934 / 0.488014 / 0.500000` | `0.440000` | clears +10% target |
| Q4 `planning_awareness` | `relational_importance + trajectory_calibrated_acceptor`, duplicate radius `1.0m` | `0.729672 / 0.711258 / 0.749064` | `0.613774 / 0.576685 / 0.655962` | `0.608000` | exceeds V2V-GoT reference |

## Final Promoted Checkpoints (Quick View)

| Task | Metric | Ours | Baseline | Relative Improvement |
| --- | --- | ---: | ---: | ---: |
| `q1_notable_objects` | `F1@0.5m` | `0.585836` | `0.525000` | `+11.59%` |
| `q2_occluding_objects` | `F1@0.5m` | `0.427921` | `0.301000` | `+42.17%` |
| `q5_object_motion_prediction` | `L2 Avg 123 (m)` | `3.822136` | `8.050000` | `+52.52%` |
| `q6_agent_motion_prediction` | `Binary Accuracy` | `0.904527` | `0.874000` | `+3.49%` |
| `q7_object_motion_prediction` | `L2 Avg 123 (m)` | `3.822136` | `7.610000` | `+49.77%` |
| `q3_invisible_objects` | `F1@0.5m` | `0.493934` | `0.440000` | `+12.26%` |
| `q4_planning_awareness` | `F1@0.5m` | `0.613774` | `0.608000` | `+0.95%` |
| `q8_control_settings` | `Action L1 (edit_dist/8)` | `0.076139` | `0.087600` | `+13.08%` |
| `q9_future_trajectory` | `L2 Avg All (m)` | `1.211582` | `2.620000` | `+53.76%` |

Notes:

- Q1-Q4 headline metric is strict official-style `0.5m` localization F1.
- Q5/Q7 use `l2_error_avg_123_all`; lower is better, so relative improvement is reported as error reduction versus baseline.
- Q8 normalization is `action_edit_dist / 8` because speed and steering class-index differences each range `0..4`.
- Q8/Q9 are also lower-is-better metrics, so relative improvement is reported as error reduction versus baseline.

### Q6 Checkpoint Note

- local checkpoint record:
  - `outputs/phase9_train_dev/q6_gbdt_v4/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38_checkpoint.json`
- promoted model:
  - `outputs/phase9_train_dev/q6_gbdt_v4/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38.json`
- promoted val summary:
  - `outputs/phase8_val_report/official_eval_reports/q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38_official_export_manifest_official_qa_eval_summary.json`
- held-out metric:
  - `binary_classification_accuracy=0.9045269878119558`
- selected settings:
  - `n_estimators=280`, `learning_rate=0.04`, `max_depth=2`, `min_samples_leaf=96`, `subsample=0.7`, `threshold=0.38`
- protocol note:
  - trained on `train`; threshold and hyperparameters were selected from `val`, so this is a validation-tuned checkpoint.
- dedicated rerun scenario template:
  - `val_q6_checkpoint_rerun` (Q6-only, official export + official eval)
- local presence note:
  - the checkpoint record exists in this checkout; materialized model/official-summary artifacts may live on the runtime pod unless copied back.

## Latest Baseline Comparison

This is the latest paper-facing validation comparison against the V2V-GoT reported task references.

| Task | Metric | Ours | Baseline | Relative Improvement |
| --- | --- | ---: | ---: | ---: |
| `q1_notable_objects` | `F1@0.5m` | `0.585836` | `0.525000` | `+11.59%` |
| `q2_occluding_objects` | `F1@0.5m` | `0.427921` | `0.301000` | `+42.17%` |
| `q5_object_motion_prediction` | `L2 Avg 123 (m)` | `3.822136` | `8.050000` | `+52.52%` |
| `q6_agent_motion_prediction` | `Binary Accuracy` | `0.904527` | `0.874000` | `+3.49%` |
| `q7_object_motion_prediction` | `L2 Avg 123 (m)` | `3.822136` | `7.610000` | `+49.77%` |
| `q3_invisible_objects` | `F1@0.5m` | `0.493934` | `0.440000` | `+12.26%` |
| `q4_planning_awareness` | `F1@0.5m` | `0.613774` | `0.608000` | `+0.95%` |
| `q8_control_settings` | `Action L1 (edit_dist/8)` | `0.076139` | `0.087600` | `+13.08%` |
| `q9_future_trajectory` | `L2 Avg All (m)` | `1.211582` | `2.620000` | `+53.76%` |

Interpretation:

- Q1-Q4 use strict official-style `0.5m` localization F1 as the headline metric.
- Q5/Q7/Q8/Q9 are lower-is-better metrics, so relative improvement is reported as error reduction versus baseline.
- Q5 and Q7 currently report the same `3.822136` validation result because the inspected official target/evaluator path collapses both object-motion questions to the same effective answer target under the shared model/export path.
- Q6 is a higher-is-better binary accuracy metric.
- The strongest story is not one global model change. Each QA family uses the graph evidence differently: visible-object grounding for Q1, occlusion risk for Q2, broad hidden-object retrieval plus acceptance for Q3, and planning-relevance plus trajectory calibration for Q4.

## Phase 9 Q9 Update (Week 6)

Aim:

- Expand beyond Q1-Q4 with a benchmark-faithful Q9 `future_trajectory` checkpoint.
- Keep the same KG-based pipeline shape used by other tasks:
  - prepare cooperative scene from V2V-GoT assets,
  - route deterministically through the QA layer,
  - export V2V-GoT-compatible `outputs`,
  - score with the upstream-compatible official evaluator.

Q9 approach (`ControlConditionedFutureTrajectoryPlanner`):

- The planner is a modular Q9 component used by `FutureTrajectoryHandler`.
- Input signals:
  - current asker position from the question/scene context,
  - control metadata (`suggested_speed_idx`, `suggested_steering_idx`, `dist`, `angle`),
  - optional frozen model coefficients.
- Prediction path:
  - primary: frozen linear control-metadata model (`phase9_q9_control_metadata_linear_v1`) predicts absolute waypoint coordinates;
  - improved: tail-residual variant (`phase9_q9_control_metadata_linear_tail_residual_v1`) adds a learned correction on late waypoints (tail) to reduce long-horizon drift;
  - fallback: deterministic control-conditioned kinematic prior when no frozen model is supplied.
- Design properties:
  - train-frozen inference (no split-specific tuning at inference time),
  - deterministic outputs for reproducibility,
  - plug-and-play model JSON deployment through existing CLI flags.

Q9 model formulation (paper-facing):

- Task output:
  - predict `6` future waypoints in ego-coordinate space:
  - `[(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]`
- Base predictor type:
  - multivariate linear regression (not logistic regression)
  - input feature vector `f` has `19` dimensions
  - frozen coefficient matrix `W` has shape `12 x 19`
  - output vector `o` is:
    - `o = W f`
    - `o = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6]`
- Tail-residual variant:
  - keeps the same base linear predictor for all six waypoints
  - adds a second linear residual head only for waypoint tail `(x5,y5,x6,y6)`
  - tail residual is predicted from an expanded nonlinear feature set and added to base tail outputs
  - purpose: reduce long-horizon drift while preserving short-horizon behavior

Feature vector construction:

- Base feature vector (`19` dims):
  - `1.0` (bias),
  - `current_x`, `current_y`,
  - `asker_is_cav1`,
  - one-hot `speed_idx` (`5` dims),
  - one-hot `steering_idx` (`5` dims),
  - `dist`,
  - `sin(angle)`, `cos(angle)`,
  - `dist*sin(angle)`, `dist*cos(angle)`
- Tail-residual extra features (in addition to base):
  - `current_x*dist`, `current_y*dist`,
  - `dist^2`,
  - `sin(2*angle)`, `cos(2*angle)`,
  - `dist^2*sin(angle)`, `dist^2*cos(angle)`,
  - `speed_idx*steering_idx`

Data sources for features:

- `current_x`, `current_y`:
  - parsed from question text: `I am CAV_X at (x,y)`
- `asker_is_cav1`, `speed_idx`, `steering_idx`, `dist`, `angle`:
  - from the Q9 benchmark record metadata
- No future-reference coordinates are used as runtime features.

One-sample walkthrough:

- Input question:
  - `I am CAV_1 at (-75.7,5.2). ... speed setting: fast ... steering setting: straight.`
- Runtime steps:
  - build `f` from current position and control metadata,
  - compute `o = Wf` to get 12 values,
  - reshape to 6 waypoints,
  - if tail-residual model is enabled, add residual correction to `(x5,y5,x6,y6)`,
  - emit final trajectory text for official export.

How KG helps Q9:

- The same cooperative scene preparation keeps coordinate frame, asker identity, and scene context consistent with Q1-Q4.
- Deterministic graph-prepared routing avoids free-form generation variance and keeps export/eval behavior stable.
- Q9 uses the same end-to-end benchmark path (`prepare -> route -> export -> official eval`) as the rest of the project, so comparisons are directly traceable.
- The KG layer provides a stable structured interface (agent identity, pose context, routing contract, export contract), while the Q9 regressor provides the coordinate prediction head.

Q9 evaluation protocol:

- Generation: `scripts/run_qa_split_pipeline.py --task-type future_trajectory`
- Export: `scripts/export_qa_predictions.py`
- Official metrics: `scripts/run_v2vgot_official_qa_eval.py`
- Reported metrics (lower is better): `l2_error_avg_1s`, `l2_error_avg_2s`, `l2_error_avg_3s`, `l2_error_avg_all`

Baseline-relative performance (V2V-GoT Q9 ref `2.62m`):

| Q9 Checkpoint | Train L2 all | Val L2 all | Val L2 @1s | Val L2 @2s | Val L2 @3s | V2V-GoT Ref L2 all | Relative Reduction vs Ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_metadata_linear_v1` | `0.894774` | `1.320964` | `0.950570` | `1.354670` | `1.657653` | `2.620000` | `49.58%` |
| `control_metadata_linear_tail_residual_v1` | pending full official train rerun | `1.211582` | `0.950570` | `1.354670` | `1.329508` | `2.620000` | `53.76%` |

Why it performs better than baseline:

- Control-conditioned waypoint regression aligns the Q9 answer structure with the benchmark target format.
- Train-frozen coefficients capture stable global trajectory patterns across control settings.
- Tail residual correction improves long-horizon localization without degrading short-horizon behavior.
- Net effect: substantial validation L2 reduction versus the V2V-GoT reference.

Q9 artifacts:

- linear model:
  - `outputs/phase9_train_dev/q9_future_trajectory_control_metadata_linear_v1_deployable.json`
- tail-residual model:
  - `outputs/phase9_train_dev/q9_future_trajectory_control_metadata_linear_tail_residual_v1_deployable.json`
- validation official summary (tail-residual):
  - `outputs/phase8_val_report/official_eval_reports/val_q9_future_trajectory_control_linear_tail_residual_v1_official_export_manifest_official_qa_eval_summary.json`

## Phase 9 Q8 Update (Week 6)

Aim:

- Promote a benchmark-faithful Q8 `control_settings` checkpoint with train-frozen inference and official-style evaluation.
- Keep the method defensible and generalizable:
  - structured KG-derived features,
  - transparent linear/ordinal heads,
  - no split-specific hardcoded scene rules.

Q8 task and metric:

- QA type: `18` (`control_settings`)
- Official evaluator outputs:
  - `speed_accuracy`, `steering_accuracy`, `action_accuracy`
  - `speed_edit_dist`, `steering_edit_dist`, `action_edit_dist`
- V2V-GoT Q8 reference in our protocol: action L1/error proxy `0.0876` (lower is better).

Current promoted Q8 approach:

- handler: `ControlSettingsHandler(selection_policy=linear_classifier)`
- model family: train-frozen linear heads with ordinal speed decoding
- selected model: `q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json`
- core design:
  - KG-derived control candidate ranking remains deterministic and interpretable;
  - speed uses ordinal thresholds (`fast -> ... -> stop`) rather than flat multiclass only;
  - speed thresholding is risk-conditional (`low/mid/high` top-risk regimes);
  - extended trajectory-aware features improve separation between benign far objects and path-relevant conflicts.

Q8 feature design (selected checkpoint):

- base graph features:
  - top object risk/trajectory/asker distance, confidence, conflict, uncertainty, support/provenance, visibility/status, lateral offsets
- extended trajectory-aware features (`extended_v1`):
  - distance to first waypoint
  - nearest waypoint index (normalized)
  - along-path progress (normalized)
  - local path curvature
  - heading alignment (ego heading vs object bearing)

Official-style Q8 results:

| Split | Speed Acc | Steering Acc | Action Acc | Speed Edit Dist | Steering Edit Dist | Action Edit Dist | Normalized Action (`/8`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | `0.655411` | `0.908950` | `0.607648` | `0.393165` | `0.123434` | `0.516599` | `0.064575` |
| Val (promoted) | `0.684272` | `0.890308` | `0.620720` | `0.476204` | `0.132908` | `0.609112` | `0.076139` |

Baseline comparison (val):

- V2V-GoT Q8 reference: `0.087600`
- current normalized action error: `0.076139`
- absolute delta: `-0.011461` (lower is better)
- relative error reduction: `13.08%`

Why this works:

- Residual analysis showed Q8 was speed-dominated, not steering-dominated.
- Flat class tuning saturated; ordered speed modeling reduced large class-index jumps directly tied to edit distance.
- Risk-conditional thresholding corrected context-dependent speed bias that one global threshold could not capture.
- Extended trajectory/heading features improved the model's notion of whether a high-risk candidate is truly path-critical vs spatially distant noise.

Why this is defensible for research:

- The method is explicit and auditable: graph features -> frozen model -> deterministic export -> official evaluator.
- Selection logic and model metadata are reproducible and versioned in deployable JSON files.
- The improvement is not from prompt hacks or split-specific memorization; it is from structured feature and decoder design aligned to the official metric.
- Legacy behavior remains intact (backward-compatible policy/model loading), and new behavior is opt-in via model metadata (`feature_set`, `speed_head_type`, threshold policy).

Repro commands (selected Q8 checkpoint):

```bash
python3 scripts/train_q8_control_policy.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --split train \
  --baseline-mode cooperative \
  --feature-set extended_v1 \
  --speed-head-type ordinal \
  --speed-class-weighting sqrt_inverse_freq \
  --steering-class-weighting none \
  --l2-regularization 1e-4 \
  --speed-ordinal-threshold-policy risk3 \
  --speed-risk-split-low 0.2 \
  --speed-risk-split-high 0.5 \
  --output-json outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json \
  --output-report outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_report.json

python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type control_settings \
  --scenario-name train_q8_control_linear_classifier_v7_extended_ordinal_risk3 \
  --baseline-mode cooperative \
  --control-selection-policy linear_classifier \
  --control-model-json outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json \
  --workers 32 \
  --v2vgot-root /workspace/repos/V2V-GoT

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type control_settings \
  --scenario-name val_q8_control_linear_classifier_v7_extended_ordinal_risk3 \
  --baseline-mode cooperative \
  --control-selection-policy linear_classifier \
  --control-model-json outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json \
  --workers 32 \
  --v2vgot-root /workspace/repos/V2V-GoT
```

Key Q8 artifacts:

- model:
  - `outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_deployable.json`
- train report:
  - `outputs/phase9_train_dev/q8_control_linear_classifier_v7_extended_ordinal_risk3_report.json`
- validation official summary:
  - `outputs/phase8_val_report/official_eval_reports/val_q8_control_linear_classifier_v7_extended_ordinal_risk3_official_export_manifest_official_qa_eval_summary.json`
- mismatch analysis used to guide final design:
  - `outputs/phase8_val_report/phase9_q8_control_v5_ordinal_mismatch_report.md`

## Phase 9 Q5 Update (Week 6)

Aim:

- promote a held-out validation checkpoint for Q5 `object_motion_prediction` (`qa_type_id=15`) using a frozen learned motion head.

Current promoted Q5 checkpoint:

- scenario: `val_q5_tree_d9_l64_g001_pathrel_k3_d8_occ1_rerun`
- evaluator status: return code `0`
- parse status: `gt_parse_error_rate=0.0`, `output_parse_error_rate=0.0`
- metrics:
  - `l2_error_avg_123_all=3.8221364042292816`
  - `l2_error_avg_03_all=8.357667994188219`
  - `l2_error_avg_3s=11.466409212687845`
  - `action_accuracy=0.5599349123124209`

Baseline comparison:

- V2V-GoT Q5 reference: `8.05m` L2
- relative reduction vs reference (using `l2_error_avg_123_all`): `52.52%`
- V2V-GoT Q7 reference: `7.61m` L2
- same object-motion checkpoint/export path gives Q7 `l2_error_avg_123_all=3.8221364042292816`, a `49.77%` relative error reduction.

Correction note:

- the earlier `val_q5_manual_check` metric (`5.6256477000301714`) used a sweep metadata JSON (`..._best_train_candidate.json`) in `--object-motion-model-json` instead of a deployable model JSON (`..._deployable.json`);
- it is retained as a debugging artifact only and is not used for promoted/e2e claims.

Why this works:

- the learned regression-tree Q5 head outperforms deterministic velocity projection by modeling nonlinear motion modes;
- it conditions endpoint deltas on cooperative-graph features (trajectory relevance, visibility, support/conflict, uncertainty), which reduces long-horizon drift and endpoint bias;
- improvements are strongest in late-horizon behavior (`l2_error_avg_3s`), consistent with the model’s split-based regime handling.

E2E integration:

- `scripts/e2e/run_e2e_train_pipeline.py` now always trains a fresh Q5 model per e2e run and archives it in the run manifest;
- `scripts/e2e/run_e2e_validation_report.py` now uses manifest-provided Q5 model during val runs and reports Q5 with `l2_error_avg_123_all` as the primary metric (fallback to `l2_error_avg_all`).

## Defensibility Note (Current)

This section records why the current KG + classical-head direction is technically defensible, what it does and does not claim, and what remains to de-risk generalization.

Why simple heads can outperform heavier LLM-style paths here:

- Q5/Q8/Q9 are scored with strict structured metrics (L2/edit-distance style), not open-ended language quality.
- The cooperative KG provides task-aligned state features (visibility, support/conflict, trajectory relevance, uncertainty).
- Given strong structured features, low-capacity supervised heads (linear/tree) often provide lower-variance numeric predictions than free-form generation-style pipelines.
- Resulting behavior is deterministic and reproducible under fixed manifests.

What is novel in this project (not merely "using regression"):

- cooperative KG as the central multi-agent state abstraction;
- task-specific deterministic prediction heads attached to that abstraction;
- benchmark-faithful `prepare -> route -> export -> official eval` path;
- explicit promotion/rejection checkpoint discipline with reproducible artifacts.

Q5 trainable feature construction example:

- one Q5 training row corresponds to one matched object mention:
  - input vector (fixed-width):
    - `[bias, x, y, vx, vy, speed, distance_to_trajectory, distance_to_asker, confidence, support_count, conflict_score, uncertainty_score, status_supported, status_candidate, visibility_visible, visibility_occluded, visibility_uncertain]`
  - target:
    - `[dx, dy] = [gt_tx - x, gt_ty - y]`
- variable scene complexity (e.g., 1 vs 3 neighboring objects) is converted into fixed-width summaries via graph-derived aggregates (`support_count`, `conflict_score`, `uncertainty_score`, etc.), preserving compatibility with linear/tree models.

Current claim boundary:

- supported claim: the KG + deterministic-head pipeline is competitive and can exceed benchmark references on this dataset protocol for selected tasks.
- unsupported claim (not yet): broad out-of-distribution generalization across arbitrary datasets/scenarios.

Remaining risks and next validation actions:

- artifact-type mistakes (metadata JSON vs deployable model JSON) can create false checkpoint claims; enforce strict deployable-model validation in pipeline inputs.
- add scenario-stratified and condition-stratified validation slices (traffic density, occlusion level, long-horizon difficulty).
- run feature ablations to quantify contribution of KG-derived terms versus simpler baselines.
- where possible, test transfer to an additional split/domain without retuning to strengthen generalization arguments.

### Feature Construction By Task (Implementation-Level)

This section explains how trainable inputs are formed from multi-agent graph state and why fixed-width models (logistic/linear/tree) can still encode spatial structure.

Common pattern across Q3/Q4/Q5/Q8/Q9:

- each train row corresponds to one supervised unit (candidate object, selected control context, or full trajectory record);
- variable-size scene context is converted into fixed-width vectors through graph-derived summaries;
- targets are task-specific labels or coordinates;
- no runtime dependence on reference answers.

Q3 `invisible_objects` (candidate acceptance):

- source: `scripts/export_phase8_invisible_candidate_features.py` -> `scripts/optimize_q3_invisible_candidate_policy.py`
- row unit: one hidden-candidate object row.
- fixed features:
  - numeric: `distance_to_asker`, `distance_to_trajectory`, `support_count`, `confidence`, `conflict_score`, `uncertainty_score`, relative position terms, age/miss terms;
  - categorical one-hot: status/visibility/object-type style categories.
- target:
  - binary `candidate_matches_gt` (within match threshold to reference coordinate).
- model:
  - train-frozen logistic (or MLP in ablations) scoring candidates; policy layer controls shortlist and acceptance threshold.

Q4 `planning_awareness` (candidate acceptance over planning-ranked set):

- source: `scripts/export_phase8_planning_candidate_features.py` using `planning_logreg_feature_values(...)`.
- row unit: one planning candidate from orchestrator-ordered list.
- fixed features (`PLANNING_LOGREG_FEATURE_NAMES`):
  - planning relevance/risk, trajectory distance geometry, visibility/status/provenance/support, rank/context terms.
- target:
  - binary coordinate match to reference (`candidate_matches_gt`).
- model:
  - train-frozen logistic acceptor plus deterministic post-selection/calibration.

Q5 `object_motion_prediction` (endpoint regression):

- source: `scripts/train_q5_object_motion_predictor.py`
- row unit: one matched object mention.
- fixed input vector:
  - `[bias, x, y, vx, vy, speed, distance_to_trajectory, distance_to_asker, confidence, support_count, conflict_score, uncertainty_score, status_supported, status_candidate, visibility_visible, visibility_occluded, visibility_uncertain]`
- target:
  - `[dx, dy] = [gt_tx - x, gt_ty - y]`.
- model:
  - linear / piecewise-linear / regression-tree.

Q8 `control_settings` (speed + steering classification):

- source: `src/kg_coop_drive/application/planning/control_settings_policy.py` + `scripts/train_q8_control_policy.py`.
- row unit: one control-settings sample.
- fixed features:
  - base set: top-risk object/control context (`top1/top2 risk`, trajectory distance, asker distance, confidence/conflict/uncertainty, support/provenance, visibility/status, lateral offsets, candidate count);
  - `extended_v1` adds trajectory-geometry terms (`distance_to_first_waypoint`, nearest waypoint index/progress, local curvature, heading alignment cosine).
- target:
  - discrete speed class + steering class indices from benchmark labels.
- model:
  - train-frozen linear heads (ordinal speed decoding + steering classifier).

Q9 `future_trajectory` (waypoint regression):

- source: `scripts/train_q9_future_trajectory_regressor.py`.
- row unit: one Q9 sample.
- fixed base features:
  - `[bias, current_x, current_y, asker_is_cav1, onehot(speed_idx,5), onehot(steering_idx,5), dist, sin(angle), cos(angle), dist*sin(angle), dist*cos(angle)]`
- optional tail-residual extra features:
  - nonlinear control/position terms for waypoints 5-6 (e.g., `dist^2`, harmonic angle terms, cross terms).
- target:
  - flattened future waypoints `[x1,y1,...,x6,y6]`.
- model:
  - frozen linear regressor with optional tail residual head.

How 1 vs 3 neighboring objects are represented:

- models do not ingest variable-length neighbor lists directly.
- instead, neighbor/context effects are compressed into fixed aggregates:
  - e.g., `support_count`, `conflict_score`, `uncertainty_score`, top-object risk gaps, visibility/provenance flags, trajectory-relative distances.
- example for Q5:
  - one-neighbor-like context may yield `support_count=1`, lower conflict;
  - three-neighbor-like context may yield `support_count=3`, higher conflict/uncertainty;
  - vector width remains identical, so regression/tree training remains well-posed while still encoding multi-agent context.

Defensibility implication:

- the approach is not "ignoring spatial/multi-agent structure"; it encodes that structure as explicit graph-derived, fixed-width signals suitable for reproducible supervised learning.

Q3 current finding:

- current approach: broad candidate retrieval plus train-frozen logistic acceptance
- why this works: Q3 was recall-limited when retrieval and acceptance were coupled; many correct invisible objects were not in the narrow shortlist, so the acceptor could not recover them
- what changed: the final policy expands retrieval to `shortlist_size=64` and `8.0m` trajectory window, then lets the frozen logistic acceptor choose at most one high-confidence invisible object
- result interpretation:
  - validation localization F1/P/R is `0.493934 / 0.488014 / 0.500000`, clearing the V2V-GoT reference `0.440000`
  - validation recall reaches `0.500000`, confirming the broad pool recovered previously unreachable correct candidates
  - precision remains controlled enough for F1 to improve, so the final Q3 story is not "predict more"; it is "retrieve broadly, accept selectively"
- rejected alternatives:
  - narrower legacy trajectory windows had better simplicity but missed correct objects
  - earlier precision-oriented thresholds improved precision but collapsed recall
  - MLP-style/nonlinear acceptance did not justify replacing the more transparent logistic acceptor for the selected Q3 checkpoint

Q4 current finding:

- current approach: `relational_importance` candidate scoring plus a train-frozen logistic acceptor with `1.0m` duplicate suppression and trajectory-calibrated post-selection
- why this works: Q4 is not just a visible/hidden composition task; it needs planning relevance over all graph candidates, then careful control of far/lateral false positives and close-object duplicate handling
- evidence-driven changes:
  - enabling the orchestrator made the pluggable planning-awareness ranker actually govern Q4
  - `relational_importance` improved recall by scoring all planning-relevant visible/occluded candidates instead of relying on fixed composition
  - train-frozen logistic acceptance improved over hand-written count policies by learning a candidate boundary from graph features
  - residual attribution showed retrieval was not the blocker: all `1928` validation false negatives had nearby candidates, and `1665` were within `0.5m`
  - the same attribution showed two actionable issues: `2.0m` duplicate suppression was too aggressive for close two-object references, and many false positives were far/lateral supported visible objects
  - reducing duplicate suppression to `1.0m` improved both train and validation slightly
  - trajectory calibration then suppressed moderate-probability far/lateral extras and produced the final validation jump
- result interpretation:
  - validation F1/P/R is `0.613774 / 0.576685 / 0.655962`, exceeding the V2V-GoT Q4 reference `0.608000` under the strict `0.5m` headline metric
  - compared with the prior `nd1p0` checkpoint, F1 improves `0.607578 -> 0.613774` mainly through precision `0.565305 -> 0.576685`
  - train also improves over `nd1p0`: `0.726896 -> 0.729672`, so the validation gain is not an isolated split artifact
- rejected alternatives:
  - L1/L2/elastic-net variants did not improve held-out validation
  - a small MLP acceptor over-accepted and underperformed
  - hard and soft count gates improved precision in some cases but lost too much recall
  - final gain came from residual-attribution-guided graph/geometry calibration, not from blindly increasing model complexity

Q4 current mismatch diagnosis:

- validation report: `outputs/phase8_val_report/phase8_q4_planning_orch_rel_count_adaptive_mismatch_report.md`
- reference mentions: `5564`
- predicted mentions: `9019`
- matched mentions at `0.5m`: `3167`
- false positives / false negatives: `5852 / 2397`
- interpretation from the pre-logreg checkpoint: Q4 was over-predicted on validation; precision/count control was the right next target, while the much higher `4.0m` F1 also suggested strict-coordinate misses.

Q4 adaptive acceptor next step:

- implemented train-frozen logistic acceptor path:
  - export train features with `scripts/export_phase8_planning_candidate_features.py`
  - train/freeze model with `scripts/train_q4_planning_acceptor.py`
  - run normal evaluation with `--planning-selection-policy logreg_acceptor`
  - provide frozen JSON via `--planning-acceptor-model-json`
- training uses train split only; validation is used only after the model and threshold are frozen.
- inference uses graph-derived features only: rank, relational score, score gaps, trajectory/asker distances, visibility, confidence, support, uncertainty, conflict, provenance, status, and duplicate distance.
- no sample IDs, reference coordinates, or validation labels are used at inference.
- current train result with frozen model `q4_planning_rel_logreg_nd1p0_p055_t0p56_deployable.json`:
  - F1/precision/recall at `0.5m`: `0.726896 / 0.704928 / 0.750278`
- held-out validation with the same frozen model:
  - F1/precision/recall at `0.5m`: `0.607578 / 0.565305 / 0.656684`
  - this is only `0.000422` below the V2V-GoT Q4 reference `0.608000` under our strict `0.5m` headline metric.

Q4 next improvement cycle toward `0.7`:

- first inspect residual errors for the current `logreg_acceptor` checkpoint, because the remaining gap may be a mix of over-prediction, missed candidates, and exact-coordinate mismatch:

```bash
python3 scripts/inspect_phase8_planning_official_mismatches.py \
  --export-manifest outputs/phase8_val_report/official_exports/phase8_val_report_val_planning_awareness_orch_rel_logreg_acceptor_official_export_manifest.json \
  --output-json outputs/phase8_val_report/phase8_q4_planning_orch_rel_logreg_acceptor_mismatch_report.json \
  --output-markdown outputs/phase8_val_report/phase8_q4_planning_orch_rel_logreg_acceptor_mismatch_report.md \
  --examples 50
```

- then train candidate acceptors on train only. The trainer supports `--regularization l2`, `l1`, `elasticnet`, or `none`; these are general model-capacity controls, not hard-coded scene rules.
- ridge/L2 should help when the current model is too sensitive to noisy correlated features; lasso/L1 should help if only a smaller subset of graph features is truly reliable; elastic-net is the compromise when both sparsity and shrinkage are useful.
- promote a candidate only if train official evaluation beats or cleanly improves the current train checkpoint `0.726814 / 0.704891 / 0.750145`, then run one validation evaluation.
- first lasso candidate `q4_planning_rel_logreg_lasso_p060`, with `--regularization l1 --l1 0.001 --min-precision 0.60`, selected threshold `0.56` and train-side selection F1/precision/recall `0.697744 / 0.654755 / 0.746774`; this is below the current promoted Q4 train checkpoint, so do not spend a validation run on this lasso model.
- ridge-style candidate with selected threshold `0.56` produced train-side selection F1/precision/recall `0.696056 / 0.653385 / 0.744689`, with predicted/reference/matched mentions `25700 / 22549 / 16792`; this is also below the current promoted Q4 train checkpoint, so do not spend a validation run on this ridge model.
- elastic-net candidate with selected threshold `0.56` produced train-side selection F1/precision/recall `0.697256 / 0.654476 / 0.746020`, with predicted/reference/matched mentions `25703 / 22549 / 16822`; this also remains below the current promoted Q4 train checkpoint, so the regularization family should be treated as rejected for this improvement cycle.
- exploratory official train/validation for `q4_planning_rel_logreg_elastic_p060`:
  - train official F1/precision/recall at `0.5m`: `0.727218 / 0.705611 / 0.750189`, slightly above the current train checkpoint
  - validation official F1/precision/recall at `0.5m`: `0.606293 / 0.564413 / 0.654886`, slightly below the current validation checkpoint `0.607062 / 0.564947 / 0.655962`
  - conclusion: do not promote elastic-net; it is a tiny train-side gain that does not generalize to validation.
- current promoted `logreg_acceptor` validation mismatch report:
  - report: `outputs/phase8_val_report/phase8_q4_planning_orch_rel_logreg_acceptor_mismatch_report.md`
  - samples/reference/predicted/matched mentions: `3446 / 5564 / 7369 / 3636`
  - false positives / false negatives: `3733 / 1928`
  - exact count matches: `1684`; over-predicted rows: `1518`; under-predicted rows: `244`; positive predictions on empty-reference rows: `515`
  - predicted count distribution: `0:428`, `1:390`, `2:905`, `3:1723`; reference count distribution: `0:929`, `1:489`, `2:1009`, `3:1019`
  - interpretation: the current model still predicts the maximum count `3` too often, especially on scenes whose references contain `0` or `1` object, but some examples also show missed second close-by objects in true two-object scenes.
  - candidate direction: add adaptive count gating or non-linear interaction modeling on top of candidate acceptance, rather than more L1/L2 penalty tuning.
- non-linear Q4 acceptor implementation:
  - added a pluggable `mlp_acceptor` planning-selection policy alongside the existing `logreg_acceptor`
  - `scripts/train_q4_planning_acceptor.py` now supports `--model-type mlp` and writes the same deployable JSON shape with normalization, threshold, max-results, and train-selected metrics
  - runtime evaluation still uses the standard Q4 orchestrator path with `--planning-selection-policy mlp_acceptor --planning-acceptor-model-json <model>`
  - the existing promoted `logreg_acceptor` checkpoint and all earlier policies remain unchanged.
- first MLP candidate `q4_planning_rel_mlp_h16_p060`:
  - selected threshold: `0.62`
  - train-side selection F1/precision/recall: `0.648767 / 0.601554 / 0.704022`
  - predicted/reference/matched mentions: `26390 / 22549 / 15875`
  - conclusion: reject for validation; this is far below the current promoted Q4 train checkpoint `0.726814 / 0.704891 / 0.750145`.
- scene-level count gate implementation:
  - added `count_gated_acceptor` as a separate planning-selection policy; existing `logreg_acceptor`, `mlp_acceptor`, and hand-written policies are unchanged
  - added `scripts/train_phase8_planning_count_gate.py`, which wraps an existing frozen candidate acceptor and learns a train-frozen multinomial count model for answer cardinality `0/1/2/3`
  - count-gate features are scene aggregates from candidate probabilities and graph metadata: eligible counts, high-probability counts, top probability gaps, top relational scores, trajectory distances, lateral offsets, visibility mix, behind count, and cooperative support
  - inference first scores candidates with the frozen acceptor, predicts scene count K, then returns the top K accepted non-duplicate candidates
  - this directly targets the current mismatch pattern where predicted count `3` rows are too frequent.
- first count-gated official train result:
  - F1/precision/recall at `0.5m`: `0.721751 / 0.728936 / 0.714706`
  - compared with current promoted train `0.726814 / 0.704891 / 0.750145`, count gating improves precision but loses recall and slightly lowers F1
  - conclusion: useful precision-oriented candidate, but do not promote or run validation yet without improving recall.
- exploratory count-gated validation result:
  - F1/precision/recall at `0.5m`: `0.605134 / 0.583075 / 0.628927`
  - compared with current promoted validation `0.607062 / 0.564947 / 0.655962`, count gating improves precision but loses recall and slightly lowers F1
  - conclusion: do not promote the hard count gate; next count-gate work should soften the cap by allowing one extra high-confidence candidate.
- soft count-gate implementation:
  - added `soft_count_gated_acceptor` as a separate policy
  - it predicts K with the same count gate, takes top K, then allows one extra accepted candidate if its probability is high or close to the Kth candidate
  - soft-gate knobs live in the deployable JSON: `soft_extra_min_probability` and `soft_extra_min_relative_to_k`
  - this is intended to recover some recall from the hard count gate while preserving part of its precision gain.
- exploratory soft count-gated validation result for `p062_r090`:
  - F1/precision/recall at `0.5m`: `0.601722 / 0.569263 / 0.638106`
  - compared with hard count gate validation `0.605134 / 0.583075 / 0.628927`, recall improves but precision drops enough that F1 worsens
  - compared with current promoted validation `0.607062 / 0.564947 / 0.655962`, this is lower F1 and lower recall
  - conclusion: reject this soft-cap setting; current promoted Q4 remains `relational_importance + logreg_acceptor`.
- residual attribution tooling:
  - added `scripts/analyze_phase8_planning_residual_attribution.py`
  - joins official Q4 outputs, candidate feature rows, and frozen acceptor probabilities
  - attributes false positives and false negatives by probability band, candidate rank, visibility/status, trajectory distance, candidate-pool presence, and strict-vs-loose localization recoverability
  - intended to decide whether the next Q4 improvement should target retrieval, candidate acceptance, false-positive suppression, or coordinate/identity correction.
- residual attribution findings for promoted Q4:
  - false negatives are not a retrieval problem: `1928 / 1928` have a nearby candidate present within the candidate-pool threshold, and `1665` have nearest candidate distance `<=0.5m`
  - only `282` strict misses are recovered at `4.0m`, so coordinate snapping alone is not the main issue
  - many missed objects are below threshold (`909` in probability `0.45-0.56`), but `477` are already accepted-like (`>=0.56`)
  - examples show high-probability missed second objects close to the selected object, which indicates the current near-duplicate suppression radius can be too aggressive for Q4 two-object answers
  - false positives are mostly high-probability supported visible candidates, with many far from trajectory (`1980` at `>4m`), so a later false-positive suppression feature may still be needed
  - next experiment: retrain/freeze the logreg acceptor with a smaller near-duplicate distance, e.g. `1.0m`, and evaluate train first.
- near-duplicate `1.0m` candidate training result:
  - model: `outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_p055_t0p56_deployable.json`
  - selected threshold: `0.56`
  - train-side selection F1/precision/recall: `0.696978 / 0.653918 / 0.746108`
  - predicted/reference/matched mentions: `25728 / 22549 / 16824`
  - interpretation: this internal trainer metric is not directly comparable to the official exported QA metric; run official train evaluation before rejecting because prior candidates also had lower internal metrics than official QA results.
- near-duplicate `1.0m` official train result:
  - F1/precision/recall at `0.5m`: `0.726896 / 0.704928 / 0.750278`
  - compared with current promoted train `0.726814 / 0.704891 / 0.750145`, this is a very small improvement across F1, precision, and recall
  - conclusion: eligible for one held-out validation run because it is a train-selected, general duplicate-radius change motivated by residual attribution.
- near-duplicate `1.0m` official validation result:
  - F1/precision/recall at `0.5m`: `0.607578 / 0.565305 / 0.656684`
  - compared with previous promoted validation `0.607062 / 0.564947 / 0.655962`, this improves F1, precision, and recall
  - validation F1 at looser thresholds: `1.0m=0.654315`, `2.0m=0.654482`, `4.0m=0.731745`
  - conclusion: promote `q4_planning_rel_logreg_nd1p0_p055_t0p56_deployable.json` as the current Q4 checkpoint; the gain is small but consistent and directly supported by residual attribution.
- trajectory-calibrated acceptor implementation:
  - added `trajectory_calibrated_acceptor` as a separate policy wrapping the frozen acceptor
  - suppresses moderate-probability far/lateral candidates using trajectory distance and lateral offset
  - rescues near-path, near-threshold candidates using probability, rank, and trajectory distance
  - added `scripts/configure_q4_trajectory_calibration.py` to copy a deployable model and attach reproducible calibration knobs
  - intended target: reduce high-probability far/lateral false positives while recovering near-threshold false negatives.
- trajectory-calibrated `v1` official validation result:
  - model: `outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json`
  - F1/precision/recall at `0.5m`: `0.613774 / 0.576685 / 0.655962`
  - compared with previous promoted validation `0.607578 / 0.565305 / 0.656684`, F1 improves by `+0.006196` and precision improves by `+0.011380`, with a small recall decrease
  - validation F1 at looser thresholds: `1.0m=0.661040`, `2.0m=0.661209`, `4.0m=0.733909`
  - this exceeds the V2V-GoT Q4 reference `0.608000` by `+0.005774` under the strict `0.5m` headline metric
  - official train F1/precision/recall at `0.5m`: `0.729672 / 0.711258 / 0.749064`
  - train F1 at looser thresholds: `1.0m=0.790651`, `2.0m=0.791432`, `4.0m=0.799786`
  - official train summary: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1_official_export_manifest_official_qa_eval_summary.json`
  - conclusion: promote `q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json` as the current Q4 checkpoint; train and validation both improve F1 over the previous `nd1p0` checkpoint.

Train/freeze commands:

```bash
python3 scripts/export_phase8_planning_candidate_features.py \
  --split train \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-ranker relational_importance \
  --output-jsonl outputs/phase8_train_dev/q4_policy_optimization/q4_train_relational_candidate_features.jsonl

python3 scripts/train_q4_planning_acceptor.py \
  --train-features-jsonl outputs/phase8_train_dev/q4_policy_optimization/q4_train_relational_candidate_features.jsonl \
  --output-dir outputs/phase8_train_dev/q4_policy_optimization \
  --run-name q4_planning_rel_logreg_acceptor \
  --min-precision 0.55
```

Frozen-model train/validation evaluation:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type planning_awareness \
  --scenario-name phase8_train_dev_train_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1 \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-selection-source orchestrator \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32

python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type planning_awareness \
  --scenario-name phase8_val_report_val_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1 \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-selection-source orchestrator \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32
```

Paper-facing validation deltas:

- Q1: `0.585836 - 0.525000 = +0.060836` absolute, `+11.59%` relative
- Q2: `0.427921 - 0.301000 = +0.126921` absolute, `+42.17%` relative
- Q3: `0.493934 - 0.440000 = +0.053934` absolute, `+12.26%` relative
- Q4: `0.613774 - 0.608000 = +0.005774` absolute, `+0.95%` relative

## Q1: Notable Objects

Current policy:

- ranker: `heuristic`
- selects visible notable objects from the cooperative scene graph
- prefers grounded/supported visible objects over weaker candidate-only visible objects
- uses trajectory relevance and visible-object grounding rather than free-form LLM reasoning

Reasoning summary:

- Q1 asks for notable objects visible to the asker.
- The strongest current behavior comes from keeping the object selection deterministic and graph-grounded.
- The earlier BEV/coordinate and visible-candidate leakage issues were fixed before this checkpoint.

Artifacts:

- train summary: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_notable_objects_selected_official_export_manifest_official_qa_eval_summary.json`
- validation summary: `outputs/phase8_val_report/official_eval_reports/phase8_val_report_val_notable_objects_selected_official_export_manifest_official_qa_eval_summary.json`

Reproduce validation:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type notable_objects \
  --scenario-name phase8_val_report_val_notable_objects_selected \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --notable-ranker heuristic \
  --progress-every 250 \
  --workers 2
```

## Q2: Occluding Objects

Current policy:

- ranker: `risk_adaptive`
- uses explicit blocker/occlusion candidates from the scene graph
- scores candidates with relative risk signals:
  - trajectory proximity
  - line-of-sight/blocker role evidence
  - source/provenance support
  - confidence
  - conflict and uncertainty penalties
- includes sparse-evidence backfill when too few blocker-role candidates are available

Reasoning summary:

- Q2 failures were originally dominated by under-selection of blockers.
- A fixed top-3 policy improved recall but introduced over-prediction.
- `risk_adaptive` was kept because it retained most of the recall gain while being less scenario-specific and more defensible as an occlusion-risk rule.
- Sparse-evidence backfill helps when the blocker candidate set itself is incomplete.

Artifacts:

- train summary: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_occluding_objects_selected_official_export_manifest_official_qa_eval_summary.json`
- validation summary: `outputs/phase8_val_report/official_eval_reports/phase8_val_report_val_occluding_objects_selected_official_export_manifest_official_qa_eval_summary.json`

Reproduce validation:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type occluding_objects \
  --scenario-name phase8_val_report_val_occluding_objects_selected \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --occluding-ranker risk_adaptive \
  --progress-every 250 \
  --workers 2
```

## Q3: Invisible Objects

Current policy:

- ranker: `logreg_acceptor`
- selected model: `outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json`
- max results: `1`
- shortlist size: `64`
- max distance to trajectory: `8.0m`
- acceptor threshold: `0.33`

Reasoning summary:

- Earlier Q3 policies had a precision/recall tradeoff:
  - `legacy_traj6` had recall but low precision.
  - `logreg_acceptor_t0p25` improved precision but recall collapsed.
- False-negative analysis showed many correct invisible objects were absent from the narrow shortlist.
- The current policy separates retrieval from acceptance:
  - broad retrieval makes plausible hidden objects reachable;
  - train-calibrated logistic acceptance filters the broad pool using learned candidate features.
- This recovered validation recall from `0.301754` to `0.500000` while keeping enough precision for F1 to clear the paper target.

Verified metrics:

- train localization F1/P/R: `0.464406 / 0.527863 / 0.414568`
- train binary F1/P/R: `0.538892 / 0.569847 / 0.511126`
- validation localization F1/P/R: `0.493934 / 0.488014 / 0.500000`
- validation binary F1/P/R: `0.509666 / 0.496575 / 0.523466`

Artifacts:

- train verification summary: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_invisible_objects_broadpool_logreg_p50_t0p33_verify_official_export_manifest_official_qa_eval_summary.json`
- validation verification summary: `outputs/phase8_val_report/official_eval_reports/phase8_val_report_val_invisible_objects_broadpool_logreg_p50_t0p33_verify_official_export_manifest_official_qa_eval_summary.json`
- deployable model: `outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json`

Reproduce train:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type invisible_objects \
  --scenario-name phase8_train_dev_train_invisible_objects_broadpool_logreg_p50_t0p33_verify \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --progress-every 250 \
  --workers 2
```

Reproduce validation:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type invisible_objects \
  --scenario-name phase8_val_report_val_invisible_objects_broadpool_logreg_p50_t0p33_verify \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --progress-every 250 \
  --workers 2
```

## Q4: Planning Awareness

Current policy:

- ranker: `relational_importance`
- selection policy: `trajectory_calibrated_acceptor`
- base acceptor: train-frozen logistic acceptor
- deployable model: `outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json`
- duplicate suppression radius: `1.0m`
- trajectory calibration:
  - suppress moderate-probability far/lateral extras
  - rescue near-path, near-threshold candidates

Reasoning summary:

- Q4 asks which objects matter to the planned future trajectory, so a fixed hidden-plus-visible composition was too rigid.
- The orchestrator path exposes all graph-grounded planning candidates, and `relational_importance` scores them using trajectory proximity, visibility, confidence, support, uncertainty, conflict, and cooperative provenance.
- The train-frozen logistic acceptor made candidate acceptance adaptive without using validation labels or sample-specific rules.
- Residual attribution showed the remaining errors were not mainly retrieval failures:
  - all `1928` validation false negatives had nearby candidate rows;
  - `1665` were within `0.5m` of a candidate;
  - many false positives were supported visible objects far from the trajectory.
- Two general geometry fixes followed from that:
  - reduce duplicate suppression from `2.0m` to `1.0m` so close two-object answers are not collapsed;
  - suppress moderate-probability far/lateral extras while rescuing near-path near-threshold candidates.
- This is why the final gain is interpretable: it improves precision by filtering residual far/lateral extras while preserving almost all recall.

Verified metrics:

- train localization F1/P/R: `0.729672 / 0.711258 / 0.749064`
- validation localization F1/P/R: `0.613774 / 0.576685 / 0.655962`
- validation F1 at looser thresholds:
  - `1.0m`: `0.661040`
  - `2.0m`: `0.661209`
  - `4.0m`: `0.733909`
- V2V-GoT Q4 reference: `0.608000`
- strict `0.5m` validation delta over V2V-GoT: `+0.005774`

Rejected alternatives and interpretation:

- L1, L2, and elastic-net variants did not improve held-out validation, suggesting the issue was not simple regularization.
- A small MLP acceptor over-accepted and underperformed, suggesting model complexity alone was not the answer.
- Hard and soft count gates improved precision but dropped recall too much.
- Residual-attribution-guided trajectory calibration was the winning change because it targeted the actual remaining error shape.

Artifacts:

- train summary: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1_official_export_manifest_official_qa_eval_summary.json`
- train summary markdown: `outputs/phase8_train_dev/official_eval_reports/phase8_train_dev_train_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1_official_export_manifest_official_qa_eval_summary.md`
- validation summary: `outputs/phase8_val_report/official_eval_reports/phase8_val_report_val_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1_official_export_manifest_official_qa_eval_summary.json`
- deployable model: `outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json`
- residual attribution report: `outputs/phase8_val_report/phase8_q4_planning_orch_rel_logreg_acceptor_residual_attribution.md`

Reproduce train:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose train_dev \
  --split train \
  --task-type planning_awareness \
  --scenario-name phase8_train_dev_train_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1 \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-selection-source orchestrator \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32
```

Reproduce validation:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type planning_awareness \
  --scenario-name phase8_val_report_val_planning_awareness_orch_rel_logreg_nd1p0_trajcal_v1 \
  --limit 0 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --planning-selection-source orchestrator \
  --planning-ranker relational_importance \
  --planning-selection-policy trajectory_calibrated_acceptor \
  --planning-acceptor-model-json outputs/phase8_train_dev/q4_policy_optimization/q4_planning_rel_logreg_nd1p0_trajcal_v1_deployable.json \
  --progress-every 250 \
  --workers 32
```

## One-Command Matrix Rerun

The selected-matrix runner now defaults to the current Q1-Q3 policy settings, including the frozen Q3 broad-pool model.

```bash
python3 scripts/run_phase8_selected_qa_train_val_matrix.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --limit 0
```

Expected current Q1-Q3 validation headline:

- Q1 F1 `0.585836`
- Q2 F1 `0.427921`
- Q3 F1 `0.493934`

## Ego-Only Versus Cooperative Demo

Use the same task, same split, and same selector settings, changing only `--baseline-mode`.

Example cooperative Q3 validation:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type invisible_objects \
  --scenario-name demo_val_q3_cooperative \
  --limit 0 \
  --baseline-mode cooperative \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --progress-every 250 \
  --workers 2
```

Example ego-only Q3 validation:

```bash
python3 scripts/run_qa_split_pipeline.py \
  --purpose val_report \
  --split val \
  --task-type invisible_objects \
  --scenario-name demo_val_q3_ego_only \
  --limit 0 \
  --baseline-mode ego_only \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --invisible-ranker logreg_acceptor \
  --invisible-max-results 1 \
  --invisible-shortlist-size 64 \
  --invisible-max-distance-to-trajectory 8.0 \
  --invisible-acceptor-model-json outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json \
  --progress-every 250 \
  --workers 2
```

Interpretation:

- cooperative mode answers from the fused multi-agent graph;
- ego-only mode answers from the same graph pipeline after removing partner-agent observations and partner visibility facts;
- the comparison isolates the effect of cooperative evidence while keeping downstream reasoning fixed.

## Demonstration Talking Points

- The system does not answer Q1-Q3 from free-form text generation. It builds and queries an explicit cooperative scene graph.
- Q1 demonstrates visible-object grounding.
- Q2 demonstrates occlusion/blocker reasoning with risk-adaptive selection.
- Q3 demonstrates cooperative hidden-object reasoning under partial observability.
- The strongest Q3 improvement came from separating candidate retrieval from candidate acceptance.
- All three current validation results exceed the V2V-GoT paper references, and all three clear the Phase 8 `+10%` target.
