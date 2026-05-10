# Project Info: Cooperative Scene-Graph QA Architecture

## Purpose

This note captures the project's core architecture: storing cooperative driving scenes as explicit graphs, querying those graphs for task-relevant evidence, and training compact task-aware models over graph-derived features. V2V-GoT-QA is the current benchmark used to evaluate this architecture, not the full boundary of the method.

Short version:

- The contribution is a cooperative scene-graph reasoning layer for autonomous driving.
- Multi-agent perception, visibility, provenance, uncertainty, motion, and planning context are stored explicitly.
- Downstream tasks query this graph and form features for simple task-aware models.
- V2V-GoT provides the typed evaluation surface and official metrics.
- The benchmark adapter is V2V-GoT-specific, but the scene-graph representation and task-head pattern are broader.

## Current Evaluation Architecture

For V2V-GoT-QA evaluation, the project follows a typed QA flow.

1. A V2V-GoT QA record is loaded.
2. The record is classified into a benchmark task type.
3. A cooperative scene graph is built from perception, pose, trajectory, and QA metadata.
4. The scene is enriched with associations, object candidates, visibility, support, temporal, and relation features.
5. A router sends the sample to the appropriate task handler.
6. The handler produces official-style answer text.
7. The official evaluator scores the exported answers.

Important implementation points:

- `V2VGoTQABenchmarkAdapter.classify_record` uses `qa_type_id` first, with keyword fallback only for older or incomplete records.
- `V2VGoTSceneAdapter.build_scene` extracts structured scene seeds from the raw V2V-GoT record.
- `V2VGoTQAPhase5AEvaluator.prepare_sample` loads processed assets and enriches the scene before routing.
- `V2VGoTQARouter.answer` dispatches by task type.
- `SceneQueryEngine` provides reusable query primitives over the scene graph.

Relevant files:

- `src/kg_coop_drive/infrastructure/v2vgot_benchmark_adapter.py`
- `src/kg_coop_drive/infrastructure/v2vgot_scene_adapter.py`
- `src/kg_coop_drive/application/qa/v2vgotqa_evaluator.py`
- `src/kg_coop_drive/application/qa/v2vgotqa_router.py`
- `src/kg_coop_drive/application/scene_graph/query_engine.py`

## Benchmark Router Design

The current V2V-GoT adapter uses a static router because the benchmark records expose known QA task families and official QA type IDs.

The benchmark contains known QA task families:

| Task | QA type | Meaning |
| --- | ---: | --- |
| Q1 | `11` | notable objects |
| Q2 | `12` | occluding objects |
| Q3 | `13` | invisible objects |
| Q4 | `14` | planning awareness |
| Q5 | `15` | object motion prediction |
| Q6 | `16` | agent motion prediction |
| Q7 | `17` | object motion prediction |
| Q8 | `18` | control settings |
| Q9 | `19` | future trajectory |

The router does not infer arbitrary intent from arbitrary text. Instead, it uses the benchmark task identity to select the correct graph-query/model handler. Each handler then encodes the task-specific readout over the shared cooperative scene graph.

This is an evaluation adapter for a broader architecture. The general method is not "hard-code V2V-GoT"; the general method is "build a cooperative scene graph, then train/query task-aware heads over grounded graph features."

## How V2V-GoT Differs

The original V2V-GoT method is based on a graph-of-thoughts MLLM pipeline.

V2V-GoT uses fixed QA nodes. Later nodes receive context from earlier nodes. For example, prediction and planning nodes are generated with previous QA answers as context, then passed to a LLaVA-style MLLM for answer generation.

Important nuance:

- V2V-GoT does send generated prompts to an MLLM.
- But those prompts are not arbitrary user prompts.
- They are created by fixed task-node generation scripts.
- The graph structure is predefined: NQ1, NQ2, NQ3, and so on through NQ9.

So V2V-GoT is also structured. Its generalization comes from the MLLM and training distribution, but its benchmark execution still depends on fixed graph nodes and task-specific prompt construction.

Our system replaces the MLLM answer generator with explicit KG construction, query logic, and trained task models.

## Comparison

| Dimension | V2V-GoT | This project |
| --- | --- | --- |
| Task organization | Fixed graph QA nodes | V2V-GoT adapter routes fixed benchmark tasks |
| Main answer engine | MLLM/LLaVA text generation | KG query handlers and trained task models |
| Context passing | Prior QA answers appended to later prompts | Structured scene graph enrichment |
| Interpretability | Lower; answer produced by model generation | Higher; objects/features/ranking/model choices are inspectable |
| General method | Prompt/context construction around graph nodes | Cooperative scene graph plus task-aware readouts |
| Benchmark defensibility | Official graph/prompt pipeline | Official task IDs, train/val split, official evaluator |

## Train/Evaluation Protocol Framing

The clean way to describe our protocol is by task family, not by individual V2V-GoT prompt node.

We group the V2V-GoT QA nodes into reusable driving task families:

- visible-object grounding;
- occlusion reasoning;
- hidden-object discovery;
- planning awareness;
- object motion prediction;
- agent motion prediction;
- control selection;
- future trajectory prediction.

Each family consumes the same cooperative scene graph but learns or configures its own compact readout:

- selectors for object-grounding tasks;
- acceptors for hidden/planning-object tasks;
- classifiers for agent/control decisions;
- regressors for motion and trajectory outputs.

These readouts are trained or selected using the official train split and then evaluated as frozen artifacts on the held-out V2V-GoT-QA evaluation split. Q5 and Q7 are treated as the same object-motion prediction family in our architecture because the inspected official evaluator/targets expose the same effective motion-prediction problem under the shared model/export path.

This mirrors the train-to-held-out-evaluation discipline of the released V2V-GoT pipeline, but the modeling unit is different:

- V2V-GoT fine-tunes an MLLM over graph-of-thought prompts;
- our method trains task-family-specific graph readouts over an explicit cooperative scene graph.

So the contribution is not that we optimize static QA IDs. The contribution is that we turn driving scenes into a reusable structured substrate and train simple task-aware heads over that substrate.

## Why The Benchmark Mapping Is Defensible

Static task mapping is defensible for V2V-GoT evaluation because the benchmark itself is typed.

The official evaluation is not asking a free-form assistant to answer arbitrary natural language. It evaluates specific QA families with specific expected output formats. In that setting, using `qa_type_id` is appropriate because it avoids accidental ambiguity and keeps the method aligned with the benchmark contract.

The strongest defense is:

- The benchmark defines the task ontology.
- The official records expose the QA type.
- The official evaluator scores by QA type.
- Therefore, routing by QA type is a faithful use of the benchmark metadata.

The important distinction is that the mapping is benchmark-specific, while the representation and task-head pattern are more general.

## What Is Generalized Today

Current generalization happens in five layers.

1. Scene-graph schema:
   The core abstraction is a cooperative driving graph with objects, agents, trajectories, relations, visibility, provenance, uncertainty, and temporal evidence.

2. Benchmark-node coverage:
   The router covers the V2V-GoT task families rather than only one hand-written case.

3. Shared scene representation:
   The same cooperative scene graph stores objects, tracks, agent poses, provenance, visibility, occlusion, relations, trajectories, and support evidence.

4. Shared query primitives:
   Handlers reuse selection, filtering, relation, trajectory, and attribute query operations instead of duplicating raw parsing logic.

5. Learned task models:
   Several tasks use train-split-trained models or learned scoring functions rather than fixed constants or pretrained answer weights.

This means the system can generalize across scenes and tasks that can be expressed with the same driving-scene graph concepts. It does not require a large end-to-end MLLM for every task; it can train smaller task-aware models over grounded graph features.

## What Would Be Needed For Open-Ended Generalization

To generalize beyond benchmark task IDs and toward open-ended language, the next architectural step would be an explicit `QueryIntent` layer.

The intent representation could include:

- task family, such as object selection, object motion, agent motion, control, or trajectory;
- target entity type, such as vehicle, pedestrian, obstacle, CAV, or ego;
- constraints, such as visible, occluded, invisible, near planned path, source agent, time horizon, or support level;
- required output fields, such as object coordinates, action label, future waypoint, control command, or trajectory;
- scoring objective, such as localization F1, binary accuracy, action accuracy, or L2 error.

Then two frontends could compile into the same intent:

- a benchmark frontend using `qa_type_id`;
- a language frontend using a parser or LLM planner.

The handler layer would remain grounded and constrained. The LLM would be used to propose structured intent, not to hallucinate the final answer directly.

This would make the project more general while preserving the safety and auditability advantages of the current KG approach.

## Q5 And Q7 Note

Q5 and Q7 are separate benchmark questions, but in the current validation data they can evaluate identically when the exported records contain the same object-motion answer targets.

Observed behavior:

- Q5 uses `qa_type_id=15`.
- Q7 uses `qa_type_id=17`.
- Both are evaluated by the official object-motion prediction evaluator.
- In the inspected validation split, Q5 and Q7 rows have matching sample IDs and identical ground-truth answer text.
- Therefore, when the same model/export logic is used, the official metrics are expected to be identical.

This does not necessarily mean the code is broken. It means that, for this split and evaluator path, the two QA nodes are scored against the same effective target. The distinction between Q5 and Q7 may live more in the V2V-GoT graph context and prompt framing than in the final official motion target.

For reporting, this should be stated clearly rather than hidden.

## Defensible Research Framing

A clean way to describe the method:

> We propose a cooperative scene-graph reasoning layer for autonomous driving. Multi-agent perception, visibility, provenance, uncertainty, motion, and planning context are stored explicitly as graph structure. Downstream tasks query this graph and train compact task-aware models over grounded features. V2V-GoT-QA is the current benchmark instantiation, while the reusable contribution is the graph representation and task-aware reasoning stack.

This framing is honest and strong.

This avoids overclaiming open-language generality while making the actual contribution clear:

- cooperative KG construction;
- explicit scene grounding;
- reusable query primitives;
- task-specific learned prediction models;
- reproducible train/validation/e2e artifacts;
- official evaluator compatibility.

## How The KG Is Queried

The KG is represented as a `CooperativeScene`. It is not a database server; it is an in-memory structured scene graph for each QA sample.

Core scene entities:

- `AgentContext`: participating CAVs, poses, velocities, and planned trajectories.
- `ObservationEvidence`: one object observation from one source agent.
- `ObjectTrack`: a fused object belief with position, velocity, confidence, status, uncertainty, conflict, observations, and provenance.
- `VisibilityFact`: whether an object is visible, occluded, or uncertain for an agent.
- `RelationFact`: derived scene relations such as `near_trajectory`, `near_first_waypoint`, `path_relevant`, `cooperatively_supported`, and `low_conflict`.
- `Trajectory`: the asker/ego future path used for path relevance and planning features.

The query layer exposes reusable graph operations:

- select all tracked objects;
- filter by semantic type;
- filter by visibility state for one agent;
- filter by source/provenance agent;
- filter by relation type;
- filter by distance to future trajectory;
- read attributes such as confidence, position, support count, uncertainty, and conflict;
- trace provenance back to source agents and observations.

The task handlers build richer ranking/model features from these primitives. The important point is that the final answer is grounded in graph objects and graph-derived measurements, not directly generated from raw language.

## How The Scene Becomes Rich

Each QA sample starts as a lightweight scene seed, then the evaluator enriches it using processed V2V-GoT assets.

The enrichment pipeline does the following:

1. Loads processed detections, visibility assets, object metadata, agent poses, and trajectories.
2. Associates observations to existing object tracks.
3. Promotes unmatched observations into candidate tracks.
4. Associates cross-agent observations.
5. Enriches tracks with cooperative support evidence.
6. Merges nearby tracks.
7. Computes track quality, uncertainty, conflict, and support features.
8. Infers visibility with respect to the asker.
9. Builds relations such as path relevance and cooperative support.
10. For motion tasks, also loads the previous frame and performs temporal track updates to estimate velocity.

This is how information from multiple vehicles is preserved and blended:

- every fused object keeps provenance through `source_agent_ids` and `observation_ids`;
- support count tells whether one or multiple agents observed the object;
- conflict and uncertainty keep track of disagreement/noisiness instead of hiding it;
- visibility facts are agent-relative, so an object can be visible to another CAV but occluded to the asker;
- relations encode task-relevant geometry after fusion, such as proximity to the planned path;
- temporal enrichment estimates motion from current and previous graph states.

This is a key defensibility point. The cooperative contribution is not just "more boxes." It is a structured representation of who saw what, how reliable it is, where it lies relative to the ego plan, and whether it matters for downstream driving questions.

## Question-By-Question Behavior

| Task | What The Question Asks | Main KG Information Used | Output Scored By Evaluator |
| --- | --- | --- | --- |
| Q1 notable objects | Which visible objects matter now? | visible objects, distance to trajectory, distance to first waypoint, support, confidence, conflict, uncertainty, path relevance | object localization F1 |
| Q2 occluding objects | Which visible objects block view of hidden relevant objects? | visible objects, hidden/occluded objects, bearing alignment, depth ordering, hidden-object trajectory relevance, support, conflict | object localization F1 |
| Q3 invisible objects | Which important objects are not visible to the asker? | occluded objects, distance to trajectory, distance to asker, road-region features, support, confidence, uncertainty, conflict, learned acceptor probability | object localization F1 |
| Q4 planning awareness | Which objects should affect planning? | candidate object score, visibility, path distance, confidence, source-agent count, observation count, uncertainty, conflict, cooperative support, diversity/count features | object localization F1 |
| Q5 object motion | Where will relevant objects move? | selected path-relevant objects, current position, velocity, visibility, status, support, conflict, uncertainty, path-relative features | object/action metrics and future L2 |
| Q6 agent motion | Is another CAV's motion notable/relevant? | other CAV planned trajectory, ego future path, path overlap, endpoint distance, heading alignment, nearby object counts, learned classifier | binary accuracy |
| Q7 object motion | Similar final motion target to Q5 in current val split, with different graph-node framing in V2V-GoT | same object-motion graph/model features as Q5 in our current handler | object/action metrics and future L2 |
| Q8 control settings | What speed/steering should ego use? | top risk objects, trajectory distance, asker distance, visibility, support, confidence, conflict, uncertainty, lateral offset, local path geometry | action L1/edit distance |
| Q9 future trajectory | What future path should ego follow? | current ego position, control metadata, speed/steering context, distance/angle metadata, optional learned trajectory residuals | trajectory L2 |

## Q1: Notable Objects

Q1 is an object-selection problem over objects visible to the asker.

The handler ranks visible tracks using:

- distance to the future trajectory;
- distance to the first waypoint;
- distance to the asker;
- confidence;
- support count;
- object status;
- conflict and uncertainty;
- relation confidence for path relevance, first-waypoint proximity, cooperative support, and low conflict.

The answer is object IDs rendered into official-style text. The exporter converts those IDs into object coordinate answers for the official evaluator.

Why this is KG-native:

- visibility is agent-relative;
- path relevance is geometric and relation-based;
- cooperative support is explicit provenance, not just an averaged confidence score.

## Q2: Occluding Objects

Q2 is not just "find visible objects." It asks for visible objects that plausibly block the asker's view of important hidden objects.

The handler builds visible/hidden pairs:

- visible candidates come from objects visible to the asker;
- hidden candidates come from occluded or uncertain objects;
- pairs are scored by bearing alignment from the asker, depth ordering, distance between objects, and hidden-object distance to the future trajectory;
- the visible blocker receives additional support from confidence, provenance, conflict, status, and path distance.

This is a good example of graph reasoning: the selected object is not necessarily important by itself. It is important because of a relation between a visible object, a hidden object, the asker's pose, and the planned path.

## Q3: Invisible Objects

Q3 selects important objects that the asker cannot currently see.

The candidate set is mostly occluded objects. The handler uses:

- distance to the planned trajectory;
- distance to the asker;
- object confidence;
- support count;
- object status;
- conflict and uncertainty;
- relative position to the asker;
- road-region and temporal/backtracking guards;
- optional train-calibrated logistic/MLP acceptor.

For learned acceptors, per-candidate features include rank, role score, relative x/y, distance to asker, distance to trajectory, support count, confidence, conflict, uncertainty, age, miss count, and status flags.

The model answers a focused question: "from the graph-produced hidden-object shortlist, which candidates should be accepted as official Q3 answers?"

## Q4: Planning Awareness

Q4 asks which objects should be considered for planning.

This is broader than Q1-Q3. An object can be planning-relevant because it is visible and near the path, hidden and risky, cooperatively supported, uncertain, conflicting, or important as part of the overall scene context.

The planning-awareness layer computes per-object features:

- distance to trajectory;
- distance to first waypoint;
- distance to asker;
- visibility flags;
- confidence;
- source-agent count;
- observation count;
- uncertainty and conflict;
- whether the object is cooperative;
- whether the asker observed it;
- status flags;
- rank gaps and local diversity against higher-ranked objects.

Some variants also compute scene-level count-gate features:

- number of candidates;
- number above probability thresholds;
- top probabilities and gaps;
- top scores and gaps;
- top object distances;
- visible/occluded/cooperative counts among the top candidates.

This makes Q4 a bridge between object retrieval and planning. It is not only asking "where is an object?" It asks "which parts of the cooperative scene should influence a planning decision?"

## Q5 And Q7: Object Motion

Q5 and Q7 are both evaluated as object-motion prediction in the current official path.

The handler first selects relevant object tracks:

- objects are ranked by distance to the asker's trajectory;
- status and confidence break ties;
- the model can limit `selection_max_objects`;
- the model can include occluded/uncertain objects even when they are farther from the trajectory.

For each selected object, the learned object-motion predictor builds features such as:

- current object `x`, `y`;
- velocity `vx`, `vy`, and speed;
- distance to trajectory;
- distance to asker;
- confidence;
- support count;
- conflict and uncertainty;
- status flags;
- visibility flags;
- object position relative to the asker;
- vector from object to the asker's goal;
- distance from object to the asker's goal;
- closest point on asker's path relative to the object;
- normalized closest-path index;
- object velocity projected toward/lateral to the asker's goal.

The regression tree then predicts future endpoint deltas. The output is rendered as object motion text and exported to the official evaluator.

Why the regression tree helped:

- motion behavior is not globally linear;
- path-relative regimes matter;
- objects near the ego path behave differently from far objects;
- occluded/supported/candidate objects have different noise profiles;
- tree splits can isolate these regimes before predicting deltas.

Important Q5/Q7 nuance:

- Q5 and Q7 are separate V2V-GoT graph nodes.
- However, in the inspected validation data, paired Q5/Q7 rows had matching sample IDs and identical ground-truth answer text.
- Both are scored by the same official object-motion evaluator.
- Therefore identical Q5/Q7 metrics are expected when the same exported motion predictions are used.

This should be reported carefully. It does not mean the architecture cannot distinguish Q5 and Q7; it means the current official target for this split collapses them to the same effective motion-answer target.

## Q6: Agent Motion

Q6 predicts whether another CAV's motion is notable.

The learned predictor uses features from the other agent and the ego path:

- other CAV planned final displacement;
- other CAV planned final distance;
- maximum step in the other CAV's planned trajectory;
- minimum distance from other CAV to the asker's future path;
- asker's future path length and max step;
- overlap ratio between other CAV path and asker's path;
- endpoint distance to asker's final point;
- heading alignment between the other CAV's plan and the asker's goal;
- curvature proxy of the other CAV path;
- whether the other CAV is ahead along the asker's goal direction;
- nearby object counts around the asker;
- nearby dynamic object counts.

The output includes both motion text and a notable/not-notable label. The official Q6 score is binary accuracy.

This is a planning-context classifier, not simply a velocity threshold. It asks whether another agent's planned motion matters to the asker.

## Q8: Control Settings

Q8 predicts speed and steering labels.

The control policy first ranks risk objects using:

- distance to the planned trajectory;
- distance to the asker;
- visibility state;
- candidate status;
- asker support;
- multi-agent support.

The learned control model then uses a feature vector with:

- top object risk;
- second object risk;
- risk gap;
- top object distance to trajectory;
- top object distance to asker;
- confidence;
- uncertainty;
- conflict;
- support count;
- visible/uncertain/occluded flags;
- candidate flag;
- asker-supported flag;
- multi-supported flag;
- lateral offset;
- number of selected top objects.

The extended feature set adds local trajectory geometry:

- distance to first waypoint;
- normalized nearest waypoint index;
- normalized along-path progress;
- local path curvature;
- heading alignment.

The model predicts speed and steering classes. This is where object retrieval becomes a planning-layer feature vector.

## Q9: Future Trajectory

Q9 predicts a future trajectory for the asker.

The planner intentionally avoids simply replaying the ground-truth future trajectory field. It uses:

- current position parsed from the question or scene;
- suggested speed and steering metadata;
- asker CAV identity;
- distance and angle metadata;
- sine/cosine expansions;
- optional learned linear tail residuals.

If no learned model is available, the fallback is a control-conditioned kinematic prior: speed controls step size, steering controls heading angle, and points are rolled forward.

This makes Q9 the final planning task: object/scene understanding and control context are turned into a future path that the evaluator scores with trajectory L2.

## One End-To-End Example

Consider a scene where the asker CAV plans to continue forward through an intersection.

The processed assets contain:

- an ego-visible vehicle ahead near the planned path;
- another vehicle visible only to a cooperating CAV on the right;
- a partially occluded object behind the visible vehicle;
- another CAV with its own planned trajectory;
- ego control metadata suggesting slow/straight motion.

The graph construction would create:

- object tracks for the visible vehicle, hidden/right-side vehicle, and occluded object;
- provenance showing which CAV observed each track;
- visibility facts saying which objects are visible/occluded to the asker;
- support/conflict/uncertainty scores;
- relations showing which objects are near the planned path or first waypoint;
- temporal velocities if previous-frame evidence is available.

Then the task handlers read the same graph differently:

- Q1 selects the visible vehicle near the path as notable.
- Q2 may select that same visible vehicle as an occluder if it aligns with a hidden object behind it.
- Q3 selects the hidden/right-side or occluded object if it is path-relevant and not visible to the asker.
- Q4 may include both the visible blocker and hidden path-relevant object as planning-awareness objects.
- Q5/Q7 select path-relevant object tracks and predict their next endpoint using velocity plus path-relative learned features.
- Q6 checks whether the other CAV's planned path overlaps or conflicts with the asker's future path.
- Q8 converts top object risk, visibility, lateral offset, and path geometry into speed/steering labels.
- Q9 converts current position plus control metadata into a future trajectory.

The same cooperative scene graph supports all nine questions. The difference is not that each question gets a completely separate world model. The difference is the readout: each task asks for a different projection of the same structured cooperative scene.

## Why This Is Different

The project is different from pure cooperative perception because it does not stop at better detections. It turns cooperative perception outputs into a queryable reasoning object.

It is different from pure MLLM prompting because the answer is not generated from unconstrained text alone. The system first builds explicit objects, relations, provenance, visibility, and trajectory features, then uses task-specific query/model layers to produce benchmark answers.

It is different from a normal object detector/tracker because:

- it preserves multi-agent provenance;
- it models agent-relative visibility;
- it keeps uncertainty and conflict as first-class features;
- it builds path-relative and planning-relative relations;
- it supports multiple downstream QA/planning tasks from one graph.

The novelty claim should be phrased around the architecture:

> A cooperative driving scene graph can act as an interpretable middle layer between multi-agent perception and downstream QA/planning tasks. Instead of relying only on black-box prompt generation or flat perception outputs, downstream tasks can query grounded graph structure and train simple task-aware models over shared cooperative features.

This is broader than V2V-GoT. V2V-GoT gives us a concrete benchmark with typed questions and official metrics. The underlying approach is that scene-graph storage improves cooperation and reasoning quality because it preserves information that is usually flattened away: who observed an object, whether the ego can see it, whether another CAV supports it, how uncertain it is, whether evidence conflicts, how it relates to the ego path, and how it moves over time.

## Position Relative To Driving LLM/VLM Work

A popular current direction is to adapt LLMs/VLMs to autonomous driving using instruction tuning, visual question answering, chain-of-thought, or end-to-end language-guided planning.

Representative examples:

- [DriveLM](https://arxiv.org/abs/2312.14150) argues that driving decisions are naturally multi-step and proposes Graph VQA over perception, prediction, and planning QA pairs.
- [LMDrive](https://arxiv.org/abs/2312.07488) frames closed-loop driving as language-guided end-to-end driving with multimodal sensor data and natural-language instructions.
- [DriveMLM](https://arxiv.org/abs/2312.09245) uses a multimodal LLM for behavior planning and standardizes language decisions into vehicle-control decision states.
- [Dolphins](https://research.nvidia.com/labs/avg/publication/ma.cao.etal.arxiv2023/) instruction-tunes a multimodal driving assistant with grounded chain-of-thought over video/image, text instructions, and historical controls.
- [DriveVLM](https://arxiv.org/abs/2402.12289) uses VLM reasoning modules for scene description, scene analysis, and hierarchical planning, while also motivating a hybrid system because of spatial-reasoning and compute limitations.
- [EM-VLM4AD](https://arxiv.org/abs/2403.19838) explicitly targets efficiency, noting that large LLM backbones and image encoders are expensive for real-time driving systems.
- [ST-VLM](https://arxiv.org/abs/2503.19355) focuses on kinematic instruction tuning because generic VLMs still struggle with traveled distance, speed, movement direction, and related spatio-temporal quantities.

These papers support the broader motivation: language and vision-language models are useful for interaction, semantic abstraction, and broad prior knowledge, but driving also demands precise spatial, temporal, provenance, and planning-grounded reasoning.

Our approach is an interesting complementary direction rather than a rejection of LLMs.

The core argument:

> A driving LLM must learn or infer scene structure inside its latent representation. A cooperative scene graph stores that structure explicitly. Once the structure is explicit, many downstream tasks can be solved by small query/model heads that are cheaper, easier to audit, easier to retrain, and easier to transfer across datasets with compatible scene schemas.

Why this can be more generalizable in a fundamental sense:

1. Generalization through explicit invariants:
   A graph schema encodes stable driving invariants: agents, objects, poses, visibility, provenance, relations, trajectories, uncertainty, and temporal motion. These concepts recur across datasets even when camera views, prompt wording, object frequencies, or annotation styles change.

2. Generalization by swapping adapters:
   A new dataset mainly needs a new adapter from its detections/tracks/poses into the graph schema. The query and task-head pattern can remain similar. In a pure VLM setting, a new dataset often requires new instruction data, prompt engineering, and model tuning to teach the model the same latent structure again.

3. Generalization by task-specific heads:
   If a new task asks for risk ranking, control choice, motion endpoint, or trajectory adjustment, we can form features from the same graph and train a compact head. We do not need to re-train a full multimodal language model for every task.

4. Generalization through auditability:
   When a graph-based model fails, we can inspect whether the issue came from detection, association, visibility, relation building, candidate selection, or the learned head. This makes improvement more systematic than only reading a generated explanation after an LLM answer.

5. Generalization under resource constraints:
   Compact models over structured features can run cheaply and repeatedly. This matters for driving because real-time systems care about latency, memory, reproducibility, and deployment cost.

6. Generalization to cooperation:
   Multi-agent driving needs provenance. The question "why do we believe this object exists?" is not only visual; it depends on which CAV observed it, whether another CAV corroborated it, and whether reports conflict. A graph naturally stores this. A prompt-only system must compress it into text or tokens and hope the model uses it consistently.

The claim should be careful:

- LLMs/VLMs may generalize better to open-ended language, unusual semantic categories, and human interaction.
- Graph-based task heads may generalize better to repeated structured driving tasks where geometry, visibility, provenance, and planning context dominate.
- The strongest future system may combine both: use an LLM to map open language into structured intent, but keep final scene grounding, candidate selection, and safety-critical scoring in the graph/model layer.

So the project can be framed as a middle path:

> Instead of betting that a large VLM will internally learn all driving structure from prompts and instruction data, we externalize the structure as a cooperative graph. This gives us a reusable substrate for many tasks, and lets us train smaller specialized models that are transparent enough to debug and cheap enough to run in systematic evaluations.

## Generalization To Other Driving Tasks And Datasets

The architecture can generalize if the new task or dataset can provide or approximate the same scene-graph schema:

- agent poses;
- object observations or tracks;
- source-agent provenance;
- object confidence;
- visibility or occlusion estimates;
- planned or candidate ego trajectories;
- temporal history for velocity;
- task-specific labels for training heads.

For another dataset, most reusable pieces are:

- the `CooperativeScene` schema;
- observation association and track enrichment;
- visibility and relation construction;
- query primitives;
- task handler interfaces;
- train/eval artifact discipline.

The parts likely requiring adaptation are:

- dataset adapters;
- coordinate-frame normalization;
- visibility estimation;
- object taxonomy mapping;
- official answer rendering;
- evaluator wrappers;
- learned model retraining.

This is a grounded form of generalization: the architecture transfers, but the dataset adapter and learned heads must be validated on the new benchmark. The claim is not that one trained checkpoint works everywhere. The claim is that the representation, query interface, and task-head pattern can be reused: build the graph, extract task-relevant features, train a compact model, and evaluate it against that task's metric.

## Caveats And Nuances

The main caveats:

- The current V2V-GoT evaluation uses typed task IDs; open-ended natural-language parsing would require an additional intent layer.
- Static task routing is an evaluation-adapter choice, not the core research contribution.
- Learned heads must be trained only on train split and selected carefully to avoid validation overfitting.
- Q5/Q7 identical validation metrics need to be disclosed because the official target appears effectively identical in the inspected split.
- KG quality depends on upstream processed assets. Bad detections, poses, timestamps, or visibility estimates can directly affect answers.
- The evaluator rewards specific output formats, so export/rendering correctness is part of the method.
- Some features are benchmark-specific, such as V2V-GoT control metadata for Q8/Q9.
- A new dataset may not expose the same QA type IDs, so an intent/parser layer would be needed for broader natural-language deployment.

The strongest honest claim is therefore:

> The method is generalizable as a cooperative driving reasoning architecture, not as a plug-and-play universal QA parser. It provides a reusable schema, query interface, provenance model, and planning-feature extraction pipeline. New datasets or tasks require adapters and retrained heads, but not a complete redesign of the reasoning stack.

An even shorter defense:

> We are not claiming that static QA IDs are the general idea. The general idea is that cooperative driving scenes should be stored as explicit graphs, and that downstream reasoning/planning tasks can be solved by querying those graphs and training compact task-aware heads over the resulting features.
