# G-MARK Ablation and OPV2V Extension Plan

## Goal

This document proposes a concrete extension plan for strengthening **G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs** through targeted ablations and an additional evaluation on **OPV2V**.

The main objective is to address the strongest likely reviewer concern:

> Are the reported improvements due to the proposed provenance-aware cooperative graph, or are they mainly due to benchmark-specific feature engineering and task-specific readouts?

The proposed ablations and OPV2V experiments are designed to make the contribution more defensible by isolating the value of:

1. cooperative evidence,
2. provenance-aware graph construction,
3. conservative fusion and candidate retention,
4. uncertainty and conflict modeling,
5. graph-derived readouts,
6. robustness under localization noise and communication delay.

---

## Why OPV2V is a Good Extension Dataset

OPV2V is a simulated vehicle-to-vehicle cooperative perception dataset built using OpenCDA and CARLA. It contains multi-agent driving scenes with different numbers of connected vehicles, 11,464 frames, and 232,913 annotated 3D vehicle bounding boxes across more than 70 scenes. It was designed specifically for benchmarking V2V cooperative perception and includes baselines for early, late, and intermediate fusion.

Relevant sources:

- OPV2V paper: https://arxiv.org/abs/2109.07644
- OPV2V project page: https://mobility-lab.seas.ucla.edu/opv2v/
- OpenCOOD codebase: https://github.com/DerrickXuNu/OpenCOOD
- OpenCOOD documentation: https://opencood.readthedocs.io/

OPV2V is useful for G-MARK because it provides the core ingredients needed to construct cooperative scene graphs:

| G-MARK Requirement | OPV2V Support |
|---|---|
| Multiple connected agents | Multiple CAVs per scene |
| Shared coordinate reasoning | Agent poses and calibration through OpenCOOD/OpenCDA pipeline |
| Object hypotheses | 3D vehicle bounding boxes |
| Cooperative evidence | Per-agent LiDAR observations and annotations |
| Occlusion and partial observability | Multi-view scenes with severe occlusion cases |
| Controlled robustness tests | OpenCOOD supports GPS noise and communication-delay simulation |

This makes OPV2V a strong dataset for testing whether the G-MARK graph construction and fusion logic generalizes beyond V2V-GoT-QA.

### Validity Check

This extension is valid if it is presented as a graph-construction and cooperative-fusion validation, not as a second QA benchmark.

What is well supported:

- OPV2V is explicitly a multi-agent V2V cooperative perception dataset.
- Each scenario contains multiple connected automated vehicles with per-agent sensor data and metadata.
- The metadata includes agent pose, calibration, speed, future planning trajectories, and object annotations.
- OPV2V/OpenCOOD supports standard cooperative-perception comparisons and controlled localization-noise / communication-delay settings.
- OPV2V therefore gives enough structure to test whether G-MARK's cooperative graph construction, provenance tracking, candidate retention, and conservative fusion behave sensibly outside V2V-GoT-QA.

What must be handled carefully:

- OPV2V does not directly provide the V2V-GoT Q1-Q9 QA tasks. Any OPV2V experiment should be called a proxy structural validation, not a QA benchmark result.
- If using ground-truth annotations as "observations," the experiment must define them as per-agent visible object annotations. OPV2V metadata annotates surrounding vehicles that receive at least one LiDAR point from that agent, which is useful for partner-only visibility analysis. Do not treat global labels as if every agent observed every object.
- If using detector outputs, the measured result mixes detector quality with graph quality. That is more realistic, but it is less clean for isolating the G-MARK graph contribution.
- OPV2V contains vehicles as the main annotated object class. It is suitable for vehicle-object recovery and fusion analysis, but not for validating pedestrian/cyclist/general-object reasoning unless additional labels are introduced.
- Standard OPV2V/OpenCOOD perception results are usually reported with detection metrics such as 3D AP. Our distance-threshold graph metrics are useful and aligned with V2V-GoT-style localization, but they should be reported as graph/proxy metrics unless we also run the official OpenCOOD detection protocol.

---

## Paper-Level Role of the OPV2V Experiment

The OPV2V experiment should not be framed as a replacement for the current V2V-GoT-QA evaluation. Instead, it should be framed as a **cross-benchmark structural validation**.

### Current V2V-GoT-QA Result

The existing paper shows that G-MARK improves over V2V-GoT reference values on Q1 to Q9. This supports the claim that explicit graph-based reasoning can perform well on cooperative driving QA tasks.

### Missing Evidence

The current draft does not yet fully prove that:

1. the cooperative graph itself matters,
2. multi-agent evidence is responsible for the gains,
3. provenance and candidate retention are necessary,
4. the framework generalizes beyond the V2V-GoT-QA task formatting.

### OPV2V Addition

OPV2V can be used to show that the same graph construction principle works on a separate V2V cooperative perception dataset. The OPV2V evaluation should focus on object recovery, cooperative recall, fusion quality, and robustness, rather than trying to reproduce all Q1 to Q9 tasks.

Recommended framing:

> To verify that G-MARK is not only a V2V-GoT-QA task adapter, we further evaluate the cooperative graph construction and fusion components on OPV2V. Since OPV2V is a V2V cooperative perception benchmark rather than a QA benchmark, we use graph-compatible proxy tasks that measure cooperative object recovery, provenance-supported recall, duplicate suppression, and robustness to localization noise and communication delay.

---

## Proposed OPV2V Graph Construction

For each timestamp, construct a G-MARK graph from OPV2V records as follows.

### Agent Nodes

Each connected autonomous vehicle is represented as an agent node:

```text
Agent node:
  id: vehicle_id
  pose: SE(3) pose in world or ego frame
  timestamp: frame timestamp
  sensor_type: LiDAR/camera if available
```

### Object Observation Nodes

Each per-agent detected or annotated object becomes an observation-level entity before fusion. In ground-truth validation mode, this should use per-agent annotations rather than global scene labels, so that agent-specific visibility and partner-only cases remain meaningful:

```text
Observation node:
  source_agent: vehicle_id
  object_type: vehicle
  bbox_3d: 3D bounding box
  center: x, y, z
  confidence: detection confidence, if using detector output
  timestamp: frame timestamp
```

If using ground-truth annotations for the first structural validation, confidence can be set to 1.0. If using detector outputs from an OpenCOOD baseline, confidence should be the detector confidence.

### Fused Object Hypothesis Nodes

Compatible observations are associated into object hypotheses:

```text
Object hypothesis node:
  fused_center: weighted center
  fused_bbox: representative or averaged box
  support_agents: set of agents that observed it
  support_count: number of supporting observations
  provenance: list of source observations
  uncertainty: uncertainty score
  conflict: disagreement score
  candidate_status: supported / candidate / ambiguous
```

### Edges and Relations

The graph should include simple but useful relations:

```text
observed_by(object, agent)
near(object_i, object_j)
front_of(object_i, ego)
left_of(object_i, ego)
right_of(object_i, ego)
behind(object_i, ego)
cooperatively_supported(object)
partner_only(object)
low_conflict(object)
high_uncertainty(object)
```

Optional relations:

```text
near_ego_path(object)
occluding_candidate(object_i, object_j)
```

These optional relations require a planned or proxy ego path. If OPV2V does not expose a directly usable future ego trajectory for every scene, use a simple ego-forward lane/path proxy only for auxiliary analysis, not as a main benchmark claim.

---

## Core OPV2V Proxy Tasks

Because OPV2V is not a QA benchmark, the evaluation should use proxy tasks aligned with the G-MARK contribution.

## Task 1: Cooperative Object Recovery

### Question

Does the cooperative graph recover objects that are missed or poorly observed by the ego vehicle alone?

### Setup

Compare two settings:

1. **Ego-only graph:** build graph using only ego vehicle observations.
2. **Cooperative graph:** build graph using ego plus connected-vehicle observations.

### Prediction

The output is a set of object centers or 3D boxes in the ego/shared frame.

### Metric

Use one or more of:

```text
Recall@IoU threshold
Recall@distance threshold
F1@distance threshold
3D AP, if using standard OpenCOOD detector outputs
```

For consistency with the current G-MARK paper, a distance-threshold metric is easier to align:

```text
Localization Recall@0.5m / 1.0m / 2.0m
Localization F1@0.5m / 1.0m / 2.0m
```

### Expected Claim

> Cooperative graph construction improves object recovery compared with ego-only graph construction, especially for objects that are visible to another agent but weakly observed or missed by the ego vehicle.

---

## Task 2: Partner-Only Object Recovery

### Question

Can G-MARK preserve objects that are only observed by non-ego agents?

### Setup

Define a ground-truth object as **partner-only** if it is matched by at least one non-ego agent observation but not matched by an ego observation. This definition is valid only when the observations are per-agent annotations or per-agent detector outputs. It is not valid if the experiment starts from a global object list shared by every agent.

Evaluate whether the final cooperative graph retains this object as a supported or candidate object node.

### Metric

```text
Partner-only Recall = recovered partner-only objects / total partner-only objects
Partner-only Precision = correct partner-only graph nodes / predicted partner-only graph nodes
Partner-only F1
```

### Why This Matters

This directly supports the hidden-object and partial-observability motivation from the current paper.

### Expected Claim

> Candidate retention and provenance-aware graph construction improve recall for partner-only objects, which are precisely the cases where cooperative perception should help most.

---

## Task 3: Duplicate and False-Merge Analysis

### Question

Does conservative graph fusion avoid over-merging nearby objects while still reducing duplicate hypotheses?

### Compared Methods

1. **Naive late fusion:** concatenate all object detections from all agents.
2. **Aggressive merge:** merge any nearby same-class objects under a loose threshold.
3. **G-MARK conservative fusion:** merge only type-consistent and geometrically compatible objects; retain ambiguous observations as candidates.

### Metrics

```text
Duplicate Rate:
  number of duplicate graph nodes matched to the same ground-truth object

False Merge Rate:
  number of fused graph nodes whose support observations match different ground-truth objects

Object F1:
  localization-aware F1 after fusion
```

### Expected Claim

> Conservative fusion reduces duplicate predictions compared with naive late fusion while avoiding the false merges introduced by overly aggressive fusion.

---

## Task 4: Provenance Reliability Analysis

### Question

Are objects supported by multiple agents more reliable than objects supported by a single agent?

### Setup

Group graph object nodes by support count:

```text
support_count = 1
support_count = 2
support_count >= 3
```

For each group, compute match quality against ground truth.

### Metrics

```text
Precision by support count
Localization error by support count
Conflict score by support count
Uncertainty score by support count
```

### Expected Claim

> Multi-agent support is correlated with higher precision and lower localization error, supporting the use of provenance as a first-class graph attribute.

This ablation is important because it justifies storing provenance rather than only storing fused coordinates.

---

## Task 5: Robustness to Localization Noise and Communication Delay

### Question

How stable is G-MARK when agent poses are noisy or observations are delayed?

### Setup

Use OpenCOOD's support for adding GPS noise and communication delay.

Evaluate under multiple conditions:

```text
Clean setting
Small pose noise
Medium pose noise
Large pose noise
Communication delay
Pose noise + communication delay
```

### Compared Methods

1. Ego-only graph
2. Naive late fusion
3. G-MARK without uncertainty/conflict
4. Full G-MARK

### Metrics

```text
Object F1
Recall
Precision
Duplicate rate
False merge rate
Mean localization error
```

### Expected Claim

> Explicit uncertainty and conflict modeling make G-MARK more stable under noisy cooperative alignment than naive fusion variants.

This is one of the strongest possible ablations because real cooperative perception must handle localization error and communication delay.

---

## Recommended Ablation Set

The main ablation table should isolate one design choice at a time.

| Variant | Cooperative Evidence | Provenance | Candidate Retention | Uncertainty/Conflict | Conservative Fusion | Purpose |
|---|---:|---:|---:|---:|---:|---|
| Ego-only | No | No | Yes | Yes | Yes | Tests value of V2V cooperation |
| Naive late fusion | Yes | No | No | No | No | Tests whether graph fusion is better than concatenation |
| Aggressive merge | Yes | Partial | No | No | No | Tests false-merge risk |
| No provenance | Yes | No | Yes | Yes | Yes | Tests value of source support |
| No candidate retention | Yes | Yes | No | Yes | Yes | Tests hidden/partner-only recall |
| No uncertainty/conflict | Yes | Yes | Yes | No | Yes | Tests quality modeling |
| Full G-MARK | Yes | Yes | Yes | Yes | Yes | Proposed method |

---

## Main OPV2V Result Table Template

Use this as the main OPV2V validation table.

| Method | Object F1 ↑ | Recall ↑ | Precision ↑ | Partner-Only Recall ↑ | Duplicate Rate ↓ | False Merge Rate ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ego-only graph |  |  |  |  |  |  |
| Naive late fusion |  |  |  |  |  |  |
| Aggressive merge |  |  |  |  |  |  |
| G-MARK w/o provenance |  |  |  |  |  |  |
| G-MARK w/o candidate retention |  |  |  |  |  |  |
| G-MARK w/o uncertainty/conflict |  |  |  |  |  |  |
| Full G-MARK |  |  |  |  |  |  |

### Interpretation Pattern

The strongest expected result pattern would be:

1. Full G-MARK improves recall over ego-only.
2. Full G-MARK improves duplicate rate over naive late fusion.
3. Full G-MARK reduces false merges compared with aggressive merge.
4. Removing candidate retention hurts partner-only recall.
5. Removing provenance hurts precision or confidence calibration.
6. Removing uncertainty/conflict hurts robustness under noise.

---

## Robustness Table Template

| Method | Clean F1 ↑ | Pose Noise F1 ↑ | Delay F1 ↑ | Noise + Delay F1 ↑ | False Merge Rate ↓ |
|---|---:|---:|---:|---:|---:|
| Ego-only graph |  |  |  |  |  |
| Naive late fusion |  |  |  |  |  |
| G-MARK w/o uncertainty/conflict |  |  |  |  |  |
| Full G-MARK |  |  |  |  |  |

### Interpretation Pattern

A good result would show that full G-MARK does not only perform well in clean settings, but degrades more gracefully when pose noise or delay is introduced.

---

## How This Strengthens the Current Paper

The current G-MARK paper makes four main claims:

1. cooperative scene graphs preserve useful multi-agent evidence,
2. provenance-aware fusion supports reasoning under partial observability,
3. conservative fusion helps retain ambiguous but useful hypotheses,
4. graph-derived readouts improve downstream reasoning and prediction tasks.

The OPV2V extension strengthens these claims as follows:

| Paper Claim | OPV2V Evidence |
|---|---|
| Cooperation helps | Ego-only vs cooperative graph |
| Provenance matters | Full G-MARK vs no provenance |
| Candidate retention matters | Full G-MARK vs no candidate retention |
| Conservative fusion matters | Full G-MARK vs naive/aggressive fusion |
| Robustness matters | Clean vs noisy pose/delay settings |
| Not benchmark-specific | Evaluation beyond V2V-GoT-QA |

---

## Suggested New Paper Subsection

The following subsection can be added after the main V2V-GoT-QA results or in the ablation section.

### Cross-Benchmark Structural Validation on OPV2V

To evaluate whether the proposed cooperative graph construction generalizes beyond the V2V-GoT-QA benchmark format, we conduct an additional structural validation on OPV2V. OPV2V is a simulated V2V cooperative perception dataset containing multi-agent driving scenes with synchronized connected vehicles and annotated 3D vehicle bounding boxes. Since OPV2V is a perception benchmark rather than a question-answering benchmark, we do not directly evaluate the Q1-Q9 task families. Instead, we define graph-compatible proxy tasks that isolate the core mechanisms of G-MARK: cooperative object recovery, partner-only object retention, duplicate suppression, false-merge avoidance, and robustness to pose noise and communication delay.

For each OPV2V frame, we construct an ego-centered cooperative scene graph from the available connected-vehicle observations. Object observations from different agents are transformed into a shared coordinate frame, associated using semantic and geometric compatibility, and fused into object hypothesis nodes. Each hypothesis stores source-agent provenance, support count, uncertainty, conflict, and candidate status. We compare the full G-MARK graph against ego-only construction, naive late fusion, aggressive geometric merging, and variants that remove provenance, candidate retention, or uncertainty/conflict features.

This evaluation is intended to test the structural contribution of G-MARK rather than its benchmark-specific QA adapter. Improvements over ego-only graph construction would indicate that cooperative evidence improves object recovery. Improvements over naive late fusion and aggressive merging would indicate that conservative graph fusion reduces duplicate predictions and false merges. Degradation after removing provenance, candidate retention, or uncertainty/conflict features would provide causal evidence that these graph attributes contribute to the final representation.

---

## Suggested Claims After OPV2V Ablations

Use these only if the results support them.

### Conservative Claim

> OPV2V experiments show that the proposed graph construction pipeline generalizes beyond V2V-GoT-QA and provides a controlled setting for isolating the effect of cooperative evidence, provenance, and conservative fusion.

### Stronger Claim

> Across OPV2V ablations, full G-MARK improves cooperative object recovery while reducing duplicate hypotheses and false merges compared with naive fusion variants. Removing provenance, candidate retention, or uncertainty/conflict modeling degrades the graph's ability to preserve useful partner-supported objects, supporting the design choices used in the V2V-GoT-QA experiments.

### Avoid This Claim Unless You Have Strong Evidence

> G-MARK outperforms all cooperative perception methods on OPV2V.

This would be risky unless you directly compare against standard OpenCOOD baselines under the same detector, same inputs, and same metrics.

---

## Important Cautions

### 1. Do Not Overclaim Planning on OPV2V

OPV2V is primarily a cooperative perception dataset. Unless you construct a reliable planning proxy, avoid claiming that OPV2V validates Q8/Q9-style planning.

Better wording:

> OPV2V validates the cooperative graph construction and fusion components, while V2V-GoT-QA remains the primary benchmark for downstream reasoning and planning tasks.

### 2. Separate Ground-Truth Graph Validation from Detector-Based Evaluation

There are two possible evaluation modes:

#### Mode A: Ground-truth object graph validation

Use per-agent annotations as object observations to test graph construction, association, provenance, and fusion logic.

This is cleaner for testing the graph itself.

Important detail:

- The graph input should be the objects annotated for each agent at a timestamp, not a single global label set copied to every agent.
- This keeps ego-only, cooperative, and partner-only definitions meaningful.

#### Mode B: Detector-output graph validation

Use predictions from an OpenCOOD detector to test end-to-end perception-to-graph behavior.

This is more realistic but introduces detector noise.

Important detail:

- If reporting against OpenCOOD baselines, use the same detector family, split, sensor inputs, and metric definitions.
- Otherwise, describe the result as a graph-level proxy experiment rather than an OPV2V detection leaderboard comparison.

Recommended approach:

1. Start with Mode A to validate graph logic.
2. Add Mode B if time permits.

### 3. Keep OPV2V as Supporting Evidence

The main paper should still center on V2V-GoT-QA if the contribution is driving reasoning. OPV2V should be used to strengthen the ablation and generalization story.

---

## Final Recommended Experiment Package

If time is limited, run only these four experiments:

1. **Ego-only vs cooperative graph**
2. **Naive late fusion vs conservative G-MARK fusion**
3. **Full G-MARK vs no candidate retention**
4. **Clean vs pose-noise robustness**

These four experiments would already address the most important reviewer concerns.

If time allows, add:

5. **Full G-MARK vs no provenance**
6. **Full G-MARK vs no uncertainty/conflict**
7. **Partner-only object recall**
8. **Duplicate and false-merge analysis**

---

## Final Takeaway

The OPV2V extension should be used to support the following core argument:

> G-MARK is not only a V2V-GoT-QA task adapter. Its cooperative graph construction provides a reusable structured representation for V2V scenes. OPV2V allows us to isolate the graph-level mechanisms, including cooperative evidence, provenance, candidate retention, conservative fusion, and robustness to noisy multi-agent alignment.

This would make the paper much more defensible because it directly connects each major design choice to an ablation-backed result.
