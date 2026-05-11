# G-MARK OPV2V Extension: Context, Approach, And Plan

## Purpose

This document defines a clean plan for extending G-MARK beyond the current V2V-GoT-QA evaluation using OPV2V.

The goal is not to turn OPV2V into another QA benchmark. The goal is to test whether the core G-MARK idea, an explicit cooperative scene graph with provenance-aware fusion and graph-derived readouts, remains useful on a separate cooperative driving dataset.

This extension should answer one reviewer-critical question:

> Are the gains coming from a reusable cooperative graph representation, or mostly from V2V-GoT-specific task engineering?

OPV2V is useful because it gives us multi-agent cooperative perception scenes with agent poses, per-agent sensor data, annotations, and controlled noise/delay settings. That makes it a good setting for graph-level validation.

## High-Level Claim

The intended claim is:

> G-MARK is not only a V2V-GoT-QA adapter. Its cooperative scene-graph construction, provenance tracking, candidate retention, and conservative fusion can be evaluated independently on OPV2V using graph-compatible perception proxy tasks.

The claim should stay at the graph-structure level unless we also run official OpenCOOD detector baselines.

Safe wording:

> We use OPV2V as a cross-benchmark structural validation of the cooperative graph construction and fusion components.

Risky wording:

> G-MARK outperforms all OPV2V cooperative perception methods.

That stronger claim would require matching the standard OpenCOOD evaluation protocol, detector inputs, splits, and metrics.

## Relationship To Current V2V-GoT-QA Work

The current V2V-GoT-QA results evaluate downstream reasoning and prediction tasks:

- object grounding;
- occlusion reasoning;
- invisible object discovery;
- planning awareness;
- object motion prediction;
- agent motion prediction;
- control selection;
- future trajectory prediction.

Those results show that graph-derived task heads can perform well under the official V2V-GoT-QA evaluator.

OPV2V tests a different layer:

- whether cooperative graph construction works on another dataset;
- whether multi-agent evidence improves object recovery;
- whether provenance helps explain reliability;
- whether conservative fusion reduces duplicates and false merges;
- whether uncertainty/conflict modeling improves robustness under pose noise or delay.

So the two evaluations are complementary:

| Evaluation | Main Question | What It Validates |
| --- | --- | --- |
| V2V-GoT-QA | Can graph-derived task heads answer driving QA tasks? | Downstream task performance |
| OPV2V | Does the cooperative graph/fusion mechanism generalize structurally? | Graph construction and multi-agent fusion |

## Core Design Principle

The OPV2V extension should preserve the same G-MARK architecture:

1. Build per-agent observation nodes.
2. Transform observations into a shared frame.
3. Associate compatible observations.
4. Create fused object hypothesis nodes.
5. Preserve provenance from every source agent.
6. Retain ambiguous but plausible candidate objects.
7. Estimate uncertainty and conflict.
8. Derive graph relations.
9. Evaluate graph-level outputs against ground truth.

The important thing is that OPV2V should reuse the same conceptual graph contract, not introduce a separate one-off pipeline.

## Non-Regression Guardrail

The OPV2V extension must not break or rewrite the current V2V-GoT-QA architecture.

Development rule:

> Add dataset adapters, experiment wrappers, and ablation configurations around the existing graph contracts. Do not change the current task handlers, e2e training/evaluation flow, or V2V-GoT adapter behavior unless the change is strictly backwards-compatible and verified against the existing results.

Practical constraints:

- Keep OPV2V loading in a separate adapter module.
- Keep OPV2V scripts separate from the existing V2V-GoT e2e scripts.
- Reuse existing domain concepts such as agents, observations, object tracks, provenance, relations, uncertainty, and conflict.
- Add extension points only where the current architecture already expects dataset-specific conversion.
- Preserve the current V2V-GoT-QA outputs, official export format, trained model manifests, and validation-report behavior.
- Any shared utility change must be covered by a quick regression check on the existing V2V-GoT path.

The OPV2V work should therefore be an adapter-and-ablation extension, not an architectural rewrite.

## Proposed Graph Schema

### Agent Node

```text
agent_id
timestamp
pose_world
pose_ego
sensor_metadata
speed
optional_future_plan
```

### Observation Node

One object observation from one source agent.

```text
observation_id
source_agent_id
timestamp
object_type
bbox_3d
center
heading
velocity_if_available
confidence
```

For ground-truth graph validation, confidence can be `1.0`.

For detector-output validation, confidence should come from the detector.

Important: use per-agent annotations or per-agent detector outputs. Do not copy a global label set into every agent, because that would destroy the meaning of ego-only and partner-only visibility.

### Object Hypothesis Node

The fused graph-level object after association.

```text
track_id
fused_center
fused_bbox
object_type
support_agent_ids
support_count
source_observation_ids
candidate_status
uncertainty
conflict
duplicate_risk
```

### Relations

Minimum useful relations:

```text
observed_by(object, agent)
cooperatively_supported(object)
partner_only(object)
near(object_i, object_j)
front_of(object, ego)
behind(object, ego)
left_of(object, ego)
right_of(object, ego)
low_conflict(object)
high_uncertainty(object)
```

Optional relations:

```text
near_ego_path(object)
occluding_candidate(object_i, object_j)
```

Use optional path/planning relations only if OPV2V metadata gives a reliable future plan for the selected scenario. Otherwise keep planning out of the OPV2V claim.

## Evaluation Modes

### Mode A: Ground-Truth Graph Validation

Use per-agent annotations as observations.

This is the cleanest first experiment because it isolates graph construction from detector quality.

It answers:

- Can the graph associate per-agent object observations correctly?
- Does cooperative evidence recover partner-only objects?
- Does conservative fusion reduce duplicates without false merging?
- Does provenance correlate with object reliability?

Limit:

- This does not prove end-to-end detector performance.

### Mode B: Detector-Output Graph Validation

Use detector outputs from an OpenCOOD-compatible model as observations.

This is more realistic because the graph consumes noisy perception predictions.

It answers:

- Does G-MARK still help when upstream detections are imperfect?
- Does uncertainty/conflict modeling help under detector and alignment noise?
- How does graph fusion compare with naive late fusion or aggressive merging?

Limit:

- The result mixes detector quality and graph quality.
- Any comparison to OPV2V/OpenCOOD baselines must use the same detector, split, inputs, and metric definitions.

Recommended order:

1. Implement Mode A first.
2. Add Mode B only after the graph-level experiment is stable.

## Main Experiments

### Experiment 1: Ego-Only vs Cooperative Graph

Question:

> Does adding connected-vehicle evidence improve object recovery?

Compare:

- ego-only graph;
- cooperative graph.

Metrics:

```text
Recall@0.5m / 1.0m / 2.0m
Precision@0.5m / 1.0m / 2.0m
F1@0.5m / 1.0m / 2.0m
mean localization error
```

Expected value:

- cooperative graph should improve recall, especially for objects weakly visible or invisible to the ego vehicle.

### Experiment 2: Partner-Only Object Recovery

Question:

> Does the graph retain objects observed only by other connected vehicles?

Definition:

An object is partner-only if it is matched by at least one non-ego agent observation and not matched by an ego observation.

Metrics:

```text
partner-only recall
partner-only precision
partner-only F1
```

Expected value:

- candidate retention and provenance should improve partner-only recall.

### Experiment 3: Duplicate And False-Merge Analysis

Question:

> Does conservative fusion avoid both duplicate clutter and incorrect merges?

Compare:

- naive late fusion;
- aggressive geometric merge;
- full G-MARK conservative fusion.

Metrics:

```text
duplicate rate
false merge rate
object F1
mean localization error
```

Expected value:

- naive late fusion should have more duplicates;
- aggressive merge should have more false merges;
- G-MARK should balance duplicate suppression and merge safety.

### Experiment 4: Provenance Reliability

Question:

> Are objects with multi-agent support more reliable?

Group predictions by support count:

```text
support_count = 1
support_count = 2
support_count >= 3
```

Metrics:

```text
precision by support count
localization error by support count
uncertainty by support count
conflict by support count
```

Expected value:

- higher support should generally correlate with higher precision and/or lower localization error;
- high conflict should identify cases where support is geometrically inconsistent.

### Experiment 5: Robustness To Pose Noise And Delay

Question:

> Does uncertainty/conflict-aware graph fusion degrade more gracefully under noisy cooperative alignment?

Conditions:

```text
clean
small localization noise
medium localization noise
large localization noise
communication delay
localization noise + communication delay
```

Compare:

- ego-only graph;
- naive late fusion;
- G-MARK without uncertainty/conflict;
- full G-MARK.

Metrics:

```text
object F1
recall
precision
duplicate rate
false merge rate
mean localization error
```

Expected value:

- full G-MARK should be more stable than naive fusion when alignment is imperfect.

## Ablation Matrix

| Variant | Cooperative Evidence | Provenance | Candidate Retention | Uncertainty/Conflict | Conservative Fusion | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ego-only | No | No | Yes | Yes | Yes | Isolate value of connected-vehicle evidence |
| Naive late fusion | Yes | No | No | No | No | Test whether graph fusion is better than concatenation |
| Aggressive merge | Yes | Partial | No | No | No | Test false-merge risk |
| No provenance | Yes | No | Yes | Yes | Yes | Test value of source support |
| No candidate retention | Yes | Yes | No | Yes | Yes | Test partner-only / hidden-object recall |
| No uncertainty/conflict | Yes | Yes | Yes | No | Yes | Test quality modeling |
| Full G-MARK | Yes | Yes | Yes | Yes | Yes | Proposed method |

## Result Table Template

| Method | F1@1.0m ↑ | Recall@1.0m ↑ | Precision@1.0m ↑ | Partner-Only Recall ↑ | Duplicate Rate ↓ | False Merge Rate ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego-only graph |  |  |  |  |  |  |
| Naive late fusion |  |  |  |  |  |  |
| Aggressive merge |  |  |  |  |  |  |
| G-MARK w/o provenance |  |  |  |  |  |  |
| G-MARK w/o candidate retention |  |  |  |  |  |  |
| G-MARK w/o uncertainty/conflict |  |  |  |  |  |  |
| Full G-MARK |  |  |  |  |  |  |

## Preliminary Structural Result

Using the downloaded OPV2V `test-001` subset, we ran a read-only structural visibility/recovery ablation over 1000 frames from 6 scenarios.

Important correction:

- An earlier smoke test grouped observations by OPV2V annotation IDs.
- Diagnostics showed same-ID cross-agent observations could be tens of meters apart.
- Therefore annotation IDs should not be treated as stable global object identity in this extracted subset.
- The result below uses geometry-derived object targets instead.

Current target definition:

- collect all per-agent vehicle observations in a frame;
- cluster them geometrically with `gt_cluster_radius=1.0m` to form the target object set;
- evaluate ego-only, naive late fusion, and graph fusion against this geometry-derived target;
- use annotation IDs only as diagnostics, not as object identity.

This is not yet a detector-output benchmark. It is a graph-structure ablation that tests whether cooperative evidence and conservative geometric fusion have measurable value before adding detector noise.

### Main Geometry-Derived Result

Configuration:

```text
frames = 1000
scenarios = 6
gt_cluster_radius = 1.0m
association_radius = 1.0m
match_radius = 1.0m
```

| Method | Recall ↑ | Precision ↑ | F1 ↑ | Partner-Only Recall ↑ | Duplicate Rate ↓ | False Merge Rate ↓ | Avg Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego-only | `0.475743` | `0.999810` | `0.644711` | `0.000000` | `0.000190` | `0.000000` | `10.544` |
| Naive late fusion | `1.000000` | `0.974493` | `0.987082` | `1.000000` | `0.025507` | `0.000000` | `22.739` |
| Geometry graph fusion | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.000000` | `22.159` |

Interpretation:

- Ego-only has high precision but low recall because it misses objects observed only by partner vehicles.
- Naive late fusion recovers partner-only objects but retains duplicate observations.
- Conservative geometry graph fusion preserves partner-only recovery while suppressing duplicate clutter.

Defensible early claim:

> On OPV2V geometry-derived object targets, cooperative graph fusion recovers objects missed by ego-only perception and removes duplicate clutter introduced by naive late fusion.

### Association-Radius Ablation

This sweep keeps `gt_cluster_radius=1.0m` and `match_radius=1.0m`, while changing the graph fusion association radius.

| Association Radius | Geometry Recall ↑ | Geometry Precision ↑ | Geometry F1 ↑ | Partner-Only Recall ↑ | Duplicate Rate ↓ | False Merge Rate ↓ | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.75m` | `1.000000` | `0.986906` | `0.993410` | `1.000000` | `0.013094` | `0.000000` | too conservative; leaves duplicates |
| `1.0m` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.000000` | best clean setting on this subset |
| `1.5m` | `0.981452` | `1.000000` | `0.990639` | `0.966859` | `0.000000` | `0.018898` | starts over-merging |
| `2.0m` | `0.976985` | `1.000000` | `0.988358` | `0.959456` | `0.000000` | `0.023558` | more over-merging |
| `3.0m` | `0.806399` | `0.908439` | `0.854383` | `0.764569` | `0.000000` | `0.123996` | aggressive fusion fails |

This sweep gives a useful causal ablation:

- too-small association radius leaves duplicate clutter;
- conservative radius around `1.0m` gives high recall and duplicate suppression;
- aggressive radii over-merge nearby objects and degrade recall/F1.

Required next step:

- optionally validate with detector outputs from an OpenCOOD model;
- optionally test robustness by perturbing partner poses before geometry fusion.

### Synthetic Partner-Noise Robustness

We also ran a synthetic robustness check where Gaussian xy noise is applied only to non-ego/partner observations before method prediction. The geometry-derived target set remains clean. This simulates cooperative alignment error while keeping the experiment isolated from detector quality.

This is our controlled graph-level perturbation, not the official OpenCOOD wild-setting protocol.

Implementation detail:

```text
x_noisy = x + Normal(0, noise_std)
y_noisy = y + Normal(0, noise_std)
```

Only non-ego observations are perturbed. Ego observations and the geometry-derived target set remain clean.

Configuration:

```text
frames = 1000
scenarios = 6
gt_cluster_radius = 1.0m
match_radius = 1.0m
noise_std = 0.0, 0.25, 0.5, 1.0, 1.5m
conservative graph radius = 1.0m
aggressive graph radius = 3.0m
```

| Noise Std | Method | Recall ↑ | Precision ↑ | F1 ↑ | Partner-Only Recall ↑ | Duplicate Rate ↓ | False Merge Rate ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00m` | Ego-only | `0.475743` | `0.999810` | `0.644711` | `0.000000` | `0.000190` | `0.000000` |
| `0.00m` | Naive late fusion | `1.000000` | `0.974493` | `0.987082` | `1.000000` | `0.025507` | `0.000000` |
| `0.00m` | Conservative graph `r=1.0m` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.000000` |
| `0.00m` | Aggressive graph `r=3.0m` | `0.806399` | `0.908439` | `0.854383` | `0.764569` | `0.000000` | `0.123996` |
| `0.25m` | Naive late fusion | `0.999774` | `0.974273` | `0.986859` | `0.999570` | `0.025111` | `0.000000` |
| `0.25m` | Conservative graph `r=1.0m` | `0.998240` | `0.994515` | `0.996374` | `0.996643` | `0.004856` | `0.001529` |
| `0.25m` | Aggressive graph `r=3.0m` | `0.815515` | `0.914895` | `0.862351` | `0.797796` | `0.000000` | `0.119532` |
| `0.50m` | Naive late fusion | `0.929013` | `0.905317` | `0.917012` | `0.864595` | `0.023000` | `0.000000` |
| `0.50m` | Conservative graph `r=1.0m` | `0.927434` | `0.920991` | `0.924201` | `0.861582` | `0.006274` | `0.004033` |
| `0.50m` | Aggressive graph `r=3.0m` | `0.774584` | `0.866212` | `0.817840` | `0.714470` | `0.000000` | `0.115216` |
| `1.00m` | Naive late fusion | `0.681935` | `0.664541` | `0.673126` | `0.393303` | `0.015128` | `0.000000` |
| `1.00m` | Conservative graph `r=1.0m` | `0.680626` | `0.674508` | `0.677553` | `0.390807` | `0.002683` | `0.008900` |
| `1.00m` | Aggressive graph `r=3.0m` | `0.581209` | `0.648947` | `0.613213` | `0.342429` | `0.000000` | `0.112315` |
| `1.50m` | Naive late fusion | `0.585857` | `0.570913` | `0.578289` | `0.210037` | `0.013193` | `0.000000` |
| `1.50m` | Conservative graph `r=1.0m` | `0.584864` | `0.579658` | `0.582249` | `0.208143` | `0.001923` | `0.012166` |
| `1.50m` | Aggressive graph `r=3.0m` | `0.497315` | `0.554549` | `0.524375` | `0.183955` | `0.000000` | `0.113929` |

Interpretation:

- Conservative graph fusion is strongest in the clean and mild-noise regimes.
- At `0.25m` noise, conservative graph fusion keeps near-perfect recall while reducing duplicate clutter relative to naive late fusion.
- At `0.5m` noise, conservative graph fusion slightly improves F1 and precision over naive late fusion while keeping duplicate rate much lower.
- At large noise (`1.0m` and `1.5m`), both naive and conservative fusion degrade because the partner observations no longer match the clean target well. The remaining advantage of conservative fusion is mostly duplicate suppression and modest precision/F1 improvement.
- Aggressive fusion performs poorly across settings because it over-merges nearby objects, producing a consistently high false-merge rate.

Careful claim:

> Conservative graph fusion improves duplicate control and clean/mild-noise robustness, but it does not eliminate large cooperative alignment errors. This motivates keeping uncertainty/conflict as explicit graph attributes rather than blindly merging partner observations.

### OpenCOOD Wild-Setting Extension

To strengthen the robustness story, the next step is to align with OpenCOOD's existing wild-setting noise parameters rather than using only our synthetic observation perturbation.

OpenCOOD exposes these relevant parameters:

```text
wild_setting:
  async: true/false
  async_mode: sim/real
  async_overhead: <milliseconds>
  loc_err: true/false
  xyz_std: <meters>
  ryp_std: <degrees-like angular std used by OpenCOOD pose helper>
```

OpenCOOD documentation describes:

- `async_overhead`: communication delay in milliseconds;
- `loc_err`: whether localization error is enabled;
- `xyz_std`: standard deviation of positional GPS error;
- `ryp_std`: standard deviation of angular GPS error.

Relevant local OpenCOOD files inspected:

```text
/Users/bhavya/Desktop/ms_projects/OpenCOOD/docs/md_files/data_intro.md
/Users/bhavya/Desktop/ms_projects/OpenCOOD/docs/md_files/config_tutorial.md
/Users/bhavya/Desktop/ms_projects/OpenCOOD/opencood/data_utils/datasets/basedataset.py
/Users/bhavya/Desktop/ms_projects/OpenCOOD/opencood/hypes_yaml/point_pillar_v2xvit.yaml
/Users/bhavya/Desktop/ms_projects/OpenCOOD/opencood/hypes_yaml/point_pillar_where2comm.yaml
```

How this should be used in our work:

1. Keep the current synthetic perturbation result as a controlled graph-only sensitivity check.
2. Add an OpenCOOD-style robustness script that applies pose-level perturbations before transforming partner observations into ego/shared frame.
3. Evaluate the same methods:
   - ego-only;
   - naive late fusion;
   - conservative geometry graph fusion;
   - aggressive geometry graph fusion.
4. Report this as OpenCOOD-style pose/delay robustness, not as official OpenCOOD detector benchmarking unless detector predictions are included.

Recommended first OpenCOOD-style settings:

| Setting | `async` | `async_overhead` | `loc_err` | `xyz_std` | `ryp_std` | Purpose |
| --- | --- | ---: | --- | ---: | ---: | --- |
| clean | false | `0` | false | `0.0` | `0.0` | reference |
| loc small | false | `0` | true | `0.1` | `0.1` | mild pose noise |
| loc standard | false | `0` | true | `0.2` | `0.2` | OpenCOOD-style common noisy value |
| delay sim | true | `100` | false | `0.0` | `0.0` | 100ms simulated delay |
| loc + delay | true | `100` | true | `0.2` | `0.2` | combined noisy setting |

Careful claim after this extension:

> The synthetic perturbation study isolates graph sensitivity to partner observation misalignment, while the OpenCOOD-style wild-setting study tests the same graph-fusion choices under noise parameters used by the cooperative-perception benchmark infrastructure.

Published OpenCOOD baseline context:

- OpenCOOD's OPV2V LiDAR-track table reports detector AP, not graph-proxy object-recovery F1.
- Example OPV2V AP@0.7 values from the OpenCOOD README include:
  - Naive Late + PointPillar: `0.668`;
  - Cooper + PointPillar: `0.696`;
  - Attentive Fusion + PointPillar: `0.735`;
  - V2VNet + PointPillar: `0.734`;
  - CoBEVT + PointPillar: `0.773`.
- OpenCOOD's V2XSet table includes perfect/noisy AP columns, but that is for V2XSet rather than OPV2V.

How to use these baselines:

- cite them as cooperative perception baselines showing the standard detection benchmark context;
- do not compare their AP numbers directly to our OPV2V graph-proxy F1 numbers;
- if we later run detector-output graph validation with OpenCOOD predictions, then AP-style comparison becomes more direct.

### OpenCOOD-Style Wild-Setting Result

We ran the standalone OpenCOOD-style graph ablation over 1000 OPV2V `test-001` frames using:

```text
gt_cluster_radius = 1.0m
graph radii = 1.0m conservative, 3.0m aggressive
seed = 25
```

At `match_radius=1.0m`, localization noise follows the expected pattern:

| Setting | Method | Recall ↑ | Precision ↑ | F1 ↑ | Partner-Only Recall ↑ | Duplicate Rate ↓ | False Merge Rate ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | Naive late fusion | `1.000000` | `0.974493` | `0.987082` | `1.000000` | `0.025507` | `0.000000` |
| clean | Conservative graph `r=1.0m` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.000000` |
| clean | Aggressive graph `r=3.0m` | `0.806399` | `0.908439` | `0.854383` | `0.764569` | `0.000000` | `0.123996` |
| loc standard | Naive late fusion | `0.999955` | `0.974449` | `0.987037` | `0.999914` | `0.025507` | `0.000000` |
| loc standard | Conservative graph `r=1.0m` | `0.998782` | `0.995144` | `0.996959` | `0.997676` | `0.004811` | `0.001169` |
| loc standard | Aggressive graph `r=3.0m` | `0.810235` | `0.912668` | `0.858406` | `0.791771` | `0.000000` | `0.124187` |

At stricter `match_radius=0.5m`, `loc_standard_delay200` also supports the same conclusion:

| Setting | Method | Recall ↑ | Precision ↑ | F1 ↑ | Partner-Only Recall ↑ | Duplicate Rate ↓ | False Merge Rate ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loc standard + delay 200ms | Naive late fusion | `0.963401` | `0.938828` | `0.950955` | `0.930189` | `0.016184` | `0.000000` |
| loc standard + delay 200ms | Conservative graph `r=1.0m` | `0.962318` | `0.958813` | `0.960562` | `0.928123` | `0.000000` | `0.001169` |
| loc standard + delay 200ms | Aggressive graph `r=3.0m` | `0.749628` | `0.844398` | `0.794196` | `0.706120` | `0.000000` | `0.124187` |

Important nuance:

- Delay-only settings (`delay100`, `delay200`) were effectively identical to clean in this graph-level emulation.
- This means the current adapter-level delay simulation is not strong enough to support a standalone communication-delay robustness claim.
- The useful and defensible result is about OpenCOOD-style localization noise and combined localization-noise settings.

Careful claim:

> Under OpenCOOD-style localization noise (`xyz_std=0.2`, `ryp_std=0.2`, seed `25`), conservative graph fusion preserves high object-recovery F1 while reducing duplicate clutter relative to naive late fusion. Aggressive fusion consistently over-merges, producing high false-merge rates. Delay-only effects were inconclusive in the graph-level emulation and should not be overclaimed without detector-output inference.

## Implementation Plan

### Local OpenCOOD Inspection Notes

Initial read-only inspection of the local OpenCOOD clone at `/Users/bhavya/Desktop/ms_projects/OpenCOOD` found:

- the repo is present, but the OPV2V dataset directory is not present locally yet;
- standard config paths expect:

```text
opv2v_data_dumping/train
opv2v_data_dumping/validate
opv2v_data_dumping/test
```

- OpenCOOD builds a `scenario_database` with this structure:

```text
scenario -> cav_id -> timestamp -> yaml/lidar/camera paths
```

- the first non-negative CAV id is treated as ego in OpenCOOD's dataset path;
- each timestamp YAML is expected to expose fields such as:

```text
lidar_pose
vehicles
ego_speed
```

- `vehicles` is the per-agent object annotation dictionary used by OpenCOOD for ground-truth object centers;
- OpenCOOD transforms between CAV coordinate frames with `x1_to_x2(x1_pose, x2_pose)`, where pose format is:

```text
[x, y, z, roll, yaw, pitch]
```

- OpenCOOD's helper path already has concepts for async delay and localization noise:

```text
wild_setting.async
wild_setting.async_mode
wild_setting.async_overhead
wild_setting.loc_err
wild_setting.xyz_std
wild_setting.ryp_std
```

These findings support the adapter-first plan. The first implementation should read OPV2V YAML files directly and convert the per-agent `vehicles` entries into G-MARK observation nodes, without importing the full OpenCOOD training stack or changing the V2V-GoT pipeline.

### Phase 1: Dataset Inspection

Deliverables:

- inspect OPV2V directory structure;
- confirm train/validate/test split availability;
- inspect one scenario;
- parse agent IDs, timestamps, poses, and per-agent annotations;
- confirm coordinate frame conventions.

Output:

```text
outputs/opv2v_inspection/
```

### Phase 2: OPV2V Adapter

Deliverables:

- implement an OPV2V scene adapter;
- convert per-agent annotations into observation nodes;
- create ego-only and cooperative scene graph modes;
- serialize graph snapshots for debugging.

Likely files:

```text
src/kg_coop_drive/infrastructure/opv2v_scene_adapter.py
src/kg_coop_drive/application/opv2v_graph_eval.py
scripts/inspect_opv2v_scene.py
```

### Phase 3: Graph Association And Fusion

Deliverables:

- transform per-agent observations into ego/shared frame;
- associate observations using class and geometry gates;
- compute fused centers/boxes;
- record support agents and source observations;
- compute uncertainty/conflict;
- retain ambiguous candidates.

Reuse existing G-MARK concepts where possible:

- observation evidence;
- object tracks;
- provenance records;
- relation facts;
- uncertainty/conflict fields.

### Phase 4: Metrics And Ablations

Deliverables:

- implement localization matching;
- compute F1/precision/recall at distance thresholds;
- compute partner-only recall;
- compute duplicate and false-merge rates;
- run the ablation matrix.

Likely script:

```text
scripts/run_opv2v_graph_ablation.py
```

### Phase 5: Noise And Delay Robustness

Deliverables:

- either use OpenCOOD noise/delay settings directly;
- or add a controlled perturbation layer for pose and timestamp alignment;
- compare clean/noisy/delayed variants.

Keep this phase separate from the clean graph validation so the first result is easy to interpret.

### Phase 6: Paper Summary

Deliverables:

- compact result table;
- one figure showing ego-only vs cooperative graph;
- one ablation table;
- one paragraph explaining why OPV2V supports generalization beyond V2V-GoT-QA.

## Minimal Experiment Package

If time is limited, run only:

1. Ego-only vs cooperative graph.
2. Naive late fusion vs conservative G-MARK fusion.
3. Full G-MARK vs no candidate retention.
4. Clean vs pose-noise robustness.

This is enough to address the main reviewer concern without over-expanding the paper.

## Caveats

- OPV2V is simulated, while V2V4Real/V2V-GoT are grounded in real cooperative driving assets.
- OPV2V primarily annotates vehicles, so object-class diversity is limited.
- OPV2V validates graph construction and cooperative fusion more directly than downstream QA reasoning.
- Ground-truth graph validation isolates graph logic but is not an end-to-end perception result.
- Detector-output validation is more realistic but needs careful baseline control.
- Planning claims should remain with V2V-GoT-QA unless OPV2V planning metadata is explicitly used and validated.

## Final Positioning

The OPV2V extension should support this argument:

> V2V-GoT-QA evaluates whether G-MARK's graph-derived task heads can answer structured cooperative driving questions. OPV2V separately tests whether the underlying cooperative graph construction and fusion mechanisms generalize to another V2V dataset. Together, these evaluations make the method stronger: V2V-GoT-QA validates task performance, while OPV2V validates the reusable graph substrate.
