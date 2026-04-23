# Cooperative Data Generation Plan

## Goal

Identify what cooperative data is currently available, what is missing for this project, and what it would take to generate or recover the missing data needed for reliable cooperative graph reasoning experiments.

This note is based on:

- the local `kg_coop_drive` repository
- the local `V2V-GoT` repository
- the local `auto_drive_copy` repository
- external benchmark and simulation references

## Executive Summary

The main missing piece is not the benchmark QA JSON itself. The missing piece is the **timestamp-aligned processed perception data** needed to run real cooperative experiments frame by frame.

Right now, the local V2V-GoT checkout contains:

- the benchmark-style QA JSON files under `.../npy/co_llm/`
- the code paths that expect per-frame predictions, GT arrays, and visibility arrays
- the scripts that generate QA from those processed arrays

But it does **not** currently contain the actual per-frame `.npy` assets locally:

- `*_pred.npy`
- `*_pred_score.npy`
- `*_gt.npy`
- `*_gt_object_id.npy`
- `*_gt_object_id_visible_to_*.npy`
- `*_gt_object_id_invisible_to_*.npy`

So the fastest path is:

1. recover the missing processed V2V-GoT / V2V4Real assets in the expected format, if possible
2. if they are unavailable, generate equivalent cooperative perception assets ourselves
3. validate the generated assets against research-standard checks before using them for QA evaluation

## What Currently Exists

### 1. In `kg_coop_drive`

The current repo already supports:

- canonical scene construction from benchmark records
- loading processed per-frame GT / prediction / visibility arrays when present
- local graph construction
- deterministic query execution
- local multi-frame validation

Relevant files:

- [v2vgot_processed_assets.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive/infrastructure/v2vgot_processed_assets.py:1)
- [demo_phase2_scene_query.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/scripts/demo_phase2_scene_query.py:1)
- [validate_phase3_local_graphs.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/scripts/validate_phase3_local_graphs.py:1)

Important implication:

- the graph and reasoning pipeline is already ready to consume processed cooperative data
- the bottleneck is the data layer, not the graph layer

### 2. In local `V2V-GoT`

The repo README and DMSTrack docs clearly expect a processed asset package with GT and prediction arrays.

Relevant local references:

- [V2V-GoT README](/Users/bhavya/Desktop/ms_projects/V2V-GoT/README.md:1)
- [DMSTrack DATA.md](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/docs/DATA.md:1)

Those docs describe a structure like:

```text
official_models/
  no_fusion_keep_all/
    npy/
      ego/
      1/
      co_llm/
```

The OpenCOOD / DMSTrack code also shows how these arrays are normally written.

Relevant local references:

- [temp_qa_generation.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/temp_qa_generation.py:1)
- [infrence_utils.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/infrence_utils.py:161)

That code writes:

- `pcd.npy`
- `pred.npy`
- `gt.npy`
- `pred_score.npy`
- `gt_object_id.npy`
- optional feature maps
- visibility ID lists per agent

### 3. In local `auto_drive_copy`

This repo is useful mostly as evaluation and structured-QA inspiration, not as a cooperative data generator.

Useful aspects:

- deterministic question-answer evaluation
- template-based QA splits
- structured comparison of predictions and gold answers

Relevant local references:

- [agent.py](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/code/agent.py:1)
- [tools.py](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/tools.py:1)
- [eval_shard.py](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/eval_shard.py:1)

## What Is Missing

### 1. Missing local processed arrays

In the current local V2V-GoT checkout:

- `official_models/no_fusion_keep_all/npy/` exists
- `official_models/train_no_fusion_keep_all/npy/` exists
- but they currently only contain `co_llm/` JSON datasets
- they do not contain the per-frame `.npy` arrays the graph loader expects

This is the most important immediate gap.

### 2. Missing synchronized multi-agent prediction coverage

The cooperative graph code expects both agents to contribute predictions at the same timestamp.

What is missing experimentally is:

- enough frames where both `CAV_EGO` and `CAV_1` have aligned prediction outputs
- visibility and GT artifacts aligned to those same timestamps

Without that, strong cooperative fusion experiments are blocked.

### 3. Missing a data-generation workflow under our control

At the moment, the project depends on upstream processed assets.

For research independence, we need our own reproducible workflow that can:

- generate multi-agent observations
- generate aligned GT and visibility labels
- optionally derive benchmark-style QA pairs
- export to a format our graph code can consume directly

## What We Need To Do

There are two realistic paths.

### Path A: Recover the official processed assets

This is the fastest and least risky path if the data is available.

What we would do:

- obtain the missing `dataset_processed_features_and_gt.zip` or equivalent archives referenced by V2V-GoT / DMSTrack
- restore the expected `official_models/.../npy` folder contents
- verify that both agent-specific prediction folders and GT / visibility files are present
- rerun the current graph pipeline on those assets

Why this path is strong:

- maximum compatibility with the existing V2V-GoT benchmark
- least schema drift
- easiest justification for benchmark-faithful experiments

Main risk:

- the exact assets may be inaccessible, incomplete, or still not cover the needed synchronized cases

### Path B: Generate new cooperative processed data ourselves

This is the stronger long-term research path.

What we would generate:

- per-agent observations for the same scene and timestamp
- shared world GT
- visibility / occlusion labels per agent
- synchronized agent poses
- optional feature maps if later LLM or feature-fusion experiments need them
- benchmark-style QA derived from the resulting structured scene

Why this path is valuable:

- gives full control over scenario design
- lets us target missing edge cases like occlusion and partial observability
- allows systematic comparison between single-agent and cooperative reasoning

Main risk:

- higher engineering effort
- stronger burden to justify fidelity and realism

## Recommended Approach

The recommended plan is:

### Phase 1: Attempt benchmark-faithful recovery

First try to recover the missing processed V2V-GoT / V2V4Real assets.

Reason:

- if successful, this gets us to real cooperative experiments fastest
- it preserves the strongest connection to the benchmark used in the current project

### Phase 2: Build a reproducible synthetic cooperative generator

In parallel or immediately after, build our own generation pipeline for missing cases.

Reason:

- even if official assets are recovered, we still need controlled scenarios for occlusion, partial visibility, and agent-complementary evidence
- this is where the more novel experiments will come from

## How We Would Generate the Missing Data Properly

## Option 1: Regenerate V2V4Real-style processed arrays

This means reproducing the missing `.npy` assets in the exact format expected by the local pipeline.

Required inputs:

- V2V4Real raw dataset or upstream processed inputs
- model checkpoints for the detector used in `no_fusion_keep_all`
- OpenCOOD / DMSTrack inference pipeline

Expected outputs:

- `0000_gt.npy`
- `0000_gt_object_id.npy`
- `ego/0000_pred.npy`
- `ego/0000_pred_score.npy`
- `1/0000_pred.npy`
- `1/0000_pred_score.npy`
- `0000_gt_object_id_visible_to_ego.npy`
- `0000_gt_object_id_visible_to_1.npy`
- `0000_gt_object_id_invisible_to_ego.npy`
- `0000_gt_object_id_invisible_to_1.npy`

Why this is attractive:

- it plugs directly into the existing loader and graph code
- it stays benchmark-compatible

What it takes:

- raw or hidden upstream data access
- detector inference environment matching V2V-GoT / OpenCOOD expectations
- GPU compute
- substantial disk space

## Option 2: Generate synthetic cooperative data in simulation

This means creating controlled multi-agent scenarios and exporting them in a format compatible with our graph pipeline.

Strong candidate toolchain:

- CARLA
- OpenCDA
- optionally SUMO for traffic co-simulation

External support for this direction:

- OpenCDA is explicitly designed for cooperative driving automation simulation and evaluation: <https://opencda-documentation.readthedocs.io/>
- OPV2V is generated from OpenCDA + CARLA and emphasizes reproducible multi-agent scenes with saved configuration seeds: <https://mobility-lab.seas.ucla.edu/opv2v/>
- V2Xverse explicitly provides multi-agent driving dataset generation and closed-loop evaluation: <https://collaborativeperception.github.io/V2Xverse/>

What we should generate:

- multi-agent synchronized poses
- per-agent LiDAR and optionally camera observations
- GT object states and IDs in a shared global frame
- agent-specific visibility / occlusion labels
- scenario metadata
- benchmark-style question targets

Best use case:

- create exactly the missing scenarios the current benchmark slice does not give us
- especially occluded objects, complementary views, and multi-agent support cases

## What Infra / Models Will Be Required

## Minimal path: benchmark-faithful regeneration

Infra:

- one or more GPUs for detector inference
- enough disk for full `.npy` processed assets
- V2V-GoT + DMSTrack + OpenCOOD compatible environment

Models:

- the detector corresponding to `no_fusion_keep_all`
- optionally CoBEVT / V2X-ViT / other fusion backbones if we want comparative processed outputs

Engineering tasks:

- reconstruct correct file layout
- run per-frame inference
- export GT and visibility arrays
- validate timestamp alignment

## Stronger research path: synthetic generation

Infra:

- CARLA simulator
- OpenCDA
- optional SUMO integration
- scenario configuration management
- storage for generated logs and annotations

Models:

- none, if starting from simulator ground truth only
- optional detector models if we want realistic noisy observations rather than oracle boxes
- optional learned fusion models later for comparison against graph fusion

Engineering tasks:

- scenario generation
- multi-agent sensor synchronization
- coordinate-frame export
- visibility and occlusion labeling
- export into graph-consumable processed format

## How To Justify and Back the Approach

The justification should be:

### 1. We are not replacing the benchmark casually

We still anchor the project to V2V-GoT / V2V4Real as the benchmark target.

The generated data is meant to:

- recover missing processed cooperative artifacts
- provide controlled experiments where the benchmark assets are incomplete

### 2. The generated format matches the project’s reasoning interface

The graph pipeline expects:

- synchronized agent observations
- GT identities
- visibility information
- spatial alignment

So the generated data is not arbitrary; it is designed to satisfy the exact reasoning assumptions already implemented in the repo.

### 3. Simulation is already standard in cooperative perception research

This is well-supported by existing benchmarks:

- OPV2V: OpenCDA + CARLA simulated V2V benchmark
- V2X-Sim: synthetic synchronized multi-agent V2X dataset with preprocessing pipelines
- V2Xverse: simulation platform for collaborative autonomous driving with dataset generation

This makes synthetic generation easy to justify as a standard research fallback or complement.

## Existing Benchmarks We Should Consider

### 1. V2V4Real

Best for:

- real-world V2V cooperative perception
- staying closest to the current V2V-GoT benchmark story

Why it matters here:

- it already matches the two-vehicle cooperative framing
- our current benchmark adaptation is based on it

Source:

- <https://mobility-lab.seas.ucla.edu/v2v4real/>

### 2. OPV2V

Best for:

- simulated V2V scenes
- reproducible multi-agent generation
- scenarios with several connected vehicles and occlusion

Why it matters here:

- useful inspiration for our own synthetic generator
- especially strong if we need more than two vehicles or controlled reproducibility

Source:

- <https://mobility-lab.seas.ucla.edu/opv2v/>

### 3. V2X-Sim 2.0

Best for:

- synchronized multi-agent synthetic data
- preprocessing and benchmarking patterns
- V2X settings including RSUs

Why it matters here:

- shows how a synthetic benchmark is turned into task-specific preprocessed data
- can inspire our export and validation workflow

Sources:

- <https://ai4ce.github.io/V2X-Sim/>
- <https://coperception.readthedocs.io/en/stable/tools/det/>

### 4. V2X-Real

Best for:

- larger-scale real-world multi-agent V2X settings
- multiple vehicles plus infrastructure

Why it matters here:

- useful if the project later expands beyond two CAVs
- good justification for moving from V2V-only to broader V2X experiments

Source:

- <https://mobility-lab.seas.ucla.edu/v2x-real/>

### 5. DAIR-V2X

Best for:

- real-world vehicle-infrastructure cooperation

Why it matters here:

- less aligned with current V2V framing
- but useful if infrastructure-supported graph reasoning becomes relevant later

Source:

- <https://arxiv.org/abs/2204.05575>

## Can We Evaluate Whether the Generated Data Is Good Enough?

Yes. We should evaluate both **data validity** and **task usefulness**.

## A. Data validity checks

These should be required before any model benchmarking.

### 1. Synchronization

Check:

- same scene ID and timestamp across all agents
- aligned ego poses and transforms
- no missing agent files for a claimed cooperative frame

### 2. Geometry consistency

Check:

- transformed object positions agree across agents within a tolerance
- GT IDs are consistent across views
- sensor poses and object coordinates remain physically plausible

### 3. Visibility labeling quality

Check:

- visible / invisible assignments match line-of-sight or simulator visibility
- objects marked invisible are not trivially observable in that agent view

### 4. Temporal consistency

Check:

- object IDs persist across consecutive frames
- tracks do not flicker excessively without a cause

### 5. Distribution sanity

Check:

- object counts per frame
- distance distribution
- occlusion frequency
- per-agent detection coverage

## B. Task usefulness checks

These tell us whether the dataset is actually useful for the research question.

### 1. Single-agent vs cooperative gap

The generated data should include enough cases where:

- one agent alone is insufficient
- another agent contributes complementary evidence

Otherwise, the cooperative setting is not actually being tested.

### 2. Benchmark-style query coverage

The dataset should support questions about:

- visible objects
- object count
- relation facts
- trajectory-nearness
- occlusion and partial observability

### 3. Ground-truth answer derivability

For each generated QA sample, the gold answer should be derivable deterministically from:

- GT objects
- agent visibility
- trajectory
- relation rules

This is crucial for research credibility.

## C. Standard metrics we should use

For perception-style quality:

- AP / recall for object detection if detector outputs are involved
- association quality for cross-agent matching
- visibility-label precision / recall if labels are inferred

For graph / QA quality:

- exact-match QA accuracy
- count accuracy
- relation accuracy
- visibility-filter accuracy
- single-agent vs cooperative delta on the same samples

For temporal quality:

- object persistence statistics
- identity consistency across frames

For benchmark credibility:

- train / val / test separation by scenario
- fixed random seeds for simulation
- reproducible generation configs

## Industry / Research Standard Expectations

To claim the generated data is credible, we should be able to show:

- synchronized multi-agent timestamps
- calibrated, shared coordinate frames
- consistent object identities
- deterministic ground-truth generation
- documented scenario generation process
- reproducible splits
- quantitative validation metrics

That is roughly the minimum expected standard in current cooperative perception datasets and papers.

## Recommended Next Steps

### Immediate

1. Verify whether the missing V2V-GoT processed zip archives can be obtained.
2. If yes, restore them and run the current graph pipeline end to end.
3. If no, define the exact export schema we need for synthetic generation.

### Short term

4. Build a small synthetic pilot generator for 2 CAVs with:
   - synchronized poses
   - GT object IDs
   - visibility labels
   - per-agent observations
5. Export in the same naming format expected by `V2VGoTProcessedAssetLoader`.
6. Run `kg_coop_drive` on that pilot data and confirm the graph behaves correctly.

### After pilot validation

7. Add scenario families specifically targeting:
   - occlusion
   - partial visibility
   - complementary evidence between agents
   - near-trajectory notable objects
8. Derive deterministic QA samples from the generated scenes.
9. Compare:
   - single-agent graph reasoning
   - cooperative graph reasoning
   on the same generated benchmark slices.

## Bottom Line

What is missing is not the benchmark questions themselves. What is missing is the processed, synchronized, per-frame cooperative perception data needed to make the graph pipeline fully operational in multi-agent mode.

The best practical strategy is:

- first try to recover the official V2V-GoT / V2V4Real processed arrays
- then build a synthetic cooperative data-generation pipeline for controlled missing cases

That gives us:

- benchmark faithfulness where possible
- research flexibility where the benchmark assets are incomplete
- a defensible path for evaluating cooperative graph reasoning by current research standards

## Sources

External:

- V2V4Real: <https://mobility-lab.seas.ucla.edu/v2v4real/>
- OPV2V: <https://mobility-lab.seas.ucla.edu/opv2v/>
- V2X-Real: <https://mobility-lab.seas.ucla.edu/v2x-real/>
- OpenCDA docs: <https://opencda-documentation.readthedocs.io/>
- V2Xverse: <https://collaborativeperception.github.io/V2Xverse/>
- CoPerception / V2X-Sim detection preprocessing: <https://coperception.readthedocs.io/en/stable/tools/det/>
- V2X-Sim project: <https://ai4ce.github.io/V2X-Sim/>

Local:

- [README.md](/Users/bhavya/Desktop/ms_projects/V2V-GoT/README.md:1)
- [DMSTrack DATA.md](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/docs/DATA.md:1)
- [temp_qa_generation.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/temp_qa_generation.py:1)
- [infrence_utils.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/infrence_utils.py:161)
- [v2vgot_processed_assets.py](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive/infrastructure/v2vgot_processed_assets.py:1)
