# V2V4Real, Processed Cooperative Data, and a No-Fusion Synthesis Plan

## Why this note exists

This note explains, at a conceptual level:

- what `V2V4Real` is
- what the processed `.npy` files in the V2V-GoT ecosystem actually represent
- how those files are formed from synchronized multi-agent perception
- how we could build something similar to the `no_fusion_keep_all` data for our own occlusion-focused use case

The goal is to make the data pipeline easy to explain now, and reusable later if we decide to synthesize new cooperative data ourselves.

## What V2V4Real is

`V2V4Real` is a **real-world cooperative perception dataset** for autonomous driving.

Conceptually:

- two vehicles drive through the same traffic scene at the same time
- each vehicle has its own sensors
- each vehicle has its own pose and localization information
- the dataset keeps the two views synchronized
- objects are annotated consistently across time and across views

So the dataset is not just “what one car sees.” It is a **multi-agent view of the same world**.

That makes it useful for cooperative driving research, especially questions like:

- what does ego miss because of occlusion?
- what does the second vehicle see that ego cannot?
- how much does multi-agent cooperation improve object detection or reasoning?

In simple terms:

> V2V4Real is a real-world, synchronized, two-vehicle perception dataset designed for cooperative autonomous driving.

## What the dataset contains conceptually

At a high level, each synchronized frame gives us:

- the state of the world at one time
- vehicle A's view of that world
- vehicle B's view of that world
- a way to align those views into a common frame
- ground-truth object annotations

This is the key idea:

- **same scene**
- **same time**
- **different viewpoints**

Because the vehicles are synchronized and localized, we can compare:

- what both vehicles detect
- what only one vehicle detects
- which objects are visible or occluded for each vehicle

## What the `.npy` files actually are

The `.npy` files are **processed perception artifacts**.

They are not:

- raw video
- raw LiDAR logs
- natural-language descriptions
- graph files

They are NumPy arrays storing machine-readable outputs of the perception pipeline.

Examples of what they store:

- predicted 3D object boxes
- prediction scores
- ground-truth boxes
- ground-truth object IDs
- visibility / invisibility labels for each agent
- optional feature maps used by downstream multimodal / reasoning models

So conceptually:

> the `.npy` files are a compact, structured export of the cooperative perception state at each timestamp.

## How the processed data is formed conceptually

The conceptual pipeline looks like this:

```text
Real multi-agent driving scene
        ->
Synchronized sensor capture for each vehicle
        ->
Localization / calibration / common-frame alignment
        ->
Per-agent perception model inference
        ->
Post-processing into boxes, scores, IDs, visibility, features
        ->
Saved per-frame .npy artifacts
        ->
QA generation / graph reasoning / downstream evaluation
```

### Step 1: Capture the same scene from multiple vehicles

The starting point is a shared traffic scene.

Each vehicle captures:

- LiDAR
- cameras
- localization

The important part is that both vehicles are recording the **same moment** from different positions.

### Step 2: Synchronize and align the data

This is the foundation of cooperative perception.

The dataset uses:

- timestamps
- poses
- calibration / transformations

to make sure that vehicle A and vehicle B can be interpreted in one common spatial frame.

This is what allows later steps like:

- matching the same physical object across agents
- comparing visibility across agents
- building a shared scene graph

### Step 3: Run perception per agent

A perception model is run on each vehicle’s data.

Depending on the experiment setting, that model may be:

- no fusion
- early fusion
- attention fusion
- V2X-ViT
- CoBEVT
- or another fusion variant

For our purposes, the important mode is:

- `no_fusion_keep_all`

This means, conceptually:

- keep each vehicle’s predictions separate
- do not collapse them into one fused detector output too early
- preserve agent-specific evidence

That is exactly what a graph-based cooperative reasoning pipeline wants.

### Step 4: Convert inference outputs into saved structured arrays

After inference, the system saves structured outputs such as:

- predicted boxes
- scores
- GT boxes
- GT IDs
- visibility labels
- optional intermediate feature maps

These are written as `.npy` files, one timestamp at a time.

Why this is useful:

- very fast to load
- preserves geometry and confidence numerically
- easy to use in downstream scripts
- avoids rerunning the full perception model every time

## What `no_fusion_keep_all` means conceptually

`no_fusion_keep_all` is a useful intermediate representation because it preserves **per-agent perception outputs**.

Instead of saying:

- “here is one final fused answer”

it says:

- “here is what ego predicted”
- “here is what CAV_1 predicted”
- “here is the shared GT”
- “here is what each agent could or could not see”

That is powerful for us because our graph pipeline needs to reason over:

- overlap
- disagreement
- complementary visibility
- cross-agent support
- uncertainty

So if someone asks why this format matters, the answer is:

> `no_fusion_keep_all` preserves the raw multi-agent evidence before aggressive fusion, which makes it ideal for explicit graph-based cooperative reasoning.

## Where the synchronization comes from conceptually

This is important to understand clearly.

The synchronization is not something our graph code invents later.

It comes from the dataset / perception pipeline design itself:

- shared scene time
- synchronized or aligned vehicle captures
- known ego poses
- known partner poses
- common coordinate transforms
- consistent object identity labels

Our graph code then consumes that synchronized processed state.

So the graph does not solve synchronization from scratch.
It assumes the upstream data has already been synchronized and aligned enough to support multi-agent reasoning.

## Where V2V-GoT fits into this

V2V-GoT sits on top of this cooperative perception stack and adds:

- benchmark QA tasks
- multimodal reasoning
- graph-of-thought style chaining

In our case, we are not primarily using the LLM part right now.

We are using the processed cooperative perception artifacts to build:

- a canonical scene representation
- local graphs
- cooperative scene graphs
- deterministic query-based reasoning

So the relationship is:

```text
V2V4Real / cooperative perception
        ->
processed per-frame .npy artifacts
        ->
V2V-GoT QA / reasoning ecosystem
        ->
our knowledge graph pipeline
```

## If we want to build something similar tomorrow for occluded vehicles, how would we do it?

If the goal is:

- create new cooperative data
- keep it similar in spirit to `no_fusion_keep_all`
- specifically focus on occluded vehicles

then the right approach is:

- keep the same conceptual format
- keep per-agent outputs separate
- explicitly preserve visibility and occlusion labels

### High-level plan

```text
Create synchronized multi-agent scenes
        ->
Generate or collect per-agent observations
        ->
Run per-agent no-fusion perception
        ->
Save GT, IDs, predictions, scores, visibility arrays
        ->
Load into our graph pipeline
        ->
Compare ego-only vs cooperative reasoning
```

## Two ways to build this

### Option A: Extend the current V2V-GoT / V2V4Real pipeline

This is the closest to the current benchmark setup.

What we would do:

- keep using the current V2V-GoT / OpenCOOD / DMSTrack ecosystem
- generate more processed `no_fusion_keep_all`-style assets
- target scenarios where one agent is occluded and another has a better view

This is the most benchmark-faithful path.

### Option B: Build a synthetic cooperative generator

This is the more controllable research path.

We would:

- simulate two or more vehicles
- create controlled occlusion scenarios
- run a no-fusion perception pipeline per vehicle
- export the same style of per-agent arrays

This is the better option when we want:

- more occlusion cases
- more controlled comparisons
- more than two agents
- richer failure / recovery scenarios

## A practical one-week outline for synthesizing no-fusion occlusion data

If we want a first usable version in roughly a week, the fastest realistic plan is:

### Day 1: Lock the target output schema

Decide that the generated dataset must export, at minimum:

- GT boxes
- GT object IDs
- ego predicted boxes
- ego predicted scores
- partner predicted boxes
- partner predicted scores
- visibility IDs per agent
- optional pose metadata if needed for QA generation

The key rule:

- match the `no_fusion_keep_all` structure closely enough that our current graph loader can read it with minimal changes

### Day 2: Define occlusion scenario templates

Create a small set of scenario types such as:

- lead vehicle blocks ego’s view of a farther vehicle
- parked truck blocks side-street visibility
- intersection crossing where one vehicle sees around a corner
- highway merge where one agent sees a hidden merging car earlier

The goal is not volume yet.
The goal is to build **high-value cooperative scenes**.

### Day 3: Generate synchronized multi-agent scenes

Using the cooperative driving codebase / simulator stack:

- place two vehicles in the same scene
- synchronize timestamps
- save each vehicle’s pose
- save the shared world GT

For synthetic data, a likely toolchain would be:

- OpenCDA / CARLA for simulation
- OpenCOOD-compatible data formatting

For benchmark extension, use the existing V2V-GoT / OpenCOOD data format where possible.

### Day 4: Run per-agent no-fusion perception

For each agent independently:

- run the detector on that agent’s own observation only
- do not fuse partner features yet
- save predicted boxes and scores separately

This is critical because we want to preserve:

- what each agent believed on its own
- before graph fusion or reasoning

### Day 5: Derive visibility / occlusion labels

For each GT object and each agent:

- visible
- invisible / occluded

This can come from:

- simulator visibility / line-of-sight
- geometric ray tests
- dataset annotation rules

This step is important because cooperative benefit is often about:

- object invisible to ego
- object visible to partner

### Day 6: Export and load into our current graph pipeline

Save the generated data using the same conceptual conventions as `no_fusion_keep_all`.

Then immediately test:

- can our processed asset loader read it?
- can we build scene graphs from it?
- do cross-agent matches appear?
- does the visibility filter behave sensibly?

This is the earliest useful end-to-end milestone.

### Day 7: Validate and inspect cooperative benefit

Pick a handful of scenes and compare:

- ego-only graph
- cooperative graph

Look for:

- extra supported tracks from the second vehicle
- recovered occluded objects
- better visibility-aware reasoning
- more correct answers on QA-style queries

## How to use the existing V2V-GoT / V2V codebase for this

Conceptually, we should reuse the upstream stack for:

- scene formatting
- synchronized timestamp handling
- pose / transform conventions
- inference and post-processing
- per-agent prediction export

Then we reuse our own repo for:

- graph construction
- cross-agent reasoning
- deterministic querying
- evaluation of cooperative benefit

So the split of responsibilities is:

### Upstream cooperative perception stack

Use it for:

- generating per-agent detections
- exporting GT / ID / visibility arrays
- maintaining compatibility with existing data conventions

### Our graph pipeline

Use it for:

- building structured scene facts
- comparing ego-only vs cooperative understanding
- measuring whether cooperation helps under occlusion

## What should the synthetic data preserve to stay useful?

If we synthesize new data later, it should preserve these properties:

- synchronized timestamps across agents
- known poses for each agent
- consistent world object IDs
- per-agent predictions kept separate
- explicit visibility / occlusion labels
- enough occlusion cases where cooperation actually matters

If those are preserved, then the data will be useful for our graph reasoning pipeline.

If those are missing, then it will be difficult to justify the dataset as a proper cooperative benchmark.

## What this means for us right now

Right now, after the recovery work:

- we already have a real cooperative processed dataset that is sufficient to test the current pipeline
- we do not need to synthesize new data immediately just to make the code work
- but it is still worth planning a synthetic no-fusion occlusion generator for later controlled experiments

So the right mental model is:

- **current V2V4Real / V2V-GoT assets** let us validate the cooperative pipeline now
- **future synthetic no-fusion occlusion data** would let us test our own research questions more directly

## Short explanation you can say aloud

> V2V4Real is a real-world cooperative perception dataset where two vehicles observe the same scene at the same time. Their sensor data is synchronized using timestamps and poses, and the perception pipeline converts each frame into structured NumPy arrays such as predicted 3D boxes, confidence scores, ground-truth boxes, object IDs, and visibility labels. In the `no_fusion_keep_all` setting, each vehicle’s predictions are kept separate, which is exactly what we need for graph-based cooperative reasoning. If we want to build a similar dataset later for occluded-vehicle scenarios, we should keep the same idea: synchronized multi-agent scenes, separate per-agent perception outputs, shared GT identities, and explicit visibility labels, then export them in the same structured format for the graph pipeline to consume.

## Sources

External:

- V2V4Real project page: <https://mobility-lab.seas.ucla.edu/v2v4real/>
- V2V4Real GitHub: <https://github.com/ucla-mobility/V2V4Real>
- CVPR 2023 V2V4Real summary: <https://cvpr.thecvf.com/virtual/2023/poster/22810>

Local repo references that align with this understanding:

- [README.md](/Users/bhavya/Desktop/ms_projects/V2V-GoT/README.md:1)
- [DATA.md](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/docs/DATA.md:1)
- [infrence_utils.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/infrence_utils.py:1)
