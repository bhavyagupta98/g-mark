# Paper Detail Notes

## Why this note

This note captures the extra details that need to be made explicit in the paper or proposal:

- the high-level framework
- the role of each component
- the datasets we will use
- the state-of-the-art baselines we should compare against
- the initial motivating results we already have
- the current scope and limitations

The goal is to make the project description concrete enough for a conference-style research narrative.

## One-sentence idea

We propose a provenance-aware, uncertainty-aware knowledge-graph reasoning layer for cooperative driving, where each connected vehicle first forms a local structured scene representation, then exchanges structured facts with other vehicles, and finally performs QA or scene reasoning over a fused cooperative graph rather than relying only on implicit feature fusion or free-form LLM reasoning.

## High-level framework

The current idea can be described as a five-stage pipeline.

### 1. Multi-agent perception input

The system starts from synchronized cooperative driving scenes in the `V2V4Real / V2V-GoT` ecosystem.

For each frame, we have access to:

- vehicle poses
- future ego trajectory
- ground-truth object boxes and IDs
- per-agent prediction outputs
- per-agent visibility / invisibility labels

Conceptually, this gives us a shared scene observed from multiple vehicles at the same timestamp.

### 2. Local graph construction

Each vehicle constructs its own local graph from its own evidence.

The local graph contains:

- objects
- positions
- support confidence
- visibility
- ego-relative relations
- provenance of observations
- uncertainty / conflict signals

The purpose of this stage is to preserve what each vehicle independently knows before cooperative fusion happens.

### 3. Cooperative graph fusion

Local scene facts from multiple vehicles are then aligned and fused into a shared cooperative graph.

The fusion stage should handle:

- timestamp alignment
- object association across vehicles
- support attachment from multiple vehicles
- conflict handling
- candidate vs supported distinction
- provenance preservation

This is the core technical area where the knowledge graph becomes more than a data structure. It becomes the explicit fusion substrate.

### 4. Deterministic graph reasoning

Once the graph is built, reasoning is done through deterministic operations such as:

- object selection
- visibility filtering
- relation filtering
- trajectory-nearness filtering
- attribute reads
- pairwise comparison
- provenance tracing

This keeps reasoning:

- inspectable
- reproducible
- less prone to hidden assumptions

### 5. Downstream QA

The final scene understanding or QA answer is generated from the graph state.

This allows us to compare:

- single-agent graph reasoning
- cooperative graph reasoning
- LLM-based or non-graph baselines

on the same questions.

## Where the novelty lies

The novelty is not simply “we use a graph.”

The stronger and more defensible novelty claims are:

### 1. Explicit structured cooperative world modeling

Most cooperative perception systems fuse features or predictions implicitly.

Our approach explicitly represents:

- objects
- relations
- visibility
- provenance
- uncertainty

at the scene level.

### 2. Provenance-aware graph fusion

The cooperative graph preserves which vehicle contributed which evidence.

This matters because it enables:

- interpretability
- uncertainty-aware reasoning
- conflict-aware fusion

### 3. Candidate vs supported object handling

The graph can preserve weaker hypotheses instead of prematurely collapsing all beliefs into a single final object list.

This is especially important in occluded or ambiguous scenes.

### 4. Deterministic querying over the cooperative graph

Instead of asking an LLM to reason directly over raw perception outputs, we query a structured graph with fixed semantics.

### 5. Stronger behavior under partial observability

The main expected value of the approach is in scenarios where:

- one vehicle is occluded
- another vehicle has complementary evidence
- the fused graph can recover a better world model than any single vehicle alone

## Detailed component expectations

The paper should make clear what each component is expected to contribute.

### Local graph module

Expected role:

- represent each vehicle’s view faithfully
- establish the single-agent baseline
- avoid mixing cooperative and local errors too early

### Alignment and synchronization module

Expected role:

- ensure graph fusion happens only across comparable scene states
- align the same physical object across vehicles
- preserve coordinate consistency

### Graph fusion module

Expected role:

- merge complementary evidence
- avoid over-merging conflicting detections
- preserve uncertainty where the data remains ambiguous

### Query engine

Expected role:

- provide a deterministic reasoning substrate
- make results reproducible
- allow direct comparison between local and cooperative graph outputs

### Optional uncertainty / scoring extensions

Possible future research direction:

- ensemble-style reasoning across graph hypotheses
- energy-based selection over competing graph states
- confidence-aware graph arbitration inspired by structured reasoning systems like KLDrive

These are not strictly required for the first paper version, but they are plausible extensions if the core graph pipeline works and stronger uncertainty handling becomes necessary.

## Datasets we will use

The paper should name the datasets clearly.

### 1. V2V4Real

Primary cooperative perception dataset.

Why:

- real-world
- synchronized two-vehicle cooperative setup
- directly aligned with the upstream perception ecosystem we are already using

This is the best dataset for the core cooperative perception and graph construction experiments.

### 2. V2V-GoT-QA / V2V-QA

Primary QA benchmark layer on top of cooperative perception outputs.

Why:

- gives benchmark-style questions
- allows evaluation of graph-based scene reasoning
- aligns with the current V2V-GoT ecosystem we are already integrating with

### 3. Potential later dataset for controlled synthesis

If later experiments need stronger targeted occlusion scenarios, we may introduce:

- a synthetic V2V4Real-like no-fusion occlusion dataset

This would be justified only if the current benchmark data is insufficient for controlled occlusion-specific evaluation.

## Why we intentionally use two vehicles first

This should be stated clearly so it does not look like a limitation we are ignoring.

Our current focus is intentionally on two vehicles because:

- V2V4Real itself is a two-vehicle cooperative dataset
- two agents are already sufficient to expose the main cooperative reasoning problem
- they let us isolate alignment, fusion, provenance, and occlusion recovery cleanly
- adding more agents too early would increase complexity without first validating the core fusion mechanism

Defensible statement:

> We intentionally start with the two-agent setting because it is the minimal cooperative setting that still captures the essential challenges of multi-agent scene alignment, graph fusion, provenance-aware reasoning, and occlusion recovery, while remaining aligned with the available benchmark data.

## State-of-the-art and comparison baselines

The paper should separate **perception baselines** from **reasoning baselines**.

### A. Cooperative perception baselines on V2V4Real

The V2V4Real codebase and paper benchmark several strong cooperative perception methods. These are the main upstream perception baselines we should acknowledge and potentially compare against:

- F-Cooper
- V2VNet
- Attentive Fusion
- V2X-ViT
- CoBEVT

These models are important because they represent the dominant family of feature-fusion or cooperative perception approaches.

### B. Strong internal baseline for our project

This is the most important direct comparison for the first stage:

- ego-only / single-agent graph reasoning
- cooperative graph reasoning

This tells us whether cooperation over the graph actually helps.

### C. V2V-LLM baseline

Relevant for QA / multimodal reasoning comparison:

- `V2V-LLM` is a strong baseline because it already frames cooperative autonomous driving as a multimodal LLM problem without the explicit graph fusion layer we are proposing.

### D. V2V-GoT baseline

Relevant as a reasoning benchmark baseline:

- `V2V-GoT` adds graph-of-thought reasoning over the multimodal cooperative setup

This is not a perfect apples-to-apples comparison, because our contribution is more about an explicit graph substrate than a graph-of-thought prompting strategy, but it is still an important reference point.

## How we should position the comparison

We should be careful not to overclaim.

The clean positioning is:

- compare our graph pipeline against the single-agent version of itself
- compare against non-graph cooperative reasoning baselines where possible
- compare against V2V-LLM / V2V-GoT on benchmark-style QA if the task protocol is compatible

In other words:

> our first burden is not to beat every upstream cooperative perception model on raw detection metrics, but to show that explicit graph-based cooperative reasoning gives value over weaker or less-structured reasoning baselines.

## Initial motivating results we already have

The paper should include early results that justify the research direction.

After recovering the missing processed cooperative assets and fixing the root-selection logic, the current implementation now:

- loads both ego and partner predictions
- loads per-agent visibility files
- builds cooperative scenes from the correct `no_fusion_keep_all` root
- performs cross-agent observation association
- performs cross-agent support attachment

Observed example signals from the current pipeline:

- `has_pred_for_ego = True`
- `has_pred_for_cav1 = True`
- `has_visibility_for_ego = True`
- `has_visibility_for_cav1 = True`
- `Loaded observations = 14`
- `Loaded visibility facts = 4`
- `cross_agent_match_count = 2`
- `Attached 1 cross-agent matches onto 1 existing tracks`

This is meaningful because it shows the system is no longer only operating in a single-agent fallback mode.

The local validation also became richer after the cooperative assets were properly loaded:

- average objects per frame increased to `5.20`
- average relations per frame increased to `11.00`
- average visibility facts per frame increased to `3.20`
- average supported tracks per frame increased to `1.60`

These are not final benchmark results, but they are strong enough to motivate the research:

- the recovered cooperative assets are materially changing the graph
- the code path for cooperative reasoning is active
- the graph becomes richer and more informative when the correct assets are used

## What we will evaluate

The paper should present a clear evaluation plan.

### 1. Local vs cooperative graph comparison

For the same frames:

- local graph from ego only
- cooperative graph from both vehicles

Measure differences in:

- supported object count
- candidate object count
- visibility fact count
- cross-agent match count
- provenance richness

### 2. QA-style task evaluation

Tasks:

- visible object existence
- object count
- relation reasoning
- near-trajectory reasoning
- provenance-aware inspection

Metrics:

- exact-match QA accuracy
- count accuracy
- relation accuracy
- visibility-filter accuracy

### 3. Occlusion-focused analysis

Most important experimental angle:

- identify cases where ego misses an object or has weaker evidence
- measure whether the cooperative graph recovers that information

### 4. Error analysis

We should inspect:

- over-merged objects
- unresolved candidates
- false cross-agent matches
- uncertainty propagation failures

## Planned ablations

The paper should include ablations that isolate what matters.

### 1. Single-agent vs cooperative graph

Most essential ablation.

### 2. Provenance-aware vs provenance-agnostic fusion

Does preserving source-agent identity actually help?

### 3. Supported-only vs supported-plus-candidate reasoning

Do candidates help recover missing objects, or mostly add noise?

### 4. Visibility-aware vs visibility-agnostic querying

How much do explicit visibility facts matter for QA correctness?

### 5. Conservative fusion vs stronger fusion policy

Later-stage ablation if needed:

- current conservative support attachment
- stronger uncertainty-aware or conflict-aware fusion

### 6. Upstream processed root comparison

Potential later ablation:

- `no_fusion_keep_all`
- `cobevt`
- `attfuse`
- `v2xvit`

This helps separate improvements from:

- upstream perception quality
- downstream graph reasoning quality

## Current limitations

The paper should be explicit about scope limits.

### 1. Two-agent focus

Current work is not yet a 3+ vehicle cooperative graph framework.

### 2. Conservative fusion

The current fusion design prioritizes safety and interpretability over aggressive merging.

### 3. Benchmark dependence

The current experiments depend on the V2V4Real / V2V-GoT processed ecosystem.

### 4. No final uncertainty-aware graph scoring yet

Ideas like ensemble or energy-based graph selection are future opportunities, not yet fully realized components.

## Short project positioning statement

Our project fits as an explicit structured reasoning layer on top of cooperative perception outputs. We use V2V4Real-based processed multi-agent perception data and V2V-GoT-style QA tasks to build and evaluate a provenance-aware, uncertainty-aware cooperative knowledge graph. The main hypothesis is that explicit graph fusion and deterministic graph querying will improve robustness and interpretability over single-agent reasoning and less-structured cooperative reasoning, especially under occlusion and partial visibility.

## Sources

External sources used for the dataset and baseline positioning:

- V2V4Real CVPR 2023 overview: https://cvpr.thecvf.com/virtual/2023/poster/22810
- V2V4Real GitHub: https://github.com/ucla-mobility/V2V4Real
- V2V4Real paper page: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_V2V4Real_A_Real-World_Large-Scale_Dataset_for_Vehicle-to-Vehicle_Cooperative_Perception_CVPR_2023_paper.html
- V2X-ViT official publication page: https://link.springer.com/chapter/10.1007/978-3-031-19842-7_7
- V2V-LLM overview: https://research.nvidia.com/labs/twn/publication/cvprw_2025_v2vllm/
- V2V-GoT preprint metadata: https://papers.cool/arxiv/2509.18053v3
