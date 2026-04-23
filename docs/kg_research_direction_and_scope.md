# Knowledge Graph Cooperative Driving Research Direction

## Project Idea

The core idea of this project is to build an **explicit knowledge-graph-based reasoning layer** for cooperative driving.

Instead of relying only on:

- implicit feature fusion
- black-box multimodal reasoning
- or free-form text reasoning over raw perception outputs

we aim to:

1. let each vehicle construct a **local structured scene representation**
2. share structured scene facts across vehicles
3. fuse those facts into a **cooperative knowledge graph**
4. answer QA or scene-understanding questions over that fused graph

In simple terms:

> each vehicle first understands the world locally, then the vehicles exchange structured scene knowledge, and finally reasoning happens over a cooperative graph rather than over raw detections alone.

## What We Are Building

At a high level, our approach has four stages.

### 1. Local graph construction

Each vehicle builds a local graph from its own evidence, including:

- objects
- positions
- confidence
- visibility
- ego-relative spatial relations
- provenance of supporting observations

### 2. Cooperative fact exchange

Vehicles share structured scene outputs rather than only latent embeddings.

These shared outputs may include:

- object hypotheses
- support evidence
- visibility-related facts
- geometric relations
- provenance and confidence signals

### 3. Cooperative graph fusion

The system then fuses local scene knowledge into a cooperative graph by:

- aligning observations across agents
- merging matched object beliefs
- preserving provenance
- handling candidate vs supported objects
- carrying uncertainty forward instead of overclaiming certainty

### 4. Graph-based reasoning and QA

The fused graph is then queried using deterministic reasoning operations such as:

- object selection
- relation filtering
- visibility filtering
- trajectory-nearness filtering
- attribute lookup
- provenance tracing

The goal is to answer scene-level questions more robustly and more transparently than purely implicit fusion or free-form reasoning.

## Dataset Choice

We are grounding the current implementation in the **V2V4Real / V2V-GoT ecosystem**.

Conceptually:

- `V2V4Real` provides the synchronized two-vehicle cooperative perception data
- processed model outputs are exported as `.npy` artifacts
- `V2V-GoT` builds QA and reasoning tasks on top of that ecosystem
- our work builds an explicit knowledge-graph reasoning layer on top of the processed cooperative perception outputs

So the current project is not creating a new benchmark from scratch.
It is building a new structured reasoning approach on top of an established cooperative perception setup.

## Why We Are Starting With Two Vehicles

We are intentionally starting with **two ego-like vehicles / two connected autonomous vehicles**.

This is not a weakness in the design. It is a deliberate scope choice.

### Why this is intentional

1. **It matches the available benchmark data**

The current V2V4Real / V2V-GoT cooperative setup is fundamentally centered on two synchronized vehicles.

2. **It is the cleanest first test of cooperative reasoning**

With two agents, we can already study the central problem:

- one vehicle misses something
- another vehicle sees it
- the system should combine their knowledge better than a single-agent approach

3. **It keeps the first fusion problem tractable**

The hardest early research questions already appear with two vehicles:

- object alignment
- synchronization
- visibility disagreement
- conflict handling
- provenance-aware merging

Adding more agents too early would increase engineering and evaluation complexity before the two-agent case is well understood.

4. **It gives a defensible experimental control**

Two-agent cooperative reasoning is the natural first step before scaling to:

- more vehicles
- infrastructure nodes
- richer V2X settings

### Defensible statement

> We intentionally focus on the two-agent cooperative setting first because it is the minimal setting that still captures the key challenges of cooperative scene reasoning, while remaining aligned with the available benchmark data and keeping graph fusion interpretable and experimentally tractable.

## Where the Knowledge Graph Fits

The knowledge graph does not replace raw perception.
It sits **between perception outputs and higher-level reasoning**.

The conceptual pipeline is:

```text
Multi-agent perception data
    ->
Processed per-agent detections / GT / visibility
    ->
Local knowledge graphs
    ->
Cooperative graph fusion
    ->
Deterministic graph reasoning / QA
```

So the knowledge graph is the **explicit intermediate world model**.

That is important because it allows:

- inspectable reasoning
- provenance-aware fusion
- uncertainty-aware decision making
- controlled comparison between single-agent and cooperative reasoning

## Where the Novelty Likely Lies

The novelty is not simply “using a graph.”

The more defensible novelty is:

### 1. Explicit structured cooperative world modeling

Each agent contributes structured scene facts rather than only latent feature maps.

### 2. Provenance-aware graph fusion

The cooperative graph remembers:

- which agent supported which object
- which facts came from GT vs perception
- where uncertainty or disagreement still exists

### 3. Uncertainty-aware object handling

Instead of flattening all detections into one certainty level, the graph distinguishes:

- supported objects
- candidate objects
- uncertain or unresolved cases

### 4. Deterministic graph querying

Reasoning is done through explicit graph operations rather than purely hidden or free-form model behavior.

### 5. Cooperative reasoning under partial observability

The strongest expected value of the method is in cases where:

- one vehicle is occluded
- another vehicle has complementary evidence
- structured graph fusion improves what can be inferred about the scene

## Scope of the Current Work

The current scope is:

- two-vehicle cooperative scene understanding
- frame-level and near-temporal scene reasoning
- structured graph construction
- deterministic query-based reasoning
- benchmark-aligned QA-style evaluation

The current work is **not** trying to solve everything at once.

It is intentionally focused on:

- building the representation layer properly
- validating local and cooperative graph behavior
- then evaluating whether cooperative graph reasoning helps over a single-agent baseline

## What We Plan To Do Next

The near-term plan is:

### 1. Properly test the cooperative path on the recovered processed assets

Now that the cooperative `no_fusion_keep_all` data is correctly loaded, we need to evaluate:

- ego-only graph behavior
- cooperative graph behavior
- where the two differ

### 2. Identify informative frames

We need to find scenes where:

- both agents contribute observations
- one agent has complementary evidence
- the cooperative graph differs meaningfully from ego-only reasoning

### 3. Measure cooperative benefit

We need to quantify whether the cooperative graph improves:

- visibility-aware scene understanding
- object support quality
- relation-aware queries
- QA-style answers

### 4. Decide whether stronger fusion logic is needed

After observing the current cooperative behavior, we can decide whether to add more advanced graph-fusion strategies such as:

- better conflict resolution
- stronger multi-observation merging
- uncertainty-aware scoring
- ensemble or energy-based reasoning ideas

## What We Will Evaluate

We plan to evaluate both **graph quality** and **task-level usefulness**.

### A. Graph construction quality

- number of supported objects
- number of candidate objects
- number of cross-agent matches
- number of enriched tracks from cross-agent evidence
- visibility fact coverage
- relation fact coverage

### B. Single-agent vs cooperative comparison

For the same timestamp or scene:

- ego-only graph output
- cooperative graph output

We will compare whether cooperation changes:

- object support
- visibility reasoning
- relation reasoning
- final selected objects for QA

### C. QA-style task performance

Questions involving:

- visible object existence
- object count
- relative position
- near-trajectory filtering
- provenance-aware selection

### D. Robustness under occlusion or incomplete view

This is likely the most important experimental theme.

We want to evaluate whether cooperation helps when:

- ego cannot see an object clearly
- a second vehicle provides complementary evidence

## Planned Ablations

To make the research defensible, we should plan ablations early.

### 1. Single-agent vs cooperative graph

This is the primary ablation.

Compare:

- local graph from ego only
- cooperative graph from both vehicles

### 2. No provenance vs provenance-aware graph

Compare:

- graph fusion without explicit source tracking
- graph fusion with provenance preserved

This tests whether provenance actually helps interpretability or conflict handling.

### 3. Supported-only vs supported-plus-candidate reasoning

Compare:

- only high-confidence supported objects
- reasoning that also includes lower-confidence candidates

This tests whether candidates help recovery or just add noise.

### 4. Visibility-aware vs visibility-agnostic querying

Compare:

- queries that ignore explicit visibility facts
- queries that require visibility-aware filtering

This tests whether visibility reasoning improves QA correctness.

### 5. Cooperative graph fusion variants

As the project grows, compare conservative baseline fusion against stronger variants such as:

- nearest-neighbor object alignment only
- provenance-weighted merging
- uncertainty-aware selection
- ensemble-style or energy-based graph scoring

### 6. Different processed perception roots

Later, it may be useful to compare:

- `no_fusion_keep_all`
- `cobevt`
- `v2xvit`
- `attfuse`

This would help separate:

- benefits from upstream perception quality
- benefits from downstream graph reasoning

## Limitations of the Current Scope

There are important limitations we should state clearly.

### 1. Current focus is only two agents

This is intentional, but it means the work is not yet evaluating more complex multi-agent traffic cooperation.

### 2. The current graph fusion is still conservative

The present implementation:

- matches close observations
- attaches support conservatively
- keeps uncertain candidates rather than over-merging

This is useful for reliability, but may under-use some cooperative evidence.

### 3. Benchmark dependence

The current evaluation is grounded in the V2V4Real / V2V-GoT ecosystem.

This is a strength for comparability, but may limit scenario diversity.

### 4. Stronger controlled occlusion cases may still require synthetic data

Even with the recovered processed assets, later work may still need a custom synthetic dataset to stress-test the method under targeted occlusion conditions.

## What Success Would Look Like

A realistic success claim for this project would be:

- we build a reliable cooperative graph pipeline on top of multi-agent perception outputs
- we show that the graph supports transparent, provenance-aware, deterministic reasoning
- we show that cooperative graph reasoning improves over a single-agent baseline on at least some benchmark-aligned tasks
- we demonstrate especially meaningful benefit in partial-visibility or occluded cases

That would be a strong and defensible contribution.

## Short Summary

Our approach is to use V2V4Real / V2V-GoT cooperative perception outputs as the basis for explicit knowledge-graph scene reasoning. Each vehicle first forms a local graph from its own evidence, then structured scene facts are exchanged and fused into a cooperative graph, and finally QA is answered over that fused graph using deterministic queries. We intentionally focus on two vehicles because this is the minimal setting that still captures the key challenges of cooperative reasoning while staying aligned with the available benchmark data. The main evaluation plan is to compare ego-only and cooperative graph reasoning, especially in cases of partial visibility and occlusion, and to run ablations around provenance, visibility, candidate handling, and fusion strategy.
