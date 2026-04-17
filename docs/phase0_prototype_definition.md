# Phase 0 Prototype Definition

## Purpose

Define the first narrow prototype clearly enough that we can build it iteratively without mixing too many unknowns.

## First Prototype

The first prototype should answer:

"Can we build local scene graphs for two agents, fuse them into a cooperative graph for a single frame, and answer a small deterministic QA subset using structured graph queries?"

## Scope

In scope:

- two agents
- single-frame reasoning
- object existence
- object counting
- basic relative position
- visibility-aware filtering if available from data or heuristics
- deterministic graph queries

Out of scope for the first prototype:

- full multi-turn graph-of-thought reasoning
- planning and trajectory generation
- long-horizon prediction
- raw networking
- learned fusion models
- full LLM planner

## Input

Expected inputs for the prototype:

- per-agent scene observations for the same frame
- ego pose and shared frame alignment information
- object-level detections or ground-truth boxes
- basic metadata such as agent id and timestamp

## Output

Expected outputs for the prototype:

- one local graph per agent
- one fused cooperative graph
- deterministic answers for a small QA subset
- query traces showing how the answer was produced

## Success Criteria

We consider the first prototype successful if:

- the same physical object from two agents can be merged correctly in simple examples
- the fused graph contains provenance for each fused object
- deterministic graph queries answer a small hand-picked QA subset correctly
- at least a few examples show an advantage over single-agent reasoning

## Failure Conditions

We should stop and debug before scaling if:

- coordinate alignment is unclear
- object identity merges are obviously wrong
- graph queries rely on hidden assumptions not present in the graph
- outputs cannot be inspected and traced back to graph facts

## Why This Prototype

This keeps the initial unknowns limited to:

- data adaptation
- schema definition
- local graph construction
- cross-agent alignment
- object association
- fusion
- deterministic querying

It intentionally delays the LLM planner until the graph layer is validated.
