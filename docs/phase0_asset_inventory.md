# Phase 0 Asset Inventory

## Objective

Record the external assets we already have, what role they can play in this project, and what they do not solve yet.

## Available Repositories

### `V2V-GoT`

Path:

- `/Users/bhavya/Desktop/ms_projects/V2V-GoT`

What it gives us:

- cooperative driving benchmark context
- V2V4Real data plumbing
- DMSTrack and OpenCOOD integration
- QA generation scripts
- evaluation scripts for QA and grounding outputs
- baseline structure for cooperative reasoning experiments

Useful references:

- [README.md](/Users/bhavya/Desktop/ms_projects/V2V-GoT/README.md:1)
- [temp_qa_generation.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/temp_qa_generation.py:25)
- [eval_v2v4real_3d_grounding.py](/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/eval_v2v4real_3d_grounding.py:243)

What it does not directly provide:

- a cooperative knowledge graph abstraction
- provenance-aware fused world modeling
- typed graph query tools
- tool-constrained KG reasoning

### `auto_drive_copy`

Path:

- `/Users/bhavya/Desktop/ms_projects/auto_drive_copy`

What it gives us:

- typed tool-based reasoning pattern
- scene binding abstraction
- relation and counting tools
- constrained execution flow for LLM reasoning
- a useful reference for how to keep the model grounded in structured operations

Useful references:

- [agent.py](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/code/agent.py:36)
- [agent_readme.md](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/code/agent_readme.md:1)
- [tools.py](/Users/bhavya/Desktop/ms_projects/auto_drive_copy/tools.py:1)

What it does not directly provide:

- multi-agent cooperative fusion
- V2V object association
- provenance-aware cooperative graphs
- cooperative visibility reasoning

## Available Conceptual Building Blocks

### From V2V-GoT

- task and benchmark framing
- QA chaining ideas
- cooperative perception context
- evaluation entry points

### From KLDrive-style code

- constrained reasoning loop
- typed tool interface
- safety-oriented execution design
- deterministic graph access pattern

## Recommended Reuse Strategy

- Reuse V2V-GoT for dataset understanding, benchmark structure, and evaluation entry points.
- Reuse `auto_drive_copy` as a design reference for tool-constrained reasoning, not as a drop-in solution.
- Build the cooperative KG, association, fusion, provenance, and visibility modules fresh in this repository.

## Known Gaps To Fill Ourselves

- canonical scene fact schema
- local graph construction
- structured V2V packet abstraction
- coordinate and time alignment
- object association
- fusion logic
- provenance modeling
- visibility modeling
- deterministic graph query engine
- benchmark adapter for cooperative KG reasoning
