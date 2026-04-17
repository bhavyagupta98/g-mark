# Phase 1 Dataset Note

## Goal

Phase 1 is about understanding what data is locally available, what the first prototype can realistically use, and what assumptions must wait until the full processed V2V-GoT assets are present.

## What Was Inspected

Primary local source:

- `/Users/bhavya/Desktop/ms_projects/V2V-GoT`

Files directly inspected:

- [LLaVA/playground/data/V2V4Real/data.json](/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/playground/data/V2V4Real/data.json:1)
- [LLaVA/playground/data/V2V4Real/v2v4real_dataset_for_llava_train.json](/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/playground/data/V2V4Real/v2v4real_dataset_for_llava_train.json:1)
- [LLaVA/playground/data/V2V4Real/v2v4real_dataset_for_llava_val.json](/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/playground/data/V2V4Real/v2v4real_dataset_for_llava_val.json:1)

## Key Findings

### 1. Split metadata is available locally

`data.json` provides:

- train and val splits
- sequence IDs for each split
- cumulative frame counts via `len_record`
- expected `llm_data_path` locations for processed cooperative QA data

This is enough to define split-aware adapters and dataset metadata models now.

### 2. LLaVA-style cooperative detection samples are available locally

The train and val JSON files contain list-based sample records with fields such as:

- `id`
- `conversations`
- `scenario_index`
- `local_timestamp_index`
- `global_timestamp_index`

These records contain prompt-style cooperative perception text. A representative prompt includes:

- individual detection outputs for `Agent ego`
- individual detection outputs for `Agent 1`
- a target cooperative detection result

This means the locally available repo state is enough to:

- inspect benchmark-facing sample format
- study how multi-agent information is represented textually
- design dataset adapters and parsers

### 3. The expected processed `official_models/.../npy/co_llm` assets are not currently present locally

The metadata points to paths such as:

- `../DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm`
- `../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm`

Those paths are expected by the benchmark code, but the corresponding local directories are not present in the current checkout.

So at this moment, we do not yet have the full processed graph/QA-ready dataset artifacts that the original V2V-GoT pipeline expects.

## Implications for Our Implementation

### What we can do immediately

- build modular dataset metadata models
- build a V2V-GoT dataset inspector
- build an adapter interface for benchmark metadata and sample parsing
- define the canonical schema independently of the missing processed assets
- plan the first narrow QA subset

### What we should postpone until processed assets are available

- loading the original `co_llm` QA dataset files directly
- running the full V2V-GoT evaluation pipeline as-is
- building final graph adapters against unknown final sample payloads

## Recommended Phase 1 Scope Decision

The first prototype should be based on:

- split metadata from `data.json`
- prompt/sample inspection from the local LLaVA JSON files
- a dataset adapter abstraction that can later support:
  - metadata-only mode
  - processed-asset mode

This lets us keep moving without hardcoding assumptions that depend on files not yet downloaded.

## Recommended First Supported Task Slice

For the graph system, start with tasks that should map cleanly to structured querying:

- object existence
- object count
- basic relative position
- simple visibility-aware reasoning later once object-level scene facts are available

Do not start Phase 2 assuming:

- long-horizon planning
- rich intent prediction
- full graph-of-thought chaining

## What Phase 1 Should Produce

Before Phase 1 is complete, we should have:

- a modular dataset inspection layer
- a written record of what local assets exist
- a clear note distinguishing benchmark metadata from full processed assets
- a narrowed first task slice for the KG prototype

## Exit Check for Phase 1

We should be comfortable moving to Phase 2 when:

- we can describe the currently available data accurately
- we know what is missing
- our adapters are designed to tolerate both current and future data states
- the first prototype task remains narrow and realistic
