# Bootstrap Artifacts

## What the Data Is in Bootstrap

### `dataset_jsons.zip`

This is the benchmark JSON archive.

It provides:

- `co_llm` QA dataset JSONs under `DMSTrack/V2V4Real/official_models/.../npy/co_llm`
- train and val benchmark files
- `v2vgot`, `v2vllmq5`, and `nq*` task JSONs

These are the main files we use to understand the benchmark-facing sample structure in Phase 1.

### `dataset_processed_features_and_gt.zip`

This is the processed perception and ground-truth archive.

It is expected to provide:

- processed features used by the original V2V-GoT pipeline
- ground-truth data
- additional supporting assets for perception and evaluation

This archive matters more for deeper integration and full pipeline reproduction than for the first JSON-level Phase 1 inspection.

## What the Model Is in Bootstrap

### `model_ckpt.zip`

This is the model checkpoint archive.

It provides pretrained checkpoints under the V2V-GoT LLaVA checkpoint tree, including task-specific checkpoints for:

- V2V-GoT
- V2V-LLM variants

These checkpoints are useful when you want to:

- reproduce the original paper's inference flow
- run pretrained benchmark baselines
- compare our KG system against the original learned systems

They are not required for the earliest schema and dataset-understanding steps, but they are useful for later evaluation and baseline reproduction.

## Practical Meaning for Phase 1

For Phase 1, the most important bootstrap artifact is:

- `dataset_jsons.zip`

For later benchmarking and reproduction, the more important additions are:

- `dataset_processed_features_and_gt.zip`
- `model_ckpt.zip`
