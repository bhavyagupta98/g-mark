# V2V-GoT Table-I Reproduction Runbook

This note captures how to reproduce the V2V-GoT planning-table rows that require
released or trained LLaVA LoRA checkpoints. It is intentionally separate from the
normal `kg_coop_drive` e2e flow so existing Q1-Q9 training and validation flows
remain unaffected.

## Current Asset State

The released `model_ckpt.zip` provides at least:

- `v2vllmq5_10ep_both_shallow_f2/checkpoint-490`
- `v2vgot_10ep_both_shallow_f2/checkpoint-4330`

The following V2V-LLM baseline variant checkpoint directories may exist after
unzip, but the actual `checkpoint-490` subfolders can be missing:

- `v2vllmq5_10ep_both_shallow_f2_ego_only/checkpoint-490`
- `v2vllmq5_10ep_both_shallow_f2_early/checkpoint-490`
- `v2vllmq5_10ep_both_shallow_f2_attfuse/checkpoint-490`
- `v2vllmq5_10ep_both_shallow_f2_v2xvit/checkpoint-490`
- `v2vllmq5_10ep_both_shallow_f2_cobevt/checkpoint-490`

These are separate LoRA checkpoints in the upstream script design. Do not symlink
the base V2V-LLM checkpoint into these directories if the goal is a defensible
reproduction.

## Why These Checkpoints Matter

V2V-GoT Table I compares planning performance for:

- no fusion;
- early fusion;
- intermediate-fusion backbones such as AttFuse, V2X-ViT, and CoBEVT;
- V2V-LLM;
- V2V-GoT.

The perception-fusion rows should be interpreted carefully. They are not raw
perception backbones directly producing planning trajectories. In the released
V2V-GoT scripts, each row corresponds to a perception or feature source feeding
a downstream V2V-LLM-style planning model, and the final generated planning
answer is scored by the common evaluator. For example, the CoBEVT row is best
read as:

```text
CoBEVT-derived perception features + V2V-LLM-style planning head
```

not as:

```text
CoBEVT alone performs the reasoning/planning task
```

In the released scripts, the five baseline variants are not just a metric flag.
They point to different model names and different feature sources. For example:

- `ego_only` uses `--ego_only True`;
- `early` uses `--feature_source early`;
- `attfuse` uses `--feature_source attfuse`;
- `v2xvit` uses `--feature_source v2xvit`;
- `cobevt` uses `--feature_source cobevt`.

Therefore, rerunning those rows from scratch requires the matching LoRA
checkpoint for each variant, or retraining those LoRA adapters.

## Audit Commands

From `kg_coop_drive`:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/build_v2vgot_table1_reproduction.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --audit-only
```

To inspect which checkpoints exist:

```bash
find /workspace/repos/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora \
  -type d \
  -name "checkpoint-490" \
  | sort
```

To inspect whether variant directories exist:

```bash
find /workspace/repos/V2V-GoT -type d \
  \( -name "*ego_only*" -o -name "*early*" -o -name "*attfuse*" -o -name "*v2xvit*" -o -name "*cobevt*" \) \
  | sort
```

## Download Released Assets

If `model_ckpt.zip` is not present:

```bash
cd /workspace/repos/V2V-GoT

python3 -m pip install -U huggingface_hub hf_transfer hf_xet
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download eddyhkchiu/V2V-GoT-QA model_ckpt.zip \
  --repo-type dataset \
  --local-dir /workspace/repos/V2V-GoT
```

Then unzip without overwriting existing files:

```bash
cd /workspace/repos/V2V-GoT
unzip -n model_ckpt.zip -d /workspace/repos/V2V-GoT
```

## Train Missing Variant Checkpoints

### Kubernetes Job

For unattended training on the PVC-mounted VM/cluster, use the dedicated job:

```bash
kubectl delete job v2vgot-baseline-training -n seelab --ignore-not-found
kubectl apply -f k8s/v2vgot-baseline-training-job.yaml
kubectl logs -f job/v2vgot-baseline-training -n seelab
```

The job trains the variants listed in `TRAIN_VARIANTS` sequentially and stores
checkpoints under:

```text
/workspace/repos/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/
```

Default variants:

```text
ego_only early attfuse v2xvit cobevt
```

The checked-in job targets A100 and uses the upstream training batch defaults:

```text
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
```

For A10-style GPUs, change these to `4` and `8` respectively. That preserves the
effective batch size of the upstream script (`4 * 8 = 32`) while reducing
per-step GPU memory. The job also disables W&B logging and applies the dependency
fixes for `peft`, `accelerate`, and `hf_transfer`.

To train only one variant, edit the job env before applying:

```yaml
env:
  - name: TRAIN_VARIANTS
    value: "ego_only"
```

### Manual Commands

The upstream training scripts contain `cd LLaVA`, so launch them from the
V2V-GoT repository root, not from inside `LLaVA`.

```bash
cd /workspace/repos/V2V-GoT
```

If the `llava` conda environment is not already configured:

```bash
source /opt/conda/etc/profile.d/conda.sh

conda create -n llava python=3.10 -y
conda activate llava

cd /workspace/repos/V2V-GoT/LLaVA
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

The upstream `pyproject.toml` pins `accelerate==0.21.0` but leaves `peft`
unpinned. Newer `peft` releases expect newer Accelerate APIs such as
`clear_device_cache`, causing:

```text
ImportError: cannot import name 'clear_device_cache' from 'accelerate.utils.memory'
```

Fix the environment by pinning PEFT to the older LLaVA-compatible release:

```bash
conda activate llava
cd /workspace/repos/V2V-GoT/LLaVA

python -m pip install --force-reinstall --no-deps \
  accelerate==0.21.0 \
  peft==0.4.0

python - <<'PY'
import accelerate, peft, transformers
print("accelerate", accelerate.__version__)
print("peft", peft.__version__)
print("transformers", transformers.__version__)
from transformers import Trainer
print("Trainer import OK")
PY
```

If `HF_HUB_ENABLE_HF_TRANSFER=1` is set in the shell, install `hf_transfer`
inside the `llava` environment too, or disable the variable for training:

```bash
conda activate llava
python -m pip install hf_transfer
```

or:

```bash
unset HF_HUB_ENABLE_HF_TRANSFER
```

On A10, skip `flash-attn` unless we know it builds cleanly in the image. The
training scripts do not explicitly require calling `flash-attn` from the shell.

Return to the V2V-GoT root before launching the upstream scripts:

```bash
cd /workspace/repos/V2V-GoT
```

Train the missing no-fusion/ego-only variant:

```bash
source scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_ego_only.sh
```

Train the early-fusion variant:

```bash
source scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_early.sh
```

Train the AttFuse variant:

```bash
source scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_attfuse.sh
```

Train the V2X-ViT variant:

```bash
source scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_v2xvit.sh
```

Train the CoBEVT variant:

```bash
source scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_cobevt.sh
```

After training, verify:

```bash
find /workspace/repos/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora \
  -type d \
  -name "checkpoint-490" \
  | sort
```

Then rerun the asset audit:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/build_v2vgot_table1_reproduction.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --audit-only
```

## Runtime Estimate

These are rough estimates for one missing variant:

- A100 80GB: about 1-3 hours;
- RTX A6000: about 2-5 hours;
- A10: about 5-10+ hours and may need lower batch settings.

For all five variants, sequential training can plausibly take:

- A100 80GB: about 5-15 hours;
- RTX A6000: about 10-25 hours;
- A10: about 25-50+ hours.

Before training all five, run one variant first and confirm that it creates the
expected `checkpoint-490` output.

## Reproduce Available Rows

If only the released base checkpoints are available, reproduce the released
V2V-LLM and V2V-GoT rows:

### Current Smoke Command

Use this before deleting the current GPU pod or before launching the full
baseline-training job:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/build_v2vgot_table1_reproduction.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --run-v2v-baselines \
  --run-v2vgot \
  --only-method V2V-LLM \
  --only-method V2V-GoT \
  --skip-missing \
  --output-dir outputs/v2vgot_table1_reproduction
```

The command above is the recommended smoke simulation before training the five
missing variant checkpoints. It uses the currently available released
`V2V-LLM` and `V2V-GoT` checkpoints, skips missing variant checkpoints, parses
the generated upstream evaluator logs, and emits:

```text
outputs/v2vgot_table1_reproduction/table1_reproduction_summary.json
outputs/v2vgot_table1_reproduction/table1_reproduction_report.md
```

To run only V2V-GoT:

```bash
python3 scripts/build_v2vgot_table1_reproduction.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --run-v2vgot \
  --output-dir outputs/v2vgot_table1_reproduction
```

To run only the released base V2V-LLM row:

```bash
python3 scripts/build_v2vgot_table1_reproduction.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --run-v2v-baselines \
  --only-method V2V-LLM \
  --skip-missing \
  --output-dir outputs/v2vgot_table1_reproduction
```

If the variant checkpoints are still missing, use `--skip-missing` and
`--only-method` as shown above, or train the missing checkpoints first.

## Defensible Reporting Shortcut

If time is limited, the clean reporting path is:

1. Reproduce the released V2V-LLM and V2V-GoT rows using available checkpoints.
2. Use the five perception-fusion baseline rows as reported by V2V-GoT Table I,
   explicitly marked as reported/cited numbers.
3. Add the G-MARK row from the official evaluator output.

This is defensible because the unreleased/missing variant checkpoints are not
silently replaced with another checkpoint, and reproduced versus reported rows
remain clearly separated.

When adding G-MARK, label it by the evidence it actually uses. The current
paper-facing row should be written as `G-MARK (structured V2V-GoT-QA evidence)`
or equivalent. Do not label it as `G-MARK + CoBEVT` unless we implement a real
adapter from CoBEVT outputs into the G-MARK graph.

## Add A G-MARK Row On The Released Table-I/Q9 Split

The normal G-MARK validation flow uses
`v2v4real_3d_grounding_qa_dataset_v2vgot.json`. For a Table-I-style planning
row, run G-MARK on the released Q9 planning file used by the upstream
V2V-GoT/V2V-LLM scripts:

```text
DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json
```

Use the dedicated wrapper:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/run_gmark_table1_q9_eval.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --manifest-json /workspace/repos/kg_coop_drive/outputs/e2e_runs/r5/e2e_model_manifest.json \
  --run-name gmark_table1_q9_full \
  --method-name "G-MARK" \
  --workers 32 \
  --progress-every 250
```

The script runs the future-trajectory pipeline with:

```text
--file-name v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json
--task-type future_trajectory
--future-trajectory-model-json <q9_model_json from manifest>
```

It writes:

```text
outputs/v2vgot_table1_reproduction/gmark_q9/gmark_table1_q9_full_table1_row.json
outputs/v2vgot_table1_reproduction/gmark_q9/gmark_table1_q9_full_table1_row.md
```

These outputs contain the report-ready L2 1s/2s/3s/average and collision-rate
1s/2s/3s/average metrics parsed from the official evaluator summary. They also
include measured G-MARK communication cost as average compact serialized KG
payload size per released Q9 sample:

```text
Comm(MB) = mean_bytes(compact_json(LocalGraphSerializer(prepared_gmark_scene))) / 1e6
```

This is intentionally separate from the V2V-GoT paper's feature-tensor
communication accounting. It is a G-MARK-specific structured-message budget, so
caption/report text should state that communication is measured as serialized
KG payload size unless we later add a feature-tensor-equivalent estimator.
