# V2V-GoT Table I Reproduction Report

## Asset Audit

| Status | Asset | Path |
| --- | --- | --- |
| OK | `V2V-GoT root` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT` |
| OK | `V2V-LLM Q5 QA JSON` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vllmq5.json` |
| OK | `V2V-GoT Q9 QA JSON` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json` |
| OK | `Q9 collision GT root` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy` |
| OK | `LLaVA model_vqa_loader` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/llava/eval/model_vqa_loader.py` |
| OK | `V2V-LLM eval script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh` |
| OK | `V2V-GoT Q9 eval script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_nq9.sh` |
| OK | `No Fusion inference script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_ego_only.sh` |
| MISSING | `No Fusion checkpoint` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_ego_only/checkpoint-490` |
| OK | `No Fusion feature source `no_fusion_keep_all`` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm` |
| OK | `Early Fusion inference script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_early.sh` |
| MISSING | `Early Fusion checkpoint` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_early/checkpoint-490` |
| MISSING | `Early Fusion feature source `early`` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/early/npy/co_llm` |
| OK | `AttFuse inference script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_attfuse.sh` |
| MISSING | `AttFuse checkpoint` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_attfuse/checkpoint-490` |
| MISSING | `AttFuse feature source `attfuse`` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/attfuse/npy/co_llm` |
| OK | `V2X-ViT inference script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_v2xvit.sh` |
| MISSING | `V2X-ViT checkpoint` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_v2xvit/checkpoint-490` |
| MISSING | `V2X-ViT feature source `v2xvit`` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/v2xvit/npy/co_llm` |
| OK | `CoBEVT inference script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_cobevt.sh` |
| MISSING | `CoBEVT checkpoint` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_cobevt/checkpoint-490` |
| MISSING | `CoBEVT feature source `cobevt`` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/cobevt/npy/co_llm` |
| OK | `V2V-LLM inference script` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2.sh` |
| MISSING | `V2V-LLM checkpoint` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2/checkpoint-490` |
| OK | `V2V-LLM feature source `no_fusion_keep_all`` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm` |
| MISSING | `V2V-GoT checkpoint` | `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2/checkpoint-4330` |

## Planning Table

| Method | Family | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | L2 Avg ↓ | CR 1s ↓ | CR 2s ↓ | CR 3s ↓ | CR Avg ↓ | Comm MB ↓ | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `No Fusion` | `reported_v2vgot_table_i` | 3.47 | 5.79 | 8.26 | 5.84 | 1.48 | 4.24 | 7.72 | 4.48 | 0.0000 | `V2V-GoT Table I` |
| `Early Fusion` | `reported_v2vgot_table_i` | 3.48 | 5.61 | 7.82 | 5.63 | 1.16 | 3.51 | 5.66 | 3.44 | 1.9208 | `V2V-GoT Table I` |
| `AttFuse` | `reported_v2vgot_table_i` | 3.65 | 6.21 | 8.75 | 6.20 | 1.19 | 4.41 | 6.38 | 3.99 | 0.4008 | `V2V-GoT Table I` |
| `V2X-ViT` | `reported_v2vgot_table_i` | 3.46 | 5.80 | 8.19 | 5.81 | 1.45 | 4.24 | 6.59 | 4.09 | 0.4008 | `V2V-GoT Table I` |
| `CoBEVT` | `reported_v2vgot_table_i` | 3.38 | 5.42 | 7.46 | 5.42 | 1.31 | 4.41 | 5.75 | 3.82 | 0.4008 | `V2V-GoT Table I` |
| `V2V-LLM` | `reported_v2vgot_table_i` | 2.90 | 4.91 | 6.98 | 4.93 | 0.75 | 2.87 | 4.93 | 2.85 | 0.4068 | `V2V-GoT Table I` |
| `V2V-GoT` | `reported_v2vgot_table_i` | 1.65 | 2.63 | 3.59 | 2.62 | 0.12 | 1.92 | 3.45 | 1.83 | 0.4068 | `V2V-GoT Table I` |
| `G-MARK full` | `gmark_local` | 0.95 | 1.35 | 1.33 | 1.21 | 0.00 | 0.00 | 0.00 | 0.00 | - | `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/phase8_val_report/official_eval_reports/future_trajectory_qa_type_19_official_eval.log` |

## Notes

- `reported_v2vgot_table_i` rows are literature values from V2V-GoT Table I.
- `reproduced_*` rows are parsed from local V2V-GoT evaluator logs when available.
- `gmark_local` rows are parsed from local G-MARK Q9 official-evaluator logs.
- Running V2V-LLM/V2V-GoT inference requires GPU, the released LoRA checkpoints, processed feature folders, and the `llava` conda environment.
- Communication values for reproduced baselines use V2V-GoT Table I accounting unless a separate communication estimator is added.
