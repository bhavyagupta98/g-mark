# Clean G-MARK Table-I/Q9 Row `latency_full_e2e_parallel_prefetch_20260519_q9_clean_model`

- method: `G-MARK clean Q9`
- train split file: `v2v4real_3d_grounding_qa_dataset_v2vgot.json`
- released validation Q9 file: `v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json`
- train rows: `12290`
- usable train rows: `11925`
- filtered train/validation overlap rows: `365`
- skipped train rows (temporary window): `0`
- model: `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/latency_analysis/latency_full_e2e_parallel_prefetch_20260519/q9_clean_model/latency_full_e2e_parallel_prefetch_20260519_q9_clean_model_clean_q9_model.json`
- official summary: `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/latency_analysis/latency_full_e2e_parallel_prefetch_20260519/q9_clean_model/official_eval_reports/latency_full_e2e_parallel_prefetch_20260519_q9_clean_model_official_export_manifest_official_qa_eval_summary.json`

Leakage policy: this row excludes `dist`, `angle`, `suggested_speed_idx`, `suggested_steering_idx`, `future_trajectory_str_in_ego`, and `future_trajectory_str_in_self` from the model inputs. It uses current position plus non-target scene/KG aggregates only.

| Method | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | L2 Avg ↓ | CR 1s ↓ | CR 2s ↓ | CR 3s ↓ | CR Avg ↓ | Comm MB ↓ | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `G-MARK clean Q9` | - | - | - | - | - | - | - | - | - | `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/latency_analysis/latency_full_e2e_parallel_prefetch_20260519/q9_clean_model/official_eval_reports/latency_full_e2e_parallel_prefetch_20260519_q9_clean_model_official_export_manifest_official_qa_eval_summary.json` |
