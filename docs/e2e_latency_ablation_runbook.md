# E2E + Latency + Ablation Runbook (Current)

Date: 2026-05-26  
Scope: Current reproducible commands for:
- end-to-end train/validation,
- latency analysis,
- G-MARK ablations.

This runbook reflects the current Q9 policy:
- Q9 in E2E uses the clean Q8-context ElasticNet path.
- Legacy metadata-based Q9 behavior is deprecated.

---

## 1) E2E Train + Validation

### 1.1 E2E Train

```bash
python3 scripts/e2e/run_e2e_train_pipeline.py \
  --run-name phase9_e2e_cleanq9_v1 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --workers 32 \
  --progress-every 250
```

What this does:
- trains/uses selected Q1-Q9 pipeline components,
- trains Q3/Q4/Q5/Q6/Q7/Q8 heads as configured,
- builds clean Q9 model via Q8-context ElasticNet sweep,
- runs train-split official-style evaluations,
- writes run manifest.

Primary outputs:
- `outputs/e2e_runs/phase9_e2e_cleanq9_v1/e2e_model_manifest.json`
- `outputs/e2e_runs/phase9_e2e_cleanq9_v1/e2e_model_manifest.md`
- `outputs/e2e_runs/phase9_e2e_cleanq9_v1/models/`
- `outputs/e2e_runs/phase9_e2e_cleanq9_v1/train_eval/`

### 1.2 E2E Validation Report (from manifest)

```bash
python3 scripts/e2e/run_e2e_validation_report.py \
  --manifest-json outputs/e2e_runs/phase9_e2e_cleanq9_v1/e2e_model_manifest.json \
  --workers 32 \
  --progress-every 250
```

Optional (auto-pick latest manifest):

```bash
python3 scripts/e2e/run_e2e_validation_report.py \
  --workers 32 \
  --progress-every 250
```

Primary outputs:
- `outputs/e2e_runs/phase9_e2e_cleanq9_v1/val_eval/e2e_validation_summary.json`
- `outputs/e2e_runs/phase9_e2e_cleanq9_v1/val_eval/e2e_validation_summary.md`

---

## 2) Latency Analysis

### 2.1 Full Q1-Q9 Latency Run

```bash
python3 scripts/run_full_latency_analysis.py \
  --run-name latency_full_e2e_parallel_prefetch_20260519 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --workers 32 \
  --progress-every 250 \
  --temporal-execution-mode parallel_prefetch
```

What this does:
- runs task-wise latency collection for Q1-Q9,
- records per-sample timing breakdowns,
- writes combined latency JSONL + summarized tables.

Primary outputs (under `outputs/latency_analysis/<run-name>/`):
- `<run-name>_combined_latency.jsonl`
- `<run-name>_latency_summary.json`
- `<run-name>_latency_summary.md`
- per-task latency JSONLs

### 2.2 Summarize Any Latency JSONL

```bash
python3 scripts/report_latency_breakdown.py \
  --latency-jsonl outputs/latency_analysis/<run>/<scenario>_latency.jsonl \
  --output-markdown outputs/latency_analysis/<run>/<scenario>_summary.md
```

---

## 3) G-MARK Ablation Runs

Reference runbook:
- `docs/gmark_ablation_runbook.md`

### 3.1 Validation-only ablations (fastest ablation mode)

```bash
python3 scripts/e2e/run_gmark_ablation_report.py \
  --run-name gmark_ablation_validation_only_v1 \
  --manifest-json outputs/e2e_runs/phase9_e2e_cleanq9_v1/e2e_model_manifest.json \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --workers 32 \
  --progress-every 250
```

### 3.2 Train+Validation ablations (stronger ablation evidence)

```bash
python3 scripts/e2e/run_gmark_ablation_report.py \
  --run-name gmark_ablation_v1 \
  --manifest-json outputs/e2e_runs/phase9_e2e_cleanq9_v1/e2e_model_manifest.json \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --workers 32 \
  --progress-every 250 \
  --run-trained-validation
```

### 3.3 Example subset ablations

```bash
python3 scripts/e2e/run_gmark_ablation_report.py \
  --run-name gmark_ablation_subset_v1 \
  --manifest-json outputs/e2e_runs/phase9_e2e_cleanq9_v1/e2e_model_manifest.json \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --workers 32 \
  --progress-every 250 \
  --graph-ablation-mode no_provenance \
  --graph-ablation-mode no_candidate_retention \
  --graph-ablation-mode no_uncertainty_conflict \
  --graph-ablation-mode no_graph_relations \
  --graph-ablation-mode flat_non_graph_readout
```

Primary outputs:
- `outputs/gmark_ablations/<run-name>/ablation_summary.json`
- `outputs/gmark_ablations/<run-name>/ablation_summary.md`

---

## 4) Kubernetes Ablation Job (optional)

K8s spec:
- `k8s/gmark-ablation-job.yaml`

Typical commands:

```bash
kubectl delete job gmark-ablation-report gmark-ablation-report-v2 -n seelab --ignore-not-found
kubectl apply -f k8s/gmark-ablation-job.yaml
kubectl get pods -n seelab -l app=gmark-ablation-report-v2 -o wide
kubectl logs job/gmark-ablation-report-v2 -n seelab --tail=200
```

---

## 5) Suggested Execution Order

1. Run E2E train (`run_e2e_train_pipeline.py`).
2. Run E2E val report (`run_e2e_validation_report.py`).
3. Run full latency analysis.
4. Run validation-only ablations.
5. Run train+validation ablations (if needed for paper-grade evidence).

---

## 6) Source Files (for maintenance)

- E2E train: `scripts/e2e/run_e2e_train_pipeline.py`
- E2E val: `scripts/e2e/run_e2e_validation_report.py`
- Full latency: `scripts/run_full_latency_analysis.py`
- Latency breakdown: `scripts/report_latency_breakdown.py`
- Ablations: `scripts/e2e/run_gmark_ablation_report.py`
- Ablation train pipeline: `scripts/e2e/run_gmark_ablation_train_pipeline.py`
- Ablation docs: `docs/gmark_ablation_runbook.md`, `docs/gmark_ablation_interpretation.md`
