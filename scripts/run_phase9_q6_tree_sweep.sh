#!/usr/bin/env bash
set -euo pipefail

V2VGOT_ROOT="${V2VGOT_ROOT:-/workspace/repos/V2V-GoT}"
RUN_NAME="${RUN_NAME:-phase9_q6_tree_sweep_v1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase9_train_dev/${RUN_NAME}}"
VAL_OUT_ROOT="${VAL_OUT_ROOT:-outputs/phase8_val_report}"
WORKERS="${WORKERS:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"

mkdir -p "${OUT_ROOT}"

declare -a GRID=(
  "d6_l32_g0.0005"
  "d6_l64_g0.0005"
  "d6_l96_g0.0005"
  "d8_l32_g0.0005"
  "d8_l64_g0.0005"
  "d8_l96_g0.0005"
  "d10_l32_g0.0005"
  "d10_l64_g0.0005"
  "d10_l96_g0.0005"
  "d8_l64_g0.0002"
  "d8_l64_g0.001"
  "d10_l64_g0.0002"
  "d10_l64_g0.001"
  "d12_l64_g0.0005"
  "d12_l96_g0.0005"
)

echo "q6_sweep_start run=${RUN_NAME} count=${#GRID[@]}"

for cfg in "${GRID[@]}"; do
  depth="$(echo "${cfg}" | awk -F'_' '{print $1}' | sed 's/d//')"
  leaf="$(echo "${cfg}" | awk -F'_' '{print $2}' | sed 's/l//')"
  gain="$(echo "${cfg}" | awk -F'_' '{print $3}' | sed 's/g//')"
  model_json="${OUT_ROOT}/${cfg}_deployable.json"
  report_json="${OUT_ROOT}/${cfg}_train_report.json"
  echo "train ${cfg}"
  python3 scripts/train_q6_agent_motion_notability.py \
    --v2vgot-root "${V2VGOT_ROOT}" \
    --split train \
    --model-family regression_tree \
    --tree-max-depth "${depth}" \
    --tree-min-leaf "${leaf}" \
    --tree-min-gain "${gain}" \
    --decision-threshold 0.5 \
    --output-json "${model_json}" \
    --output-report "${report_json}" >/dev/null

  tag="val_q6_${RUN_NAME}_${cfg}"
  echo "eval ${cfg}"
  python3 scripts/evaluate_qa_router.py \
    --split val \
    --limit 0 \
    --task-type agent_motion_prediction \
    --baseline-mode cooperative \
    --workers "${WORKERS}" \
    --progress-every "${PROGRESS_EVERY}" \
    --agent-motion-model-json "${model_json}" \
    --output-jsonl "${VAL_OUT_ROOT}/${tag}.jsonl" >/dev/null

  python3 - <<PY
import json, pathlib
tag="${tag}"
p=pathlib.Path("${VAL_OUT_ROOT}/${tag}_manifest.json")
p.write_text(json.dumps({
  "split":"val",
  "scenario_name":tag,
  "runs":[{"task_type":"agent_motion_prediction","output_jsonl":"${VAL_OUT_ROOT}/${tag}.jsonl"}]
}, indent=2), encoding="utf-8")
PY

  python3 scripts/export_qa_predictions.py \
    --manifest "${VAL_OUT_ROOT}/${tag}_manifest.json" \
    --output-dir "${VAL_OUT_ROOT}/official_exports" \
    --split val \
    --scenario-name "${tag}" \
    --task-type agent_motion_prediction >/dev/null

  python3 scripts/run_v2vgot_official_qa_eval.py \
    --export-manifest "${VAL_OUT_ROOT}/official_exports/${tag}_official_export_manifest.json" \
    --output-dir "${VAL_OUT_ROOT}/official_eval_reports" \
    --tools-dir "${VAL_OUT_ROOT}/official_exports/tools" \
    --task-type agent_motion_prediction \
    --num-future-waypoints 1 \
    --npy-save-path "${V2VGOT_ROOT}/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy" \
    --v2vgot-root "${V2VGOT_ROOT}" >/dev/null

  python3 - <<PY
import json
p="${VAL_OUT_ROOT}/official_eval_reports/${tag}_official_export_manifest_official_qa_eval_summary.json"
d=json.load(open(p))
acc=float(d["runs"][0]["metrics"]["binary_classification_accuracy"])
print(f"result {acc:.6f} ${cfg}")
PY
done

python3 - <<PY
import glob,json,os
root="${VAL_OUT_ROOT}/official_eval_reports"
rows=[]
for p in glob.glob(f"{root}/val_q6_${RUN_NAME}_*_official_export_manifest_official_qa_eval_summary.json"):
    d=json.load(open(p))
    acc=float(d["runs"][0]["metrics"]["binary_classification_accuracy"])
    rows.append((acc,p))
rows.sort(reverse=True)
for acc,p in rows[:15]:
    print(f"{acc:.6f}\t{p}")
if rows:
    print("BEST",rows[0][0],rows[0][1])
PY

echo "q6_sweep_done run=${RUN_NAME}"

