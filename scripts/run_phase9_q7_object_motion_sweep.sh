#!/usr/bin/env bash
set -euo pipefail

V2VGOT_ROOT="${V2VGOT_ROOT:-/workspace/repos/V2V-GoT}"
RUN_NAME="${RUN_NAME:-phase9_q7_object_motion_sweep_v1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase9_train_dev/${RUN_NAME}}"
VAL_OUT_ROOT="${VAL_OUT_ROOT:-outputs/phase8_val_report}"
WORKERS="${WORKERS:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"

mkdir -p "${OUT_ROOT}"

declare -a GRID=(
  "tree_d6_l48_g0.01"
  "tree_d7_l48_g0.01"
  "tree_d8_l64_g0.01"
  "tree_d9_l64_g0.01"
  "tree_d9_l96_g0.01"
  "gbdt_n160_lr0.04_d2_l48_s0.7"
  "gbdt_n220_lr0.04_d2_l64_s0.7"
  "gbdt_n280_lr0.035_d2_l64_s0.7"
  "gbdt_n240_lr0.04_d3_l96_s0.7"
)

echo "q7_object_motion_sweep_start run=${RUN_NAME} count=${#GRID[@]}"

for cfg in "${GRID[@]}"; do
  model_json="${OUT_ROOT}/${cfg}.json"
  report_json="${OUT_ROOT}/${cfg}_train_report.json"
  echo "train ${cfg}"

  if [[ "${cfg}" == tree_* ]]; then
    depth="$(echo "${cfg}" | awk -F'_' '{print $2}' | sed 's/d//')"
    leaf="$(echo "${cfg}" | awk -F'_' '{print $3}' | sed 's/l//')"
    gain="$(echo "${cfg}" | awk -F'_' '{print $4}' | sed 's/g//')"
    python3 scripts/train_q7_object_motion_predictor.py \
      --v2vgot-root "${V2VGOT_ROOT}" \
      --split train \
      --baseline-mode cooperative \
      --model-family regression_tree \
      --tree-max-depth "${depth}" \
      --tree-min-leaf "${leaf}" \
      --tree-min-gain "${gain}" \
      --max-match-distance 2.0 \
      --max-abs-delta 120.0 \
      --output-json "${model_json}" \
      --output-report "${report_json}" >/dev/null
  else
    n="$(echo "${cfg}" | awk -F'_' '{print $2}' | sed 's/n//')"
    lr="$(echo "${cfg}" | awk -F'_' '{print $3}' | sed 's/lr//')"
    depth="$(echo "${cfg}" | awk -F'_' '{print $4}' | sed 's/d//')"
    leaf="$(echo "${cfg}" | awk -F'_' '{print $5}' | sed 's/l//')"
    subsample="$(echo "${cfg}" | awk -F'_' '{print $6}' | sed 's/s//')"
    python3 scripts/train_q7_object_motion_predictor.py \
      --v2vgot-root "${V2VGOT_ROOT}" \
      --split train \
      --baseline-mode cooperative \
      --model-family gradient_boosting \
      --gbdt-n-estimators "${n}" \
      --gbdt-learning-rate "${lr}" \
      --gbdt-max-depth "${depth}" \
      --gbdt-min-samples-leaf "${leaf}" \
      --gbdt-subsample "${subsample}" \
      --max-match-distance 2.0 \
      --max-abs-delta 120.0 \
      --output-json "${model_json}" \
      --output-report "${report_json}" >/dev/null
  fi

  tag="val_q7_${RUN_NAME}_${cfg}"
  python3 scripts/run_qa_split_pipeline.py \
    --purpose val_report \
    --split val \
    --task-type object_motion_prediction \
    --qa-type-id 17 \
    --scenario-name "${tag}" \
    --baseline-mode cooperative \
    --v2vgot-root "${V2VGOT_ROOT}" \
    --object-motion-model-json "${model_json}" \
    --workers "${WORKERS}" \
    --progress-every "${PROGRESS_EVERY}" >/dev/null

  python3 - <<PY
import json
p="${VAL_OUT_ROOT}/official_eval_reports/${tag}_official_export_manifest_official_qa_eval_summary.json"
d=json.load(open(p))
m=d["runs"][0]["metrics"]
l2=float(m.get("l2_error_avg_123_all", m.get("l2_error_avg_all")))
print(f"result {l2:.6f} ${cfg}")
PY
done

python3 - <<PY
import glob,json
root="${VAL_OUT_ROOT}/official_eval_reports"
rows=[]
for p in glob.glob(f"{root}/val_q7_${RUN_NAME}_*_official_export_manifest_official_qa_eval_summary.json"):
    d=json.load(open(p))
    m=d["runs"][0]["metrics"]
    l2=float(m.get("l2_error_avg_123_all", m.get("l2_error_avg_all")))
    rows.append((l2,p))
rows.sort()
for l2,p in rows:
    print(f"{l2:.6f}\t{p}")
if rows:
    print("BEST",rows[0][0],rows[0][1])
PY

echo "q7_object_motion_sweep_done run=${RUN_NAME}"
