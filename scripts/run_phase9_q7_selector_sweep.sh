#!/usr/bin/env bash
set -euo pipefail

V2VGOT_ROOT="${V2VGOT_ROOT:-/workspace/repos/V2V-GoT}"
RUN_NAME="${RUN_NAME:-phase9_q7_selector_sweep_v1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase9_train_dev/${RUN_NAME}}"
VAL_OUT_ROOT="${VAL_OUT_ROOT:-outputs/phase8_val_report}"
WORKERS="${WORKERS:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"

mkdir -p "${OUT_ROOT}"

declare -a GRID=(
  "k1_d4_occ1"
  "k1_d6_occ1"
  "k1_d8_occ1"
  "k2_d4_occ1"
  "k2_d6_occ1"
  "k2_d8_occ1"
  "k2_d8_occ0"
  "k3_d6_occ1"
  "k3_d8_occ1"
)

echo "q7_selector_sweep_start run=${RUN_NAME} count=${#GRID[@]}"

for cfg in "${GRID[@]}"; do
  k="$(echo "${cfg}" | awk -F'_' '{print $1}' | sed 's/k//')"
  distance="$(echo "${cfg}" | awk -F'_' '{print $2}' | sed 's/d//')"
  occ="$(echo "${cfg}" | awk -F'_' '{print $3}' | sed 's/occ//')"
  occ_arg="--selection-include-occluded-uncertain"
  if [[ "${occ}" == "0" ]]; then
    occ_arg="--no-selection-include-occluded-uncertain"
  fi

  model_json="${OUT_ROOT}/${cfg}.json"
  report_json="${OUT_ROOT}/${cfg}_train_report.json"
  tag="val_q7_${RUN_NAME}_${cfg}"

  echo "train ${cfg}"
  python3 scripts/train_q7_object_motion_predictor.py \
    --v2vgot-root "${V2VGOT_ROOT}" \
    --split train \
    --baseline-mode cooperative \
    --model-family regression_tree \
    --tree-max-depth 9 \
    --tree-min-leaf 64 \
    --tree-min-gain 0.01 \
    --selection-max-objects "${k}" \
    --selection-max-distance-to-trajectory "${distance}" \
    "${occ_arg}" \
    --output-json "${model_json}" \
    --output-report "${report_json}" >/dev/null

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
precision=float(m.get("binary_precision", 0.0))
f1=float(m.get("localization", {}).get("0.5", {}).get("f1", 0.0))
print(f"result l2={l2:.6f} precision={precision:.6f} loc_f1_0.5={f1:.6f} ${cfg}")
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

echo "q7_selector_sweep_done run=${RUN_NAME}"
