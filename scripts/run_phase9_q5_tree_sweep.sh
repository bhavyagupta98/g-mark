#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

V2VGOT_ROOT="${V2VGOT_ROOT:-/workspace/repos/V2V-GoT}"
RUN_NAME="${RUN_NAME:-phase9_q5_tree_sweep_v1}"
OUT_DIR="${OUT_DIR:-outputs/phase9_train_dev/${RUN_NAME}}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
RUN_VAL_FINAL="${RUN_VAL_FINAL:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
AUTO_PARALLEL="${AUTO_PARALLEL:-0}"
AUTO_PARALLEL_SHARDS="${AUTO_PARALLEL_SHARDS:-2}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-0}"
SWEEP_PROFILE="${SWEEP_PROFILE:-narrow_q5_v2}"
export RUN_NAME OUT_DIR LIMIT WORKERS PROGRESS_EVERY RUN_VAL_FINAL
export SHARD_COUNT SHARD_INDEX AUTO_PARALLEL AUTO_PARALLEL_SHARDS MAX_EXPERIMENTS SWEEP_PROFILE

mkdir -p "${OUT_DIR}"

if [[ "${AUTO_PARALLEL}" == "1" ]]; then
  if [[ "${AUTO_PARALLEL_SHARDS}" -lt 2 ]]; then
    echo "invalid_auto_parallel_shards ${AUTO_PARALLEL_SHARDS}"
    exit 1
  fi
  echo "auto_parallel_start run=${RUN_NAME} shards=${AUTO_PARALLEL_SHARDS}"
  pids=()
  for idx in $(seq 0 $((AUTO_PARALLEL_SHARDS - 1))); do
    child_run="${RUN_NAME}_s${idx}"
    child_out="outputs/phase9_train_dev/${child_run}"
    echo "auto_parallel_spawn shard=${idx}/${AUTO_PARALLEL_SHARDS} run=${child_run}"
    env \
      V2VGOT_ROOT="${V2VGOT_ROOT}" \
      RUN_NAME="${child_run}" \
      OUT_DIR="${child_out}" \
      LIMIT="${LIMIT}" \
      WORKERS="${WORKERS}" \
      PROGRESS_EVERY="${PROGRESS_EVERY}" \
      RUN_VAL_FINAL=0 \
      SHARD_COUNT="${AUTO_PARALLEL_SHARDS}" \
      SHARD_INDEX="${idx}" \
      AUTO_PARALLEL=0 \
      bash "$0" &
    pids+=($!)
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "auto_parallel_failed"
    exit 1
  fi

  COMBINED_DIR="outputs/phase9_train_dev/${RUN_NAME}_combined"
  mkdir -p "${COMBINED_DIR}"
  COMBINED_BEST="${COMBINED_DIR}/${RUN_NAME}_best_train_candidate_combined.json"
  COMBINED_TSV="${COMBINED_DIR}/${RUN_NAME}_best_train_candidates.tsv"

  python3 - <<'PY'
import json
import os
from pathlib import Path

run_name = os.environ.get("RUN_NAME", "phase9_q5_tree_sweep_v1")
shards = int(os.environ.get("AUTO_PARALLEL_SHARDS", "2"))
repo_root = Path.cwd()
combined_dir = repo_root / "outputs" / "phase9_train_dev" / f"{run_name}_combined"
combined_best = combined_dir / f"{run_name}_best_train_candidate_combined.json"
combined_tsv = combined_dir / f"{run_name}_best_train_candidates.tsv"

rows = []
for idx in range(shards):
    child_run = f"{run_name}_s{idx}"
    p = repo_root / "outputs" / "phase9_train_dev" / child_run / f"{child_run}_best_train_candidate.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    cid = str(d.get("candidate_id", ""))
    if not cid:
        continue
    rows.append(
        {
            "run_name": child_run,
            "candidate_id": cid,
            "model_path": str(repo_root / "outputs" / "phase9_train_dev" / child_run / f"{cid}_deployable.json"),
            "summary_json": str(d.get("summary_json", "")),
            "l2_error_avg_123_all": float(d.get("l2_error_avg_123_all", 1e18)),
            "l2_error_avg_03_all": float(d.get("l2_error_avg_03_all", 1e18)),
            "l2_error_avg_3s": float(d.get("l2_error_avg_3s", 1e18)),
            "action_accuracy": float(d.get("action_accuracy", 0.0)),
            "output_parse_error_rate": float(d.get("output_parse_error_rate", 1e18)),
            "gt_parse_error_rate": float(d.get("gt_parse_error_rate", 1e18)),
        }
    )

rows.sort(key=lambda r: (r["l2_error_avg_123_all"], r["output_parse_error_rate"], -r["action_accuracy"]))
combined_dir.mkdir(parents=True, exist_ok=True)
with combined_tsv.open("w", encoding="utf-8") as f:
    f.write("run_name\tcandidate_id\tl2_123\tl2_03\tl2_3s\taction_acc\tparse_out\tparse_gt\tmodel_path\n")
    for r in rows:
        f.write(
            f"{r['run_name']}\t{r['candidate_id']}\t{r['l2_error_avg_123_all']:.6f}\t"
            f"{r['l2_error_avg_03_all']:.6f}\t{r['l2_error_avg_3s']:.6f}\t"
            f"{r['action_accuracy']:.6f}\t{r['output_parse_error_rate']:.6f}\t"
            f"{r['gt_parse_error_rate']:.6f}\t{r['model_path']}\n"
        )

best = rows[0] if rows else {}
combined_best.write_text(json.dumps(best, indent=2), encoding="utf-8")
print(f"combined_candidates={len(rows)}")
if best:
    print(
        "combined_best "
        f"run={best['run_name']} id={best['candidate_id']} "
        f"l2_123={best['l2_error_avg_123_all']:.6f} "
        f"action_acc={best['action_accuracy']:.6f}"
    )
print(f"combined_tsv={combined_tsv}")
print(f"combined_best_json={combined_best}")
PY

  if [[ "${RUN_VAL_FINAL}" == "1" ]]; then
    BEST_MODEL="$(python3 - <<'PY'
import json
import os
from pathlib import Path
run_name = os.environ.get("RUN_NAME", "phase9_q5_tree_sweep_v1")
p = Path("outputs/phase9_train_dev") / f"{run_name}_combined" / f"{run_name}_best_train_candidate_combined.json"
d = json.loads(p.read_text(encoding="utf-8"))
print(str(d.get("model_path", "")))
PY
)"
    if [[ -n "${BEST_MODEL}" && -f "${BEST_MODEL}" ]]; then
      VAL_SCENARIO="val_q5_${RUN_NAME}_combined_final"
      echo "eval_val_final_combined scenario=${VAL_SCENARIO}"
      python3 scripts/run_qa_split_pipeline.py \
        --purpose val_report \
        --split val \
        --task-type object_motion_prediction \
        --scenario-name "${VAL_SCENARIO}" \
        --baseline-mode cooperative \
        --object-motion-model-json "${BEST_MODEL}" \
        --workers "${WORKERS}" \
        --progress-every "${PROGRESS_EVERY}" \
        --v2vgot-root "${V2VGOT_ROOT}" \
        >/dev/null
      VAL_SUMMARY="outputs/phase8_val_report/official_eval_reports/${VAL_SCENARIO}_official_export_manifest_official_qa_eval_summary.json"
      echo "val_final_done summary=${VAL_SUMMARY}"
    else
      echo "val_final_skip reason=no_combined_best_model"
    fi
  fi

  echo "auto_parallel_done run=${RUN_NAME}"
  exit 0
fi

echo "q5_sweep_start run=${RUN_NAME} profile=${SWEEP_PROFILE} limit=${LIMIT} workers=${WORKERS} val_final=${RUN_VAL_FINAL} shard=${SHARD_INDEX}/${SHARD_COUNT}"

if [[ "${SWEEP_PROFILE}" == "narrow_q5_v2" ]]; then
  # Narrow sweep around current best: d8_l64_g0.01_m2.0_a120
  DEPTHS=(7 8 9)
  LEAVES=(64 96)
  GAINS=(0.008 0.01 0.012)
  MATCH_DISTS=(2.0)
  MAX_DELTAS=(100 120 140)
else
  DEPTHS=(4 6 8)
  LEAVES=(32 64 128)
  GAINS=(0.005 0.01)
  MATCH_DISTS=(1.5 2.0 2.5)
  MAX_DELTAS=(80 120 160)
fi

if [[ "${SHARD_COUNT}" -lt 1 ]]; then
  echo "invalid_shard_count ${SHARD_COUNT}"
  exit 1
fi
if [[ "${SHARD_INDEX}" -lt 0 || "${SHARD_INDEX}" -ge "${SHARD_COUNT}" ]]; then
  echo "invalid_shard_index ${SHARD_INDEX} for shard_count ${SHARD_COUNT}"
  exit 1
fi

candidate_counter=0
processed_counter=0

# Optional curated list gate, used only when MAX_EXPERIMENTS>0.
if [[ "${SWEEP_PROFILE}" == "narrow_q5_v2" ]]; then
  PRIORITY_CANDIDATES=(
    "d8_l64_g0.01_m2.0_a120"
    "d8_l96_g0.01_m2.0_a120"
    "d7_l64_g0.01_m2.0_a120"
    "d9_l64_g0.01_m2.0_a120"
    "d8_l64_g0.008_m2.0_a120"
    "d8_l64_g0.012_m2.0_a120"
    "d8_l64_g0.01_m2.0_a100"
    "d8_l64_g0.01_m2.0_a140"
    "d9_l96_g0.01_m2.0_a120"
    "d7_l96_g0.01_m2.0_a120"
  )
else
  PRIORITY_CANDIDATES=(
    "d6_l64_g0.01_m2.0_a120"
    "d8_l64_g0.01_m2.0_a120"
    "d4_l64_g0.01_m2.0_a120"
    "d6_l32_g0.01_m2.0_a120"
    "d6_l128_g0.01_m2.0_a120"
    "d6_l64_g0.005_m2.0_a120"
    "d6_l64_g0.01_m1.5_a120"
    "d6_l64_g0.01_m2.5_a120"
    "d6_l64_g0.01_m2.0_a80"
    "d6_l64_g0.01_m2.0_a160"
  )
fi

if [[ "${MAX_EXPERIMENTS}" -gt 0 ]]; then
  echo "max_experiments=${MAX_EXPERIMENTS}"
fi

for depth in "${DEPTHS[@]}"; do
  for leaf in "${LEAVES[@]}"; do
    for gain in "${GAINS[@]}"; do
      for match_dist in "${MATCH_DISTS[@]}"; do
        for max_delta in "${MAX_DELTAS[@]}"; do
          if (( candidate_counter % SHARD_COUNT != SHARD_INDEX )); then
            candidate_counter=$((candidate_counter + 1))
            continue
          fi
          CANDIDATE_ID="d${depth}_l${leaf}_g${gain}_m${match_dist}_a${max_delta}"

          if [[ "${MAX_EXPERIMENTS}" -gt 0 ]]; then
            keep=0
            for pc in "${PRIORITY_CANDIDATES[@]}"; do
              if [[ "${CANDIDATE_ID}" == "${pc}" ]]; then
                keep=1
                break
              fi
            done
            if [[ "${keep}" -ne 1 ]]; then
              candidate_counter=$((candidate_counter + 1))
              continue
            fi
            if [[ "${processed_counter}" -ge "${MAX_EXPERIMENTS}" ]]; then
              candidate_counter=$((candidate_counter + 1))
              continue
            fi
          fi

          MODEL_PATH="${OUT_DIR}/${CANDIDATE_ID}_deployable.json"
          REPORT_PATH="${OUT_DIR}/${CANDIDATE_ID}_train_report.json"
          SCENARIO="train_q5_${RUN_NAME}_${CANDIDATE_ID}"

          echo "train_model ${CANDIDATE_ID}"
          python3 scripts/train_q5_object_motion_predictor.py \
            --v2vgot-root "${V2VGOT_ROOT}" \
            --split train \
            --baseline-mode cooperative \
            --model-family regression_tree \
            --tree-max-depth "${depth}" \
            --tree-min-leaf "${leaf}" \
            --tree-min-gain "${gain}" \
            --max-match-distance "${match_dist}" \
            --max-abs-delta "${max_delta}" \
            --limit "${LIMIT}" \
            --output-json "${MODEL_PATH}" \
            --output-report "${REPORT_PATH}" \
            >/dev/null

          echo "eval_train ${CANDIDATE_ID}"
          python3 scripts/run_qa_split_pipeline.py \
            --purpose train_dev \
            --split train \
            --task-type object_motion_prediction \
            --scenario-name "${SCENARIO}" \
            --baseline-mode cooperative \
            --object-motion-model-json "${MODEL_PATH}" \
            --workers "${WORKERS}" \
            --progress-every "${PROGRESS_EVERY}" \
            --limit "${LIMIT}" \
            --v2vgot-root "${V2VGOT_ROOT}" \
            >/dev/null

          candidate_counter=$((candidate_counter + 1))
          processed_counter=$((processed_counter + 1))
        done
      done
    done
  done
done

echo "shard_done processed=${processed_counter}"

SUMMARY_PATH="${OUT_DIR}/${RUN_NAME}_train_metric_summary.tsv"
BEST_PATH="${OUT_DIR}/${RUN_NAME}_best_train_candidate.json"

python3 - <<'PY'
import json
from pathlib import Path
import re

repo_root = Path.cwd()
run_name = __import__("os").environ.get("RUN_NAME", "phase9_q5_tree_sweep_v1")
out_dir = Path(__import__("os").environ.get("OUT_DIR", f"outputs/phase9_train_dev/{run_name}"))
summary_path = Path(__import__("os").environ.get("SUMMARY_PATH", str(out_dir / f"{run_name}_train_metric_summary.tsv")))
best_path = Path(__import__("os").environ.get("BEST_PATH", str(out_dir / f"{run_name}_best_train_candidate.json")))

pattern = re.compile(rf"^train_q5_{re.escape(run_name)}_(.+)_official_export_manifest_official_qa_eval_summary\.json$")
eval_dir = repo_root / "outputs" / "phase8_train_dev" / "official_eval_reports"

rows = []
for p in sorted(eval_dir.glob("*.json")):
    m = pattern.match(p.name)
    if not m:
        continue
    candidate_id = m.group(1)
    data = json.loads(p.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        continue
    run = runs[0]
    metrics = run.get("metrics", {})
    rows.append(
        {
            "candidate_id": candidate_id,
            "summary_json": str(p),
            "returncode": int(run.get("returncode", 1)),
            "l2_error_avg_123_all": float(metrics.get("l2_error_avg_123_all", 1e18)),
            "l2_error_avg_03_all": float(metrics.get("l2_error_avg_03_all", 1e18)),
            "l2_error_avg_3s": float(metrics.get("l2_error_avg_3s", 1e18)),
            "action_accuracy": float(metrics.get("action_accuracy", 0.0)),
            "output_parse_error_rate": float(metrics.get("output_parse_error_rate", 1e18)),
            "gt_parse_error_rate": float(metrics.get("gt_parse_error_rate", 1e18)),
        }
    )

rows_ok = [r for r in rows if r["returncode"] == 0]
rows_ok.sort(key=lambda r: (r["l2_error_avg_123_all"], r["output_parse_error_rate"], -r["action_accuracy"]))

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    f.write("candidate_id\tl2_123\tl2_03\tl2_3s\taction_acc\tparse_out\tparse_gt\tsummary_json\n")
    for r in rows_ok:
        f.write(
            f"{r['candidate_id']}\t{r['l2_error_avg_123_all']:.6f}\t{r['l2_error_avg_03_all']:.6f}\t"
            f"{r['l2_error_avg_3s']:.6f}\t{r['action_accuracy']:.6f}\t"
            f"{r['output_parse_error_rate']:.6f}\t{r['gt_parse_error_rate']:.6f}\t{r['summary_json']}\n"
        )

best = rows_ok[0] if rows_ok else {}
best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

print(f"candidates_total={len(rows)} candidates_ok={len(rows_ok)}")
if rows_ok:
    print(
        "best_train "
        f"id={best['candidate_id']} "
        f"l2_123={best['l2_error_avg_123_all']:.6f} "
        f"l2_03={best['l2_error_avg_03_all']:.6f} "
        f"action_acc={best['action_accuracy']:.6f}"
    )
print(f"summary_tsv={summary_path}")
print(f"best_json={best_path}")
PY

if [[ "${RUN_VAL_FINAL}" == "1" ]]; then
  BEST_ID="$(python3 - <<'PY'
import json
from pathlib import Path
import os
run_name = os.environ.get("RUN_NAME", "phase9_q5_tree_sweep_v1")
out_dir = Path(os.environ.get("OUT_DIR", f"outputs/phase9_train_dev/{run_name}"))
best_path = out_dir / f"{run_name}_best_train_candidate.json"
best = json.loads(best_path.read_text(encoding="utf-8"))
print(best.get("candidate_id", ""))
PY
)"
  if [[ -z "${BEST_ID}" ]]; then
    echo "val_final_skip reason=no_best_candidate"
    exit 0
  fi
  BEST_MODEL="${OUT_DIR}/${BEST_ID}_deployable.json"
  VAL_SCENARIO="val_q5_${RUN_NAME}_${BEST_ID}_final"
  echo "eval_val_final ${BEST_ID}"
  python3 scripts/run_qa_split_pipeline.py \
    --purpose val_report \
    --split val \
    --task-type object_motion_prediction \
    --scenario-name "${VAL_SCENARIO}" \
    --baseline-mode cooperative \
    --object-motion-model-json "${BEST_MODEL}" \
    --workers "${WORKERS}" \
    --progress-every "${PROGRESS_EVERY}" \
    --v2vgot-root "${V2VGOT_ROOT}" \
    >/dev/null

  VAL_SUMMARY="outputs/phase8_val_report/official_eval_reports/${VAL_SCENARIO}_official_export_manifest_official_qa_eval_summary.json"
  echo "val_final_done scenario=${VAL_SCENARIO} summary=${VAL_SUMMARY}"
fi

echo "q5_sweep_done run=${RUN_NAME}"
