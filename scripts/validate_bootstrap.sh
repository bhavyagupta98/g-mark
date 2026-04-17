#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPOS_DIR="${REPOS_DIR:-${WORKSPACE}/repos}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${WORKSPACE}/artifacts/v2vgot}"
V2VGOT_DIR="${V2VGOT_DIR:-${REPOS_DIR}/V2V-GoT}"

print_header() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

check_path() {
  local label="$1"
  local path="$2"
  if [ -e "${path}" ]; then
    echo "[PASS] ${label}: ${path}"
  else
    echo "[FAIL] ${label}: ${path}"
    return 1
  fi
}

check_dir_has_files() {
  local label="$1"
  local path="$2"
  if [ -d "${path}" ] && [ -n "$(find "${path}" -maxdepth 1 -type f | head -n 1)" ]; then
    echo "[PASS] ${label}: ${path}"
  else
    echo "[FAIL] ${label}: ${path}"
    return 1
  fi
}

main() {
  local status=0

  print_header "Bootstrap Validation"

  check_path "Workspace directory" "${WORKSPACE}" || status=1
  check_path "Repos directory" "${REPOS_DIR}" || status=1
  check_path "Artifacts directory" "${ARTIFACT_DIR}" || status=1

  print_header "Repository Checks"
  check_path "kg_coop_drive repo" "${REPOS_DIR}/kg_coop_drive/.git" || status=1
  check_path "auto_drive_copy repo" "${REPOS_DIR}/auto_drive_copy/.git" || status=1
  check_path "V2V-GoT repo" "${V2VGOT_DIR}/.git" || status=1

  print_header "Artifact Checks"
  check_path "dataset_jsons.zip" "${ARTIFACT_DIR}/dataset_jsons.zip" || status=1
  if [ "${DOWNLOAD_PROCESSED_FEATURES:-true}" = "true" ]; then
    check_path "dataset_processed_features_and_gt.zip" "${ARTIFACT_DIR}/dataset_processed_features_and_gt.zip" || status=1
  fi
  if [ "${DOWNLOAD_MODEL_CKPT:-true}" = "true" ]; then
    check_path "model_ckpt.zip" "${ARTIFACT_DIR}/model_ckpt.zip" || status=1
  fi

  print_header "Extracted Dataset Checks"
  check_dir_has_files \
    "Val co_llm JSONs" \
    "${V2VGOT_DIR}/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm" || status=1
  check_dir_has_files \
    "Train co_llm JSONs" \
    "${V2VGOT_DIR}/DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm" || status=1

  check_path \
    "Val V2V-GoT QA dataset" \
    "${V2VGOT_DIR}/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vgot.json" || status=1
  check_path \
    "Train V2V-GoT QA dataset" \
    "${V2VGOT_DIR}/DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vgot.json" || status=1

  print_header "Summary"
  if [ "${status}" -eq 0 ]; then
    echo "[PASS] Bootstrap validation completed successfully."
  else
    echo "[FAIL] Bootstrap validation found missing files."
  fi

  return "${status}"
}

main "$@"
