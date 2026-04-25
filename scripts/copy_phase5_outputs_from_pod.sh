#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAMESPACE="${NAMESPACE:-seelab}"
POD_NAME="${POD_NAME:-kg-coop-runtime}"
REMOTE_DIR="${1:-/workspace/repos/kg_coop_drive/outputs/phase8_baselines}"
LOCAL_DIR="${2:-${REPO_ROOT}/outputs/$(basename "${REMOTE_DIR}")}"

mkdir -p "$(dirname "${LOCAL_DIR}")"

echo "[INFO] namespace: ${NAMESPACE}"
echo "[INFO] pod: ${POD_NAME}"
echo "[INFO] remote_dir: ${REMOTE_DIR}"
echo "[INFO] local_dir: ${LOCAL_DIR}"
echo "[INFO] Copying files from pod to local workspace..."

kubectl -n "${NAMESPACE}" cp "${POD_NAME}:${REMOTE_DIR}" "${LOCAL_DIR}"

echo "[PASS] Copied Phase 5 outputs to ${LOCAL_DIR}"
