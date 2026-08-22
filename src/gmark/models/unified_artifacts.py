from __future__ import annotations

import base64
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from gmark.features.leakage_checks import Q9_LEAKAGE_FIELDS

LOGGER = logging.getLogger("unified_artifacts")


REQUIRED = {
    "object_retrieval": "object_retrieval/object_retrieval_logreg_shared_q1_q4.json",
    "motion_q57": "motion_regression/motion_elasticnet_q5_q7.json",
    "motion_q9": "motion_regression/motion_elasticnet_q9_clean.json",
    "q6": "scene_action/scene_action_gbdt_q6.json",
    "q8_speed": "scene_action/scene_action_gbdt_q8_speed.json",
    "q8_steering": "scene_action/scene_action_gbdt_q8_steering.json",
}


def load_unified_artifact(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def validate_artifact_schema(artifact: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if "feature_names" not in artifact or not isinstance(artifact.get("feature_names"), list):
        errs.append("missing feature_names")
    if artifact.get("family") == "object_retrieval":
        if artifact.get("shared_across_tasks") is not True:
            errs.append("object_retrieval shared_across_tasks must be true")
        if not isinstance(artifact.get("per_task_thresholds"), dict):
            errs.append("missing per_task_thresholds")
    if artifact.get("q9_clean_only") is True and artifact.get("leakage_check_passed") is not True:
        errs.append("q9 leakage_check_passed must be true")
    return errs


def load_all_unified_artifacts(artifact_dir: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = Path(artifact_dir)
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    invalid: dict[str, list[str]] = {}

    for key, rel in REQUIRED.items():
        path = root / rel
        if not path.exists():
            missing.append(key)
            continue
        artifact = load_unified_artifact(path)
        errs = validate_artifact_schema(artifact)
        if errs:
            invalid[key] = errs
        loaded[key] = artifact

    summary = {
        "artifact_dir": str(root),
        "found_keys": sorted(loaded.keys()),
        "missing_keys": missing,
        "invalid": invalid,
    }
    return loaded, summary


def _to_vector(model_input: dict[str, Any], feature_names: list[str], *, qa_type_id: int | None = None) -> np.ndarray:
    vec = np.zeros(len(feature_names), dtype=np.float32)
    for idx, name in enumerate(feature_names):
        if name.startswith("task_onehot::"):
            if qa_type_id is None:
                continue
            vec[idx] = _task_onehot_value(name, int(qa_type_id))
            continue
        value = model_input.get(name, 0.0)
        try:
            vec[idx] = np.float32(value)
        except (TypeError, ValueError):
            vec[idx] = np.float32(0.0)
    return vec


def _task_onehot_value(name: str, qa_type_id: int) -> float:
    mapping = {
        "task_onehot::is_q1_notable": 11,
        "task_onehot::is_q2_occluding": 12,
        "task_onehot::is_q3_invisible": 13,
        "task_onehot::is_q4_planning": 14,
    }
    target = mapping.get(name)
    return 1.0 if target == qa_type_id else 0.0


def predict_with_object_retrieval_artifact(rows: list[dict[str, Any]], artifact: dict[str, Any], config: dict[str, Any] | None = None) -> list[float]:
    del config
    feat_names = [str(x) for x in artifact.get("feature_names", [])]
    weights = np.asarray(artifact.get("model_weights", []), dtype=np.float32)
    intercept = float(artifact.get("intercept", 0.0))
    out: list[float] = []
    for row in rows:
        vec = _to_vector(row.get("model_input", {}), feat_names, qa_type_id=int(row.get("qa_type_id", -1) or -1))
        z = float(np.dot(vec, weights) + intercept)
        out.append(1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0))))
    return out


def predict_with_motion_artifact(rows: list[dict[str, Any]], artifact: dict[str, Any], config: dict[str, Any] | None = None) -> list[list[float]]:
    del config
    feat_names = [str(x) for x in artifact.get("feature_names", [])]
    if artifact.get("q9_clean_only") is True:
        present = {n.lower() for n in feat_names}
        bad = sorted([f for f in Q9_LEAKAGE_FIELDS if f in present])
        if bad:
            raise ValueError(f"Q9 leakage fields present at inference: {bad}")

    coeff = np.asarray(artifact.get("model_coefficients", []), dtype=np.float32)
    intercept = np.asarray(artifact.get("intercept", []), dtype=np.float32)
    center = np.asarray(artifact.get("target_center", []), dtype=np.float32)
    std_info = artifact.get("standardization", {}) if isinstance(artifact.get("standardization"), dict) else {}
    enabled = bool(std_info.get("enabled", False))
    mu = np.asarray(std_info.get("mean", []), dtype=np.float32)
    sigma = np.asarray(std_info.get("std", []), dtype=np.float32)
    sigma = np.where(np.abs(sigma) < 1e-8, 1.0, sigma)

    preds: list[list[float]] = []
    for row in rows:
        vec = _to_vector(row.get("model_input", {}), feat_names)
        if enabled and mu.size == vec.size and sigma.size == vec.size:
            vec = (vec - mu) / sigma
        raw = np.matmul(coeff, vec) + intercept
        raw = raw + center
        preds.append([float(x) for x in raw.tolist()])
    return preds


def _load_pickled_model_b64(blob: str) -> Any:
    return pickle.loads(base64.b64decode(blob.encode("ascii")))


def predict_with_scene_action_artifact(rows: list[dict[str, Any]], artifact: dict[str, Any], config: dict[str, Any] | None = None) -> list[Any]:
    del config
    feat_names = [str(x) for x in artifact.get("feature_names", [])]
    model_blob = artifact.get("serialized_model_b64", "")
    if not isinstance(model_blob, str) or not model_blob:
        raise ValueError("scene-action artifact missing serialized_model_b64")
    model = _load_pickled_model_b64(model_blob)

    x = np.zeros((len(rows), len(feat_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        x[i, :] = _to_vector(row.get("model_input", {}), feat_names)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return [float(p[1]) for p in probs]
    pred = model.predict(x)
    return [int(v) for v in pred.tolist()]
