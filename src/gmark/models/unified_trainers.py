from __future__ import annotations

import base64
import json
import math
import pickle
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gmark.features.leakage_checks import Q9_LEAKAGE_FIELDS, assert_no_leakage_features

try:
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
    from sklearn.linear_model import ElasticNet, LogisticRegression  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    GradientBoostingClassifier = None  # type: ignore
    ElasticNet = None  # type: ignore
    LogisticRegression = None  # type: ignore


@dataclass(frozen=True)
class TrainResult:
    head_name: str
    artifact_relpath: str
    artifact_payload: dict[str, Any]
    metrics: dict[str, Any]


LOGGER = logging.getLogger("unified_heads_trainers")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.flush()
    tmp.replace(path)


def _serialize_model_b64(model: Any) -> str:
    raw = pickle.dumps(model)
    return base64.b64encode(raw).decode("ascii")


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = float((tp + tn) / max(1, len(y_true)))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    metrics = {
        "mae": mae,
        "rmse": rmse,
    }
    if y_true.shape[1] % 2 == 0:
        pairs = y_true.shape[1] // 2
        l2s = []
        for i in range(pairs):
            dx = err[:, 2 * i]
            dy = err[:, 2 * i + 1]
            l2s.append(np.sqrt(dx * dx + dy * dy))
        if l2s:
            metrics["avg_l2"] = float(np.mean(np.concatenate([x.reshape(-1, 1) for x in l2s], axis=1)))
    return metrics


def _iter_labeled_rows(path: Path, max_rows: int = 0, log_every: int = 0, head_name: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    LOGGER.info("loading_rows head=%s path=%s max_rows=%d", head_name, path, max_rows)
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
            if log_every > 0 and idx % log_every == 0:
                LOGGER.info("load_progress head=%s lines=%d rows=%d", head_name, idx, len(rows))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    LOGGER.info("loaded_rows head=%s rows=%d", head_name, len(rows))
    return rows


def _iter_labeled_rows_gen(path: Path, max_rows: int = 0) -> Any:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        count = 0
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break


def safe_feature_matrix_builder(
    rows: list[dict[str, Any]],
    *,
    required_qa_types: set[int] | None = None,
    extra_feature_fn: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], list[str], list[dict[str, Any]], dict[str, int]]:
    filtered: list[dict[str, Any]] = []
    skipped = {"malformed": 0, "missing_label": 0, "wrong_task": 0}
    feature_union: set[str] = set()
    for row in rows:
        qa = int(row.get("qa_type_id", -1) or -1)
        if required_qa_types is not None and qa not in required_qa_types:
            skipped["wrong_task"] += 1
            continue
        if not bool(row.get("has_label", False)):
            skipped["missing_label"] += 1
            continue
        model_input = row.get("model_input", {})
        if not isinstance(model_input, dict):
            skipped["malformed"] += 1
            continue
        filtered.append(row)
        feature_union.update(str(k) for k in model_input.keys())

    feature_names = sorted(feature_union)
    if extra_feature_fn is not None:
        feature_names.extend(extra_feature_fn("__feature_names_only__"))

    n = len(filtered)
    d = len(feature_names)
    x = np.zeros((n, d), dtype=np.float32)
    qa_types: list[int] = []
    sample_ids: list[str] = []
    meta_rows: list[dict[str, Any]] = []

    for i, row in enumerate(filtered):
        qa = int(row.get("qa_type_id", -1) or -1)
        qa_types.append(qa)
        sample_ids.append(str(row.get("sample_id", "")))
        meta_rows.append({
            "sample_id": str(row.get("sample_id", "")),
            "qa_type_id": qa,
            "candidate_id": str(row.get("candidate_id", "")),
        })
        model_input = row.get("model_input", {})
        for j, name in enumerate(feature_names):
            if extra_feature_fn is not None and name.startswith("task_onehot::"):
                continue
            value = model_input.get(name, 0.0)
            try:
                x[i, j] = np.float32(value)
            except (TypeError, ValueError):
                x[i, j] = np.float32(0.0)

        if extra_feature_fn is not None:
            extra_values = extra_feature_fn(qa)
            for name, value in extra_values.items():
                idx = feature_names.index(name)
                x[i, idx] = np.float32(value)

    y_dummy = np.zeros(n, dtype=np.float32)
    return x, y_dummy, feature_names, qa_types, sample_ids, meta_rows, skipped


def threshold_search_binary(
    probs: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
    threshold_metric: str = "f1",
    max_results: int | None = None,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t = 0.5
    best = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "accuracy": 0.0}

    by_sample: dict[str, list[int]] = defaultdict(list)
    for idx, sid in enumerate(sample_ids):
        by_sample[sid].append(idx)

    for t in thresholds:
        pred = np.zeros_like(labels)
        for idxs in by_sample.values():
            selected = [i for i in idxs if probs[i] >= t]
            if max_results is not None and max_results > 0 and len(selected) > max_results:
                selected = sorted(selected, key=lambda i: float(probs[i]), reverse=True)[:max_results]
            for i in selected:
                pred[i] = 1
        m = compute_binary_metrics(labels, pred)
        score = m.get(threshold_metric, m["f1"])
        if score > best.get(threshold_metric, best["f1"]):
            best_t = float(t)
            best = m
    return best_t, best


def train_shared_object_retrieval_logreg(
    rows_path: Path,
    model_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    output_path: Path,
    max_rows: int = 0,
    overwrite: bool = False,
    log_every: int = 0,
) -> TrainResult:
    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is required for Stage 3 training")

    required = {11, 12, 13, 14}

    task_feature_names = [
        "task_onehot::is_q1_notable",
        "task_onehot::is_q2_occluding",
        "task_onehot::is_q3_invisible",
        "task_onehot::is_q4_planning",
    ]

    def _task_features(qa_type: Any) -> dict[str, float] | list[str]:
        if qa_type == "__feature_names_only__":
            return task_feature_names
        qa = int(qa_type)
        return {
            "task_onehot::is_q1_notable": 1.0 if qa == 11 else 0.0,
            "task_onehot::is_q2_occluding": 1.0 if qa == 12 else 0.0,
            "task_onehot::is_q3_invisible": 1.0 if qa == 13 else 0.0,
            "task_onehot::is_q4_planning": 1.0 if qa == 14 else 0.0,
        }

    LOGGER.info("object_retrieval streaming_pass=1 path=%s", rows_path)
    feature_union: set[str] = set()
    n_rows = 0
    rows_total = 0
    skipped = {"malformed": 0, "missing_label": 0, "wrong_task": 0}
    for row in _iter_labeled_rows_gen(rows_path, max_rows=max_rows):
        rows_total += 1
        qa = int(row.get("qa_type_id", -1) or -1)
        if qa not in required:
            skipped["wrong_task"] += 1
            continue
        if not bool(row.get("has_label", False)):
            skipped["missing_label"] += 1
            continue
        label = row.get("label", {})
        if not isinstance(label, dict) or label.get("label_type") != "binary" or label.get("label") is None:
            skipped["malformed"] += 1
            continue
        model_input = row.get("model_input", {})
        if not isinstance(model_input, dict):
            skipped["malformed"] += 1
            continue
        feature_union.update(str(k) for k in model_input.keys())
        n_rows += 1
        if log_every > 0 and n_rows % log_every == 0:
            LOGGER.info("object_retrieval pass1_progress kept_rows=%d", n_rows)

    if n_rows == 0:
        raise RuntimeError("No valid object-retrieval labeled rows found")
    feature_names = sorted(feature_union)
    feature_names.extend(_task_features("__feature_names_only__"))  # type: ignore[arg-type]
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    x = np.zeros((n_rows, len(feature_names)), dtype=np.float32)
    y_arr = np.zeros(n_rows, dtype=np.int32)
    kept_qa: list[int] = []
    kept_ids: list[str] = []

    LOGGER.info("object_retrieval streaming_pass=2 rows=%d features=%d", n_rows, len(feature_names))
    write_i = 0
    for row in _iter_labeled_rows_gen(rows_path, max_rows=max_rows):
        qa = int(row.get("qa_type_id", -1) or -1)
        if qa not in required or not bool(row.get("has_label", False)):
            continue
        label = row.get("label", {})
        if not isinstance(label, dict) or label.get("label_type") != "binary" or label.get("label") is None:
            continue
        model_input = row.get("model_input", {})
        if not isinstance(model_input, dict):
            continue
        for k, v in model_input.items():
            idx = feature_to_idx.get(str(k))
            if idx is None:
                continue
            try:
                x[write_i, idx] = np.float32(v)
            except (TypeError, ValueError):
                pass
        for name, value in _task_features(qa).items():  # type: ignore[union-attr]
            x[write_i, feature_to_idx[name]] = np.float32(value)
        y_arr[write_i] = int(label.get("label", 0))
        kept_qa.append(qa)
        kept_ids.append(str(row.get("sample_id", "")))
        write_i += 1
        if log_every > 0 and write_i % log_every == 0:
            LOGGER.info("object_retrieval pass2_progress written_rows=%d", write_i)

    if write_i != n_rows:
        x = x[:write_i, :]
        y_arr = y_arr[:write_i]

    if len(np.unique(y_arr)) < 2:
        raise RuntimeError("Object retrieval labels contain only one class; cannot train logreg")
    LOGGER.info(
        "object_retrieval train_matrix_rows=%d features=%d positives=%d negatives=%d",
        x.shape[0],
        x.shape[1],
        int(np.sum(y_arr == 1)),
        int(np.sum(y_arr == 0)),
    )

    l2 = float(model_cfg.get("l2", 0.01))
    c_value = 1.0 / max(l2, 1e-9)
    class_weight = str(model_cfg.get("class_weighting", "balanced"))
    if class_weight == "none":
        cw = None
    else:
        cw = class_weight
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        class_weight=cw,
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    model.fit(x, y_arr)
    probs = model.predict_proba(x)[:, 1]

    thresholds: dict[str, float] = {}
    metrics_per_task: dict[str, dict[str, float]] = {}
    max_results_cfg = model_cfg.get("max_results_by_task", {}) if isinstance(model_cfg.get("max_results_by_task"), dict) else {}
    for qa in sorted(required):
        idx = [i for i, q in enumerate(kept_qa) if q == qa]
        if not idx:
            continue
        t_probs = probs[idx]
        t_labels = y_arr[idx]
        t_sids = [kept_ids[i] for i in idx]
        raw_max = max_results_cfg.get(qa, max_results_cfg.get(str(qa)))
        max_results = int(raw_max) if isinstance(raw_max, int) or (isinstance(raw_max, str) and raw_max.isdigit()) else None
        thr, task_metrics = threshold_search_binary(
            t_probs,
            t_labels,
            t_sids,
            threshold_metric=str(model_cfg.get("threshold_metric", "f1")),
            max_results=max_results,
        )
        thresholds[str(qa)] = thr
        metrics_per_task[str(qa)] = task_metrics
        LOGGER.info("object_retrieval threshold qa_type_id=%d threshold=%.3f f1=%.4f", qa, thr, task_metrics.get("f1", 0.0))

    pred_default = (probs >= 0.5).astype(int)
    overall = compute_binary_metrics(y_arr, pred_default)

    payload = {
        "family": "object_retrieval",
        "model_type": "logreg",
        "shared_across_tasks": True,
        "qa_type_ids": [11, 12, 13, 14],
        "feature_names": feature_names,
        "task_one_hot_feature_names": task_feature_names,
        "normalization": None,
        "model_weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "classes": model.classes_.tolist(),
        "per_task_thresholds": thresholds,
        "per_task_max_results": max_results_cfg,
        "train_metrics_overall": overall,
        "train_metrics_per_qa_type": metrics_per_task,
        "positive_negative_counts": {
            str(k): {"count": int(v)} for k, v in Counter(kept_qa).items()
        },
        "row_filter_summary": skipped,
        "config_snapshot": run_cfg,
        "creation_timestamp": utc_timestamp(),
    }
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"SKIP_ARTIFACT_EXISTS:{output_path}")
    save_json_artifact(output_path, payload)
    return TrainResult(
        head_name="object_retrieval_shared",
        artifact_relpath=str(output_path),
        artifact_payload=payload,
        metrics={"overall": overall, "per_task": metrics_per_task},
    )


def _extract_reg_targets(rows: list[dict[str, Any]], qa_types: set[int]) -> tuple[list[dict[str, Any]], np.ndarray]:
    keep: list[dict[str, Any]] = []
    y: list[list[float]] = []
    for row in rows:
        qa = int(row.get("qa_type_id", -1) or -1)
        if qa not in qa_types:
            continue
        if not bool(row.get("has_label", False)):
            continue
        label = row.get("label", {})
        if not isinstance(label, dict) or label.get("label_type") != "regression":
            continue
        target = label.get("target", [])
        if not isinstance(target, list) or not target:
            continue
        try:
            vec = [float(v) for v in target]
        except (TypeError, ValueError):
            continue
        keep.append(row)
        y.append(vec)
    if not y:
        return [], np.zeros((0, 0), dtype=float)
    target_dim = len(y[0])
    compat = [vec for vec in y if len(vec) == target_dim]
    keep = [r for r, vec in zip(keep, y) if len(vec) == target_dim]
    return keep, np.asarray(compat, dtype=float)


def train_elasticnet_motion_heads(
    rows_path: Path,
    model_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    output_dir: Path,
    max_rows: int = 0,
    overwrite: bool = False,
    log_every: int = 0,
) -> list[TrainResult]:
    if ElasticNet is None:
        raise RuntimeError("scikit-learn is required for Stage 3 training")

    rows = _iter_labeled_rows(rows_path, max_rows=max_rows, log_every=log_every, head_name="motion_regression")
    alpha = float(model_cfg.get("alpha", 0.001))
    l1_ratio = float(model_cfg.get("l1_ratio", 0.1))
    standardize = bool(model_cfg.get("standardize", True))
    results: list[TrainResult] = []

    # Shared Q5/Q7
    q57_rows, y57 = _extract_reg_targets(rows, {15, 17})
    LOGGER.info("motion_q5_q7 rows_with_targets=%d", y57.shape[0] if y57.size else 0)
    if y57.size:
        x_rows, _, feat_names, qa_types, _, _, skipped = safe_feature_matrix_builder(q57_rows, required_qa_types={15, 17})
        if x_rows.shape[0] == y57.shape[0]:
            mu = np.mean(x_rows, axis=0) if standardize else np.zeros(x_rows.shape[1], dtype=float)
            sigma = np.std(x_rows, axis=0) if standardize else np.ones(x_rows.shape[1], dtype=float)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            x_fit = (x_rows - mu) / sigma if standardize else x_rows
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
            y_mean = np.mean(y57, axis=0)
            model.fit(x_fit, y57 - y_mean)
            pred = model.predict(x_fit) + y_mean
            metrics = compute_regression_metrics(y57, pred)
            LOGGER.info(
                "motion_q5_q7 trained rows=%d features=%d target_dim=%d mae=%.4f rmse=%.4f",
                x_fit.shape[0],
                x_fit.shape[1],
                y57.shape[1],
                metrics.get("mae", 0.0),
                metrics.get("rmse", 0.0),
            )
            payload = {
                "family": "motion_regression",
                "model_type": "elasticnet",
                "qa_type_ids": [15, 17],
                "feature_names": feat_names,
                "target_names": [f"t{i}" for i in range(y57.shape[1])],
                "target_dim": int(y57.shape[1]),
                "standardization": {"enabled": standardize, "mean": mu.tolist(), "std": sigma.tolist()},
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "model_coefficients": model.coef_.tolist() if hasattr(model.coef_, "tolist") else [],
                "intercept": model.intercept_.tolist() if hasattr(model.intercept_, "tolist") else float(model.intercept_),
                "target_center": y_mean.tolist(),
                "train_metrics": metrics,
                "row_filter_summary": skipped,
                "qa_type_distribution": {str(k): int(v) for k, v in Counter(qa_types).items()},
                "config_snapshot": run_cfg,
                "creation_timestamp": utc_timestamp(),
            }
            out = output_dir / "motion_elasticnet_q5_q7.json"
            if out.exists() and not overwrite:
                pass
            else:
                save_json_artifact(out, payload)
                results.append(TrainResult("motion_q5_q7", str(out), payload, metrics))

    # Separate Q9 clean
    q9_rows, y9 = _extract_reg_targets(rows, {19})
    LOGGER.info("motion_q9 rows_with_targets=%d", y9.shape[0] if y9.size else 0)
    if y9.size:
        for row in q9_rows:
            names = row.get("feature_names", [])
            if not isinstance(names, list):
                continue
            assert_no_leakage_features((str(v) for v in names), qa_type_id=19, strict=True)
        x9, _, f9, qa9, _, _, skipped9 = safe_feature_matrix_builder(q9_rows, required_qa_types={19})
        for name in f9:
            if name.lower() in Q9_LEAKAGE_FIELDS:
                raise RuntimeError(f"Q9 leakage field present in model features: {name}")
        LOGGER.info("motion_q9 leakage_check_passed features=%d exclusion_fields=%s", len(f9), sorted(Q9_LEAKAGE_FIELDS))
        mu9 = np.mean(x9, axis=0) if standardize else np.zeros(x9.shape[1], dtype=float)
        sigma9 = np.std(x9, axis=0) if standardize else np.ones(x9.shape[1], dtype=float)
        sigma9 = np.where(sigma9 < 1e-8, 1.0, sigma9)
        x9_fit = (x9 - mu9) / sigma9 if standardize else x9
        model9 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
        y9_mean = np.mean(y9, axis=0)
        model9.fit(x9_fit, y9 - y9_mean)
        pred9 = model9.predict(x9_fit) + y9_mean
        m9 = compute_regression_metrics(y9, pred9)
        LOGGER.info(
            "motion_q9 trained rows=%d features=%d target_dim=%d mae=%.4f rmse=%.4f",
            x9_fit.shape[0],
            x9_fit.shape[1],
            y9.shape[1],
            m9.get("mae", 0.0),
            m9.get("rmse", 0.0),
        )
        payload9 = {
            "family": "motion_regression",
            "model_type": "elasticnet",
            "qa_type_ids": [19],
            "q9_clean_only": True,
            "leakage_exclusion_list": sorted(Q9_LEAKAGE_FIELDS),
            "leakage_check_passed": True,
            "feature_names": f9,
            "target_names": [f"t{i}" for i in range(y9.shape[1])],
            "target_dim": int(y9.shape[1]),
            "standardization": {"enabled": standardize, "mean": mu9.tolist(), "std": sigma9.tolist()},
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "model_coefficients": model9.coef_.tolist() if hasattr(model9.coef_, "tolist") else [],
            "intercept": model9.intercept_.tolist() if hasattr(model9.intercept_, "tolist") else float(model9.intercept_),
            "target_center": y9_mean.tolist(),
            "train_metrics": m9,
            "row_filter_summary": skipped9,
            "qa_type_distribution": {str(k): int(v) for k, v in Counter(qa9).items()},
            "config_snapshot": run_cfg,
            "creation_timestamp": utc_timestamp(),
        }
        out9 = output_dir / "motion_elasticnet_q9_clean.json"
        if out9.exists() and not overwrite:
            pass
        else:
            save_json_artifact(out9, payload9)
            results.append(TrainResult("motion_q9", str(out9), payload9, m9))

    return results


def _binary_labels(rows: list[dict[str, Any]], qa_type_id: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    keep: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in rows:
        if int(row.get("qa_type_id", -1) or -1) != qa_type_id:
            continue
        if not bool(row.get("has_label", False)):
            continue
        label = row.get("label", {})
        if not isinstance(label, dict):
            continue
        if qa_type_id == 16 and label.get("label_type") == "binary":
            keep.append(row)
            labels.append(int(label.get("label", 0)))
        elif qa_type_id == 18 and label.get("label_type") == "multitask_classification":
            keep.append(row)
            labels.append(int(label.get("speed_class", -1)))
    return keep, np.asarray(labels, dtype=int)


def _q8_head_labels(rows: list[dict[str, Any]], key: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    keep: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in rows:
        if int(row.get("qa_type_id", -1) or -1) != 18:
            continue
        if not bool(row.get("has_label", False)):
            continue
        label = row.get("label", {})
        if not isinstance(label, dict):
            continue
        label_type = str(label.get("label_type", ""))
        if label_type not in {"multitask_classification", "control"}:
            continue
        v = label.get(key)
        if not isinstance(v, int):
            continue
        keep.append(row)
        labels.append(int(v))
    return keep, np.asarray(labels, dtype=int)


def _build_class_weights(labels: np.ndarray, scheme: str) -> np.ndarray:
    weights = np.ones(labels.shape[0], dtype=np.float32)
    if labels.size == 0 or scheme == "none":
        return weights
    max_class = int(np.max(labels))
    counts = np.bincount(labels.astype(np.int32), minlength=max_class + 1).astype(np.float32)
    counts = np.where(counts <= 0.0, 1.0, counts)
    if scheme == "inverse_freq":
        class_w = 1.0 / counts
    elif scheme == "sqrt_inverse_freq":
        class_w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unsupported class-weighting scheme: {scheme}")
    class_w = class_w * (float(len(class_w)) / float(np.sum(class_w)))
    for i, label in enumerate(labels):
        weights[i] = class_w[int(label)]
    return weights


def train_gbdt_scene_action_heads(
    rows_path: Path,
    model_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    output_dir: Path,
    max_rows: int = 0,
    overwrite: bool = False,
    log_every: int = 0,
) -> list[TrainResult]:
    if GradientBoostingClassifier is None:
        raise RuntimeError("scikit-learn is required for Stage 3 training")

    rows = _iter_labeled_rows(rows_path, max_rows=max_rows, log_every=log_every, head_name="scene_action")
    n_estimators = int(model_cfg.get("n_estimators", 280))
    learning_rate = float(model_cfg.get("learning_rate", 0.04))
    max_depth = int(model_cfg.get("max_depth", 2))
    min_samples_leaf = int(model_cfg.get("min_samples_leaf", 96))
    subsample = float(model_cfg.get("subsample", 0.7))
    default_q6_thr = float(model_cfg.get("q6_default_threshold", 0.38))
    q8_speed_weighting = str(model_cfg.get("q8_speed_class_weighting", "none"))
    q8_steering_weighting = str(model_cfg.get("q8_steering_class_weighting", "none"))
    q8_head_model_type = str(model_cfg.get("q8_head_model_type", "gbdt")).strip().lower()
    q8_logreg_c = float(model_cfg.get("q8_logreg_c", 5.0))
    q8_logreg_max_iter = int(model_cfg.get("q8_logreg_max_iter", 2000))
    results: list[TrainResult] = []

    def _fit_head(head_name: str, qa: int, label_key: str | None = None, threshold: bool = False) -> TrainResult | None:
        if qa == 16:
            keep, y = _binary_labels(rows, 16)
        else:
            keep, y = _q8_head_labels(rows, label_key or "speed_class")
        if y.size == 0:
            LOGGER.warning("scene_action head=%s no rows after label filtering", head_name)
            return None
        x, _, feat_names, _, sample_ids, _, skipped = safe_feature_matrix_builder(keep, required_qa_types={qa})
        if x.shape[0] != y.shape[0]:
            min_n = min(x.shape[0], y.shape[0])
            x = x[:min_n, :]
            y = y[:min_n]
            sample_ids = sample_ids[:min_n]
        if len(np.unique(y)) < 2:
            LOGGER.warning("scene_action head=%s has <2 classes; skipping", head_name)
            return None

        LOGGER.info("scene_action head=%s rows=%d features=%d labels=%s", head_name, x.shape[0], x.shape[1], {str(k): int(v) for k, v in Counter(y.tolist()).items()})
        use_logreg = head_name in {"q8_speed", "q8_steering"} and q8_head_model_type == "logreg"
        if use_logreg:
            model = LogisticRegression(
                C=q8_logreg_c,
                solver="lbfgs",
                max_iter=q8_logreg_max_iter,
                multi_class="multinomial",
                random_state=42,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=42,
            )
        sample_weight = None
        if head_name == "q8_speed":
            sample_weight = _build_class_weights(y, q8_speed_weighting)
        elif head_name == "q8_steering":
            sample_weight = _build_class_weights(y, q8_steering_weighting)
        model.fit(x, y, sample_weight=sample_weight)
        pred = model.predict(x)
        acc = float(np.mean(pred == y))
        metrics: dict[str, Any] = {
            "accuracy": acc,
            "label_distribution": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
        }
        if head_name in {"q8_speed", "q8_steering"}:
            per_class_acc = {}
            for c in sorted(np.unique(y).tolist()):
                idx = np.where(y == c)[0]
                if idx.size > 0:
                    per_class_acc[str(int(c))] = float(np.mean(pred[idx] == y[idx]))
            metrics["per_class_accuracy"] = per_class_acc
            metrics["class_weighting"] = q8_speed_weighting if head_name == "q8_speed" else q8_steering_weighting

        selected_thr = None
        if threshold and hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)
            if probs.shape[1] == 2:
                thr, m = threshold_search_binary(probs[:, 1], y.astype(int), sample_ids, threshold_metric="f1", max_results=None)
                selected_thr = thr
                metrics.update({"precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
            else:
                selected_thr = default_q6_thr

        payload = {
            "family": "scene_action",
            "model_type": "logreg_classifier" if use_logreg else "gbdt_classifier",
            "qa_type_id": qa,
            "head_name": head_name,
            "feature_names": feat_names,
            "label_map": {str(c): int(c) for c in sorted(np.unique(y).tolist())},
            "model_params": {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "subsample": subsample,
                "q8_speed_class_weighting": q8_speed_weighting,
                "q8_steering_class_weighting": q8_steering_weighting,
                "q8_head_model_type": q8_head_model_type,
                "q8_logreg_c": q8_logreg_c,
                "q8_logreg_max_iter": q8_logreg_max_iter,
            },
            "serialized_model_b64": _serialize_model_b64(model),
            "threshold": selected_thr,
            "train_metrics": metrics,
            "row_filter_summary": skipped,
            "config_snapshot": run_cfg,
            "creation_timestamp": utc_timestamp(),
        }
        file_name = {
            "q6_binary": "scene_action_gbdt_q6.json",
            "q8_speed": "scene_action_gbdt_q8_speed.json",
            "q8_steering": "scene_action_gbdt_q8_steering.json",
        }[head_name]
        out = output_dir / file_name
        if out.exists() and not overwrite:
            return None
        save_json_artifact(out, payload)
        return TrainResult(head_name, str(out), payload, metrics)

    for r in (
        _fit_head("q6_binary", 16, threshold=bool(model_cfg.get("threshold_search", True))),
        _fit_head("q8_speed", 18, label_key="speed_class", threshold=False),
        _fit_head("q8_steering", 18, label_key="steering_class", threshold=False),
    ):
        if r is not None:
            results.append(r)

    return results
