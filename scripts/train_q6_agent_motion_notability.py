#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    GradientBoostingClassifier = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    xgb = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

NOTABLE_RE = re.compile(r"\bis a notable object\b", re.IGNORECASE)
NOT_NOTABLE_RE = re.compile(r"\bis a not notable object\b", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train frozen Q6 agent-motion notability classifier.")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--l2-regularization", type=float, default=1e-3)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--model-family",
        default="logistic",
        choices=("logistic", "regression_tree", "gbdt"),
    )
    parser.add_argument("--tree-max-depth", type=int, default=8)
    parser.add_argument("--tree-min-leaf", type=int, default=64)
    parser.add_argument("--tree-min-gain", type=float, default=1e-3)
    parser.add_argument("--gbdt-n-estimators", type=int, default=200)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-max-depth", type=int, default=3)
    parser.add_argument("--gbdt-min-samples-leaf", type=int, default=32)
    parser.add_argument("--gbdt-subsample", type=float, default=1.0)
    parser.add_argument(
        "--gbdt-backend",
        default="auto",
        choices=("auto", "sklearn", "xgboost"),
        help="GBDT implementation to use. auto preserves the historical sklearn-then-xgboost fallback.",
    )
    return parser


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _distance_to_future_path(asker_points, point_x: float, point_y: float) -> float:
    if not asker_points:
        return 999.0
    return min((((point_x - p.x) ** 2) + ((point_y - p.y) ** 2)) ** 0.5 for p in asker_points)


def _max_abs_step(points) -> float:
    if not points or len(points) < 2:
        return 0.0
    best = 0.0
    for i in range(1, len(points)):
        dx = points[i].x - points[i - 1].x
        dy = points[i].y - points[i - 1].y
        best = max(best, (dx * dx + dy * dy) ** 0.5)
    return best


def _extract_label(raw_record: dict[str, object]) -> int | None:
    conversations = raw_record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None
    answer = conversations[1].get("value", "")
    if not isinstance(answer, str):
        return None
    if NOT_NOTABLE_RE.search(answer):
        return 0
    if NOTABLE_RE.search(answer):
        return 1
    return None


@dataclass(frozen=True)
class Q6Row:
    features: list[float]
    label: int


FEATURE_NAMES = [
    "bias",
    "other_planned_final_dx",
    "other_planned_final_dy",
    "other_planned_final_dist",
    "other_planned_max_step",
    "other_min_distance_to_asker_path",
    "asker_path_length",
    "asker_path_max_step",
    "asker_nearby_object_count_5m",
    "asker_nearby_object_count_10m",
    "asker_nearby_dynamic_count_10m",
    "other_path_overlap_ratio_2m",
    "endpoint_distance_to_asker_final",
    "other_heading_alignment_with_asker_goal",
    "other_path_curvature_proxy",
    "other_ahead_of_asker_goal_flag",
]


def _build_rows(v2vgot_root: str, split: str, limit: int) -> list[Q6Row]:
    adapter = V2VGoTQABenchmarkAdapter(v2vgot_root)
    samples = adapter.load_samples(split_name=split)
    rows: list[Q6Row] = []
    for sample in samples:
        if int(sample.qa_type_id or -1) != 16:
            continue
        label = _extract_label(sample.raw_record)
        if label is None:
            continue

        scene = sample.scene
        asker = next((a for a in scene.agents if a.agent_id == scene.asker_agent_id), None)
        if asker is None:
            continue
        others = [a for a in scene.agents if a.agent_id != scene.asker_agent_id]
        if not others:
            continue
        other = min(
            others,
            key=lambda a: (((a.pose.position.x - asker.pose.position.x) ** 2) + ((a.pose.position.y - asker.pose.position.y) ** 2)) ** 0.5,
        )
        other_points = getattr(getattr(other, "planned_trajectory", None), "points", ()) or ()
        if other_points:
            final_dx = _safe_float(other_points[-1].x, 0.0)
            final_dy = _safe_float(other_points[-1].y, 0.0)
        else:
            final_dx = 0.0
            final_dy = 0.0
        final_dist = (final_dx * final_dx + final_dy * final_dy) ** 0.5
        other_max_step = _max_abs_step(other_points)

        asker_points = scene.future_trajectory.points
        min_dist_to_asker_path = _distance_to_future_path(
            asker_points,
            other.pose.position.x,
            other.pose.position.y,
        )
        asker_path_length = 0.0
        if len(asker_points) >= 2:
            for i in range(1, len(asker_points)):
                pdx = asker_points[i].x - asker_points[i - 1].x
                pdy = asker_points[i].y - asker_points[i - 1].y
                asker_path_length += (pdx * pdx + pdy * pdy) ** 0.5
        asker_path_max_step = _max_abs_step(asker_points)

        other_path_overlap_ratio_2m = 0.0
        if other_points:
            overlap = 0
            for p in other_points:
                if _distance_to_future_path(asker_points, p.x, p.y) <= 2.0:
                    overlap += 1
            other_path_overlap_ratio_2m = overlap / len(other_points)

        if asker_points:
            asker_final = asker_points[-1]
            endpoint_distance_to_asker_final = (
                ((other.pose.position.x + final_dx - asker_final.x) ** 2)
                + ((other.pose.position.y + final_dy - asker_final.y) ** 2)
            ) ** 0.5
            goal_vec_x = asker_final.x - asker.pose.position.x
            goal_vec_y = asker_final.y - asker.pose.position.y
            other_vec_x = final_dx
            other_vec_y = final_dy
            goal_norm = (goal_vec_x * goal_vec_x + goal_vec_y * goal_vec_y) ** 0.5
            other_norm = (other_vec_x * other_vec_x + other_vec_y * other_vec_y) ** 0.5
            if goal_norm > 1e-6 and other_norm > 1e-6:
                other_heading_alignment_with_asker_goal = (
                    (goal_vec_x * other_vec_x + goal_vec_y * other_vec_y) / (goal_norm * other_norm)
                )
            else:
                other_heading_alignment_with_asker_goal = 0.0
            rel_x = other.pose.position.x - asker.pose.position.x
            rel_y = other.pose.position.y - asker.pose.position.y
            proj = rel_x * goal_vec_x + rel_y * goal_vec_y
            other_ahead_of_asker_goal_flag = 1.0 if proj > 0.0 else 0.0
        else:
            endpoint_distance_to_asker_final = 999.0
            other_heading_alignment_with_asker_goal = 0.0
            other_ahead_of_asker_goal_flag = 0.0

        if len(other_points) >= 3:
            total_turn = 0.0
            turn_count = 0
            for i in range(1, len(other_points) - 1):
                ax = other_points[i].x - other_points[i - 1].x
                ay = other_points[i].y - other_points[i - 1].y
                bx = other_points[i + 1].x - other_points[i].x
                by = other_points[i + 1].y - other_points[i].y
                an = (ax * ax + ay * ay) ** 0.5
                bn = (bx * bx + by * by) ** 0.5
                if an > 1e-6 and bn > 1e-6:
                    cosv = max(-1.0, min(1.0, (ax * bx + ay * by) / (an * bn)))
                    total_turn += (1.0 - cosv)
                    turn_count += 1
            other_path_curvature_proxy = (total_turn / turn_count) if turn_count > 0 else 0.0
        else:
            other_path_curvature_proxy = 0.0

        nearby_5 = 0.0
        nearby_10 = 0.0
        nearby_dynamic_10 = 0.0
        for obj in scene.object_tracks:
            odx = obj.position.x - asker.pose.position.x
            ody = obj.position.y - asker.pose.position.y
            dist_to_asker = (odx * odx + ody * ody) ** 0.5
            if dist_to_asker <= 5.0:
                nearby_5 += 1.0
            if dist_to_asker <= 10.0:
                nearby_10 += 1.0
                v = obj.velocity
                if v is not None and ((v.x * v.x + v.y * v.y) ** 0.5) > 0.1:
                    nearby_dynamic_10 += 1.0

        rows.append(
            Q6Row(
                features=[
                    1.0,
                    final_dx,
                    final_dy,
                    final_dist,
                    other_max_step,
                    min_dist_to_asker_path,
                    asker_path_length,
                    asker_path_max_step,
                    nearby_5,
                    nearby_10,
                    nearby_dynamic_10,
                    other_path_overlap_ratio_2m,
                    endpoint_distance_to_asker_final,
                    other_heading_alignment_with_asker_goal,
                    other_path_curvature_proxy,
                    other_ahead_of_asker_goal_flag,
                ],
                label=label,
            )
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2_regularization: float) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=float)
    lr = 0.05
    for _ in range(800):
        p = _sigmoid(x @ w)
        grad = (x.T @ (p - y)) / x.shape[0]
        reg = l2_regularization * w
        reg[0] = 0.0
        grad += reg
        w -= lr * grad
    return w


def _gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    p1 = float(np.mean(labels))
    p0 = 1.0 - p1
    return 1.0 - (p0 * p0 + p1 * p1)


def _fit_tree_node(
    x: np.ndarray,
    y: np.ndarray,
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
    min_gain: float,
    split_feature_names: list[str],
) -> dict[str, object]:
    p1 = float(np.mean(y)) if y.size else 0.0
    prediction = 1 if p1 >= 0.5 else 0
    node = {"leaf": True, "prediction": prediction, "probability": p1}
    if depth >= max_depth or x.shape[0] < (2 * min_leaf):
        return node

    parent_impurity = _gini(y)
    best_gain = 0.0
    best_feature_idx = -1
    best_threshold = 0.0
    best_left_mask = None

    for feature_idx in range(x.shape[1]):
        values = x[:, feature_idx]
        quantiles = np.unique(np.quantile(values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        for threshold in quantiles:
            left_mask = values <= threshold
            right_mask = ~left_mask
            left_count = int(np.sum(left_mask))
            right_count = int(np.sum(right_mask))
            if left_count < min_leaf or right_count < min_leaf:
                continue
            left_imp = _gini(y[left_mask])
            right_imp = _gini(y[right_mask])
            weighted = (left_count / y.size) * left_imp + (right_count / y.size) * right_imp
            gain = parent_impurity - weighted
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = float(threshold)
                best_left_mask = left_mask

    if best_feature_idx < 0 or best_left_mask is None or best_gain < min_gain:
        return node

    left_mask = best_left_mask
    right_mask = ~left_mask
    left_child = _fit_tree_node(
        x[left_mask],
        y[left_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        min_gain=min_gain,
        split_feature_names=split_feature_names,
    )
    right_child = _fit_tree_node(
        x[right_mask],
        y[right_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        min_gain=min_gain,
        split_feature_names=split_feature_names,
    )
    return {
        "leaf": False,
        "feature_index": best_feature_idx,
        "feature_name": split_feature_names[best_feature_idx],
        "threshold": best_threshold,
        "gain": float(best_gain),
        "prediction": prediction,
        "probability": p1,
        "left": left_child,
        "right": right_child,
    }


def _tree_predict_prob(node: dict[str, object], features: np.ndarray) -> float:
    current = node
    while not bool(current.get("leaf", True)):
        idx = int(current.get("feature_index", -1))
        threshold = float(current.get("threshold", 0.0))
        if idx < 0 or idx >= features.shape[0]:
            break
        go_left = float(features[idx]) <= threshold
        nxt = current.get("left") if go_left else current.get("right")
        if not isinstance(nxt, dict):
            break
        current = nxt
    return float(current.get("probability", 0.0))


def _metrics_from_probs(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    acc = float(np.mean(preds == y))
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (prec + rec) == 0.0 else 2.0 * prec * rec / (prec + rec)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "positive_rate": float(np.mean(y)),
        "pred_positive_rate": float(np.mean(preds)),
    }


def _metrics(x: np.ndarray, y: np.ndarray, w: np.ndarray, threshold: float) -> dict[str, float]:
    probs = _sigmoid(x @ w)
    preds = (probs >= threshold).astype(int)
    acc = float(np.mean(preds == y))
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (prec + rec) == 0.0 else 2.0 * prec * rec / (prec + rec)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "positive_rate": float(np.mean(y)),
        "pred_positive_rate": float(np.mean(preds)),
    }


def main() -> None:
    args = build_parser().parse_args()
    rows = _build_rows(args.v2vgot_root, args.split, args.limit)
    if not rows:
        raise SystemExit("No Q6 rows found.")
    x = np.asarray([r.features for r in rows], dtype=float)
    y = np.asarray([r.label for r in rows], dtype=float)

    threshold = float(args.decision_threshold)
    if args.model_family == "regression_tree":
        split_feature_names = FEATURE_NAMES[1:]
        split_x = x[:, 1:]
        tree = _fit_tree_node(
            split_x,
            y,
            depth=0,
            max_depth=int(args.tree_max_depth),
            min_leaf=int(args.tree_min_leaf),
            min_gain=float(args.tree_min_gain),
            split_feature_names=split_feature_names,
        )
        probs = np.asarray([_tree_predict_prob(tree, row) for row in split_x], dtype=float)
        train_metrics = _metrics_from_probs(y, probs, threshold=threshold)
        model = {
            "model_type": "phase9_q6_agent_motion_notability_tree_v1",
            "split_feature_names": split_feature_names,
            "tree": tree,
            "decision_threshold": threshold,
            "tree_max_depth": int(args.tree_max_depth),
            "tree_min_leaf": int(args.tree_min_leaf),
            "tree_min_gain": float(args.tree_min_gain),
            "trained_split": args.split,
            "sample_count": int(x.shape[0]),
        }
    elif args.model_family == "gbdt":
        backend = str(args.gbdt_backend)
        if backend in {"auto", "sklearn"} and GradientBoostingClassifier is not None:
            clf = GradientBoostingClassifier(
                n_estimators=int(args.gbdt_n_estimators),
                learning_rate=float(args.gbdt_learning_rate),
                max_depth=int(args.gbdt_max_depth),
                min_samples_leaf=int(args.gbdt_min_samples_leaf),
                subsample=float(args.gbdt_subsample),
                random_state=42,
            )
            clf.fit(x, y.astype(int))
            probs = clf.predict_proba(x)[:, 1]
            train_metrics = _metrics_from_probs(y, probs, threshold=threshold)
            payload = base64.b64encode(pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
            model = {
                "model_type": "phase9_q6_agent_motion_notability_gbdt_v1",
                "feature_names": FEATURE_NAMES,
                "decision_threshold": threshold,
                "gbdt_backend": "sklearn",
                "gbdt_n_estimators": int(args.gbdt_n_estimators),
                "gbdt_learning_rate": float(args.gbdt_learning_rate),
                "gbdt_max_depth": int(args.gbdt_max_depth),
                "gbdt_min_samples_leaf": int(args.gbdt_min_samples_leaf),
                "gbdt_subsample": float(args.gbdt_subsample),
                "sklearn_pickle_b64": payload,
                "trained_split": args.split,
                "sample_count": int(x.shape[0]),
            }
        elif backend == "sklearn":
            raise SystemExit("gbdt-backend=sklearn requires scikit-learn installed.")
        elif backend in {"auto", "xgboost"} and xgb is not None:
            clf = xgb.XGBClassifier(
                n_estimators=int(args.gbdt_n_estimators),
                learning_rate=float(args.gbdt_learning_rate),
                max_depth=int(args.gbdt_max_depth),
                min_child_weight=max(1, int(args.gbdt_min_samples_leaf // 8)),
                subsample=float(args.gbdt_subsample),
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            )
            clf.fit(x, y.astype(int))
            probs = clf.predict_proba(x)[:, 1]
            train_metrics = _metrics_from_probs(y, probs, threshold=threshold)
            payload = base64.b64encode(pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
            model = {
                "model_type": "phase9_q6_agent_motion_notability_gbdt_v1",
                "feature_names": FEATURE_NAMES,
                "decision_threshold": threshold,
                "gbdt_backend": "xgboost",
                "gbdt_n_estimators": int(args.gbdt_n_estimators),
                "gbdt_learning_rate": float(args.gbdt_learning_rate),
                "gbdt_max_depth": int(args.gbdt_max_depth),
                "gbdt_min_samples_leaf": int(args.gbdt_min_samples_leaf),
                "gbdt_subsample": float(args.gbdt_subsample),
                "xgboost_pickle_b64": payload,
                "trained_split": args.split,
                "sample_count": int(x.shape[0]),
            }
        elif backend == "xgboost":
            raise SystemExit("gbdt-backend=xgboost requires xgboost installed.")
        else:
            raise SystemExit(
                "model_family=gbdt requires either sklearn or xgboost installed."
            )
    else:
        w = _fit_logistic(x, y, l2_regularization=float(args.l2_regularization))
        train_metrics = _metrics(x, y, w, threshold=threshold)
        model = {
            "model_type": "phase9_q6_agent_motion_notability_logistic_v1",
            "feature_names": FEATURE_NAMES,
            "weights": [float(v) for v in w.tolist()],
            "decision_threshold": threshold,
            "l2_regularization": float(args.l2_regularization),
            "trained_split": args.split,
            "sample_count": int(x.shape[0]),
        }

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")

    report = {
        "model_json": str(output_json),
        "model_family": args.model_family,
        "train_metrics": train_metrics,
        "sample_count": int(x.shape[0]),
    }
    report_path = Path(args.output_report).expanduser().resolve() if args.output_report else output_json.with_name(
        output_json.stem + "_train_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"q6_train_done samples={x.shape[0]}")
    print(
        "train_metrics "
        f"acc={train_metrics['accuracy']:.6f} "
        f"f1={train_metrics['f1']:.6f} "
        f"prec={train_metrics['precision']:.6f} "
        f"rec={train_metrics['recall']:.6f}"
    )
    print(f"model_json={output_json}")
    print(f"report_json={report_path}")


if __name__ == "__main__":
    main()
