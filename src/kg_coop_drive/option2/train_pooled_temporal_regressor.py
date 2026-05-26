from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kg_coop_drive.application.control_settings_policy import decide_control_settings
from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator
from kg_coop_drive.domain.benchmark import BenchmarkTaskType
from kg_coop_drive.option2.dataset import ObjectMotionDatasetBuilder

SPEED_LABELS = ("fast", "moderate", "slow", "very slow", "stop")
STEER_LABELS = ("left", "slightly left", "straight", "slightly right", "right")
Q8_SPEED_CONTROL_VALUES = {"fast": 1.0, "moderate": 0.65, "slow": 0.35, "very slow": 0.15, "stop": 0.0}
Q8_STEERING_CONTROL_VALUES = {"left": -1.0, "slightly left": -0.5, "straight": 0.0, "slightly right": 0.5, "right": 1.0}

POSITION_RE = re.compile(
    r"I am\s+(?P<agent>[A-Za-z0-9_]+)\s+at\s+"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
)
COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
NOTABLE_RE = re.compile(r"\bis a notable object\b", re.IGNORECASE)
NOT_NOTABLE_RE = re.compile(r"\bis a not notable object\b", re.IGNORECASE)

TASKS = ("q5", "q6", "q7", "q9")
Q6_FEATURES = (
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
)


@dataclass
class Row:
    x: list[float]
    y: list[float]
    mask: list[float]
    task: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Option2 pooled temporal regressor (Q5/Q6/Q7/Q9).")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--train-split", default="train", choices=("train", "val"))
    parser.add_argument("--val-split", default="val", choices=("train", "val"))
    parser.add_argument("--q8-model-json", required=True)
    parser.add_argument("--include-kg-features", action="store_true")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--graph-ablation-mode", default="full")
    parser.add_argument("--max-match-distance", type=float, default=2.0)
    parser.add_argument("--l2-alpha", type=float, default=1.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    return parser


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_current_position(question: str) -> tuple[float, float]:
    m = POSITION_RE.search(question)
    if m:
        return float(m.group("x")), float(m.group("y"))
    return (0.0, 0.0)


def _parse_waypoints(text: str, limit: int = 6) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in COORD_RE.findall(text)[:limit])


def _q8_vector(speed_label: str, steering_label: str) -> list[float]:
    s = speed_label.strip().lower()
    t = steering_label.strip().lower()
    speed_hot = [1.0 if label == s else 0.0 for label in SPEED_LABELS]
    steer_hot = [1.0 if label == t else 0.0 for label in STEER_LABELS]
    return speed_hot + steer_hot + [Q8_SPEED_CONTROL_VALUES.get(s, 0.0), Q8_STEERING_CONTROL_VALUES.get(t, 0.0)]


def _q9_features(sample, q8_model: dict[str, object], evaluator: V2VGoTQAPhase5AEvaluator, baseline_mode: str) -> tuple[list[float], list[float]] | None:
    conv = sample.raw_record.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[1], dict):
        return None
    gt_points = _parse_waypoints(str(conv[1].get("value", "")), limit=6)
    if len(gt_points) != 6:
        return None
    current_x, current_y = _parse_current_position(sample.scene.raw_question)
    asker_is_cav1 = 1.0 if str(sample.raw_record.get("asker_cav_id", "")).strip() == "1" else 0.0
    prepared = evaluator.prepare_sample(sample=sample, baseline_mode=baseline_mode)
    q8_decision = decide_control_settings(scene=prepared, selection_policy="linear_classifier", model=q8_model)
    q8 = _q8_vector(q8_decision.speed_instruction, q8_decision.steering_instruction)
    x = [1.0, current_x, current_y, asker_is_cav1] + q8
    y = []
    for px, py in gt_points:
        y.extend([px - current_x, py - current_y])
    return x, y


def _q6_label(sample) -> float | None:
    conv = sample.raw_record.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[1], dict):
        return None
    ans = str(conv[1].get("value", ""))
    if NOT_NOTABLE_RE.search(ans):
        return 0.0
    if NOTABLE_RE.search(ans):
        return 1.0
    return None


def _q6_features(sample) -> list[float] | None:
    scene = sample.scene
    asker = next((a for a in scene.agents if a.agent_id == scene.asker_agent_id), None)
    if asker is None:
        return None
    others = [a for a in scene.agents if a.agent_id != scene.asker_agent_id]
    if not others:
        return None
    other = min(
        others,
        key=lambda a: (((a.pose.position.x - asker.pose.position.x) ** 2) + ((a.pose.position.y - asker.pose.position.y) ** 2)) ** 0.5,
    )
    other_points = getattr(getattr(other, "planned_trajectory", None), "points", ()) or ()
    final_dx = _safe_float(other_points[-1].x, 0.0) if other_points else 0.0
    final_dy = _safe_float(other_points[-1].y, 0.0) if other_points else 0.0
    final_dist = math.hypot(final_dx, final_dy)

    def max_step(points) -> float:
        if not points or len(points) < 2:
            return 0.0
        return max(math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y) for i in range(1, len(points)))

    asker_points = scene.future_trajectory.points

    def dist_path(px: float, py: float) -> float:
        if not asker_points:
            return 999.0
        return min(math.hypot(px - p.x, py - p.y) for p in asker_points)

    min_dist = dist_path(other.pose.position.x, other.pose.position.y)
    asker_len = 0.0
    if len(asker_points) >= 2:
        for i in range(1, len(asker_points)):
            asker_len += math.hypot(asker_points[i].x - asker_points[i - 1].x, asker_points[i].y - asker_points[i - 1].y)

    overlap_ratio = 0.0
    if other_points:
        overlap = sum(1 for p in other_points if dist_path(p.x, p.y) <= 2.0)
        overlap_ratio = overlap / len(other_points)

    if asker_points:
        af = asker_points[-1]
        endpoint_dist = math.hypot(other.pose.position.x + final_dx - af.x, other.pose.position.y + final_dy - af.y)
        gvx, gvy = af.x - asker.pose.position.x, af.y - asker.pose.position.y
        ovx, ovy = final_dx, final_dy
        gn, on = math.hypot(gvx, gvy), math.hypot(ovx, ovy)
        heading = (gvx * ovx + gvy * ovy) / (gn * on) if gn > 1e-6 and on > 1e-6 else 0.0
        relx, rely = other.pose.position.x - asker.pose.position.x, other.pose.position.y - asker.pose.position.y
        ahead = 1.0 if (relx * gvx + rely * gvy) > 0.0 else 0.0
    else:
        endpoint_dist = 999.0
        heading = 0.0
        ahead = 0.0

    nearby5 = nearby10 = dynamic10 = 0.0
    for obj in scene.object_tracks:
        d = math.hypot(obj.position.x - asker.pose.position.x, obj.position.y - asker.pose.position.y)
        if d <= 5.0:
            nearby5 += 1.0
        if d <= 10.0:
            nearby10 += 1.0
            v = obj.velocity
            if v is not None and math.hypot(v.x, v.y) > 0.1:
                dynamic10 += 1.0

    return [
        1.0, final_dx, final_dy, final_dist, max_step(other_points), min_dist, asker_len, max_step(asker_points),
        nearby5, nearby10, dynamic10, overlap_ratio, endpoint_dist, heading, 0.0, ahead,
    ]


def _task_onehot(task: str) -> list[float]:
    return [1.0 if task == key else 0.0 for key in TASKS]


def _ridge_fit_per_dim(x: np.ndarray, y: np.ndarray, mask: np.ndarray, alpha: float) -> tuple[list[list[float]], np.ndarray]:
    # fits one linear ridge model per output dim using only rows where mask[:,d]==1
    dim = y.shape[1]
    coefs: list[list[float]] = []
    pred = np.zeros_like(y)
    for d in range(dim):
        idx = np.where(mask[:, d] > 0.5)[0]
        if idx.size == 0:
            coefs.append([0.0] * x.shape[1])
            continue
        xd = x[idx]
        yd = y[idx, d]
        reg = np.eye(x.shape[1], dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        w = np.linalg.solve(xd.T @ xd + reg, xd.T @ yd)
        coefs.append(w.tolist())
        pred[:, d] = x @ w
    return coefs, pred


def _metrics_for(rows: list[Row], y_pred: np.ndarray) -> dict[str, float]:
    if not rows:
        return {"count": 0.0}
    y_true = np.asarray([r.y for r in rows], dtype=float)
    mask = np.asarray([r.mask for r in rows], dtype=float)
    err = np.abs(y_pred - y_true) * mask
    denom = np.maximum(mask.sum(), 1.0)
    mae = float(err.sum() / denom)
    return {"count": float(len(rows)), "masked_mae": mae}


def main() -> None:
    args = build_parser().parse_args()
    q8_path = Path(args.q8_model_json).expanduser().resolve()
    if not q8_path.exists():
        raise SystemExit(f"Required Q8 model not found: {q8_path}")
    q8_model = json.loads(q8_path.read_text(encoding="utf-8"))

    obj_builder = ObjectMotionDatasetBuilder(
        v2vgot_root=args.v2vgot_root,
        file_name=args.file_name,
        baseline_mode=args.baseline_mode,
        graph_ablation_mode=args.graph_ablation_mode,
        max_match_distance=args.max_match_distance,
        include_kg_features=args.include_kg_features,
    )
    obj_dim = len(obj_builder.feature_names)
    q6_dim = len(Q6_FEATURES)
    q9_dim = 4 + len(SPEED_LABELS) + len(STEER_LABELS) + 2
    input_dim = 4 + obj_dim + q6_dim + q9_dim
    output_dim = 13  # 12 q9 waypoint deltas + 1 q6 notability score

    # Q5/Q7 rows
    q5_rows, _ = obj_builder.build(split_name=args.train_split, qa_type_ids=(15,), limit=args.train_limit, progress_every=args.progress_every)
    q7_rows, _ = obj_builder.build(split_name=args.train_split, qa_type_ids=(17,), limit=args.train_limit, progress_every=args.progress_every)
    q5_rows_val, _ = obj_builder.build(split_name=args.val_split, qa_type_ids=(15,), limit=args.val_limit, progress_every=args.progress_every)
    q7_rows_val, _ = obj_builder.build(split_name=args.val_split, qa_type_ids=(17,), limit=args.val_limit, progress_every=args.progress_every)

    evaluator = V2VGoTQAPhase5AEvaluator(args.v2vgot_root)
    adapter = obj_builder._adapter  # reuse initialized adapter

    def build_split(split: str, is_train: bool) -> list[Row]:
        rows: list[Row] = []
        if is_train:
            source_q5 = q5_rows
            source_q7 = q7_rows
        else:
            source_q5 = q5_rows_val
            source_q7 = q7_rows_val

        for item in source_q5:
            x = _task_onehot("q5") + list(item.feature_values) + [0.0] * q6_dim + [0.0] * q9_dim
            y = [0.0] * output_dim
            mask = [0.0] * output_dim
            y[0], y[1] = item.target_dx, item.target_dy
            mask[0] = 1.0
            mask[1] = 1.0
            rows.append(Row(x=x, y=y, mask=mask, task="q5"))

        for item in source_q7:
            x = _task_onehot("q7") + list(item.feature_values) + [0.0] * q6_dim + [0.0] * q9_dim
            y = [0.0] * output_dim
            mask = [0.0] * output_dim
            y[0], y[1] = item.target_dx, item.target_dy
            mask[0] = 1.0
            mask[1] = 1.0
            rows.append(Row(x=x, y=y, mask=mask, task="q7"))

        samples = tuple(s for s in adapter.load_samples(split_name=split, file_name=args.file_name))
        # Q6
        for s in samples:
            if s.task_type != BenchmarkTaskType.AGENT_MOTION_PREDICTION or int(s.qa_type_id or -1) != 16:
                continue
            label = _q6_label(s)
            feats = _q6_features(s)
            if label is None or feats is None:
                continue
            x = _task_onehot("q6") + [0.0] * obj_dim + feats + [0.0] * q9_dim
            y = [0.0] * output_dim
            mask = [0.0] * output_dim
            y[12] = label
            mask[12] = 1.0
            rows.append(Row(x=x, y=y, mask=mask, task="q6"))

        # Q9
        for s in samples:
            if s.task_type != BenchmarkTaskType.FUTURE_TRAJECTORY or int(s.qa_type_id or -1) != 19:
                continue
            out = _q9_features(s, q8_model=q8_model, evaluator=evaluator, baseline_mode=args.baseline_mode)
            if out is None:
                continue
            q9_x, q9_y = out
            x = _task_onehot("q9") + [0.0] * obj_dim + [0.0] * q6_dim + q9_x
            y = [0.0] * output_dim
            mask = [0.0] * output_dim
            for i in range(12):
                y[i] = q9_y[i]
                mask[i] = 1.0
            rows.append(Row(x=x, y=y, mask=mask, task="q9"))

        return rows

    print("[1/3] Building pooled train rows...")
    train_rows = build_split(args.train_split, is_train=True)
    print("[2/3] Building pooled val rows...")
    val_rows = build_split(args.val_split, is_train=False)
    if not train_rows or not val_rows:
        raise SystemExit("No pooled rows built.")

    x_train = np.asarray([r.x for r in train_rows], dtype=float)
    y_train = np.asarray([r.y for r in train_rows], dtype=float)
    m_train = np.asarray([r.mask for r in train_rows], dtype=float)
    x_val = np.asarray([r.x for r in val_rows], dtype=float)
    y_val = np.asarray([r.y for r in val_rows], dtype=float)
    m_val = np.asarray([r.mask for r in val_rows], dtype=float)

    print("[3/3] Fitting pooled ridge regressor...")
    coef_by_dim, train_pred = _ridge_fit_per_dim(x_train, y_train, m_train, alpha=float(args.l2_alpha))
    val_pred = np.zeros_like(y_val)
    for d, coeff in enumerate(coef_by_dim):
        w = np.asarray(coeff, dtype=float)
        val_pred[:, d] = x_val @ w

    by_task_train: dict[str, list[Row]] = {k: [] for k in TASKS}
    by_task_val: dict[str, list[Row]] = {k: [] for k in TASKS}
    for r in train_rows:
        by_task_train[r.task].append(r)
    for r in val_rows:
        by_task_val[r.task].append(r)

    # get per-task preds slice
    def task_metrics(rows: list[Row], pred_all: np.ndarray, all_rows: list[Row]) -> dict[str, float]:
        if not rows:
            return {"count": 0.0}
        idx = [all_rows.index(r) for r in rows]
        return _metrics_for(rows, pred_all[idx])

    report = {
        "module": "option2_pooled_temporal_regressor",
        "task_scope": ["q5", "q6", "q7", "q9"],
        "input_shape": {
            "feature_count": int(input_dim),
            "formula": f"4(task_onehot)+{obj_dim}(q5q7_block)+{q6_dim}(q6_block)+{q9_dim}(q9_block)",
        },
        "output_shape": {
            "target_count": int(output_dim),
            "layout": "[0..11]=q9 six-waypoint dx/dy, [12]=q6_notability_regression",
        },
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "include_kg_features": bool(args.include_kg_features),
        "l2_alpha": float(args.l2_alpha),
        "train_metrics_overall": _metrics_for(train_rows, train_pred),
        "val_metrics_overall": _metrics_for(val_rows, val_pred),
        "train_metrics_by_task": {
            k: task_metrics(by_task_train[k], train_pred, train_rows) for k in TASKS
        },
        "val_metrics_by_task": {
            k: task_metrics(by_task_val[k], val_pred, val_rows) for k in TASKS
        },
        "guards": {
            "q8_model_path_required": str(q8_path),
            "fallback_used": False,
        },
    }

    model = {
        "model_type": "option2_pooled_temporal_ridge_v1",
        "input_feature_count": int(input_dim),
        "output_target_count": int(output_dim),
        "include_kg_features": bool(args.include_kg_features),
        "l2_alpha": float(args.l2_alpha),
        "weights_by_output_dim": coef_by_dim,
    }

    out_model = Path(args.output_model_json).expanduser().resolve()
    out_report = Path(args.output_report_json).expanduser().resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(model, indent=2), encoding="utf-8")
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved_model: {out_model}")
    print(f"saved_report: {out_report}")


if __name__ == "__main__":
    main()
