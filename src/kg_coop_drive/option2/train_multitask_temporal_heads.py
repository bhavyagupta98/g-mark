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
TASKS = ("q5", "q6", "q7", "q9")


@dataclass
class HeadData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Option2 multitask temporal heads (defensible setup).")
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


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = np.eye(x.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0
    gram = x.T @ x + reg
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        # Numerical fallback for near-singular design matrices.
        return np.linalg.pinv(gram) @ rhs


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true))) if y_true.size else 0.0


def _build_q9_rows(samples, evaluator: V2VGoTQAPhase5AEvaluator, q8_model: dict[str, object], baseline_mode: str) -> tuple[list[list[float]], list[list[float]]]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for s in samples:
        if s.task_type != BenchmarkTaskType.FUTURE_TRAJECTORY or int(s.qa_type_id or -1) != 19:
            continue
        conv = s.raw_record.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[1], dict):
            continue
        gt = _parse_waypoints(str(conv[1].get("value", "")), limit=6)
        if len(gt) != 6:
            continue
        current_x, current_y = _parse_current_position(s.scene.raw_question)
        asker_is_cav1 = 1.0 if str(s.raw_record.get("asker_cav_id", "")).strip() == "1" else 0.0
        prepared = evaluator.prepare_sample(sample=s, baseline_mode=baseline_mode)
        q8_dec = decide_control_settings(scene=prepared, selection_policy="linear_classifier", model=q8_model)
        x = [1.0, current_x, current_y, asker_is_cav1] + _q8_vector(q8_dec.speed_instruction, q8_dec.steering_instruction)
        y = []
        for px, py in gt:
            y.extend([px - current_x, py - current_y])
        x_rows.append(x)
        y_rows.append(y)
    return x_rows, y_rows


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
    task_dim = len(TASKS)
    input_dim = task_dim + obj_dim + q6_dim + q9_dim

    # Build object motion rows (Q5/Q7)
    q5_train, _ = obj_builder.build(split_name=args.train_split, qa_type_ids=(15,), limit=args.train_limit, progress_every=args.progress_every)
    q7_train, _ = obj_builder.build(split_name=args.train_split, qa_type_ids=(17,), limit=args.train_limit, progress_every=args.progress_every)
    q5_val, _ = obj_builder.build(split_name=args.val_split, qa_type_ids=(15,), limit=args.val_limit, progress_every=args.progress_every)
    q7_val, _ = obj_builder.build(split_name=args.val_split, qa_type_ids=(17,), limit=args.val_limit, progress_every=args.progress_every)

    # Build Q6/Q9 rows
    adapter = obj_builder._adapter
    evaluator = V2VGoTQAPhase5AEvaluator(args.v2vgot_root)
    samples_train = tuple(adapter.load_samples(split_name=args.train_split, file_name=args.file_name))
    samples_val = tuple(adapter.load_samples(split_name=args.val_split, file_name=args.file_name))

    task_index = {name: idx for idx, name in enumerate(TASKS)}

    def _task_onehot(task_name: str) -> list[float]:
        return [1.0 if idx == task_index[task_name] else 0.0 for idx in range(task_dim)]

    def _make_unified_input(*, task_name: str, obj_block: list[float] | None, q6_block: list[float] | None, q9_block: list[float] | None) -> list[float]:
        obj = obj_block if obj_block is not None else [0.0] * obj_dim
        q6 = q6_block if q6_block is not None else [0.0] * q6_dim
        q9 = q9_block if q9_block is not None else [0.0] * q9_dim
        if len(obj) != obj_dim:
            raise ValueError(f"Object block width mismatch: expected={obj_dim} got={len(obj)}")
        if len(q6) != q6_dim:
            raise ValueError(f"Q6 block width mismatch: expected={q6_dim} got={len(q6)}")
        if len(q9) != q9_dim:
            raise ValueError(f"Q9 block width mismatch: expected={q9_dim} got={len(q9)}")
        return _task_onehot(task_name) + obj + q6 + q9

    q6_x_train, q6_y_train = [], []
    q6_x_val, q6_y_val = [], []
    for s in samples_train:
        if s.task_type == BenchmarkTaskType.AGENT_MOTION_PREDICTION and int(s.qa_type_id or -1) == 16:
            lbl = _q6_label(s)
            feats = _q6_features(s)
            if lbl is not None and feats is not None:
                q6_x_train.append(_make_unified_input(task_name="q6", obj_block=None, q6_block=feats, q9_block=None))
                q6_y_train.append([lbl])
    for s in samples_val:
        if s.task_type == BenchmarkTaskType.AGENT_MOTION_PREDICTION and int(s.qa_type_id or -1) == 16:
            lbl = _q6_label(s)
            feats = _q6_features(s)
            if lbl is not None and feats is not None:
                q6_x_val.append(_make_unified_input(task_name="q6", obj_block=None, q6_block=feats, q9_block=None))
                q6_y_val.append([lbl])

    q9_x_train, q9_y_train = _build_q9_rows(samples_train, evaluator=evaluator, q8_model=q8_model, baseline_mode=args.baseline_mode)
    q9_x_val, q9_y_val = _build_q9_rows(samples_val, evaluator=evaluator, q8_model=q8_model, baseline_mode=args.baseline_mode)
    q9_x_train = [_make_unified_input(task_name="q9", obj_block=None, q6_block=None, q9_block=v) for v in q9_x_train]
    q9_x_val = [_make_unified_input(task_name="q9", obj_block=None, q6_block=None, q9_block=v) for v in q9_x_val]

    q5_x_train = [_make_unified_input(task_name="q5", obj_block=list(r.feature_values), q6_block=None, q9_block=None) for r in q5_train]
    q5_y_train = [[r.target_dx, r.target_dy] for r in q5_train]
    q7_x_train = [_make_unified_input(task_name="q7", obj_block=list(r.feature_values), q6_block=None, q9_block=None) for r in q7_train]
    q7_y_train = [[r.target_dx, r.target_dy] for r in q7_train]
    q5_x_val = [_make_unified_input(task_name="q5", obj_block=list(r.feature_values), q6_block=None, q9_block=None) for r in q5_val]
    q5_y_val = [[r.target_dx, r.target_dy] for r in q5_val]
    q7_x_val = [_make_unified_input(task_name="q7", obj_block=list(r.feature_values), q6_block=None, q9_block=None) for r in q7_val]
    q7_y_val = [[r.target_dx, r.target_dy] for r in q7_val]

    heads: dict[str, HeadData] = {
        "q5": HeadData(np.asarray(q5_x_train, float), np.asarray(q5_y_train, float), np.asarray(q5_x_val, float), np.asarray(q5_y_val, float)),
        "q6": HeadData(np.asarray(q6_x_train, float), np.asarray(q6_y_train, float), np.asarray(q6_x_val, float), np.asarray(q6_y_val, float)),
        "q7": HeadData(np.asarray(q7_x_train, float), np.asarray(q7_y_train, float), np.asarray(q7_x_val, float), np.asarray(q7_y_val, float)),
        "q9": HeadData(np.asarray(q9_x_train, float), np.asarray(q9_y_train, float), np.asarray(q9_x_val, float), np.asarray(q9_y_val, float)),
    }

    model_heads: dict[str, object] = {}
    metrics: dict[str, object] = {}
    for name, data in heads.items():
        if data.x_train.size == 0 or data.y_train.size == 0:
            continue
        w = _ridge_fit(data.x_train, data.y_train, alpha=float(args.l2_alpha))
        pred_train = data.x_train @ w
        pred_val = data.x_val @ w if data.x_val.size else np.zeros_like(data.y_val)
        model_heads[name] = {
            "input_dim": int(data.x_train.shape[1]),
            "output_dim": int(data.y_train.shape[1]),
            "weights": w.tolist(),
        }
        head_metrics = {
            "train_rows": int(data.x_train.shape[0]),
            "val_rows": int(data.x_val.shape[0]),
            "train_mae": _mae(data.y_train, pred_train),
            "val_mae": _mae(data.y_val, pred_val),
        }
        if name == "q9" and data.y_val.size:
            e = np.linalg.norm(pred_val.reshape(-1, 6, 2) - data.y_val.reshape(-1, 6, 2), axis=2)
            head_metrics.update(
                {
                    "val_l2_error_avg_1s": float(np.mean(np.mean(e[:, :2], axis=1))),
                    "val_l2_error_avg_2s": float(np.mean(np.mean(e[:, :4], axis=1))),
                    "val_l2_error_avg_3s": float(np.mean(np.mean(e[:, :6], axis=1))),
                }
            )
        metrics[name] = head_metrics

    output = {
        "module": "option2_multitask_temporal_heads",
        "architecture": "shared_feature_space_plus_task_specific_heads",
        "input_shape": {
            "feature_count": int(input_dim),
            "task_onehot_block": int(task_dim),
            "q5q7_block": int(obj_dim),
            "q6_block": int(q6_dim),
            "q9_block": int(q9_dim),
        },
        "head_output_shapes": {
            "q5": 2,
            "q6": 1,
            "q7": 2,
            "q9": 12,
        },
        "include_kg_features": bool(args.include_kg_features),
        "l2_alpha": float(args.l2_alpha),
        "q8_model_json": str(q8_path),
        "metrics": metrics,
        "guards": {
            "q8_model_path_required": True,
            "fallback_used": False,
            "feature_source": "scene_and_q8_model_predictions_only",
        },
    }

    model = {
        "model_type": "option2_multitask_temporal_heads_ridge_v1",
        "input_dim": int(input_dim),
        "heads": model_heads,
        "include_kg_features": bool(args.include_kg_features),
    }

    out_model = Path(args.output_model_json).expanduser().resolve()
    out_report = Path(args.output_report_json).expanduser().resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(model, indent=2), encoding="utf-8")
    out_report.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved_model: {out_model}")
    print(f"saved_report: {out_report}")


if __name__ == "__main__":
    main()
