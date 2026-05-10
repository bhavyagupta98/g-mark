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
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    from sklearn.neural_network import MLPRegressor  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    GradientBoostingRegressor = None  # type: ignore
    MLPRegressor = None  # type: ignore
    make_pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.scene import VisibilityState  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

Q5_OBJECT_RE = re.compile(
    r"There is a car at "
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)\s+"
    r"(?P<action>[^.]+?)\.\s*"
    r"The predicted future trajectory is\s*\[(?P<trajectory>[^\]]+)\]",
    re.IGNORECASE | re.DOTALL,
)
POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


class ObjectMotionTrainerEntrypoint:
    qa_type_id = 15
    task_label = "Q5"

    def build_parser(self) -> argparse.ArgumentParser:
        return build_parser(task_label=self.task_label)

    def main(self) -> None:
        run_training(
            args=self.build_parser().parse_args(),
            qa_type_id=self.qa_type_id,
            task_label=self.task_label,
        )


class Q5ObjectMotionTrainer(ObjectMotionTrainerEntrypoint):
    qa_type_id = 15
    task_label = "Q5"


def build_parser(task_label: str = "Q5") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train frozen {task_label} object-motion endpoint model.")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--l2-regularization", type=float, default=1e-3)
    parser.add_argument("--max-match-distance", type=float, default=2.0)
    parser.add_argument("--max-abs-delta", type=float, default=120.0)
    parser.add_argument(
        "--model-family",
        default="linear",
        choices=("linear", "piecewise_linear", "regression_tree", "gradient_boosting", "mlp"),
        help=f"{task_label} model family to train.",
    )
    parser.add_argument("--piecewise-min-rows", type=int, default=128)
    parser.add_argument("--tree-max-depth", type=int, default=6)
    parser.add_argument("--tree-min-leaf", type=int, default=64)
    parser.add_argument("--tree-min-gain", type=float, default=0.01)
    parser.add_argument("--gbdt-n-estimators", type=int, default=240)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.04)
    parser.add_argument("--gbdt-max-depth", type=int, default=2)
    parser.add_argument("--gbdt-min-samples-leaf", type=int, default=64)
    parser.add_argument("--gbdt-subsample", type=float, default=0.7)
    parser.add_argument("--mlp-hidden-layer-sizes", default="64,32")
    parser.add_argument("--mlp-alpha", type=float, default=1e-3)
    parser.add_argument("--mlp-learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument(
        "--feature-set",
        default="auto",
        choices=("auto", "base", "path_relative"),
        help=(
            "Feature family. auto preserves current task defaults: Q5 uses base, "
            "Q7 uses path-relative interaction features."
        ),
    )
    parser.add_argument("--selection-max-objects", type=int, default=3)
    parser.add_argument("--selection-max-distance-to-trajectory", type=float, default=8.0)
    parser.set_defaults(selection_include_occluded_uncertain=True)
    parser.add_argument(
        "--selection-include-occluded-uncertain",
        dest="selection_include_occluded_uncertain",
        action="store_true",
    )
    parser.add_argument(
        "--no-selection-include-occluded-uncertain",
        dest="selection_include_occluded_uncertain",
        action="store_false",
    )
    return parser


def target_waypoint_count_for_qa_type(qa_type_id: int) -> int:
    return 1


def parse_q5_gt_answer(answer_text: str, waypoint_count: int) -> list[tuple[float, float, list[tuple[float, float]]]]:
    rows: list[tuple[float, float, list[tuple[float, float]]]] = []
    for match in Q5_OBJECT_RE.finditer(answer_text):
        points = [
            (float(raw_x), float(raw_y))
            for raw_x, raw_y in POINT_RE.findall(str(match.group("trajectory")))
        ]
        if not points:
            continue
        if len(points) < waypoint_count:
            points = points + [points[-1]] * (waypoint_count - len(points))
        points = points[:waypoint_count]
        rows.append(
            (
                float(match.group("x")),
                float(match.group("y")),
                points,
            )
        )
    return rows


def distance_to_trajectory(scene, object_track) -> float:
    if not scene.future_trajectory.points:
        return 999.0
    return min(
        ((object_track.position.x - point.x) ** 2 + (object_track.position.y - point.y) ** 2) ** 0.5
        for point in scene.future_trajectory.points
    )


def distance_to_asker(scene, object_track) -> float:
    asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
    if asker is None:
        return 999.0
    return (
        (
            (object_track.position.x - asker.pose.position.x) ** 2
            + (object_track.position.y - asker.pose.position.y) ** 2
        )
        ** 0.5
    )


def visibility_lookup(scene) -> dict[str, VisibilityState]:
    return {
        fact.object_id: fact.state
        for fact in scene.visibility_facts
        if fact.agent_id == scene.asker_agent_id
    }


FEATURE_NAMES = [
    "bias",
    "x",
    "y",
    "vx",
    "vy",
    "speed",
    "distance_to_trajectory",
    "distance_to_asker",
    "confidence",
    "support_count",
    "conflict_score",
    "uncertainty_score",
    "status_supported",
    "status_candidate",
    "visibility_visible",
    "visibility_occluded",
    "visibility_uncertain",
]

PATH_RELATIVE_FEATURE_NAMES = [
    "relative_x_to_asker",
    "relative_y_to_asker",
    "asker_goal_x_from_object",
    "asker_goal_y_from_object",
    "distance_to_asker_goal",
    "closest_asker_path_x_from_object",
    "closest_asker_path_y_from_object",
    "closest_asker_path_index_norm",
    "object_vx_along_asker_goal",
    "object_vy_lateral_to_asker_goal",
    "object_velocity_toward_asker_goal",
]


def resolve_feature_set(qa_type_id: int, raw_feature_set: str) -> str:
    if raw_feature_set != "auto":
        return raw_feature_set
    return "path_relative" if qa_type_id == 17 else "base"


def feature_names_for_qa_type(qa_type_id: int, feature_set: str = "auto") -> list[str]:
    resolved_feature_set = resolve_feature_set(qa_type_id, feature_set)
    if resolved_feature_set == "path_relative":
        return FEATURE_NAMES + PATH_RELATIVE_FEATURE_NAMES
    return FEATURE_NAMES


def feature_map(scene, object_track, visibility_state: VisibilityState | None) -> dict[str, float]:
    velocity = object_track.velocity
    vx = 0.0 if velocity is None else float(velocity.x)
    vy = 0.0 if velocity is None else float(velocity.y)
    speed = (vx * vx + vy * vy) ** 0.5
    status = str(object_track.status.value)
    support_count = 0.0
    if hasattr(object_track, "supporting_agent_ids") and getattr(object_track, "supporting_agent_ids") is not None:
        support_count = float(len(getattr(object_track, "supporting_agent_ids")))
    elif hasattr(object_track, "provenance") and getattr(object_track, "provenance") is not None:
        provenance = getattr(object_track, "provenance")
        source_agent_ids = getattr(provenance, "source_agent_ids", ())
        support_count = float(len(source_agent_ids))
    elif hasattr(object_track, "observations") and getattr(object_track, "observations") is not None:
        support_count = float(len(getattr(object_track, "observations")))

    values = {
        "bias": 1.0,
        "x": float(object_track.position.x),
        "y": float(object_track.position.y),
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "distance_to_trajectory": float(distance_to_trajectory(scene, object_track)),
        "distance_to_asker": float(distance_to_asker(scene, object_track)),
        "confidence": float(object_track.confidence),
        "support_count": support_count,
        "conflict_score": float(object_track.conflict_score),
        "uncertainty_score": float(object_track.uncertainty_score),
        "status_supported": 1.0 if status == "supported" else 0.0,
        "status_candidate": 1.0 if status == "candidate" else 0.0,
        "visibility_visible": 1.0 if visibility_state == VisibilityState.VISIBLE else 0.0,
        "visibility_occluded": 1.0 if visibility_state == VisibilityState.OCCLUDED else 0.0,
        "visibility_uncertain": 1.0 if visibility_state == VisibilityState.UNCERTAIN else 0.0,
    }
    values.update(path_relative_feature_map(scene, object_track, vx=vx, vy=vy))
    return values


def feature_vector(scene, object_track, visibility_state: VisibilityState | None, feature_names: list[str]) -> list[float]:
    values = feature_map(scene, object_track, visibility_state)
    return [float(values.get(name, 0.0)) for name in feature_names]


def path_relative_feature_map(scene, object_track, *, vx: float, vy: float) -> dict[str, float]:
    asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
    if asker is None:
        asker_x = object_track.position.x
        asker_y = object_track.position.y
    else:
        asker_x = asker.pose.position.x
        asker_y = asker.pose.position.y

    rel_x = float(object_track.position.x - asker_x)
    rel_y = float(object_track.position.y - asker_y)
    future_points = tuple(scene.future_trajectory.points)
    if future_points:
        final_x = float(future_points[-1].x)
        final_y = float(future_points[-1].y)
        closest_idx = min(
            range(len(future_points)),
            key=lambda idx: (object_track.position.x - future_points[idx].x) ** 2
            + (object_track.position.y - future_points[idx].y) ** 2,
        )
        closest_point = future_points[closest_idx]
        closest_norm = float(closest_idx) / float(max(1, len(future_points) - 1))
    else:
        final_x = asker_x
        final_y = asker_y
        closest_point = None
        closest_norm = 0.0

    goal_dx = float(final_x - object_track.position.x)
    goal_dy = float(final_y - object_track.position.y)
    goal_dist = (goal_dx * goal_dx + goal_dy * goal_dy) ** 0.5
    if goal_dist > 1e-6:
        unit_x = goal_dx / goal_dist
        unit_y = goal_dy / goal_dist
        vel_toward_goal = vx * unit_x + vy * unit_y
        vel_lateral_goal = -vx * unit_y + vy * unit_x
    else:
        vel_toward_goal = 0.0
        vel_lateral_goal = 0.0

    if closest_point is None:
        closest_dx = 0.0
        closest_dy = 0.0
    else:
        closest_dx = float(closest_point.x - object_track.position.x)
        closest_dy = float(closest_point.y - object_track.position.y)

    return {
        "relative_x_to_asker": rel_x,
        "relative_y_to_asker": rel_y,
        "asker_goal_x_from_object": goal_dx,
        "asker_goal_y_from_object": goal_dy,
        "distance_to_asker_goal": goal_dist,
        "closest_asker_path_x_from_object": closest_dx,
        "closest_asker_path_y_from_object": closest_dy,
        "closest_asker_path_index_norm": closest_norm,
        "object_vx_along_asker_goal": vel_toward_goal,
        "object_vy_lateral_to_asker_goal": vel_lateral_goal,
        "object_velocity_toward_asker_goal": vel_toward_goal,
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, l2_regularization: float) -> np.ndarray:
    xtx = x.T @ x
    if l2_regularization > 0.0:
        reg = np.eye(xtx.shape[0], dtype=float) * float(l2_regularization)
        reg[0, 0] = 0.0
        xtx = xtx + reg
    xty = x.T @ y
    return np.linalg.solve(xtx, xty)


def bucket_speed(speed: float) -> str:
    if speed < 1.0:
        return "slow"
    if speed < 5.0:
        return "mid"
    return "fast"


def bucket_traj(distance_to_trajectory_value: float) -> str:
    if distance_to_trajectory_value < 5.0:
        return "near"
    if distance_to_trajectory_value < 15.0:
        return "mid"
    return "far"


def bucket_visibility(visibility_state: VisibilityState | None) -> str:
    if visibility_state == VisibilityState.VISIBLE:
        return "visible"
    if visibility_state == VisibilityState.OCCLUDED:
        return "occluded"
    if visibility_state == VisibilityState.UNCERTAIN:
        return "uncertain"
    return "unknown"


@dataclass(frozen=True)
class Q5TrainingRow:
    features: list[float]
    target_values: list[float]
    piecewise_key: str


def _sse(targets: np.ndarray) -> float:
    if len(targets) == 0:
        return 0.0
    centered = targets - np.mean(targets, axis=0, keepdims=True)
    return float(np.sum(centered * centered))


def _fit_regression_tree_node(
    x: np.ndarray,
    y: np.ndarray,
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
    min_gain: float,
    split_feature_names: list[str],
) -> dict[str, object]:
    prediction = [float(value) for value in np.mean(y, axis=0).tolist()]
    if depth >= max_depth or len(x) < (2 * min_leaf):
        return {"leaf": True, "prediction": prediction}

    parent_sse = _sse(y)
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
            gain = parent_sse - (_sse(y[left_mask]) + _sse(y[right_mask]))
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = float(threshold)
                best_left_mask = left_mask

    if best_feature_idx < 0 or best_left_mask is None or best_gain < min_gain:
        return {"leaf": True, "prediction": prediction}

    left_mask = best_left_mask
    right_mask = ~left_mask
    left_child = _fit_regression_tree_node(
        x[left_mask],
        y[left_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        min_gain=min_gain,
        split_feature_names=split_feature_names,
    )
    right_child = _fit_regression_tree_node(
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
        "left": left_child,
        "right": right_child,
        "prediction": prediction,
    }


def _predict_regression_tree(node: dict[str, object], row: np.ndarray) -> np.ndarray:
    current = node
    while not bool(current.get("leaf", True)):
        idx = int(current["feature_index"])
        threshold = float(current["threshold"])
        go_left = float(row[idx]) <= threshold
        next_node = current["left"] if go_left else current["right"]
        if not isinstance(next_node, dict):
            break
        current = next_node
    prediction = current.get("prediction", [0.0, 0.0])
    if not isinstance(prediction, list):
        return np.zeros(2, dtype=float)
    return np.asarray([float(value) for value in prediction], dtype=float)


def parse_hidden_layer_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise SystemExit("--mlp-hidden-layer-sizes must contain positive integers, e.g. 64,32.")
    return sizes


def run_training(*, args: argparse.Namespace, qa_type_id: int, task_label: str) -> None:
    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    evaluator = V2VGoTQAPhase5AEvaluator(args.v2vgot_root)

    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name=args.split)
        if sample.task_type == BenchmarkTaskType.OBJECT_MOTION_PREDICTION
        and sample.qa_type_id == qa_type_id
    )
    if args.limit > 0:
        samples = samples[: args.limit]

    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    training_rows: list[Q5TrainingRow] = []
    gt_row_count = 0
    matched_row_count = 0
    match_distances: list[float] = []
    target_waypoint_count = target_waypoint_count_for_qa_type(qa_type_id)
    feature_set = resolve_feature_set(qa_type_id, str(args.feature_set))
    feature_names = feature_names_for_qa_type(qa_type_id, feature_set)

    for sample in samples:
        conversations = sample.raw_record.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2 or not isinstance(conversations[1], dict):
            continue
        gt_answer = str(conversations[1].get("value", ""))
        gt_rows = parse_q5_gt_answer(gt_answer, waypoint_count=target_waypoint_count)
        if not gt_rows:
            continue
        gt_row_count += len(gt_rows)

        scene = evaluator.prepare_sample(sample=sample, baseline_mode=args.baseline_mode)
        if not scene.object_tracks:
            continue
        vis_by_object = visibility_lookup(scene)

        available_tracks = list(scene.object_tracks)
        for gt_x, gt_y, gt_points in gt_rows:
            if not available_tracks:
                break
            best_idx = -1
            best_distance = float("inf")
            for idx, track in enumerate(available_tracks):
                d = ((track.position.x - gt_x) ** 2 + (track.position.y - gt_y) ** 2) ** 0.5
                if d < best_distance:
                    best_distance = d
                    best_idx = idx
            if best_idx < 0 or best_distance > args.max_match_distance:
                continue

            track = available_tracks.pop(best_idx)
            vis_state = vis_by_object.get(track.object_id)
            features = feature_vector(scene, track, vis_state, feature_names)
            target_values: list[float] = []
            for gt_tx, gt_ty in gt_points:
                target_values.extend(
                    [
                        float(gt_tx - track.position.x),
                        float(gt_ty - track.position.y),
                    ]
                )
            piecewise_key = (
                f"speed={bucket_speed(features[5])}|"
                f"traj={bucket_traj(features[6])}|"
                f"visibility={bucket_visibility(vis_state)}|"
                f"status={'supported' if features[12] > 0.5 else 'candidate' if features[13] > 0.5 else 'other'}"
            )
            training_rows.append(
                Q5TrainingRow(
                    features=features,
                    target_values=target_values,
                    piecewise_key=piecewise_key,
                )
            )
            x_rows.append(features)
            y_rows.append(target_values)
            matched_row_count += 1
            match_distances.append(best_distance)

    if not x_rows:
        raise SystemExit("No matched object-motion training rows found. Increase --max-match-distance or check input data.")

    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    global_coefficients = fit_ridge(x, y, args.l2_regularization)  # [feature_count, 2]

    if args.model_family == "linear":
        y_pred = x @ global_coefficients
        model = {
            "model_type": "phase9_q5_object_motion_linear_v1",
            "source_split": args.split,
            "source_qa_type_id": int(qa_type_id),
            "source_task_label": task_label,
            "baseline_mode": args.baseline_mode,
            "feature_set": feature_set,
            "feature_names": feature_names,
            "coefficients": global_coefficients.T.tolist(),  # [2, feature_count]
            "target_waypoint_count": int(target_waypoint_count),
            "selection_max_objects": int(args.selection_max_objects),
            "selection_max_distance_to_trajectory": float(args.selection_max_distance_to_trajectory),
            "selection_include_occluded_uncertain": bool(args.selection_include_occluded_uncertain),
            "max_abs_delta": float(args.max_abs_delta),
        }
    elif args.model_family == "piecewise_linear":
        rows_by_key: dict[str, list[Q5TrainingRow]] = {}
        for row in training_rows:
            rows_by_key.setdefault(row.piecewise_key, []).append(row)
        experts: dict[str, list[list[float]]] = {}
        expert_row_counts: dict[str, int] = {}
        for key, rows in rows_by_key.items():
            if len(rows) < args.piecewise_min_rows:
                continue
            key_x = np.asarray([item.features for item in rows], dtype=float)
            key_y = np.asarray([item.target_values for item in rows], dtype=float)
            key_coefficients = fit_ridge(key_x, key_y, args.l2_regularization)
            experts[key] = key_coefficients.T.tolist()
            expert_row_counts[key] = len(rows)
        y_pred_rows: list[list[float]] = []
        for row in training_rows:
            coeff_list = experts.get(row.piecewise_key)
            if coeff_list is None:
                pred = np.asarray(row.features, dtype=float) @ global_coefficients
            else:
                coeff = np.asarray(coeff_list, dtype=float).T
                pred = np.asarray(row.features, dtype=float) @ coeff
            y_pred_rows.append([float(pred[0]), float(pred[1])])
        y_pred = np.asarray(y_pred_rows, dtype=float)
        model = {
            "model_type": "phase9_q5_object_motion_piecewise_linear_v1",
            "source_split": args.split,
            "source_qa_type_id": int(qa_type_id),
            "source_task_label": task_label,
            "baseline_mode": args.baseline_mode,
            "feature_set": feature_set,
            "feature_names": feature_names,
            "default_coefficients": global_coefficients.T.tolist(),
            "piecewise_experts": experts,
            "piecewise_min_rows": int(args.piecewise_min_rows),
            "piecewise_expert_row_counts": expert_row_counts,
            "target_waypoint_count": int(target_waypoint_count),
            "selection_max_objects": int(args.selection_max_objects),
            "selection_max_distance_to_trajectory": float(args.selection_max_distance_to_trajectory),
            "selection_include_occluded_uncertain": bool(args.selection_include_occluded_uncertain),
            "max_abs_delta": float(args.max_abs_delta),
        }
    elif args.model_family == "regression_tree":
        split_feature_names = [name for name in feature_names if name != "bias"]
        split_feature_indices = [feature_names.index(name) for name in split_feature_names]
        x_split = x[:, split_feature_indices]
        tree = _fit_regression_tree_node(
            x_split,
            y,
            depth=0,
            max_depth=args.tree_max_depth,
            min_leaf=args.tree_min_leaf,
            min_gain=args.tree_min_gain,
            split_feature_names=split_feature_names,
        )
        y_pred = np.vstack([_predict_regression_tree(tree, row) for row in x_split])
        model = {
            "model_type": "phase9_q5_object_motion_regression_tree_v1",
            "source_split": args.split,
            "source_qa_type_id": int(qa_type_id),
            "source_task_label": task_label,
            "baseline_mode": args.baseline_mode,
            "feature_set": feature_set,
            "split_feature_names": split_feature_names,
            "tree_max_depth": int(args.tree_max_depth),
            "tree_min_leaf": int(args.tree_min_leaf),
            "tree_min_gain": float(args.tree_min_gain),
            "tree": tree,
            "target_waypoint_count": int(target_waypoint_count),
            "selection_max_objects": int(args.selection_max_objects),
            "selection_max_distance_to_trajectory": float(args.selection_max_distance_to_trajectory),
            "selection_include_occluded_uncertain": bool(args.selection_include_occluded_uncertain),
            "max_abs_delta": float(args.max_abs_delta),
        }
    elif args.model_family == "gradient_boosting":
        if GradientBoostingRegressor is None:
            raise SystemExit("model-family=gradient_boosting requires scikit-learn installed.")
        regressors = []
        pred_columns = []
        for target_index in range(y.shape[1]):
            regressor = GradientBoostingRegressor(
                n_estimators=int(args.gbdt_n_estimators),
                learning_rate=float(args.gbdt_learning_rate),
                max_depth=int(args.gbdt_max_depth),
                min_samples_leaf=int(args.gbdt_min_samples_leaf),
                subsample=float(args.gbdt_subsample),
                random_state=42 + target_index,
            )
            regressor.fit(x, y[:, target_index])
            regressors.append(regressor)
            pred_columns.append(regressor.predict(x))
        y_pred = np.column_stack(pred_columns)
        payload = base64.b64encode(pickle.dumps(regressors, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
        model = {
            "model_type": "phase9_q5_object_motion_gradient_boosting_v1",
            "source_split": args.split,
            "source_qa_type_id": int(qa_type_id),
            "source_task_label": task_label,
            "baseline_mode": args.baseline_mode,
            "feature_set": feature_set,
            "feature_names": feature_names,
            "gbdt_n_estimators": int(args.gbdt_n_estimators),
            "gbdt_learning_rate": float(args.gbdt_learning_rate),
            "gbdt_max_depth": int(args.gbdt_max_depth),
            "gbdt_min_samples_leaf": int(args.gbdt_min_samples_leaf),
            "gbdt_subsample": float(args.gbdt_subsample),
            "sklearn_pickle_b64": payload,
            "target_waypoint_count": int(target_waypoint_count),
            "selection_max_objects": int(args.selection_max_objects),
            "selection_max_distance_to_trajectory": float(args.selection_max_distance_to_trajectory),
            "selection_include_occluded_uncertain": bool(args.selection_include_occluded_uncertain),
            "max_abs_delta": float(args.max_abs_delta),
        }
    else:
        if MLPRegressor is None or StandardScaler is None or make_pipeline is None:
            raise SystemExit("model-family=mlp requires scikit-learn installed.")
        hidden_layer_sizes = parse_hidden_layer_sizes(str(args.mlp_hidden_layer_sizes))
        regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=float(args.mlp_alpha),
                learning_rate_init=float(args.mlp_learning_rate_init),
                max_iter=int(args.mlp_max_iter),
                random_state=42,
                early_stopping=True,
                n_iter_no_change=25,
            ),
        )
        regressor.fit(x, y)
        y_pred = np.asarray(regressor.predict(x), dtype=float)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        payload = base64.b64encode(pickle.dumps(regressor, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
        model = {
            "model_type": "phase9_q5_object_motion_mlp_v1",
            "source_split": args.split,
            "source_qa_type_id": int(qa_type_id),
            "source_task_label": task_label,
            "baseline_mode": args.baseline_mode,
            "feature_set": feature_set,
            "feature_names": feature_names,
            "mlp_hidden_layer_sizes": list(hidden_layer_sizes),
            "mlp_alpha": float(args.mlp_alpha),
            "mlp_learning_rate_init": float(args.mlp_learning_rate_init),
            "mlp_max_iter": int(args.mlp_max_iter),
            "sklearn_pickle_b64": payload,
            "target_waypoint_count": int(target_waypoint_count),
            "selection_max_objects": int(args.selection_max_objects),
            "selection_max_distance_to_trajectory": float(args.selection_max_distance_to_trajectory),
            "selection_include_occluded_uncertain": bool(args.selection_include_occluded_uncertain),
            "max_abs_delta": float(args.max_abs_delta),
        }

    endpoint_l2 = np.linalg.norm(y_pred - y, axis=1)
    report = {
        "model_family": args.model_family,
        "split": args.split,
        "qa_type_id": int(qa_type_id),
        "task_label": task_label,
        "baseline_mode": args.baseline_mode,
        "feature_set": feature_set,
        "feature_count": len(feature_names),
        "sample_rows": len(samples),
        "gt_rows": gt_row_count,
        "matched_rows": matched_row_count,
        "match_rate": float(matched_row_count) / float(gt_row_count) if gt_row_count > 0 else 0.0,
        "mean_match_distance": float(np.mean(match_distances)) if match_distances else 0.0,
        "selection": {
            "max_objects": int(args.selection_max_objects),
            "max_distance_to_trajectory": float(args.selection_max_distance_to_trajectory),
            "include_occluded_uncertain": bool(args.selection_include_occluded_uncertain),
        },
        "train_metrics": {
            "endpoint_l2_avg": float(np.mean(endpoint_l2)),
            "endpoint_l2_p90": float(np.quantile(endpoint_l2, 0.9)),
        },
    }

    output_model_path = Path(args.output_json).expanduser().resolve()
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    output_report_path = Path(args.output_report).expanduser().resolve() if args.output_report else None
    if output_report_path is not None:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"samples: {len(samples)}")
    print(f"gt_rows: {gt_row_count}")
    print(f"matched_rows: {matched_row_count}")
    print(f"match_rate: {report['match_rate']:.6f}")
    print(f"train_endpoint_l2_avg: {report['train_metrics']['endpoint_l2_avg']:.6f}")
    print(f"saved_model: {output_model_path}")
    if output_report_path is not None:
        print(f"saved_report: {output_report_path}")


def main() -> None:
    Q5ObjectMotionTrainer().main()


if __name__ == "__main__":
    main()
