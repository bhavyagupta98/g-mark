#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.future_trajectory_planner import (  # noqa: E402
    ControlConditionedFutureTrajectoryPlanner,
)


POSITION_RE = re.compile(
    r"I am\s+(?P<agent>[A-Za-z0-9_]+)\s+at\s+"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
)
COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen, control-conditioned Q9 trajectory prior. The model "
            "uses train answers only to learn average waypoint deltas by asker "
            "and suggested control setting; runtime prediction does not read "
            "future_trajectory_str_in_ego."
        )
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--num-future-waypoints", type=int, default=6)
    parser.add_argument(
        "--model-family",
        default="mean_delta",
        choices=("mean_delta", "linear_metadata", "linear_metadata_tail_residual"),
        help=(
            "`mean_delta` averages relative waypoints by control bucket. "
            "`linear_metadata` fits a train-frozen linear regressor over current "
            "position, speed/steering ids, and Q9 control scalars dist/angle. "
            "`linear_metadata_tail_residual` adds a tail-only residual head for "
            "waypoints 5-6 using nonlinear control features."
        ),
    )
    parser.add_argument(
        "--min-key-count",
        type=int,
        default=12,
        help="Use asker-specific control buckets only after this many train rows.",
    )
    return parser


def dataset_path(v2vgot_root: Path, split: str) -> Path:
    split_dir = "train_no_fusion_keep_all" if split == "train" else "no_fusion_keep_all"
    return (
        v2vgot_root
        / "DMSTrack"
        / "V2V4Real"
        / "official_models"
        / split_dir
        / "npy"
        / "co_llm"
        / "v2v4real_3d_grounding_qa_dataset_v2vgot.json"
    )


def parse_current_position(question: str) -> tuple[float, float] | None:
    match = POSITION_RE.search(question)
    if not match:
        return None
    return float(match.group("x")), float(match.group("y"))


def parse_waypoints(text: str, limit: int) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in COORD_RE.findall(text)[:limit])


def model_key(record: dict[str, object], asker_cav_id: str = "") -> str:
    return ControlConditionedFutureTrajectoryPlanner._model_key(
        asker_cav_id=asker_cav_id or str(record.get("asker_cav_id", "")),
        speed_idx=str(record.get("suggested_speed_idx", "")),
        steering_idx=str(record.get("suggested_steering_idx", "")),
    )


def safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def linear_features(record: dict[str, object], current: tuple[float, float]) -> list[float]:
    speed_idx = safe_int(record.get("suggested_speed_idx"))
    steering_idx = safe_int(record.get("suggested_steering_idx"))
    distance = safe_float(record.get("dist"))
    angle = safe_float(record.get("angle"))
    features = [
        1.0,
        current[0],
        current[1],
        1.0 if str(record.get("asker_cav_id", "")) == "1" else 0.0,
    ]
    features.extend(1.0 if speed_idx == index else 0.0 for index in range(5))
    features.extend(1.0 if steering_idx == index else 0.0 for index in range(5))
    features.extend(
        [
            distance,
            math.sin(angle),
            math.cos(angle),
            distance * math.sin(angle),
            distance * math.cos(angle),
        ]
    )
    return features


def linear_tail_residual_features(record: dict[str, object], current: tuple[float, float]) -> list[float]:
    base = linear_features(record, current)
    distance = safe_float(record.get("dist"))
    angle = safe_float(record.get("angle"))
    speed_idx = safe_int(record.get("suggested_speed_idx"))
    steering_idx = safe_int(record.get("suggested_steering_idx"))
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    return base + [
        current[0] * distance,
        current[1] * distance,
        distance * distance,
        math.sin(2.0 * angle),
        math.cos(2.0 * angle),
        distance * distance * sin_angle,
        distance * distance * cos_angle,
        float(speed_idx * steering_idx) if speed_idx >= 0 and steering_idx >= 0 else 0.0,
    ]


def add_delta(
    accumulator: dict[str, list[list[float]]],
    key: str,
    current: tuple[float, float],
    waypoints: tuple[tuple[float, float], ...],
) -> None:
    rows = accumulator[key]
    for index, (x, y) in enumerate(waypoints):
        rows[index][0] += x - current[0]
        rows[index][1] += y - current[1]
        rows[index][2] += 1.0


def finalize_rows(rows: list[list[float]]) -> list[list[float]]:
    finalized: list[list[float]] = []
    for sum_x, sum_y, count in rows:
        if count <= 0.0:
            finalized.append([0.0, 0.0])
        else:
            finalized.append([sum_x / count, sum_y / count])
    return finalized


def average_l2(
    records: list[dict[str, object]],
    relative_waypoints_by_key: dict[str, list[list[float]]],
    waypoint_count: int,
) -> dict[str, float]:
    errors: list[float] = []
    first_errors: list[float] = []
    for record in records:
        conversations = record.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        question = str(conversations[0].get("value", ""))
        answer = str(conversations[1].get("value", ""))
        current = parse_current_position(question)
        gt_waypoints = parse_waypoints(answer, waypoint_count)
        if current is None or len(gt_waypoints) != waypoint_count:
            continue

        key = model_key(record)
        fallback_key = model_key(record, asker_cav_id="*")
        relative = (
            relative_waypoints_by_key.get(key)
            or relative_waypoints_by_key.get(fallback_key)
            or relative_waypoints_by_key.get("__fallback__")
        )
        if not relative:
            continue
        waypoint_errors: list[float] = []
        for (dx, dy), (gt_x, gt_y) in zip(relative, gt_waypoints):
            pred_x = current[0] + dx
            pred_y = current[1] + dy
            waypoint_errors.append(((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5)
        if waypoint_errors:
            errors.append(sum(waypoint_errors) / len(waypoint_errors))
            first_errors.append(sum(waypoint_errors[:2]) / min(2, len(waypoint_errors)))

    return {
        "l2_error_avg_all": sum(errors) / len(errors) if errors else 0.0,
        "l2_error_avg_1s": sum(first_errors) / len(first_errors) if first_errors else 0.0,
        "evaluated_rows": float(len(errors)),
    }


def train_linear_metadata_model(
    records: list[dict[str, object]],
    waypoint_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    for record in records:
        conversations = record.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        question = str(conversations[0].get("value", ""))
        answer = str(conversations[1].get("value", ""))
        current = parse_current_position(question)
        waypoints = parse_waypoints(answer, waypoint_count)
        if current is None or len(waypoints) != waypoint_count:
            continue
        feature_rows.append(linear_features(record, current))
        target_rows.append([value for point in waypoints for value in point])

    x = np.asarray(feature_rows, dtype=float)
    y = np.asarray(target_rows, dtype=float)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0].T
    predictions = x @ coefficients.T
    waypoint_errors = np.linalg.norm(
        predictions.reshape(-1, waypoint_count, 2)
        - y.reshape(-1, waypoint_count, 2),
        axis=2,
    )
    report = {
        "train_rows": len(records),
        "usable_rows": len(feature_rows),
        "feature_count": int(x.shape[1]) if x.size else 0,
        "train_metrics": {
            "l2_error_avg_all": float(np.mean(np.mean(waypoint_errors, axis=1))) if len(waypoint_errors) else 0.0,
            "l2_error_avg_1s": float(np.mean(np.mean(waypoint_errors[:, :2], axis=1))) if len(waypoint_errors) else 0.0,
            "evaluated_rows": float(len(feature_rows)),
        },
    }
    model = {
        "model_type": "phase9_q9_control_metadata_linear_v1",
        "source_split": "train",
        "num_future_waypoints": waypoint_count,
        "feature_names": [
            "bias",
            "current_x",
            "current_y",
            "asker_is_cav1",
            "speed_idx_0",
            "speed_idx_1",
            "speed_idx_2",
            "speed_idx_3",
            "speed_idx_4",
            "steering_idx_0",
            "steering_idx_1",
            "steering_idx_2",
            "steering_idx_3",
            "steering_idx_4",
            "dist",
            "sin_angle",
            "cos_angle",
            "dist_sin_angle",
            "dist_cos_angle",
        ],
        "coefficients": coefficients.tolist(),
        "report": report,
    }
    return model, report


def train_linear_metadata_tail_residual_model(
    records: list[dict[str, object]],
    waypoint_count: int,
    tail_start_index: int = 4,
) -> tuple[dict[str, object], dict[str, object]]:
    feature_rows: list[list[float]] = []
    tail_feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    for record in records:
        conversations = record.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        question = str(conversations[0].get("value", ""))
        answer = str(conversations[1].get("value", ""))
        current = parse_current_position(question)
        waypoints = parse_waypoints(answer, waypoint_count)
        if current is None or len(waypoints) != waypoint_count:
            continue
        feature_rows.append(linear_features(record, current))
        tail_feature_rows.append(linear_tail_residual_features(record, current))
        target_rows.append([value for point in waypoints for value in point])

    x = np.asarray(feature_rows, dtype=float)
    xt = np.asarray(tail_feature_rows, dtype=float)
    y = np.asarray(target_rows, dtype=float)

    base_coefficients = np.linalg.lstsq(x, y, rcond=None)[0].T
    base_predictions = x @ base_coefficients.T

    tail_start_column = tail_start_index * 2
    tail_targets = y[:, tail_start_column:]
    tail_base_predictions = base_predictions[:, tail_start_column:]
    tail_residual_targets = tail_targets - tail_base_predictions

    tail_coefficients = np.linalg.lstsq(xt, tail_residual_targets, rcond=None)[0].T
    tail_residual_predictions = xt @ tail_coefficients.T

    final_predictions = base_predictions.copy()
    final_predictions[:, tail_start_column:] = (
        tail_base_predictions + tail_residual_predictions
    )

    waypoint_errors = np.linalg.norm(
        final_predictions.reshape(-1, waypoint_count, 2)
        - y.reshape(-1, waypoint_count, 2),
        axis=2,
    )
    base_waypoint_errors = np.linalg.norm(
        base_predictions.reshape(-1, waypoint_count, 2)
        - y.reshape(-1, waypoint_count, 2),
        axis=2,
    )

    report = {
        "train_rows": len(records),
        "usable_rows": len(feature_rows),
        "feature_count": int(x.shape[1]) if x.size else 0,
        "tail_feature_count": int(xt.shape[1]) if xt.size else 0,
        "tail_start_index": tail_start_index,
        "train_metrics": {
            "l2_error_avg_all": float(np.mean(np.mean(waypoint_errors, axis=1))) if len(waypoint_errors) else 0.0,
            "l2_error_avg_1s": float(np.mean(np.mean(waypoint_errors[:, :2], axis=1))) if len(waypoint_errors) else 0.0,
            "l2_error_avg_3s": float(np.mean(np.mean(waypoint_errors, axis=1))) if len(waypoint_errors) else 0.0,
            "evaluated_rows": float(len(feature_rows)),
        },
        "base_train_metrics": {
            "l2_error_avg_all": float(np.mean(np.mean(base_waypoint_errors, axis=1))) if len(base_waypoint_errors) else 0.0,
            "l2_error_avg_1s": float(np.mean(np.mean(base_waypoint_errors[:, :2], axis=1))) if len(base_waypoint_errors) else 0.0,
            "l2_error_avg_3s": float(np.mean(np.mean(base_waypoint_errors, axis=1))) if len(base_waypoint_errors) else 0.0,
            "evaluated_rows": float(len(feature_rows)),
        },
    }
    model = {
        "model_type": "phase9_q9_control_metadata_linear_tail_residual_v1",
        "source_split": "train",
        "num_future_waypoints": waypoint_count,
        "tail_start_index": tail_start_index,
        "feature_names": [
            "bias",
            "current_x",
            "current_y",
            "asker_is_cav1",
            "speed_idx_0",
            "speed_idx_1",
            "speed_idx_2",
            "speed_idx_3",
            "speed_idx_4",
            "steering_idx_0",
            "steering_idx_1",
            "steering_idx_2",
            "steering_idx_3",
            "steering_idx_4",
            "dist",
            "sin_angle",
            "cos_angle",
            "dist_sin_angle",
            "dist_cos_angle",
        ],
        "tail_feature_names": [
            "bias",
            "current_x",
            "current_y",
            "asker_is_cav1",
            "speed_idx_0",
            "speed_idx_1",
            "speed_idx_2",
            "speed_idx_3",
            "speed_idx_4",
            "steering_idx_0",
            "steering_idx_1",
            "steering_idx_2",
            "steering_idx_3",
            "steering_idx_4",
            "dist",
            "sin_angle",
            "cos_angle",
            "dist_sin_angle",
            "dist_cos_angle",
            "current_x_times_dist",
            "current_y_times_dist",
            "dist_squared",
            "sin_2angle",
            "cos_2angle",
            "dist_squared_times_sin_angle",
            "dist_squared_times_cos_angle",
            "speed_idx_times_steering_idx",
        ],
        "coefficients": base_coefficients.tolist(),
        "tail_residual_coefficients": tail_coefficients.tolist(),
        "report": report,
    }
    return model, report


def main() -> None:
    args = build_parser().parse_args()
    path = dataset_path(Path(args.v2vgot_root).expanduser().resolve(), args.split)
    with path.open("r", encoding="utf-8") as handle:
        records = [
            record
            for record in json.load(handle)
            if isinstance(record, dict) and record.get("qa_type_id") == 19
        ]

    if args.model_family == "linear_metadata":
        model, report = train_linear_metadata_model(records, args.num_future_waypoints)
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")
        if args.output_report:
            output_report = Path(args.output_report)
            output_report.parent.mkdir(parents=True, exist_ok=True)
            output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"train_rows: {report['train_rows']}")
        print(f"usable_rows: {report['usable_rows']}")
        print(f"train_metrics: {report['train_metrics']}")
        print(f"saved_model: {output_json.resolve()}")
        if args.output_report:
            print(f"saved_report: {Path(args.output_report).resolve()}")
        return
    if args.model_family == "linear_metadata_tail_residual":
        model, report = train_linear_metadata_tail_residual_model(
            records,
            args.num_future_waypoints,
        )
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")
        if args.output_report:
            output_report = Path(args.output_report)
            output_report.parent.mkdir(parents=True, exist_ok=True)
            output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"train_rows: {report['train_rows']}")
        print(f"usable_rows: {report['usable_rows']}")
        print(f"train_metrics: {report['train_metrics']}")
        print(f"base_train_metrics: {report['base_train_metrics']}")
        print(f"saved_model: {output_json.resolve()}")
        if args.output_report:
            print(f"saved_report: {Path(args.output_report).resolve()}")
        return

    accumulator = defaultdict(
        lambda: [[0.0, 0.0, 0.0] for _ in range(args.num_future_waypoints)]
    )
    key_counts: dict[str, int] = defaultdict(int)
    usable_rows = 0
    for record in records:
        conversations = record.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        question = str(conversations[0].get("value", ""))
        answer = str(conversations[1].get("value", ""))
        current = parse_current_position(question)
        waypoints = parse_waypoints(answer, args.num_future_waypoints)
        if current is None or len(waypoints) != args.num_future_waypoints:
            continue

        specific_key = model_key(record)
        control_key = model_key(record, asker_cav_id="*")
        add_delta(accumulator, specific_key, current, waypoints)
        add_delta(accumulator, control_key, current, waypoints)
        add_delta(accumulator, "__fallback__", current, waypoints)
        key_counts[specific_key] += 1
        usable_rows += 1

    relative_waypoints_by_key: dict[str, list[list[float]]] = {}
    for key, rows in accumulator.items():
        if key.startswith("asker=") and not key.startswith("asker=*"):
            if key_counts[key] < args.min_key_count:
                continue
        relative_waypoints_by_key[key] = finalize_rows(rows)

    report = {
        "train_rows": len(records),
        "usable_rows": usable_rows,
        "min_key_count": args.min_key_count,
        "specific_key_count": sum(
            1 for key in relative_waypoints_by_key if key.startswith("asker=") and not key.startswith("asker=*")
        ),
        "control_fallback_key_count": sum(
            1 for key in relative_waypoints_by_key if key.startswith("asker=*")
        ),
        "train_metrics": average_l2(records, relative_waypoints_by_key, args.num_future_waypoints),
    }
    model = {
        "model_type": "phase9_q9_control_delta_mean_v1",
        "source_split": args.split,
        "num_future_waypoints": args.num_future_waypoints,
        "min_key_count": args.min_key_count,
        "relative_waypoints_by_key": relative_waypoints_by_key,
        "report": report,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")
    if args.output_report:
        output_report = Path(args.output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"train_rows: {report['train_rows']}")
    print(f"usable_rows: {usable_rows}")
    print(f"train_metrics: {report['train_metrics']}")
    print(f"saved_model: {output_json.resolve()}")
    if args.output_report:
        print(f"saved_report: {Path(args.output_report).resolve()}")


if __name__ == "__main__":
    main()
