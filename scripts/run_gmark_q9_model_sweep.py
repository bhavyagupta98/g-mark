#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.planning.control_settings_policy import (  # noqa: E402
    SPEED_CLASSES,
    STEERING_CLASSES,
    decide_control_settings,
)
from kg_coop_drive.application.qa.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

try:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.linear_model import ElasticNet  # type: ignore
    from sklearn.multioutput import MultiOutputRegressor  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


TABLE1_Q9_FILE_NAME = "v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json"
DEFAULT_TRAIN_FILE_NAME = "v2v4real_3d_grounding_qa_dataset_v2vgot.json"

POSITION_RE = re.compile(
    r"I am\s+(?P<agent>[A-Za-z0-9_]+)\s+at\s+"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
)
COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
SPEED_LABEL_RE = re.compile(r"suggested speed setting is:\s*([^.]+)\.", re.IGNORECASE)
STEER_LABEL_RE = re.compile(r"suggested steering setting is:\s*([^.]+)\.", re.IGNORECASE)

SPEED_LABELS = tuple(SPEED_CLASSES)
STEER_LABELS = tuple(STEERING_CLASSES)
Q8_SPEED_CONTROL_VALUES = {
    "fast": 1.0,
    "moderate": 0.65,
    "slow": 0.35,
    "very slow": 0.15,
    "stop": 0.0,
}
Q8_STEERING_CONTROL_VALUES = {
    "left": -1.0,
    "slightly left": -0.5,
    "straight": 0.0,
    "slightly right": 0.5,
    "right": 1.0,
}


@dataclass(frozen=True)
class SweepModelResult:
    model_name: str
    model_json: Path
    prediction_jsonl: Path
    prediction_manifest_json: Path
    train_rows: int
    val_rows: int
    train_l2_avg: float
    val_l2_avg: float
    val_l2_1s: float
    val_l2_2s: float
    val_l2_3s: float
    official_export_manifest_json: str | None
    official_summary_json: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep multiple clean Q9 models, export per-model manifests, and "
            "optionally run official export/eval."
        )
    )
    parser.add_argument("--run-name", default=f"gmark_q9_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--output-root", default="outputs/v2vgot_table1_reproduction/gmark_q9_sweep")
    parser.add_argument("--train-file-name", default=DEFAULT_TRAIN_FILE_NAME)
    parser.add_argument("--val-file-name", default=TABLE1_Q9_FILE_NAME)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--allow-train-val-overlap", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--models", nargs="+", default=("ridge", "elasticnet", "rf"))
    parser.add_argument("--include-q8-pred-features", action="store_true")
    parser.add_argument(
        "--q8-feature-source",
        default="question_context",
        choices=("question_context", "q8_model", "legacy_jsonl"),
        help=(
            "Q8 feature source. question_context parses the Q8 context already "
            "present in the Q9 prompt; q8_model rebuilds KG scenes and reruns "
            "the Q8 model; legacy_jsonl reads --q8-predictions-jsonl."
        ),
    )
    parser.add_argument("--q8-predictions-jsonl", default="")
    parser.add_argument(
        "--q8-model-json",
        default="",
        help=(
            "Q8 model JSON used only when --q8-feature-source=q8_model. That path "
            "rebuilds KG-prepared train/val scenes and reruns Q8 inference."
        ),
    )
    parser.add_argument(
        "--q8-feature-timeout-seconds",
        type=int,
        default=0,
        help=(
            "If >0, skips one Q8 feature row with zero/default features after "
            "this many seconds. Useful for debugging pathological KG rows."
        ),
    )
    parser.add_argument(
        "--q8-feature-debug-every",
        type=int,
        default=0,
        help="If >0, logs the Q8 feature sample before processing every N rows.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-official-eval", action="store_true")
    return parser


def resolve_output_root(raw_output_root: str) -> Path:
    output_root = Path(raw_output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def parse_current_position(sample: BenchmarkSample) -> tuple[float, float]:
    match = POSITION_RE.search(sample.scene.raw_question)
    if match:
        return float(match.group("x")), float(match.group("y"))
    asker = next((agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id), None)
    if asker is None:
        return (0.0, 0.0)
    return (asker.pose.position.x, asker.pose.position.y)


def parse_waypoints(text: str, limit: int = 6) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in COORD_RE.findall(text)[:limit])


def raw_answer(sample: BenchmarkSample) -> str:
    conversations = sample.raw_record.get("conversations", [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        return ""
    second = conversations[1]
    if not isinstance(second, dict):
        return ""
    value = second.get("value", "")
    return value if isinstance(value, str) else ""


def qa_overlap_key(sample: BenchmarkSample) -> tuple[str, str, str, str]:
    return (
        str(sample.raw_record.get("scenario_index", "")),
        str(sample.raw_record.get("local_timestamp_index", "")),
        str(sample.raw_record.get("qa_type_id", sample.qa_type_id)),
        sample.scene.raw_question.strip(),
    )


def load_q9_samples(
    *,
    adapter: V2VGoTQABenchmarkAdapter,
    split_name: str,
    file_name: str,
    limit: int,
) -> tuple[BenchmarkSample, ...]:
    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name=split_name, file_name=file_name)
        if sample.task_type == BenchmarkTaskType.FUTURE_TRAJECTORY
    )
    if limit > 0:
        samples = samples[:limit]
    return samples


def extract_question_trajectory(sample: BenchmarkSample) -> tuple[tuple[float, float], ...]:
    return parse_waypoints(sample.scene.raw_question, limit=6)


def vector_norm(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def trajectory_geometry_features(sample: BenchmarkSample) -> list[float]:
    current_x, current_y = parse_current_position(sample)
    points = extract_question_trajectory(sample)
    if not points:
        return [0.0] * 10
    rel_points = [(px - current_x, py - current_y) for px, py in points]
    dists = [vector_norm(dx, dy) for dx, dy in rel_points]
    first_dx, first_dy = rel_points[0]
    last_dx, last_dy = rel_points[-1]
    first_dist = dists[0]
    last_dist = dists[-1]
    step_lengths: list[float] = []
    for idx in range(1, len(points)):
        step_lengths.append(vector_norm(points[idx][0] - points[idx - 1][0], points[idx][1] - points[idx - 1][1]))
    mean_step = float(sum(step_lengths) / len(step_lengths)) if step_lengths else 0.0
    std_step = float(np.std(np.asarray(step_lengths, dtype=float))) if step_lengths else 0.0
    heading = math.atan2(first_dy, first_dx) if first_dist > 1e-6 else 0.0
    return [
        first_dx,
        first_dy,
        first_dist,
        last_dx,
        last_dy,
        last_dist,
        mean_step,
        std_step,
        math.sin(heading),
        math.cos(heading),
    ]


def parse_q8_label_features(answer_text: str) -> list[float]:
    speed, steer = parse_q8_labels(answer_text)
    return q8_label_features(speed_label=speed, steering_label=steer)


def parse_q8_labels(answer_text: str) -> tuple[str, str]:
    speed = ""
    steer = ""
    speed_match = SPEED_LABEL_RE.search(answer_text)
    steer_match = STEER_LABEL_RE.search(answer_text)
    if speed_match:
        speed = speed_match.group(1).strip().lower()
    if steer_match:
        steer = steer_match.group(1).strip().lower()
    return speed, steer


def q8_label_features(*, speed_label: str, steering_label: str) -> list[float]:
    speed = speed_label.strip().lower()
    steer = steering_label.strip().lower()
    speed_one_hot = [1.0 if label == speed else 0.0 for label in SPEED_LABELS]
    steer_one_hot = [1.0 if label == steer else 0.0 for label in STEER_LABELS]
    return (
        speed_one_hot
        + steer_one_hot
        + [
            Q8_SPEED_CONTROL_VALUES.get(speed, 0.0),
            Q8_STEERING_CONTROL_VALUES.get(steer, 0.0),
        ]
    )


def q8_feature_width() -> int:
    return len(SPEED_LABELS) + len(STEER_LABELS) + 2


def q8_label_indices(*, speed_label: str, steering_label: str) -> tuple[int, int]:
    speed = speed_label.strip().lower()
    steer = steering_label.strip().lower()
    speed_idx = SPEED_LABELS.index(speed) if speed in SPEED_LABELS else -1
    steering_idx = STEER_LABELS.index(steer) if steer in STEER_LABELS else -1
    return speed_idx, steering_idx


def load_q8_prediction_lookup(path: str) -> dict[str, list[float]]:
    prediction_path = Path(path).expanduser()
    if not prediction_path.exists():
        raise FileNotFoundError(f"Q8 prediction file not found: {prediction_path}")
    result: dict[str, list[float]] = {}
    with prediction_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            answer_text = str(row.get("answer_text", ""))
            if sample_id:
                result[sample_id] = parse_q8_label_features(answer_text)
    return result


def load_q8_model(path: str) -> dict[str, object]:
    model_path = Path(path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Q8 model file not found: {model_path}")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Q8 model JSON: {model_path}")
    return payload


class Q8FeatureTimeoutError(TimeoutError):
    pass


def _raise_q8_feature_timeout(signum: int, frame: object) -> None:
    raise Q8FeatureTimeoutError("Q8 feature row timed out")


def predict_q8_feature_row(
    *,
    sample: BenchmarkSample,
    evaluator: V2VGoTQAPhase5AEvaluator,
    q8_model: dict[str, object],
) -> tuple[list[float], dict[str, object]]:
    prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode="cooperative")
    decision = decide_control_settings(
        scene=prepared_scene,
        selection_policy="linear_classifier",
        model=q8_model,
    )
    speed_label = decision.speed_instruction.strip().lower()
    steering_label = decision.steering_instruction.strip().lower()
    speed_idx, steering_idx = q8_label_indices(
        speed_label=speed_label,
        steering_label=steering_label,
    )
    features = q8_label_features(
        speed_label=speed_label,
        steering_label=steering_label,
    )
    metadata = {
        "q8_pred_speed_label": speed_label,
        "q8_pred_speed_idx": speed_idx,
        "q8_pred_steering_label": steering_label,
        "q8_pred_steering_idx": steering_idx,
        "q8_pred_object_ids": list(decision.object_ids),
        "q8_feature_vector": features,
        "q8_feature_status": "predicted",
    }
    return features, metadata


def q8_metadata_from_labels(*, speed_label: str, steering_label: str, status: str) -> tuple[list[float], dict[str, object]]:
    speed_idx, steering_idx = q8_label_indices(
        speed_label=speed_label,
        steering_label=steering_label,
    )
    features = q8_label_features(
        speed_label=speed_label,
        steering_label=steering_label,
    )
    metadata = {
        "q8_pred_speed_label": speed_label,
        "q8_pred_speed_idx": speed_idx,
        "q8_pred_steering_label": steering_label,
        "q8_pred_steering_idx": steering_idx,
        "q8_pred_object_ids": [],
        "q8_feature_vector": features,
        "q8_feature_status": status,
    }
    return features, metadata


def write_q8_feature_jsonl_row(
    *,
    handle: Any,
    sample: BenchmarkSample,
    elapsed: float,
    prediction_metadata: dict[str, object],
) -> None:
    handle.write(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "split_name": sample.split_name,
                "file_name": sample.file_name,
                "scenario_index": sample.raw_record.get("scenario_index"),
                "global_timestamp_index": sample.raw_record.get("global_timestamp_index"),
                "local_timestamp_index": sample.raw_record.get("local_timestamp_index"),
                "asker_cav_id": sample.raw_record.get("asker_cav_id"),
                "q8_feature_elapsed_seconds": round(elapsed, 6),
                **prediction_metadata,
            }
        )
        + "\n"
    )
    handle.flush()


def update_q8_counts(
    *,
    prediction_metadata: dict[str, object],
    speed_counts: dict[str, int],
    steering_counts: dict[str, int],
) -> None:
    speed_label = str(prediction_metadata["q8_pred_speed_label"])
    steering_label = str(prediction_metadata["q8_pred_steering_label"])
    if speed_label in speed_counts:
        speed_counts[speed_label] += 1
    if steering_label in steering_counts:
        steering_counts[steering_label] += 1


def build_q8_prediction_lookup_from_question_context(
    *,
    samples: tuple[BenchmarkSample, ...],
    output_jsonl: Path,
    progress_every: int,
) -> dict[str, list[float]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lookup: dict[str, list[float]] = {}
    speed_counts = {label: 0 for label in SPEED_LABELS}
    steering_counts = {label: 0 for label in STEER_LABELS}
    total = len(samples)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples, start=1):
            started_at = time.monotonic()
            speed_label, steering_label = parse_q8_labels(sample.scene.raw_question)
            features, prediction_metadata = q8_metadata_from_labels(
                speed_label=speed_label,
                steering_label=steering_label,
                status="parsed_from_q9_question_context",
            )
            elapsed = time.monotonic() - started_at
            lookup[sample.sample_id] = features
            update_q8_counts(
                prediction_metadata=prediction_metadata,
                speed_counts=speed_counts,
                steering_counts=steering_counts,
            )
            write_q8_feature_jsonl_row(
                handle=handle,
                sample=sample,
                elapsed=elapsed,
                prediction_metadata=prediction_metadata,
            )
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
                print(
                    f"q8_context_feature_progress: {sample.split_name} {idx}/{total} "
                    f"last_sample_id={sample.sample_id}",
                    flush=True,
                )
    print(
        "[INFO] Q8 question-context features ready: "
        f"split={samples[0].split_name if samples else 'unknown'} rows={len(lookup)} "
        f"speed_counts={speed_counts} steering_counts={steering_counts}",
        flush=True,
    )
    return lookup


def build_q8_prediction_lookup_from_model(
    *,
    samples: tuple[BenchmarkSample, ...],
    evaluator: V2VGoTQAPhase5AEvaluator,
    q8_model: dict[str, object],
    output_jsonl: Path,
    progress_every: int,
    timeout_seconds: int,
    debug_every: int,
) -> dict[str, list[float]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lookup: dict[str, list[float]] = {}
    speed_counts = {label: 0 for label in SPEED_LABELS}
    steering_counts = {label: 0 for label in STEER_LABELS}
    total = len(samples)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples, start=1):
            if debug_every > 0 and (idx == 1 or idx % debug_every == 0):
                print(
                    "q8_feature_start: "
                    f"{sample.split_name} {idx}/{total} sample_id={sample.sample_id} "
                    f"scenario={sample.raw_record.get('scenario_index')} "
                    f"global_ts={sample.raw_record.get('global_timestamp_index')} "
                    f"asker={sample.raw_record.get('asker_cav_id')}",
                    flush=True,
                )
            started_at = time.monotonic()
            timed_out = False
            if timeout_seconds > 0:
                previous_handler = signal.signal(signal.SIGALRM, _raise_q8_feature_timeout)
                signal.alarm(timeout_seconds)
            else:
                previous_handler = None
            try:
                features, prediction_metadata = predict_q8_feature_row(
                    sample=sample,
                    evaluator=evaluator,
                    q8_model=q8_model,
                )
            except Q8FeatureTimeoutError:
                timed_out = True
                features = [0.0] * q8_feature_width()
                prediction_metadata = {
                    "q8_pred_speed_label": "",
                    "q8_pred_speed_idx": -1,
                    "q8_pred_steering_label": "",
                    "q8_pred_steering_idx": -1,
                    "q8_pred_object_ids": [],
                    "q8_feature_vector": features,
                    "q8_feature_status": f"timeout_after_{timeout_seconds}s",
                }
            finally:
                if timeout_seconds > 0:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_handler)
            elapsed = time.monotonic() - started_at
            lookup[sample.sample_id] = features
            speed_label = str(prediction_metadata["q8_pred_speed_label"])
            steering_label = str(prediction_metadata["q8_pred_steering_label"])
            if speed_label in speed_counts:
                speed_counts[speed_label] += 1
            if steering_label in steering_counts:
                steering_counts[steering_label] += 1
            if timed_out or (debug_every > 0 and (idx == 1 or idx % debug_every == 0)):
                print(
                    "q8_feature_done: "
                    f"{sample.split_name} {idx}/{total} sample_id={sample.sample_id} "
                    f"elapsed={elapsed:.3f}s status={prediction_metadata['q8_feature_status']}",
                    flush=True,
                )
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "split_name": sample.split_name,
                        "file_name": sample.file_name,
                        "scenario_index": sample.raw_record.get("scenario_index"),
                        "global_timestamp_index": sample.raw_record.get("global_timestamp_index"),
                        "local_timestamp_index": sample.raw_record.get("local_timestamp_index"),
                        "asker_cav_id": sample.raw_record.get("asker_cav_id"),
                        "q8_feature_elapsed_seconds": round(elapsed, 6),
                        **prediction_metadata,
                    }
                )
                + "\n"
            )
            handle.flush()
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
                print(
                    f"q8_feature_progress: {sample.split_name} {idx}/{total} "
                    f"last_sample_id={sample.sample_id} elapsed={elapsed:.3f}s",
                    flush=True,
                )
    print(
        "[INFO] Q8 model features ready: "
        f"split={samples[0].split_name if samples else 'unknown'} rows={len(lookup)} "
        f"speed_counts={speed_counts} steering_counts={steering_counts}",
        flush=True,
    )
    return lookup


def feature_names(include_q8_pred_features: bool) -> list[str]:
    names = [
        "bias",
        "current_x",
        "current_y",
        "asker_is_cav1",
        "traj_first_dx",
        "traj_first_dy",
        "traj_first_dist",
        "traj_last_dx",
        "traj_last_dy",
        "traj_last_dist",
        "traj_mean_step",
        "traj_std_step",
        "traj_heading_sin",
        "traj_heading_cos",
    ]
    if include_q8_pred_features:
        names.extend(
            [f"q8_speed_{label.replace(' ', '_')}" for label in SPEED_LABELS]
            + [f"q8_steer_{label.replace(' ', '_')}" for label in STEER_LABELS]
            + [
                "q8_pred_speed_control_value",
                "q8_pred_steering_control_value",
            ]
        )
    return names


def build_feature_row(
    sample: BenchmarkSample,
    *,
    include_q8_pred_features: bool,
    q8_lookup: dict[str, list[float]],
) -> list[float]:
    current_x, current_y = parse_current_position(sample)
    asker_raw = str(sample.raw_record.get("asker_cav_id", "")).strip()
    asker_from_q = "1" if "I am CAV_1" in sample.scene.raw_question else ""
    asker_is_cav1 = 1.0 if (asker_raw == "1" or asker_from_q == "1") else 0.0
    row = [1.0, current_x, current_y, asker_is_cav1]
    row.extend(trajectory_geometry_features(sample))
    if include_q8_pred_features:
        row.extend(q8_lookup.get(sample.sample_id, [0.0] * q8_feature_width()))
    return row


def build_xy(
    samples: tuple[BenchmarkSample, ...],
    *,
    include_q8_pred_features: bool,
    q8_lookup: dict[str, list[float]],
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    usable = 0
    for idx, sample in enumerate(samples, start=1):
        gt_waypoints = parse_waypoints(raw_answer(sample), limit=6)
        if len(gt_waypoints) != 6:
            continue
        x_rows.append(
            build_feature_row(
                sample,
                include_q8_pred_features=include_q8_pred_features,
                q8_lookup=q8_lookup,
            )
        )
        y_rows.append([value for point in gt_waypoints for value in point])
        usable += 1
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(samples)):
            print(f"feature_progress: {idx}/{len(samples)} usable={usable}", flush=True)
    if not x_rows:
        raise ValueError("No usable Q9 rows after feature extraction.")
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), usable


def l2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(
        y_pred.reshape(-1, 6, 2) - y_true.reshape(-1, 6, 2),
        axis=2,
    )
    return {
        "l2_error_avg_1s": float(np.mean(np.mean(errors[:, :2], axis=1))),
        "l2_error_avg_2s": float(np.mean(np.mean(errors[:, :4], axis=1))),
        "l2_error_avg_3s": float(np.mean(np.mean(errors[:, :6], axis=1))),
        "l2_error_avg_all": float(np.mean(np.mean(errors, axis=1))),
    }


def train_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    print(
        f"[ridge] start fit: train_shape={x_train.shape} target_shape={y_train.shape} val_shape={x_val.shape} alpha={alpha}",
        flush=True,
    )
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    mean[0] = 0.0
    scale[0] = 1.0
    x_train_norm = (x_train - mean) / scale
    x_val_norm = (x_val - mean) / scale
    regularizer = np.eye(x_train_norm.shape[1], dtype=float) * float(alpha)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(x_train_norm.T @ x_train_norm + regularizer, x_train_norm.T @ y_train).T
    y_val_pred = x_val_norm @ coefficients.T
    model = {
        "family": "ridge",
        "alpha": float(alpha),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
    }
    print("[ridge] fit complete; prediction ready", flush=True)
    return y_val_pred, model


def train_predict_elasticnet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    alpha: float = 0.001,
    l1_ratio: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not available; elasticnet cannot run.")
    print(
        "[elasticnet] start fit: "
        f"train_shape={x_train.shape} target_shape={y_train.shape} val_shape={x_val.shape} "
        f"alpha={alpha} l1_ratio={l1_ratio}",
        flush=True,
    )
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    mean[0] = 0.0
    scale[0] = 1.0
    x_train_norm = (x_train - mean) / scale
    x_val_norm = (x_val - mean) / scale
    base = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
    reg = MultiOutputRegressor(base)
    print("[elasticnet] fitting MultiOutputRegressor...", flush=True)
    reg.fit(x_train_norm, y_train)
    print("[elasticnet] fit complete; predicting val...", flush=True)
    y_val_pred = reg.predict(x_val_norm)
    model = {
        "family": "elasticnet",
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "estimators": [
            {
                "coef": estimator.coef_.tolist(),
                "intercept": float(estimator.intercept_),
            }
            for estimator in reg.estimators_
        ],
    }
    print("[elasticnet] prediction complete", flush=True)
    return y_val_pred, model


def train_predict_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not available; random-forest cannot run.")
    print(
        "[rf] start fit: "
        f"train_shape={x_train.shape} target_shape={y_train.shape} val_shape={x_val.shape} "
        f"n_estimators={n_estimators} max_depth={max_depth}",
        flush=True,
    )
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=1,
    )
    print("[rf] fitting RandomForestRegressor...", flush=True)
    reg.fit(x_train, y_train)
    print("[rf] fit complete; predicting val...", flush=True)
    y_val_pred = reg.predict(x_val)
    model = {
        "family": "random_forest",
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
    }
    print("[rf] prediction complete", flush=True)
    return y_val_pred, model


def predict_with_saved_model(model_payload: dict[str, Any], x_val: np.ndarray) -> np.ndarray:
    family = str(model_payload.get("family", ""))
    if family == "ridge":
        mean = np.asarray(model_payload["feature_mean"], dtype=float)
        scale = np.asarray(model_payload["feature_scale"], dtype=float)
        coefficients = np.asarray(model_payload["coefficients"], dtype=float)
        return ((x_val - mean) / scale) @ coefficients.T
    if family == "elasticnet":
        mean = np.asarray(model_payload["feature_mean"], dtype=float)
        scale = np.asarray(model_payload["feature_scale"], dtype=float)
        x_val_norm = (x_val - mean) / scale
        estimators = model_payload["estimators"]
        preds = []
        for estimator in estimators:
            coef = np.asarray(estimator["coef"], dtype=float)
            intercept = float(estimator["intercept"])
            preds.append(x_val_norm @ coef + intercept)
        return np.vstack(preds).T
    raise ValueError(f"predict_with_saved_model does not support family: {family}")


def render_trajectory(flat_values: np.ndarray) -> str:
    points = []
    for idx in range(6):
        x = float(flat_values[idx * 2])
        y = float(flat_values[idx * 2 + 1])
        points.append(f"({x:.1f},{y:.1f})")
    return f"The suggested future trajectory is [{','.join(points)}]."


def write_predictions_jsonl(
    *,
    path: Path,
    val_samples: tuple[BenchmarkSample, ...],
    y_val_pred: np.ndarray,
    scenario_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(val_samples):
            record = {
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "split_name": sample.split_name,
                "task_type": BenchmarkTaskType.FUTURE_TRAJECTORY.value,
                "qa_type_id": sample.qa_type_id,
                "supported": True,
                "answer_text": render_trajectory(y_val_pred[idx]),
                "object_ids": [],
                "baseline_mode": "cooperative",
                "graph_ablation_mode": "full",
                "scenario_name": scenario_name,
            }
            handle.write(json.dumps(record) + "\n")


def write_prediction_manifest(
    *,
    path: Path,
    v2vgot_root: str,
    val_file_name: str,
    scenario_name: str,
    output_jsonl: Path,
    total_samples: int,
) -> None:
    payload = {
        "repository_root": v2vgot_root,
        "split": "val",
        "file_name": val_file_name,
        "scenario_name": scenario_name,
        "task_types": [BenchmarkTaskType.FUTURE_TRAJECTORY.value],
        "runs": [
            {
                "task_type": BenchmarkTaskType.FUTURE_TRAJECTORY.value,
                "scenario_name": scenario_name,
                "file_name": val_file_name,
                "baseline_mode": "cooperative",
                "graph_ablation_mode": "full",
                "future_trajectory_model_json": "q9_sweep_model_inline",
                "output_jsonl": str(output_jsonl),
                "supported_predictions": total_samples,
                "unsupported_predictions": 0,
                "total_samples": total_samples,
                "qa_type_ids": [19],
                "qa_type_id": 19,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(command: list[str], *, v2vgot_root: str) -> None:
    print("$ " + " ".join(command), flush=True)
    env = dict(**{"V2VGOT_ROOT": v2vgot_root}, **dict(**{}))
    env.update(**dict())
    subprocess.run(command, cwd=str(REPO_ROOT), check=True, env=env)


def run_official_export_eval(
    *,
    python_bin: str,
    v2vgot_root: str,
    prediction_manifest: Path,
    output_root: Path,
    scenario_name: str,
    val_file_name: str,
) -> tuple[str | None, str | None]:
    export_dir = output_root / f"{scenario_name}_official_exports"
    reports_dir = output_root / f"{scenario_name}_official_eval_reports"
    tools_dir = export_dir / "tools"
    export_manifest = export_dir / f"{scenario_name}_official_export_manifest.json"
    official_summary = reports_dir / f"{scenario_name}_official_export_manifest_official_qa_eval_summary.json"

    run_command(
        [
            python_bin,
            "scripts/export_qa_predictions.py",
            "--manifest",
            str(prediction_manifest),
            "--output-dir",
            str(export_dir),
            "--split",
            "val",
            "--scenario-name",
            scenario_name,
            "--task-type",
            BenchmarkTaskType.FUTURE_TRAJECTORY.value,
            "--qa-type-id",
            "19",
            "--file-name",
            val_file_name,
        ],
        v2vgot_root=v2vgot_root,
    )
    run_command(
        [
            python_bin,
            "scripts/run_v2vgot_official_qa_eval.py",
            "--export-manifest",
            str(export_manifest),
            "--output-dir",
            str(reports_dir),
            "--tools-dir",
            str(tools_dir),
            "--task-type",
            BenchmarkTaskType.FUTURE_TRAJECTORY.value,
            "--num-future-waypoints",
            "6",
            "--npy-save-path",
            str(Path(v2vgot_root) / "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy"),
            "--v2vgot-root",
            v2vgot_root,
        ],
        v2vgot_root=v2vgot_root,
    )
    return (str(export_manifest), str(official_summary) if official_summary.exists() else None)


def main() -> int:
    args = build_parser().parse_args()
    output_root = resolve_output_root(args.output_root)
    run_root = output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    adapter = V2VGoTQABenchmarkAdapter(str(Path(args.v2vgot_root).expanduser().resolve()))
    train_samples = load_q9_samples(
        adapter=adapter,
        split_name="train",
        file_name=args.train_file_name,
        limit=args.limit_train,
    )
    val_samples = load_q9_samples(
        adapter=adapter,
        split_name="val",
        file_name=args.val_file_name,
        limit=args.limit_val,
    )
    if not train_samples:
        raise ValueError("No Q9 train samples loaded.")
    if not val_samples:
        raise ValueError("No Q9 val samples loaded.")

    if not args.allow_train_val_overlap:
        val_keys = {qa_overlap_key(sample) for sample in val_samples}
        train_samples = tuple(sample for sample in train_samples if qa_overlap_key(sample) not in val_keys)
        if not train_samples:
            raise ValueError("All train samples were removed by train/val overlap filtering.")

    q8_train_lookup: dict[str, list[float]] = {}
    q8_val_lookup: dict[str, list[float]] = {}
    q8_feature_source: dict[str, object] | None = None
    if args.include_q8_pred_features:
        if args.q8_feature_source == "question_context":
            q8_feature_dir = run_root / "q8_question_context_features"
            q8_train_jsonl = q8_feature_dir / f"{args.run_name}_q8_context_features_train.jsonl"
            q8_val_jsonl = q8_feature_dir / f"{args.run_name}_q8_context_features_val.jsonl"
            print("[INFO] Building Q8 features from Q9 question context.", flush=True)
            q8_train_lookup = build_q8_prediction_lookup_from_question_context(
                samples=train_samples,
                output_jsonl=q8_train_jsonl,
                progress_every=args.progress_every,
            )
            q8_val_lookup = build_q8_prediction_lookup_from_question_context(
                samples=val_samples,
                output_jsonl=q8_val_jsonl,
                progress_every=args.progress_every,
            )
            q8_feature_source = {
                "mode": "question_context",
                "train_q8_features_jsonl": str(q8_train_jsonl),
                "val_q8_features_jsonl": str(q8_val_jsonl),
                "note": (
                    "Q8 controls are parsed from the speed/steering context already "
                    "present in the V2V-GoT Q9 prompt. This avoids rerunning KG/Q8 "
                    "feature generation and matches the staged Q8->Q9 prompt input."
                ),
            }
        elif args.q8_feature_source == "q8_model" and args.q8_model_json:
            q8_model_path = str(Path(args.q8_model_json).expanduser().resolve())
            q8_model = load_q8_model(q8_model_path)
            evaluator = V2VGoTQAPhase5AEvaluator(str(Path(args.v2vgot_root).expanduser().resolve()))
            q8_feature_dir = run_root / "q8_model_features"
            q8_train_jsonl = q8_feature_dir / f"{args.run_name}_q8_features_train.jsonl"
            q8_val_jsonl = q8_feature_dir / f"{args.run_name}_q8_features_val.jsonl"
            print(f"[INFO] Building split-correct Q8 model features from {q8_model_path}", flush=True)
            q8_train_lookup = build_q8_prediction_lookup_from_model(
                samples=train_samples,
                evaluator=evaluator,
                q8_model=q8_model,
                output_jsonl=q8_train_jsonl,
                progress_every=args.progress_every,
                timeout_seconds=args.q8_feature_timeout_seconds,
                debug_every=args.q8_feature_debug_every,
            )
            q8_val_lookup = build_q8_prediction_lookup_from_model(
                samples=val_samples,
                evaluator=evaluator,
                q8_model=q8_model,
                output_jsonl=q8_val_jsonl,
                progress_every=args.progress_every,
                timeout_seconds=args.q8_feature_timeout_seconds,
                debug_every=args.q8_feature_debug_every,
            )
            q8_feature_source = {
                "mode": "q8_model_json",
                "q8_model_json": q8_model_path,
                "train_q8_features_jsonl": str(q8_train_jsonl),
                "val_q8_features_jsonl": str(q8_val_jsonl),
                "q8_feature_timeout_seconds": args.q8_feature_timeout_seconds,
                "q8_feature_debug_every": args.q8_feature_debug_every,
                "note": (
                    "Q8 controls are predicted from KG-prepared scenes separately "
                    "for train and val, matching V2V-GoT staged Q8->Q9 context."
                ),
            }
        elif args.q8_feature_source == "legacy_jsonl" and args.q8_predictions_jsonl:
            q8_lookup = load_q8_prediction_lookup(args.q8_predictions_jsonl)
            q8_train_lookup = q8_lookup
            q8_val_lookup = q8_lookup
            q8_feature_source = {
                "mode": "q8_predictions_jsonl_legacy_shared_lookup",
                "q8_predictions_jsonl": args.q8_predictions_jsonl,
                "note": (
                    "Legacy shared lookup. Prefer --q8-model-json so train and val "
                    "Q8 features are generated from their own splits, or prefer "
                    "--q8-feature-source=question_context to reuse the staged Q8 "
                    "context already present in the Q9 prompt."
                ),
            }
            print(f"[INFO] Loaded legacy Q8 prediction features for {len(q8_lookup)} samples.", flush=True)
        else:
            raise ValueError(
                "--include-q8-pred-features requires a valid --q8-feature-source: "
                "question_context, q8_model with --q8-model-json, or legacy_jsonl "
                "with --q8-predictions-jsonl."
            )

    x_train, y_train, usable_train = build_xy(
        train_samples,
        include_q8_pred_features=args.include_q8_pred_features,
        q8_lookup=q8_train_lookup,
        progress_every=args.progress_every,
    )
    x_val, y_val, usable_val = build_xy(
        val_samples,
        include_q8_pred_features=args.include_q8_pred_features,
        q8_lookup=q8_val_lookup,
        progress_every=args.progress_every,
    )
    print(
        "[INFO] feature matrices ready: "
        f"x_train={x_train.shape} y_train={y_train.shape} usable_train={usable_train}; "
        f"x_val={x_val.shape} y_val={y_val.shape} usable_val={usable_val}",
        flush=True,
    )

    model_results: list[SweepModelResult] = []
    for model_name in args.models:
        model_name = str(model_name).strip().lower()
        print(f"\n[INFO] Training model: {model_name}", flush=True)

        if model_name == "ridge":
            y_val_pred, model_payload = train_predict_ridge(x_train, y_train, x_val, alpha=1.0)
            y_train_pred = predict_with_saved_model(model_payload, x_train)
        elif model_name == "elasticnet":
            y_val_pred, model_payload = train_predict_elasticnet(x_train, y_train, x_val, alpha=0.001, l1_ratio=0.1)
            y_train_pred = predict_with_saved_model(model_payload, x_train)
        elif model_name == "rf":
            y_val_pred, model_payload = train_predict_rf(x_train, y_train, x_val, n_estimators=200, max_depth=12)
            y_train_pred = train_predict_rf(x_train, y_train, x_train, n_estimators=200, max_depth=12)[0]
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Use ridge elasticnet rf.")

        train_metrics = l2_metrics(y_train, y_train_pred)
        val_metrics = l2_metrics(y_val, y_val_pred)
        print(
            f"[INFO] model={model_name} metrics: "
            f"train_l2_avg={train_metrics['l2_error_avg_all']:.6f} "
            f"val_l2_avg={val_metrics['l2_error_avg_all']:.6f}",
            flush=True,
        )

        scenario_name = f"{args.run_name}_{model_name}"
        model_json = run_root / f"{scenario_name}_model.json"
        pred_jsonl = run_root / f"{scenario_name}_predictions.jsonl"
        pred_manifest = run_root / f"{scenario_name}_manifest.json"

        model_record = {
            "run_name": args.run_name,
            "scenario_name": scenario_name,
            "model_name": model_name,
            "feature_names": feature_names(args.include_q8_pred_features),
            "include_q8_pred_features": bool(args.include_q8_pred_features),
            "train_rows": len(train_samples),
            "val_rows": len(val_samples),
            "usable_train_rows": usable_train,
            "usable_val_rows": usable_val,
            "train_metrics_local": train_metrics,
            "val_metrics_local": val_metrics,
            "model_payload": model_payload,
            "leakage_policy": {
                "excluded_fields": [
                    "dist",
                    "angle",
                    "suggested_speed_idx",
                    "suggested_steering_idx",
                    "future_trajectory_str_in_ego",
                    "future_trajectory_str_in_self",
                ],
                "q8_pred_feature_source": q8_feature_source,
            },
        }
        model_json.write_text(json.dumps(model_record, indent=2), encoding="utf-8")

        write_predictions_jsonl(
            path=pred_jsonl,
            val_samples=val_samples,
            y_val_pred=y_val_pred,
            scenario_name=scenario_name,
        )
        write_prediction_manifest(
            path=pred_manifest,
            v2vgot_root=str(Path(args.v2vgot_root).expanduser().resolve()),
            val_file_name=args.val_file_name,
            scenario_name=scenario_name,
            output_jsonl=pred_jsonl,
            total_samples=len(val_samples),
        )

        export_manifest_json: str | None = None
        official_summary_json: str | None = None
        if args.run_official_eval:
            export_manifest_json, official_summary_json = run_official_export_eval(
                python_bin=args.python,
                v2vgot_root=str(Path(args.v2vgot_root).expanduser().resolve()),
                prediction_manifest=pred_manifest,
                output_root=run_root,
                scenario_name=scenario_name,
                val_file_name=args.val_file_name,
            )

        model_results.append(
            SweepModelResult(
                model_name=model_name,
                model_json=model_json,
                prediction_jsonl=pred_jsonl,
                prediction_manifest_json=pred_manifest,
                train_rows=len(train_samples),
                val_rows=len(val_samples),
                train_l2_avg=float(train_metrics["l2_error_avg_all"]),
                val_l2_avg=float(val_metrics["l2_error_avg_all"]),
                val_l2_1s=float(val_metrics["l2_error_avg_1s"]),
                val_l2_2s=float(val_metrics["l2_error_avg_2s"]),
                val_l2_3s=float(val_metrics["l2_error_avg_3s"]),
                official_export_manifest_json=export_manifest_json,
                official_summary_json=official_summary_json,
            )
        )

    consolidated = {
        "run_name": args.run_name,
        "v2vgot_root": str(Path(args.v2vgot_root).expanduser().resolve()),
        "train_file_name": args.train_file_name,
        "val_file_name": args.val_file_name,
        "include_q8_pred_features": bool(args.include_q8_pred_features),
        "q8_feature_source": q8_feature_source,
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
        "models": [
            {
                "model_name": item.model_name,
                "model_json": str(item.model_json),
                "prediction_jsonl": str(item.prediction_jsonl),
                "prediction_manifest_json": str(item.prediction_manifest_json),
                "train_rows": item.train_rows,
                "val_rows": item.val_rows,
                "train_l2_avg": item.train_l2_avg,
                "val_l2_1s": item.val_l2_1s,
                "val_l2_2s": item.val_l2_2s,
                "val_l2_3s": item.val_l2_3s,
                "val_l2_avg": item.val_l2_avg,
                "official_export_manifest_json": item.official_export_manifest_json,
                "official_summary_json": item.official_summary_json,
            }
            for item in model_results
        ],
    }
    consolidated_path = run_root / f"{args.run_name}_consolidated_manifest.json"
    consolidated_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")

    print("\n============================================================")
    print("Q9 Sweep Complete")
    print("============================================================")
    print(f"run_root: {run_root}")
    print(f"consolidated_manifest: {consolidated_path}")
    print("\nModel summary:")
    for item in model_results:
        print(
            f"  - {item.model_name}: val_l2_avg={item.val_l2_avg:.4f} "
            f"(1s={item.val_l2_1s:.4f}, 2s={item.val_l2_2s:.4f}, 3s={item.val_l2_3s:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
