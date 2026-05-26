from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from kg_coop_drive.application.control_settings_policy import decide_control_settings
from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator
from kg_coop_drive.domain.benchmark import BenchmarkTaskType
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter
from kg_coop_drive.option2.models import build_regression_backend

POSITION_RE = re.compile(
    r"I am\s+(?P<agent>[A-Za-z0-9_]+)\s+at\s+"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
)
COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")

SPEED_LABELS = ("fast", "moderate", "slow", "very slow", "stop")
STEER_LABELS = ("left", "slightly left", "straight", "slightly right", "right")
Q8_SPEED_CONTROL_VALUES = {"fast": 1.0, "moderate": 0.65, "slow": 0.35, "very slow": 0.15, "stop": 0.0}
Q8_STEERING_CONTROL_VALUES = {"left": -1.0, "slightly left": -0.5, "straight": 0.0, "slightly right": 0.5, "right": 1.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Option2 Q9 trainer with mandatory Q8-context features.")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--train-split", default="train", choices=("train", "val"))
    parser.add_argument("--val-split", default="val", choices=("train", "val"))
    parser.add_argument("--q8-model-json", required=True)
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--backend", default="auto", choices=("auto", "lightgbm", "sklearn_gbdt"))
    parser.add_argument("--n-estimators", type=int, default=280)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=96)
    parser.add_argument("--subsample", type=float, default=0.7)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    return parser


def parse_current_position(question: str) -> tuple[float, float]:
    match = POSITION_RE.search(question)
    if match:
        return float(match.group("x")), float(match.group("y"))
    return (0.0, 0.0)


def parse_waypoints(text: str, limit: int = 6) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in COORD_RE.findall(text)[:limit])


def q8_feature_vector(speed_label: str, steering_label: str) -> list[float]:
    speed = speed_label.strip().lower()
    steer = steering_label.strip().lower()
    speed_one_hot = [1.0 if label == speed else 0.0 for label in SPEED_LABELS]
    steer_one_hot = [1.0 if label == steer else 0.0 for label in STEER_LABELS]
    return speed_one_hot + steer_one_hot + [
        Q8_SPEED_CONTROL_VALUES.get(speed, 0.0),
        Q8_STEERING_CONTROL_VALUES.get(steer, 0.0),
    ]


def base_feature_vector(sample) -> list[float]:
    current_x, current_y = parse_current_position(sample.scene.raw_question)
    asker_raw = str(sample.raw_record.get("asker_cav_id", "")).strip()
    asker_is_cav1 = 1.0 if asker_raw == "1" else 0.0
    return [1.0, current_x, current_y, asker_is_cav1]


def build_rows(*, samples, evaluator, q8_model, baseline_mode: str, progress_every: int) -> tuple[np.ndarray, np.ndarray, int]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    usable = 0
    for idx, sample in enumerate(samples, start=1):
        conversations = sample.raw_record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2 or not isinstance(conversations[1], dict):
            continue
        answer_text = str(conversations[1].get("value", ""))
        gt_points = parse_waypoints(answer_text, limit=6)
        if len(gt_points) != 6:
            continue
        prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode=baseline_mode)
        q8_decision = decide_control_settings(
            scene=prepared_scene,
            selection_policy="linear_classifier",
            model=q8_model,
        )
        q8_feats = q8_feature_vector(q8_decision.speed_instruction, q8_decision.steering_instruction)
        feat = base_feature_vector(sample) + q8_feats

        current_x, current_y = parse_current_position(sample.scene.raw_question)
        target = []
        for px, py in gt_points:
            target.extend([px - current_x, py - current_y])

        x_rows.append(feat)
        y_rows.append(target)
        usable += 1
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(samples)):
            print(f"q9_build_progress: {idx}/{len(samples)} usable={usable}")

    if not x_rows:
        raise SystemExit("No usable Q9 rows found.")
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), usable


def l2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(y_pred.reshape(-1, 6, 2) - y_true.reshape(-1, 6, 2), axis=2)
    return {
        "l2_error_avg_1s": float(np.mean(np.mean(errors[:, :2], axis=1))),
        "l2_error_avg_2s": float(np.mean(np.mean(errors[:, :4], axis=1))),
        "l2_error_avg_3s": float(np.mean(np.mean(errors[:, :6], axis=1))),
        "l2_error_avg_all": float(np.mean(np.mean(errors, axis=1))),
    }


def main() -> None:
    args = build_parser().parse_args()

    q8_path = Path(args.q8_model_json).expanduser().resolve()
    if not q8_path.exists():
        raise SystemExit(f"Required Q8 model JSON not found: {q8_path}")
    q8_model = json.loads(q8_path.read_text(encoding="utf-8"))

    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    evaluator = V2VGoTQAPhase5AEvaluator(args.v2vgot_root)

    train_samples = tuple(
        s for s in adapter.load_samples(split_name=args.train_split, file_name=args.file_name)
        if s.task_type == BenchmarkTaskType.FUTURE_TRAJECTORY
    )
    val_samples = tuple(
        s for s in adapter.load_samples(split_name=args.val_split, file_name=args.file_name)
        if s.task_type == BenchmarkTaskType.FUTURE_TRAJECTORY
    )
    if args.train_limit > 0:
        train_samples = train_samples[: args.train_limit]
    if args.val_limit > 0:
        val_samples = val_samples[: args.val_limit]

    print("[1/4] Building train rows...")
    x_train, y_train, train_usable = build_rows(
        samples=train_samples,
        evaluator=evaluator,
        q8_model=q8_model,
        baseline_mode=args.baseline_mode,
        progress_every=args.progress_every,
    )
    print("[2/4] Building val rows...")
    x_val, y_val, val_usable = build_rows(
        samples=val_samples,
        evaluator=evaluator,
        q8_model=q8_model,
        baseline_mode=args.baseline_mode,
        progress_every=args.progress_every,
    )

    print("[3/4] Training...")
    backend = build_regression_backend(
        backend=args.backend,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        subsample=float(args.subsample),
        num_leaves=int(args.num_leaves),
    )
    backend.fit(x_train, y_train)

    print("[4/4] Evaluating + writing artifacts...")
    train_pred = backend.predict(x_train)
    val_pred = backend.predict(x_val)

    feat_names = ["bias", "current_x", "current_y", "asker_is_cav1"]
    feat_names.extend([f"q8_speed_{label.replace(' ', '_')}" for label in SPEED_LABELS])
    feat_names.extend([f"q8_steer_{label.replace(' ', '_')}" for label in STEER_LABELS])
    feat_names.extend(["q8_pred_speed_control_value", "q8_pred_steering_control_value"])

    report = {
        "module": "option2_q9_with_q8_context",
        "task_scope": "future_trajectory_q9",
        "q8_model_json": str(q8_path),
        "feature_names": feat_names,
        "train_rows_total": len(train_samples),
        "val_rows_total": len(val_samples),
        "train_rows_usable": train_usable,
        "val_rows_usable": val_usable,
        "train_metrics": l2_metrics(y_train, train_pred),
        "val_metrics": l2_metrics(y_val, val_pred),
        "backend": backend.export_metadata(),
        "guards": {
            "q8_model_path_required": True,
            "fallback_used": False,
            "feature_source": "scene_and_q8_model_predictions_only",
        },
    }

    model_payload = {
        "model_type": "option2_q9_regressor_with_q8_context_v1",
        "q8_model_json": str(q8_path),
        "feature_names": feat_names,
        "backend": backend.export_metadata(),
    }

    out_model = Path(args.output_model_json).expanduser().resolve()
    out_report = Path(args.output_report_json).expanduser().resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved_model: {out_model}")
    print(f"saved_report: {out_report}")


if __name__ == "__main__":
    main()
