#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.planning.control_settings_policy import (  # noqa: E402
    SPEED_CLASSES,
    STEERING_CLASSES,
    build_control_feature_vector,
    control_feature_names,
    parse_speed_steering_idx,
)
from kg_coop_drive.application.qa.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.qa.v2vgotqa_evaluator import GraphAblationMode  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train frozen Q8 speed/steering linear classifiers from train split features."
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument(
        "--file-name",
        default="v2v4real_3d_grounding_qa_dataset_v2vgot.json",
        help=(
            "QA JSON filename under the split co_llm directory, or an absolute "
            "path to an isolated QA JSON."
        ),
    )
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument(
        "--graph-ablation-mode",
        default=GraphAblationMode.FULL.value,
        choices=tuple(item.value for item in GraphAblationMode),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--feature-set",
        default="base",
        choices=("base", "extended_v1"),
        help="Control feature set used for training and inference.",
    )
    parser.add_argument(
        "--speed-head-type",
        default="multiclass",
        choices=("multiclass", "ordinal"),
        help="Speed head type. ordinal uses 4 ordered binary heads.",
    )
    parser.add_argument(
        "--speed-class-weighting",
        default="none",
        choices=("none", "inverse_freq", "sqrt_inverse_freq"),
        help="Class reweighting strategy for speed head.",
    )
    parser.add_argument(
        "--steering-class-weighting",
        default="none",
        choices=("none", "inverse_freq", "sqrt_inverse_freq"),
        help="Class reweighting strategy for steering head.",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=1e-6,
        help="Ridge regularization for linear heads.",
    )
    parser.add_argument(
        "--speed-ordinal-threshold-policy",
        default="global",
        choices=("global", "risk3"),
        help="Threshold policy for ordinal speed head.",
    )
    parser.add_argument(
        "--speed-risk-split-low",
        type=float,
        default=0.2,
        help="Low-risk upper bound for risk3 ordinal threshold policy.",
    )
    parser.add_argument(
        "--speed-risk-split-high",
        type=float,
        default=0.5,
        help="Mid-risk upper bound for risk3 ordinal threshold policy.",
    )
    return parser


def one_hot(index: int, num_classes: int) -> np.ndarray:
    vector = np.zeros(num_classes, dtype=float)
    if 0 <= index < num_classes:
        vector[index] = 1.0
    return vector


def build_sample_weights(
    labels: list[int],
    num_classes: int,
    scheme: str,
) -> np.ndarray:
    weights = np.ones(len(labels), dtype=float)
    if scheme == "none" or not labels:
        return weights
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    counts = np.where(counts <= 0.0, 1.0, counts)
    if scheme == "inverse_freq":
        class_weights = 1.0 / counts
    elif scheme == "sqrt_inverse_freq":
        class_weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unsupported weighting scheme: {scheme}")
    class_weights *= float(num_classes) / float(np.sum(class_weights))
    for idx, label in enumerate(labels):
        if 0 <= label < num_classes:
            weights[idx] = class_weights[label]
    return weights


def fit_weighted_linear_head(
    x_aug: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    l2_regularization: float,
) -> np.ndarray:
    sqrt_w = np.sqrt(np.asarray(sample_weights, dtype=float))
    xw = x_aug * sqrt_w[:, None]
    yw = y * sqrt_w[:, None]
    xtx = xw.T @ xw
    if l2_regularization > 0.0:
        reg = np.eye(xtx.shape[0], dtype=float) * float(l2_regularization)
        reg[0, 0] = 0.0
        xtx = xtx + reg
    xty = xw.T @ yw
    coeff = np.linalg.solve(xtx, xty)
    return coeff.T


def average_edit_distance(pred: np.ndarray, labels: list[int]) -> float:
    if not labels:
        return 0.0
    target = np.asarray(labels, dtype=int)
    return float(np.mean(np.abs(pred.astype(int) - target)))


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_ordinal_speed(
    x_aug: np.ndarray,
    ordinal_weights: np.ndarray,
    threshold: float,
) -> np.ndarray:
    scores = x_aug @ ordinal_weights.T
    probs = sigmoid(scores)
    pred = np.sum(probs > threshold, axis=1)
    return np.clip(pred, 0, len(SPEED_CLASSES) - 1).astype(int)


def speed_risk_bucket(risk_value: float, low: float, high: float) -> str:
    if risk_value < low:
        return "low"
    if risk_value < high:
        return "mid"
    return "high"


def main() -> None:
    args = build_parser().parse_args()
    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    evaluator = V2VGoTQAPhase5AEvaluator(
        args.v2vgot_root,
        graph_ablation=args.graph_ablation_mode,
    )

    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name=args.split, file_name=args.file_name)
        if sample.task_type == BenchmarkTaskType.CONTROL_SETTINGS
    )
    if args.limit > 0:
        samples = samples[: args.limit]

    feature_rows: list[np.ndarray] = []
    speed_labels: list[int] = []
    steering_labels: list[int] = []

    for sample in samples:
        prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode=args.baseline_mode)
        feature_rows.append(
            np.asarray(
                build_control_feature_vector(prepared_scene, feature_set=args.feature_set),
                dtype=float,
            )
        )
        conversations = sample.raw_record.get("conversations", [])
        gt_answer = ""
        if isinstance(conversations, list) and len(conversations) > 1 and isinstance(conversations[1], dict):
            gt_answer = str(conversations[1].get("value", ""))
        speed_idx, steering_idx = parse_speed_steering_idx(gt_answer)
        speed_labels.append(speed_idx)
        steering_labels.append(steering_idx)

    if not feature_rows:
        raise SystemExit("No control_settings samples found for training.")

    x = np.vstack(feature_rows)
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    x_norm = (x - feature_mean) / feature_std
    x_aug = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=float), x_norm], axis=1)

    y_steering = np.vstack([one_hot(index, len(STEERING_CLASSES)) for index in steering_labels])

    speed_sample_weights = build_sample_weights(
        labels=speed_labels,
        num_classes=len(SPEED_CLASSES),
        scheme=args.speed_class_weighting,
    )
    steering_sample_weights = build_sample_weights(
        labels=steering_labels,
        num_classes=len(STEERING_CLASSES),
        scheme=args.steering_class_weighting,
    )

    speed_weights = None
    speed_ordinal_weights = None
    speed_ordinal_threshold = 0.5
    speed_ordinal_thresholds = {
        "default": 0.5,
        "low": 0.5,
        "mid": 0.5,
        "high": 0.5,
    }
    speed_pred = None
    speed_accuracy = 0.0
    speed_edit_dist = 0.0

    if args.speed_head_type == "multiclass":
        y_speed = np.vstack([one_hot(index, len(SPEED_CLASSES)) for index in speed_labels])
        speed_weights = fit_weighted_linear_head(
            x_aug=x_aug,
            y=y_speed,
            sample_weights=speed_sample_weights,
            l2_regularization=args.l2_regularization,
        )
        speed_scores = x_aug @ speed_weights.T
        speed_pred = np.argmax(speed_scores, axis=1)
        speed_accuracy = float(np.mean(speed_pred == np.asarray(speed_labels)))
        speed_edit_dist = average_edit_distance(speed_pred, speed_labels)
    else:
        ordinal_rows: list[np.ndarray] = []
        for threshold_idx in range(len(SPEED_CLASSES) - 1):
            y_binary = np.asarray([1.0 if label > threshold_idx else 0.0 for label in speed_labels], dtype=float)
            y_binary = y_binary.reshape(-1, 1)
            head_weights = fit_weighted_linear_head(
                x_aug=x_aug,
                y=y_binary,
                sample_weights=speed_sample_weights,
                l2_regularization=args.l2_regularization,
            )
            ordinal_rows.append(head_weights[0])
        speed_ordinal_weights = np.vstack(ordinal_rows)

        best_tuple: tuple[float, float, float] | None = None
        best_pred = None
        for threshold in np.linspace(0.35, 0.65, 31):
            pred = predict_ordinal_speed(x_aug, speed_ordinal_weights, float(threshold))
            edit_dist = average_edit_distance(pred, speed_labels)
            accuracy = float(np.mean(pred == np.asarray(speed_labels)))
            tie_break = -float(threshold)
            candidate = (edit_dist, -accuracy, tie_break)
            if best_tuple is None or candidate < best_tuple:
                best_tuple = candidate
                speed_ordinal_threshold = float(threshold)
                best_pred = pred
        assert best_pred is not None

        if args.speed_ordinal_threshold_policy == "risk3":
            top1_risk_values = x[:, 0]
            speed_pred = np.zeros(len(speed_labels), dtype=int)
            speed_ordinal_thresholds["default"] = float(speed_ordinal_threshold)
            for bucket_name in ("low", "mid", "high"):
                indices = [
                    idx
                    for idx, risk_value in enumerate(top1_risk_values)
                    if speed_risk_bucket(
                        float(risk_value),
                        args.speed_risk_split_low,
                        args.speed_risk_split_high,
                    )
                    == bucket_name
                ]
                if not indices:
                    speed_ordinal_thresholds[bucket_name] = float(speed_ordinal_threshold)
                    continue

                best_bucket_tuple: tuple[float, float, float] | None = None
                best_bucket_threshold = float(speed_ordinal_threshold)
                for threshold in np.linspace(0.35, 0.75, 41):
                    pred = predict_ordinal_speed(
                        x_aug[np.asarray(indices)],
                        speed_ordinal_weights,
                        float(threshold),
                    )
                    bucket_labels = [speed_labels[idx] for idx in indices]
                    edit_dist = average_edit_distance(pred, bucket_labels)
                    accuracy = float(np.mean(pred == np.asarray(bucket_labels)))
                    tie_break = -float(threshold)
                    candidate = (edit_dist, -accuracy, tie_break)
                    if best_bucket_tuple is None or candidate < best_bucket_tuple:
                        best_bucket_tuple = candidate
                        best_bucket_threshold = float(threshold)
                speed_ordinal_thresholds[bucket_name] = best_bucket_threshold

            for idx, risk_value in enumerate(top1_risk_values):
                bucket_name = speed_risk_bucket(
                    float(risk_value),
                    args.speed_risk_split_low,
                    args.speed_risk_split_high,
                )
                threshold = speed_ordinal_thresholds[bucket_name]
                pred = predict_ordinal_speed(
                    x_aug[idx : idx + 1],
                    speed_ordinal_weights,
                    float(threshold),
                )
                speed_pred[idx] = int(pred[0])
        else:
            speed_pred = best_pred
            speed_ordinal_thresholds["default"] = float(speed_ordinal_threshold)
            speed_ordinal_thresholds["low"] = float(speed_ordinal_threshold)
            speed_ordinal_thresholds["mid"] = float(speed_ordinal_threshold)
            speed_ordinal_thresholds["high"] = float(speed_ordinal_threshold)

        assert speed_pred is not None
        speed_accuracy = float(np.mean(speed_pred == np.asarray(speed_labels)))
        speed_edit_dist = average_edit_distance(speed_pred, speed_labels)

    steering_weights = fit_weighted_linear_head(
        x_aug=x_aug,
        y=y_steering,
        sample_weights=steering_sample_weights,
        l2_regularization=args.l2_regularization,
    )

    steering_scores = x_aug @ steering_weights.T
    steering_pred = np.argmax(steering_scores, axis=1)
    assert speed_pred is not None
    steering_accuracy = float(np.mean(steering_pred == np.asarray(steering_labels)))
    action_accuracy = float(
        np.mean(
            (speed_pred == np.asarray(speed_labels))
            & (steering_pred == np.asarray(steering_labels))
        )
    )
    steering_edit_dist = average_edit_distance(steering_pred, steering_labels)
    action_edit_dist = speed_edit_dist + steering_edit_dist

    report = {
        "split": args.split,
        "baseline_mode": args.baseline_mode,
        "samples": len(samples),
        "feature_dim": int(x.shape[1]),
        "training_config": {
            "feature_set": args.feature_set,
            "speed_head_type": args.speed_head_type,
            "speed_class_weighting": args.speed_class_weighting,
            "steering_class_weighting": args.steering_class_weighting,
            "l2_regularization": args.l2_regularization,
            "speed_ordinal_threshold": speed_ordinal_threshold if args.speed_head_type == "ordinal" else None,
            "speed_ordinal_threshold_policy": args.speed_ordinal_threshold_policy,
            "speed_risk_split_low": args.speed_risk_split_low,
            "speed_risk_split_high": args.speed_risk_split_high,
            "speed_ordinal_thresholds": speed_ordinal_thresholds if args.speed_head_type == "ordinal" else None,
        },
        "train_metrics": {
            "speed_accuracy": speed_accuracy,
            "steering_accuracy": steering_accuracy,
            "action_accuracy": action_accuracy,
            "speed_edit_dist": speed_edit_dist,
            "steering_edit_dist": steering_edit_dist,
            "action_edit_dist": action_edit_dist,
        },
    }
    model = {
        "model_type": "phase9_q8_control_linear_classifier_v2"
        if args.speed_head_type == "ordinal"
        else "phase9_q8_control_linear_classifier_v1",
        "feature_set": args.feature_set,
        "speed_head_type": args.speed_head_type,
        "speed_classes": list(SPEED_CLASSES),
        "steering_classes": list(STEERING_CLASSES),
        "feature_names": list(control_feature_names(args.feature_set)),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "speed_weights": speed_weights.tolist() if speed_weights is not None else [],
        "speed_ordinal_weights": (
            speed_ordinal_weights.tolist() if speed_ordinal_weights is not None else []
        ),
        "speed_ordinal_threshold": speed_ordinal_threshold,
        "speed_ordinal_threshold_policy": args.speed_ordinal_threshold_policy,
        "speed_risk_split_low": args.speed_risk_split_low,
        "speed_risk_split_high": args.speed_risk_split_high,
        "speed_ordinal_thresholds": speed_ordinal_thresholds,
        "steering_weights": steering_weights.tolist(),
        "report": report,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")

    if args.output_report:
        output_report = Path(args.output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"samples: {len(samples)}")
    print(f"train_metrics: {report['train_metrics']}")
    print(f"saved_model: {output_json.resolve()}")
    if args.output_report:
        print(f"saved_report: {Path(args.output_report).resolve()}")


if __name__ == "__main__":
    main()
