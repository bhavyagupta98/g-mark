#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.planning_awareness import (  # noqa: E402
    PLANNING_COUNT_GATE_FEATURE_NAMES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a scene-level Q4 count gate around an existing frozen candidate "
            "acceptor. The output JSON preserves the candidate model and adds a "
            "count_gate block used by --planning-selection-policy count_gated_acceptor."
        )
    )
    parser.add_argument("--train-features-jsonl", required=True)
    parser.add_argument("--candidate-model-json", required=True)
    parser.add_argument("--output-dir", default="outputs/phase8_train_dev/q4_policy_optimization")
    parser.add_argument("--run-name", default="q4_planning_rel_count_gated_logreg")
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--soft-extra-min-probability", type=float, default=0.62)
    parser.add_argument("--soft-extra-min-relative-to-k", type=float, default=0.90)
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("row_type") == "candidate":
                    rows.append(row)
    return rows


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    values = [math.exp(value - max_logit) for value in logits]
    total = sum(values)
    return [value / total for value in values] if total else [0.0 for _ in values]


def candidate_probability(model: dict[str, Any], row: dict[str, Any]) -> float:
    feature_names = [str(item) for item in model.get("feature_names", [])]
    normalization = model.get("normalization", {})
    means = normalization.get("mean", {}) if isinstance(normalization, dict) else {}
    stds = normalization.get("std", {}) if isinstance(normalization, dict) else {}
    vector = []
    for name in feature_names:
        raw_value = float(row.get(name, 0.0) or 0.0)
        mean = float(means.get(name, 0.0)) if isinstance(means, dict) else 0.0
        std = float(stds.get(name, 1.0)) if isinstance(stds, dict) else 1.0
        if std <= 0.0:
            std = 1.0
        vector.append((raw_value - mean) / std)
    if model.get("model_type") == "mlp":
        hidden = int(model.get("hidden", 0))
        w1 = [[float(value) for value in item] for item in model.get("w1", [])]
        b1 = [float(value) for value in model.get("b1", [])]
        w2 = [float(value) for value in model.get("w2", [])]
        b2 = float(model.get("b2", 0.0))
        if hidden <= 0 or len(w1) != hidden or len(b1) != hidden or len(w2) != hidden:
            return 0.0
        hidden_values = [
            math.tanh(b1[h] + sum(weight * value for weight, value in zip(w1[h], vector)))
            for h in range(hidden)
        ]
        return sigmoid(b2 + sum(weight * value for weight, value in zip(w2, hidden_values)))
    weights = [float(value) for value in model.get("weights", [])]
    bias = float(model.get("bias", 0.0))
    if len(weights) != len(vector):
        return 0.0
    return sigmoid(bias + sum(weight * value for weight, value in zip(weights, vector)))


def grouped_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id", ""))].append(row)
    return grouped


def count_gate_features(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    *,
    threshold: float,
) -> dict[str, float]:
    ordered = sorted(
        zip(rows, probabilities),
        key=lambda item: (
            -item[1],
            -float(item[0].get("relational_score", 0.0) or 0.0),
            float(item[0].get("distance_to_trajectory", 30.0) or 30.0),
            str(item[0].get("object_id", "")),
        ),
    )

    def prob(index: int) -> float:
        return float(ordered[index][1]) if len(ordered) > index else 0.0

    def value(index: int, key: str, default: float = 0.0) -> float:
        return float(ordered[index][0].get(key, default) or default) if len(ordered) > index else default

    top3 = ordered[:3]
    values = {
        "candidate_count": float(len(rows)),
        "eligible_count": float(sum(1 for _row, probability in ordered if probability >= threshold)),
        "high_prob_count_0p55": float(sum(1 for _row, probability in ordered if probability >= 0.55)),
        "high_prob_count_0p60": float(sum(1 for _row, probability in ordered if probability >= 0.60)),
        "high_prob_count_0p65": float(sum(1 for _row, probability in ordered if probability >= 0.65)),
        "p1": prob(0),
        "p2": prob(1),
        "p3": prob(2),
        "p4": prob(3),
        "p_gap_1_2": prob(0) - prob(1),
        "p_gap_2_3": prob(1) - prob(2),
        "p_gap_3_4": prob(2) - prob(3),
        "p_mean_top3": sum(prob(index) for index in range(3)) / 3.0,
        "score1": value(0, "relational_score"),
        "score2": value(1, "relational_score"),
        "score3": value(2, "relational_score"),
        "score_gap_1_2": value(0, "relational_score") - value(1, "relational_score"),
        "score_gap_2_3": value(1, "relational_score") - value(2, "relational_score"),
        "distance_to_trajectory1": value(0, "distance_to_trajectory", 30.0),
        "distance_to_trajectory2": value(1, "distance_to_trajectory", 30.0),
        "distance_to_trajectory3": value(2, "distance_to_trajectory", 30.0),
        "abs_y1": abs(value(0, "y", 30.0)),
        "abs_y2": abs(value(1, "y", 30.0)),
        "abs_y3": abs(value(2, "y", 30.0)),
        "behind_count_top3": float(sum(1 for row, _prob in top3 if float(row.get("x", 0.0) or 0.0) < 0.0)),
        "visible_count_top3": float(sum(1 for row, _prob in top3 if row.get("visibility_state") == "visible")),
        "occluded_count_top3": float(sum(1 for row, _prob in top3 if row.get("visibility_state") == "occluded")),
        "cooperative_count_top3": float(sum(1 for row, _prob in top3 if len(row.get("source_agent_ids", [])) >= 2)),
    }
    return {
        name: values.get(name, 0.0) if math.isfinite(values.get(name, 0.0)) else 0.0
        for name in PLANNING_COUNT_GATE_FEATURE_NAMES
    }


def normalization(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in PLANNING_COUNT_GATE_FEATURE_NAMES:
        values = [float(row.get(name, 0.0)) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[name] = mean
        stds[name] = math.sqrt(variance) or 1.0
    return means, stds


def vector(row: dict[str, float], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    return [
        (float(row.get(name, 0.0)) - means.get(name, 0.0)) / (stds.get(name, 1.0) or 1.0)
        for name in PLANNING_COUNT_GATE_FEATURE_NAMES
    ]


def train_softmax(
    x: list[list[float]],
    y: list[int],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    class_balance: bool,
) -> dict[str, Any]:
    classes = [0, 1, 2, 3]
    width = len(x[0])
    weights = [[0.0 for _ in range(width)] for _class in classes]
    biases = [0.0 for _class in classes]
    counts = Counter(y)
    class_weights = {
        label: (len(y) / (len(classes) * counts[label]) if counts[label] and class_balance else 1.0)
        for label in classes
    }
    for _epoch in range(epochs):
        grad_weights = [[0.0 for _ in range(width)] for _class in classes]
        grad_biases = [0.0 for _class in classes]
        for row_vector, label in zip(x, y):
            logits = [
                biases[index] + sum(weight * value for weight, value in zip(weights[index], row_vector))
                for index in range(len(classes))
            ]
            probs = softmax(logits)
            sample_weight = class_weights[label]
            for index, class_label in enumerate(classes):
                target = 1.0 if label == class_label else 0.0
                error = (probs[index] - target) * sample_weight
                grad_biases[index] += error
                for feature_index, value in enumerate(row_vector):
                    grad_weights[index][feature_index] += error * value
        scale = 1.0 / len(x)
        for index in range(len(classes)):
            biases[index] -= learning_rate * grad_biases[index] * scale
            for feature_index in range(width):
                weights[index][feature_index] -= learning_rate * (
                    grad_weights[index][feature_index] * scale + l2 * weights[index][feature_index]
                )
    return {"weights": weights, "biases": biases, "label_values": classes}


def predict_count(model: dict[str, Any], row_vector: list[float]) -> int:
    weights = model["weights"]
    biases = model["biases"]
    labels = model["label_values"]
    logits = [
        float(biases[index]) + sum(float(weight) * value for weight, value in zip(weights[index], row_vector))
        for index in range(len(labels))
    ]
    probabilities = softmax(logits)
    return int(labels[max(range(len(probabilities)), key=lambda index: probabilities[index])])


def main() -> None:
    args = build_parser().parse_args()
    train_path = Path(args.train_features_jsonl).expanduser()
    if not train_path.is_absolute():
        train_path = (REPO_ROOT / train_path).resolve()
    candidate_model_path = Path(args.candidate_model_json).expanduser()
    if not candidate_model_path.is_absolute():
        candidate_model_path = (REPO_ROOT / candidate_model_path).resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_model = json.loads(candidate_model_path.read_text(encoding="utf-8"))
    threshold = float(candidate_model.get("threshold", 0.5))
    rows = load_rows(train_path)
    scenes = grouped_rows(rows)
    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    for sample_id in sorted(scenes):
        sample_rows = scenes[sample_id]
        probabilities = [candidate_probability(candidate_model, row) for row in sample_rows]
        feature_rows.append(count_gate_features(sample_rows, probabilities, threshold=threshold))
        labels.append(min(3, max(0, int(sample_rows[0].get("gt_count", 0) or 0))))

    means, stds = normalization(feature_rows)
    x = [vector(row, means, stds) for row in feature_rows]
    model = train_softmax(
        x,
        labels,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        class_balance=not args.no_class_balance,
    )
    predictions = [predict_count(model, row_vector) for row_vector in x]
    exact = sum(1 for prediction, label in zip(predictions, labels) if prediction == label)
    over = sum(1 for prediction, label in zip(predictions, labels) if prediction > label)
    under = sum(1 for prediction, label in zip(predictions, labels) if prediction < label)
    report = {
        "run_name": args.run_name,
        "train_features_jsonl": str(train_path),
        "candidate_model_json": str(candidate_model_path),
        "samples": len(labels),
        "label_distribution": dict(Counter(labels)),
        "prediction_distribution": dict(Counter(predictions)),
        "exact_count_accuracy": exact / len(labels) if labels else 0.0,
        "over_count_rows": over,
        "under_count_rows": under,
    }
    combined_model = dict(candidate_model)
    combined_model["count_gate"] = {
        "model_type": "multinomial_logreg",
        "feature_names": list(PLANNING_COUNT_GATE_FEATURE_NAMES),
        "normalization": {"mean": means, "std": stds},
        "weights": model["weights"],
        "biases": model["biases"],
        "label_values": model["label_values"],
        "candidate_threshold": threshold,
        "soft_extra_min_probability": args.soft_extra_min_probability,
        "soft_extra_min_relative_to_k": args.soft_extra_min_relative_to_k,
        "trained_on": str(train_path),
        "candidate_model_json": str(candidate_model_path),
        "train_count_metrics": report,
    }
    model_path = output_dir / f"{args.run_name}_deployable.json"
    report_path = output_dir / f"{args.run_name}_report.json"
    model_path.write_text(json.dumps(combined_model, indent=2), encoding="utf-8")
    report["deployable_model_path"] = str(model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 72)
    print("Phase 8 Q4 Planning Count Gate Training")
    print("=" * 72)
    print(f"samples: {len(labels)}")
    print(f"label_distribution: {dict(Counter(labels))}")
    print(f"prediction_distribution: {dict(Counter(predictions))}")
    print(f"exact_count_accuracy: {report['exact_count_accuracy']:.6f}")
    print(f"over_count_rows: {over}")
    print(f"under_count_rows: {under}")
    print(f"saved_model: {model_path}")
    print(f"saved_report: {report_path}")


if __name__ == "__main__":
    main()
