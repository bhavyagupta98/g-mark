#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

NUMERIC_FEATURES = (
    "rank",
    "role_score",
    "relative_x",
    "relative_y",
    "abs_relative_x",
    "abs_relative_y",
    "distance_to_asker",
    "distance_to_trajectory",
    "support_count",
    "conflict_score",
)
CATEGORICAL_FEATURES = (
    ("status", ("confirmed", "supported", "candidate")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a transparent logistic candidate acceptor for Q3 invisible objects "
            "from the train candidate feature table."
        )
    )
    parser.add_argument("--train-features-jsonl", required=True)
    parser.add_argument("--eval-features-jsonl", default="")
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--no-class-balance", action="store_true")
    return parser


def load_feature_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def candidate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("row_type") == "candidate"]


def feature_names() -> list[str]:
    names = list(NUMERIC_FEATURES)
    for key, values in CATEGORICAL_FEATURES:
        names.extend(f"{key}={value}" for value in values)
    return names


def raw_vector(row: dict[str, object]) -> list[float]:
    values: list[float] = []
    for feature in NUMERIC_FEATURES:
        raw_value = row.get(feature, 0.0)
        values.append(float(raw_value) if isinstance(raw_value, (float, int)) else 0.0)
    for key, categories in CATEGORICAL_FEATURES:
        raw_value = str(row.get(key, ""))
        values.extend(1.0 if raw_value == category else 0.0 for category in categories)
    return values


def normalization(rows: list[dict[str, object]]) -> tuple[list[float], list[float]]:
    vectors = [raw_vector(row) for row in rows]
    if not vectors:
        raise SystemExit("No candidate rows found in train feature table.")
    width = len(vectors[0])
    means: list[float] = []
    stds: list[float] = []
    for index in range(width):
        values = [vector[index] for vector in vectors]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        means.append(mean_value)
        stds.append(math.sqrt(variance) or 1.0)
    return means, stds


def normalized_vector(row: dict[str, object], means: list[float], stds: list[float]) -> list[float]:
    return [
        (value - means[index]) / stds[index]
        for index, value in enumerate(raw_vector(row))
    ]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def train_logistic(
    rows: list[dict[str, object]],
    means: list[float],
    stds: list[float],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    class_balance: bool,
) -> tuple[float, list[float]]:
    features = [normalized_vector(row, means, stds) for row in rows]
    labels = [1.0 if row.get("candidate_matches_gt") is True else 0.0 for row in rows]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_weight = len(labels) / (2.0 * positive_count) if positive_count and class_balance else 1.0
    negative_weight = len(labels) / (2.0 * negative_count) if negative_count and class_balance else 1.0
    weights = [0.0 for _ in range(len(features[0]))]
    bias = 0.0

    for _ in range(epochs):
        grad_weights = [0.0 for _ in weights]
        grad_bias = 0.0
        for vector, label in zip(features, labels):
            prediction = sigmoid(bias + sum(weight * value for weight, value in zip(weights, vector)))
            sample_weight = positive_weight if label else negative_weight
            error = (prediction - label) * sample_weight
            grad_bias += error
            for index, value in enumerate(vector):
                grad_weights[index] += error * value

        scale = 1.0 / len(features)
        bias -= learning_rate * grad_bias * scale
        for index in range(len(weights)):
            grad = grad_weights[index] * scale + l2 * weights[index]
            weights[index] -= learning_rate * grad

    return bias, weights


def predict_probability(row: dict[str, object], means: list[float], stds: list[float], bias: float, weights: list[float]) -> float:
    vector = normalized_vector(row, means, stds)
    return sigmoid(bias + sum(weight * value for weight, value in zip(weights, vector)))


def sample_level_metrics(
    rows: list[dict[str, object]],
    candidate_probabilities: dict[int, float],
    threshold: float,
    all_rows: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    by_sample: dict[str, list[tuple[dict[str, object], float]]] = defaultdict(list)
    sample_gt_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        by_sample[sample_id].append((row, candidate_probabilities[index]))

    gt_source_rows = all_rows if all_rows is not None else rows
    sample_ids = set(by_sample)
    for row in gt_source_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        sample_ids.add(sample_id)
        gt_count = row.get("gt_count", 0)
        if isinstance(gt_count, (float, int)):
            sample_gt_counts[sample_id] = max(sample_gt_counts[sample_id], int(gt_count))

    tp = fp = fn = predicted = 0
    for sample_id in sample_ids:
        candidates = by_sample.get(sample_id, [])
        chosen_row = None
        chosen_probability = -1.0
        for row, probability in candidates:
            if probability > chosen_probability:
                chosen_row = row
                chosen_probability = probability
        matched = False
        if chosen_row is not None and chosen_probability >= threshold:
            predicted += 1
            matched = chosen_row.get("candidate_matches_gt") is True
            if matched:
                tp += 1
            else:
                fp += 1
        fn += max(sample_gt_counts[sample_id] - (1 if matched else 0), 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "predicted": float(predicted),
    }


def choose_threshold(
    rows: list[dict[str, object]],
    probabilities: dict[int, float],
    all_rows: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    candidates = [index / 100.0 for index in range(5, 96)]
    metrics = [sample_level_metrics(rows, probabilities, threshold, all_rows) for threshold in candidates]
    return max(
        metrics,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["recall"],
            -item["threshold"],
        ),
    )


def score_rows(
    rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
    means: list[float],
    stds: list[float],
    bias: float,
    weights: list[float],
    threshold: float,
) -> dict[str, object]:
    probabilities = {
        index: predict_probability(row, means, stds, bias, weights)
        for index, row in enumerate(rows)
    }
    metrics = sample_level_metrics(rows, probabilities, threshold, all_rows)
    threshold_metrics = choose_threshold(rows, probabilities, all_rows)
    return {
        "metrics_at_threshold": metrics,
        "best_threshold_on_this_split": threshold_metrics,
        "probability_summary": {
            "min": min(probabilities.values()) if probabilities else 0.0,
            "max": max(probabilities.values()) if probabilities else 0.0,
            "mean": sum(probabilities.values()) / len(probabilities) if probabilities else 0.0,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    train_path = Path(args.train_features_jsonl).expanduser().resolve()
    train_all_rows = load_feature_rows(train_path)
    train_rows = candidate_rows(train_all_rows)
    means, stds = normalization(train_rows)
    bias, weights = train_logistic(
        train_rows,
        means,
        stds,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        class_balance=not args.no_class_balance,
    )
    train_probabilities = {
        index: predict_probability(row, means, stds, bias, weights)
        for index, row in enumerate(train_rows)
    }
    train_threshold = choose_threshold(train_rows, train_probabilities, train_all_rows)
    threshold = float(train_threshold["threshold"])

    names = feature_names()
    model = {
        "model_type": "standard_library_l2_logistic_regression",
        "feature_names": names,
        "normalization": {
            "mean": means,
            "std": stds,
        },
        "bias": bias,
        "weights": weights,
        "threshold": threshold,
        "train_features_jsonl": str(train_path),
    }
    report: dict[str, object] = {
        "train_features_jsonl": str(train_path),
        "train_feature_rows": len(train_all_rows),
        "train_candidate_rows": len(train_rows),
        "selected_threshold": threshold,
        "train": score_rows(train_rows, train_all_rows, means, stds, bias, weights, threshold),
        "top_positive_weights": sorted(
            zip(names, weights),
            key=lambda item: item[1],
            reverse=True,
        )[:12],
        "top_negative_weights": sorted(
            zip(names, weights),
            key=lambda item: item[1],
        )[:12],
    }

    if args.eval_features_jsonl:
        eval_path = Path(args.eval_features_jsonl).expanduser().resolve()
        eval_all_rows = load_feature_rows(eval_path)
        eval_rows = candidate_rows(eval_all_rows)
        report["eval_features_jsonl"] = str(eval_path)
        report["eval_feature_rows"] = len(eval_all_rows)
        report["eval_candidate_rows"] = len(eval_rows)
        report["eval"] = score_rows(eval_rows, eval_all_rows, means, stds, bias, weights, threshold)

    model_path = Path(args.output_model_json).expanduser()
    report_path = Path(args.output_report_json).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Phase 8 Invisible Candidate Acceptor Training")
    print("=" * 72)
    print(f"train_features_jsonl: {train_path}")
    print(f"train_feature_rows: {len(train_all_rows)}")
    print(f"train_candidate_rows: {len(train_rows)}")
    print(f"selected_threshold: {threshold}")
    print(f"train_metrics: {json.dumps(report['train']['metrics_at_threshold'], sort_keys=True)}")
    if "eval" in report:
        print(f"eval_metrics: {json.dumps(report['eval']['metrics_at_threshold'], sort_keys=True)}")
    print(f"saved_model_json: {model_path}")
    print(f"saved_report_json: {report_path}")


if __name__ == "__main__":
    main()
