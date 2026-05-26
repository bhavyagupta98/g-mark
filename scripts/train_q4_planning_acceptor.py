#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.qa.planning_awareness import PLANNING_LOGREG_FEATURE_NAMES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and freeze a transparent logistic Q4 planning-awareness acceptor. "
            "Threshold selection uses train features only; optional eval features are "
            "reported but never used for model/threshold choice."
        )
    )
    parser.add_argument("--train-features-jsonl", required=True)
    parser.add_argument("--eval-features-jsonl", default="")
    parser.add_argument("--output-dir", default="outputs/phase8_train_dev/q4_policy_optimization")
    parser.add_argument("--run-name", default="q4_planning_logreg_acceptor")
    parser.add_argument("--model-type", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--regularization",
        choices=["l2", "l1", "elasticnet", "none"],
        default="l2",
        help=(
            "Penalty used while fitting the train-frozen acceptor. "
            "l2 is ridge-style shrinkage, l1 is lasso-style sparsity, "
            "elasticnet combines both, and none disables weight penalties."
        ),
    )
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--log-every", type=int, default=0, help="Print epoch progress every N epochs; 0 disables.")
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--near-duplicate-distance", type=float, default=2.0)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--mlp-hidden", type=int, default=24)
    parser.add_argument("--mlp-dev-fraction", type=float, default=0.15)
    parser.add_argument("--mlp-patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return [row for row in rows if row.get("row_type") == "candidate"]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def feature_names() -> list[str]:
    return list(PLANNING_LOGREG_FEATURE_NAMES)


def normalization(rows: list[dict[str, Any]], names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in names:
        values = [float(row.get(name, 0.0) or 0.0) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[name] = mean
        stds[name] = math.sqrt(variance) or 1.0
    return means, stds


def vectors(rows: list[dict[str, Any]], names: list[str], means: dict[str, float], stds: dict[str, float]) -> list[list[float]]:
    output = []
    for row in rows:
        vector = []
        for name in names:
            std = stds.get(name, 1.0) or 1.0
            vector.append((float(row.get(name, 0.0) or 0.0) - means.get(name, 0.0)) / std)
        output.append(vector)
    return output


def labels(rows: list[dict[str, Any]]) -> list[float]:
    return [1.0 if row.get("candidate_matches_gt") is True else 0.0 for row in rows]


def class_weights(y: list[float], balance: bool) -> tuple[float, float]:
    if not balance:
        return 1.0, 1.0
    positives = sum(1 for value in y if value >= 0.5)
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return 1.0, 1.0
    return len(y) / (2.0 * positives), len(y) / (2.0 * negatives)


def split_indices_by_sample(
    rows: list[dict[str, Any]],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if dev_fraction <= 0.0:
        return list(range(len(rows))), []
    by_sample: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_sample[str(row.get("sample_id", ""))].append(index)
    sample_ids = sorted(by_sample)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    dev_count = max(1, int(round(len(sample_ids) * dev_fraction)))
    dev_samples = set(sample_ids[:dev_count])
    train_indices: list[int] = []
    dev_indices: list[int] = []
    for sample_id, indices in by_sample.items():
        if sample_id in dev_samples:
            dev_indices.extend(indices)
        else:
            train_indices.extend(indices)
    if not train_indices:
        return list(range(len(rows))), []
    return sorted(train_indices), sorted(dev_indices)


def weighted_bce_loss(
    predictions: list[float],
    y: list[float],
    indices: list[int],
    *,
    positive_weight: float,
    negative_weight: float,
) -> float:
    if not indices:
        return 0.0
    loss = 0.0
    epsilon = 1e-9
    for index in indices:
        prediction = min(max(predictions[index], epsilon), 1.0 - epsilon)
        label = y[index]
        sample_weight = positive_weight if label else negative_weight
        loss -= sample_weight * (label * math.log(prediction) + (1.0 - label) * math.log(1.0 - prediction))
    return loss / len(indices)


def train_logreg(
    x: list[list[float]],
    y: list[float],
    *,
    learning_rate: float,
    epochs: int,
    regularization: str,
    l1: float,
    l2: float,
    balance: bool,
    log_every: int,
) -> dict[str, Any]:
    positive_weight, negative_weight = class_weights(y, balance)
    weights = [0.0 for _ in x[0]]
    bias = 0.0
    for epoch_index in range(epochs):
        grad_weights = [0.0 for _ in weights]
        grad_bias = 0.0
        for vector, label in zip(x, y):
            prediction = sigmoid(bias + sum(weight * value for weight, value in zip(weights, vector)))
            sample_weight = positive_weight if label else negative_weight
            error = (prediction - label) * sample_weight
            grad_bias += error
            for index, value in enumerate(vector):
                grad_weights[index] += error * value
        scale = 1.0 / len(x)
        bias -= learning_rate * grad_bias * scale
        for index in range(len(weights)):
            penalty_gradient = 0.0
            if regularization in {"l1", "elasticnet"} and l1 > 0.0:
                if weights[index] > 0.0:
                    penalty_gradient += l1
                elif weights[index] < 0.0:
                    penalty_gradient -= l1
            if regularization in {"l2", "elasticnet"} and l2 > 0.0:
                penalty_gradient += l2 * weights[index]
            weights[index] -= learning_rate * (grad_weights[index] * scale + penalty_gradient)
        if log_every > 0 and ((epoch_index + 1) % log_every == 0 or epoch_index + 1 == epochs):
            print(f"training_progress: logreg_epoch={epoch_index + 1}/{epochs}")
    return {
        "model_type": "logreg",
        "bias": bias,
        "weights": weights,
        "regularization": regularization,
        "l1": l1,
        "l2": l2,
    }


def train_mlp(
    rows: list[dict[str, Any]],
    x: list[list[float]],
    y: list[float],
    *,
    hidden: int,
    learning_rate: float,
    epochs: int,
    l2: float,
    balance: bool,
    seed: int,
    dev_fraction: float,
    patience: int,
    log_every: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    train_indices, dev_indices = split_indices_by_sample(rows, dev_fraction=dev_fraction, seed=seed)
    width = len(x[0])
    w1 = [[rng.uniform(-0.08, 0.08) for _ in range(width)] for _ in range(hidden)]
    b1 = [0.0 for _ in range(hidden)]
    w2 = [rng.uniform(-0.08, 0.08) for _ in range(hidden)]
    b2 = 0.0
    train_y = [y[index] for index in train_indices]
    positive_weight, negative_weight = class_weights(train_y, balance)
    best_loss = float("inf")
    best_state: tuple[list[list[float]], list[float], list[float], float] | None = None
    epochs_without_improvement = 0

    for epoch_index in range(epochs):
        grad_w1 = [[0.0 for _ in range(width)] for _ in range(hidden)]
        grad_b1 = [0.0 for _ in range(hidden)]
        grad_w2 = [0.0 for _ in range(hidden)]
        grad_b2 = 0.0
        for index in train_indices:
            vector = x[index]
            label = y[index]
            hidden_values = [
                math.tanh(b1[h] + sum(w1[h][i] * vector[i] for i in range(width)))
                for h in range(hidden)
            ]
            prediction = sigmoid(b2 + sum(w2[h] * hidden_values[h] for h in range(hidden)))
            sample_weight = positive_weight if label else negative_weight
            output_error = (prediction - label) * sample_weight
            grad_b2 += output_error
            for h in range(hidden):
                grad_w2[h] += output_error * hidden_values[h]
                hidden_error = output_error * w2[h] * (1.0 - hidden_values[h] ** 2)
                grad_b1[h] += hidden_error
                for i, value in enumerate(vector):
                    grad_w1[h][i] += hidden_error * value

        scale = 1.0 / len(train_indices)
        b2 -= learning_rate * grad_b2 * scale
        for h in range(hidden):
            w2[h] -= learning_rate * (grad_w2[h] * scale + l2 * w2[h])
            b1[h] -= learning_rate * grad_b1[h] * scale
            for i in range(width):
                w1[h][i] -= learning_rate * (grad_w1[h][i] * scale + l2 * w1[h][i])

        if dev_indices and patience > 0:
            current_model = {
                "model_type": "mlp",
                "hidden": hidden,
                "w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2,
            }
            dev_loss = weighted_bce_loss(
                predict_mlp(current_model, x),
                y,
                dev_indices,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
            )
            if dev_loss + 1e-6 < best_loss:
                best_loss = dev_loss
                best_state = ([row[:] for row in w1], b1[:], w2[:], b2)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        if log_every > 0 and ((epoch_index + 1) % log_every == 0 or epoch_index + 1 == epochs):
            print(f"training_progress: mlp_epoch={epoch_index + 1}/{epochs}")

    if best_state is not None:
        w1, b1, w2, b2 = best_state
    return {
        "model_type": "mlp",
        "hidden": hidden,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "l2": l2,
        "early_stopping": {
            "dev_fraction": dev_fraction,
            "patience": patience,
            "best_dev_loss": best_loss if best_state is not None else None,
        },
    }


def predict(model: dict[str, Any], x: list[list[float]]) -> list[float]:
    if model.get("model_type") == "mlp":
        return predict_mlp(model, x)
    weights = [float(value) for value in model["weights"]]
    bias = float(model["bias"])
    return [sigmoid(bias + sum(weight * value for weight, value in zip(weights, vector))) for vector in x]


def predict_mlp(model: dict[str, Any], x: list[list[float]]) -> list[float]:
    w1 = [[float(value) for value in row] for row in model["w1"]]
    b1 = [float(value) for value in model["b1"]]
    w2 = [float(value) for value in model["w2"]]
    b2 = float(model["b2"])
    hidden = int(model["hidden"])
    predictions: list[float] = []
    for vector in x:
        hidden_values = [
            math.tanh(b1[h] + sum(w1[h][i] * vector[i] for i in range(len(vector))))
            for h in range(hidden)
        ]
        predictions.append(sigmoid(b2 + sum(w2[h] * hidden_values[h] for h in range(hidden))))
    return predictions


def point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def match_count(
    predictions: list[tuple[float, float]],
    references: list[tuple[float, float]],
    threshold: float,
) -> int:
    matched_refs: set[int] = set()
    count = 0
    for prediction in predictions:
        best_index = None
        best_distance = float("inf")
        for index, reference in enumerate(references):
            if index in matched_refs:
                continue
            distance = point_distance(prediction, reference)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance <= threshold:
            matched_refs.add(best_index)
            count += 1
    return count


def select_rows(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    *,
    threshold: float,
    max_results: int,
    near_duplicate_distance: float,
) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        if probability >= threshold:
            by_sample[str(row["sample_id"])].append((row, probability))

    selected: dict[str, list[dict[str, Any]]] = {}
    for sample_id, items in by_sample.items():
        ordered = sorted(
            items,
            key=lambda item: (
                -item[1],
                -float(item[0].get("relational_score", 0.0)),
                float(item[0].get("distance_to_trajectory", 0.0)),
                str(item[0].get("object_id", "")),
            ),
        )
        sample_selected: list[dict[str, Any]] = []
        for row, probability in ordered:
            coord = (float(row["x"]), float(row["y"]))
            if any(
                point_distance(coord, (float(existing["x"]), float(existing["y"]))) <= near_duplicate_distance
                for existing in sample_selected
            ):
                continue
            accepted = dict(row)
            accepted["acceptor_probability"] = probability
            sample_selected.append(accepted)
            if len(sample_selected) >= max_results:
                break
        selected[sample_id] = sample_selected
    return selected


def evaluate_policy(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    *,
    threshold: float,
    max_results: int,
    near_duplicate_distance: float,
    match_threshold: float,
) -> dict[str, float]:
    selected = select_rows(
        rows,
        probabilities,
        threshold=threshold,
        max_results=max_results,
        near_duplicate_distance=near_duplicate_distance,
    )
    gt_by_sample: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in gt_by_sample:
            gt_by_sample[sample_id] = [
                (float(item[0]), float(item[1]))
                for item in row.get("gt_coords", [])
            ]
    predicted_mentions = 0
    reference_mentions = 0
    matched_mentions = 0
    for sample_id, references in gt_by_sample.items():
        predictions = [
            (float(row["x"]), float(row["y"]))
            for row in selected.get(sample_id, [])
        ]
        predicted_mentions += len(predictions)
        reference_mentions += len(references)
        matched_mentions += match_count(predictions, references, match_threshold)
    precision = matched_mentions / predicted_mentions if predicted_mentions else 0.0
    recall = matched_mentions / reference_mentions if reference_mentions else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "predicted_mentions": float(predicted_mentions),
        "reference_mentions": float(reference_mentions),
        "matched_mentions": float(matched_mentions),
    }


def choose_threshold(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    args: argparse.Namespace,
) -> dict[str, float]:
    candidates = [index / 100.0 for index in range(5, 96)]
    metrics = [
        evaluate_policy(
            rows,
            probabilities,
            threshold=threshold,
            max_results=args.max_results,
            near_duplicate_distance=args.near_duplicate_distance,
            match_threshold=args.match_threshold,
        )
        for threshold in candidates
    ]
    precision_floor = [
        item for item in metrics if item["precision"] >= args.min_precision
    ]
    pool = precision_floor or metrics
    return max(pool, key=lambda item: (item["f1"], item["precision"], item["recall"]))


def logreg_interpretability(model: dict[str, Any], names: list[str]) -> dict[str, list[list[Any]]]:
    if model.get("model_type") != "logreg":
        return {"top_positive_weights": [], "top_negative_weights": []}
    weights = [float(value) for value in model["weights"]]
    return {
        "top_positive_weights": [[name, weight] for name, weight in sorted(zip(names, weights), key=lambda item: item[1], reverse=True)[:10]],
        "top_negative_weights": [[name, weight] for name, weight in sorted(zip(names, weights), key=lambda item: item[1])[:10]],
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_features_jsonl).expanduser()
    if not train_path.is_absolute():
        train_path = (REPO_ROOT / train_path).resolve()
    train_rows = load_rows(train_path)
    if not train_rows:
        raise SystemExit(f"No candidate rows found in {train_path}")
    names = feature_names()
    means, stds = normalization(train_rows, names)
    train_x = vectors(train_rows, names, means, stds)
    train_y = labels(train_rows)
    if args.model_type == "mlp":
        model = train_mlp(
            train_rows,
            train_x,
            train_y,
            hidden=args.mlp_hidden,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            balance=not args.no_class_balance,
            seed=args.seed,
            dev_fraction=args.mlp_dev_fraction,
            patience=args.mlp_patience,
            log_every=args.log_every,
        )
    else:
        model = train_logreg(
            train_x,
            train_y,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            regularization=args.regularization,
            l1=args.l1,
            l2=args.l2,
            balance=not args.no_class_balance,
            log_every=args.log_every,
        )
    train_probabilities = predict(model, train_x)
    selected_train = choose_threshold(train_rows, train_probabilities, args)

    report: dict[str, Any] = {
        "run_name": args.run_name,
        "train_features_jsonl": str(train_path),
        "train_candidate_rows": len(train_rows),
        "train_positive_rows": int(sum(train_y)),
        "threshold_selection": {
            "split": "train",
            "min_precision": args.min_precision,
            **selected_train,
        },
        "training_config": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "model_type": args.model_type,
            "regularization": args.regularization,
            "l1": args.l1,
            "l2": args.l2,
            "mlp_hidden": args.mlp_hidden,
            "mlp_dev_fraction": args.mlp_dev_fraction,
            "mlp_patience": args.mlp_patience,
            "seed": args.seed,
            "class_balance": not args.no_class_balance,
            "max_results": args.max_results,
            "near_duplicate_distance": args.near_duplicate_distance,
            "match_threshold": args.match_threshold,
        },
    }

    eval_rows: list[dict[str, Any]] = []
    if args.eval_features_jsonl:
        eval_path = Path(args.eval_features_jsonl).expanduser()
        if not eval_path.is_absolute():
            eval_path = (REPO_ROOT / eval_path).resolve()
        eval_rows = load_rows(eval_path)
        eval_x = vectors(eval_rows, names, means, stds)
        eval_probabilities = predict(model, eval_x)
        report["eval_features_jsonl"] = str(eval_path)
        report["eval_at_train_threshold"] = evaluate_policy(
            eval_rows,
            eval_probabilities,
            threshold=float(selected_train["threshold"]),
            max_results=args.max_results,
            near_duplicate_distance=args.near_duplicate_distance,
            match_threshold=args.match_threshold,
        )

    frozen_model = {
        "model_type": "logreg",
        "task": "planning_awareness_q4_acceptor",
        "feature_names": names,
        "normalization": {"mean": means, "std": stds},
        **model,
        "threshold": float(selected_train["threshold"]),
        "max_results": args.max_results,
        "near_duplicate_distance": args.near_duplicate_distance,
        "match_threshold_m": args.match_threshold,
        "trained_on": str(train_path),
        "threshold_selected_on": "train",
        "train_metrics_at_threshold": selected_train,
        "interpretability": logreg_interpretability(model, names),
    }
    threshold_label = str(frozen_model["threshold"]).replace(".", "p")
    model_path = output_dir / f"{args.run_name}_t{threshold_label}_deployable.json"
    report_path = output_dir / f"{args.run_name}_report.json"
    markdown_path = output_dir / f"{args.run_name}_report.md"
    model_path.write_text(json.dumps(frozen_model, indent=2), encoding="utf-8")
    report["deployable_model_path"] = str(model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# Q4 Planning Acceptor",
                "",
                f"- train features: `{train_path}`",
                f"- candidate rows: `{len(train_rows)}`",
                f"- positive rows: `{int(sum(train_y))}`",
                f"- model type: `{args.model_type}`",
                f"- selected threshold: `{selected_train['threshold']:.2f}`",
                f"- regularization: `{args.regularization}`",
                f"- l1 / l2: `{args.l1} / {args.l2}`",
                f"- train F1/P/R: `{selected_train['f1']:.6f} / {selected_train['precision']:.6f} / {selected_train['recall']:.6f}`",
                f"- deployable model: `{model_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("=" * 72)
    print("Phase 8 Q4 Planning Acceptor Training")
    print("=" * 72)
    print(f"train_candidate_rows: {len(train_rows)}")
    print(f"train_positive_rows: {int(sum(train_y))}")
    print(f"selected_threshold: {selected_train['threshold']:.2f}")
    print(f"train_metrics: {selected_train}")
    if "eval_at_train_threshold" in report:
        print(f"eval_metrics_at_train_threshold: {report['eval_at_train_threshold']}")
    print(f"saved_model: {model_path}")
    print(f"saved_report: {report_path}")
    print(f"saved_markdown: {markdown_path}")


if __name__ == "__main__":
    main()
