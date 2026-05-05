#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
    "confidence",
    "conflict_score",
    "uncertainty_score",
    "age_frames",
    "miss_count",
)
CATEGORICAL_FEATURES = (
    ("status", ("confirmed", "supported", "candidate")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast Q3 invisible-object policy optimizer over precomputed candidate features. "
            "Trains compact candidate acceptors, sweeps train-selected policies, and writes "
            "concise interpretability reports before any expensive official rerun."
        )
    )
    parser.add_argument("--train-features-jsonl", required=True)
    parser.add_argument("--eval-features-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="q3_invisible_policy_opt")
    parser.add_argument("--baseline-f1", type=float, default=0.44, help="Reference F1 for target reporting.")
    parser.add_argument("--vp-improvement", type=float, default=0.10, help="Validation-point minimum improvement target.")
    parser.add_argument("--stretch-improvement", type=float, default=0.20, help="Stretch improvement target.")
    parser.add_argument("--models", nargs="+", default=["logreg", "mlp"], choices=["logreg", "mlp"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2", type=float, default=0.005)
    parser.add_argument("--mlp-hidden", type=int, default=24)
    parser.add_argument("--mlp-dev-fraction", type=float, default=0.15)
    parser.add_argument("--mlp-patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--max-report-policies", type=int, default=12)
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("row_type") == "candidate"]


def feature_names() -> list[str]:
    names = list(NUMERIC_FEATURES)
    for key, values in CATEGORICAL_FEATURES:
        names.extend(f"{key}={value}" for value in values)
    return names


def raw_vector(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for feature in NUMERIC_FEATURES:
        raw_value = row.get(feature, 0.0)
        values.append(float(raw_value) if isinstance(raw_value, (float, int)) else 0.0)
    for key, categories in CATEGORICAL_FEATURES:
        raw_value = str(row.get(key, ""))
        values.extend(1.0 if raw_value == category else 0.0 for category in categories)
    return values


def normalization(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    vectors = [raw_vector(row) for row in rows]
    if not vectors:
        raise SystemExit("No candidate rows found.")
    means: list[float] = []
    stds: list[float] = []
    for index in range(len(vectors[0])):
        values = [vector[index] for vector in vectors]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        stds.append(math.sqrt(variance) or 1.0)
    return means, stds


def normalized_vectors(rows: list[dict[str, Any]], means: list[float], stds: list[float]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for row in rows:
        raw = raw_vector(row)
        vectors.append([(value - means[index]) / stds[index] for index, value in enumerate(raw)])
    return vectors


def labels(rows: list[dict[str, Any]]) -> list[float]:
    return [1.0 if row.get("candidate_matches_gt") is True else 0.0 for row in rows]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def class_weights(y: list[float], enabled: bool) -> tuple[float, float]:
    if not enabled:
        return 1.0, 1.0
    positives = sum(y)
    negatives = len(y) - positives
    positive_weight = len(y) / (2.0 * positives) if positives else 1.0
    negative_weight = len(y) / (2.0 * negatives) if negatives else 1.0
    return positive_weight, negative_weight


def split_indices_by_sample(
    rows: list[dict[str, Any]],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if dev_fraction <= 0:
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
    l2: float,
    balance: bool,
) -> dict[str, Any]:
    positive_weight, negative_weight = class_weights(y, balance)
    weights = [0.0 for _ in x[0]]
    bias = 0.0
    for _ in range(epochs):
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
            weights[index] -= learning_rate * (grad_weights[index] * scale + l2 * weights[index])
    return {"model_type": "logreg", "bias": bias, "weights": weights}


def predict_logreg(model: dict[str, Any], x: list[list[float]]) -> list[float]:
    bias = float(model["bias"])
    weights = [float(value) for value in model["weights"]]
    return [sigmoid(bias + sum(weight * value for weight, value in zip(weights, vector))) for vector in x]


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
) -> dict[str, Any]:
    rng = random.Random(seed)
    train_indices, dev_indices = split_indices_by_sample(
        rows,
        dev_fraction=dev_fraction,
        seed=seed,
    )
    width = len(x[0])
    w1 = [[rng.uniform(-0.08, 0.08) for _ in range(width)] for _ in range(hidden)]
    b1 = [0.0 for _ in range(hidden)]
    w2 = [rng.uniform(-0.08, 0.08) for _ in range(hidden)]
    b2 = 0.0
    train_labels = [y[index] for index in train_indices]
    positive_weight, negative_weight = class_weights(train_labels, balance)
    best_loss = float("inf")
    best_state: tuple[list[list[float]], list[float], list[float], float] | None = None
    epochs_without_improvement = 0

    for _ in range(epochs):
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
            logit = b2 + sum(w2[h] * hidden_values[h] for h in range(hidden))
            prediction = sigmoid(logit)
            sample_weight = positive_weight if label else negative_weight
            output_error = (prediction - label) * sample_weight
            grad_b2 += output_error
            for h in range(hidden):
                grad_w2[h] += output_error * hidden_values[h]
                hidden_error = output_error * w2[h] * (1.0 - hidden_values[h] ** 2)
                grad_b1[h] += hidden_error
                for i in range(width):
                    grad_w1[h][i] += hidden_error * vector[i]

        scale = 1.0 / len(train_indices)
        b2 -= learning_rate * grad_b2 * scale
        for h in range(hidden):
            w2[h] -= learning_rate * (grad_w2[h] * scale + l2 * w2[h])
            b1[h] -= learning_rate * grad_b1[h] * scale
            for i in range(width):
                w1[h][i] -= learning_rate * (grad_w1[h][i] * scale + l2 * w1[h][i])

        if dev_indices and patience > 0:
            current_model = {"hidden": hidden, "w1": w1, "b1": b1, "w2": w2, "b2": b2}
            dev_predictions = predict_mlp(current_model, x)
            dev_loss = weighted_bce_loss(
                dev_predictions,
                y,
                dev_indices,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
            )
            if dev_loss + 1e-6 < best_loss:
                best_loss = dev_loss
                best_state = (
                    [row[:] for row in w1],
                    b1[:],
                    w2[:],
                    b2,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    if best_state is not None:
        w1, b1, w2, b2 = best_state
    return {
        "model_type": "mlp",
        "hidden": hidden,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "early_stopping": {
            "dev_fraction": dev_fraction,
            "patience": patience,
            "best_dev_loss": best_loss if best_state is not None else None,
        },
    }


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


def sample_gt_counts(all_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in all_rows:
        sample_id = str(row.get("sample_id", ""))
        gt_count = row.get("gt_count", 0)
        if sample_id and isinstance(gt_count, (float, int)):
            counts[sample_id] = max(counts[sample_id], int(gt_count))
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            counts.setdefault(sample_id, 0)
    return counts


def select_by_threshold(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    threshold: float,
) -> dict[str, tuple[int, float]]:
    by_sample: dict[str, tuple[int, float]] = {}
    for index, (row, probability) in enumerate(zip(rows, probabilities)):
        sample_id = str(row.get("sample_id", ""))
        if probability < threshold:
            continue
        previous = by_sample.get(sample_id)
        if previous is None or probability > previous[1]:
            by_sample[sample_id] = (index, probability)
    return by_sample


def metrics_for_threshold(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    all_rows: list[dict[str, Any]],
    threshold: float,
    beta: float = 1.0,
    collect_errors: bool = False,
) -> dict[str, Any]:
    selected = select_by_threshold(rows, probabilities, threshold)
    gt_counts = sample_gt_counts(all_rows, rows)
    tp = fp = fn = predicted = 0
    selected_indices: set[int] = set()
    for sample_id, (index, _probability) in selected.items():
        selected_indices.add(index)
        predicted += 1
        if rows[index].get("candidate_matches_gt") is True:
            tp += 1
        else:
            fp += 1
    for sample_id, gt_count in gt_counts.items():
        selected_index = selected.get(sample_id, (None, 0.0))[0]
        matched = selected_index is not None and rows[int(selected_index)].get("candidate_matches_gt") is True
        fn += max(gt_count - (1 if matched else 0), 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_sq = beta * beta
    fbeta = (
        (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if precision + recall
        else 0.0
    )
    result: dict[str, Any] = {
        "threshold": threshold,
        "f1": f1,
        "fbeta": fbeta,
        "beta": beta,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted": predicted,
    }
    if collect_errors:
        result["error_buckets"] = error_buckets(rows, probabilities, selected_indices)
    return result


def threshold_grid() -> list[float]:
    return [index / 100.0 for index in range(5, 96)]


def choose_policy_candidates(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    all_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grid_metrics = [
        metrics_for_threshold(rows, probabilities, all_rows, threshold, beta=1.0)
        for threshold in threshold_grid()
    ]
    policies: list[dict[str, Any]] = []
    policies.append({"policy": "best_f1", **max(grid_metrics, key=lambda item: (item["f1"], item["precision"], item["recall"]))})
    for beta in (1.25, 1.5, 2.0):
        beta_metrics = [
            metrics_for_threshold(rows, probabilities, all_rows, threshold, beta=beta)
            for threshold in threshold_grid()
        ]
        policies.append(
            {
                "policy": f"best_f{str(beta).replace('.', 'p')}",
                **max(beta_metrics, key=lambda item: (item["fbeta"], item["f1"], item["precision"])),
            }
        )
    for floor in (0.50, 0.55, 0.60, 0.65, 0.70):
        eligible = [item for item in grid_metrics if item["precision"] >= floor]
        if eligible:
            policies.append(
                {
                    "policy": f"max_recall_p{str(floor).replace('.', 'p')}",
                    **max(eligible, key=lambda item: (item["recall"], item["f1"], item["precision"])),
                }
            )
    deduped: dict[tuple[str, float], dict[str, Any]] = {}
    for policy in policies:
        deduped[(str(policy["policy"]), float(policy["threshold"]))] = policy
    return list(deduped.values())


def evaluate_policy_on_split(
    policy: dict[str, Any],
    rows: list[dict[str, Any]],
    probabilities: list[float],
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return metrics_for_threshold(
        rows,
        probabilities,
        all_rows,
        float(policy["threshold"]),
        beta=float(policy.get("beta", 1.0)),
        collect_errors=True,
    )


def bucket_numeric(prefix: str, value: Any, buckets: Counter[str]) -> None:
    if not isinstance(value, (float, int)):
        buckets[f"{prefix}=missing"] += 1
        return
    number = float(value)
    if prefix.endswith("probability"):
        if number < 0.25:
            buckets[f"{prefix}<0.25"] += 1
        elif number < 0.50:
            buckets[f"{prefix}=0.25-0.50"] += 1
        elif number < 0.75:
            buckets[f"{prefix}=0.50-0.75"] += 1
        else:
            buckets[f"{prefix}>=0.75"] += 1
    elif prefix.endswith("relative_x"):
        if number < -1:
            buckets[f"{prefix}=behind"] += 1
        elif number > 1:
            buckets[f"{prefix}=ahead"] += 1
        else:
            buckets[f"{prefix}=near_zero"] += 1
    elif prefix.endswith("abs_relative_y"):
        if number < 1:
            buckets[f"{prefix}<1m"] += 1
        elif number < 3:
            buckets[f"{prefix}=1-3m"] += 1
        else:
            buckets[f"{prefix}>=3m"] += 1
    elif prefix.endswith("distance_to_trajectory"):
        if number < 2:
            buckets[f"{prefix}<2m"] += 1
        elif number < 6:
            buckets[f"{prefix}=2-6m"] += 1
        else:
            buckets[f"{prefix}>=6m"] += 1


def add_row_buckets(label: str, row: dict[str, Any], probability: float, buckets: Counter[str]) -> None:
    buckets[f"{label}|status={row.get('status', 'missing')}"] += 1
    buckets[f"{label}|support_count={row.get('support_count', 'missing')}"] += 1
    bucket_numeric(f"{label}|probability", probability, buckets)
    bucket_numeric(f"{label}|relative_x", row.get("relative_x"), buckets)
    bucket_numeric(f"{label}|abs_relative_y", row.get("abs_relative_y"), buckets)
    bucket_numeric(f"{label}|distance_to_trajectory", row.get("distance_to_trajectory"), buckets)


def error_buckets(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    selected_indices: set[int],
) -> dict[str, int]:
    buckets: Counter[str] = Counter()
    for index, row in enumerate(rows):
        matched = row.get("candidate_matches_gt") is True
        selected = index in selected_indices
        if selected and not matched:
            add_row_buckets("selected_fp", row, probabilities[index], buckets)
        elif matched and not selected:
            add_row_buckets("missed_positive_candidate", row, probabilities[index], buckets)
    return dict(buckets.most_common(40))


def logreg_interpretability(model: dict[str, Any], names: list[str]) -> dict[str, list[list[Any]]]:
    weights = [float(value) for value in model["weights"]]
    return {
        "top_positive_weights": [[name, weight] for name, weight in sorted(zip(names, weights), key=lambda item: item[1], reverse=True)[:10]],
        "top_negative_weights": [[name, weight] for name, weight in sorted(zip(names, weights), key=lambda item: item[1])[:10]],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Q3 Invisible Policy Optimization",
        "",
        f"Baseline F1: `{report['targets']['baseline_f1']:.6f}`",
        f"VP target (+{report['targets']['vp_improvement']:.0%}): `{report['targets']['vp_target_f1']:.6f}`",
        f"Stretch target (+{report['targets']['stretch_improvement']:.0%}): `{report['targets']['stretch_target_f1']:.6f}`",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Policy | Split | F1 | P | R | Threshold | Target |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(report["leaderboard"], start=1):
        lines.append(
            "| "
            f"{index} | `{row['model']}` | `{row['policy']}` | `{row['split']}` | "
            f"`{row['f1']:.6f}` | `{row['precision']:.6f}` | `{row['recall']:.6f}` | "
            f"`{row['threshold']:.2f}` | `{row['target_status']}` |"
        )
    lines.extend(["", "## Top Error Buckets", ""])
    for row in report["leaderboard"][: min(3, len(report["leaderboard"]))]:
        key = row["result_key"]
        detail = report["results"][key]
        lines.extend([f"### `{key}`", "", "| Bucket | Count |", "| --- | ---: |"])
        for bucket, count in detail.get("eval", detail["train"]).get("error_buckets", {}).items():
            lines.append(f"| `{bucket}` | `{count}` |")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def target_status(f1: float, vp_target: float, stretch_target: float) -> str:
    if f1 >= stretch_target:
        return "stretch"
    if f1 >= vp_target:
        return "vp"
    return "below"


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_features_jsonl).expanduser().resolve()
    train_all = load_rows(train_path)
    train_candidates = candidate_rows(train_all)
    names = feature_names()
    means, stds = normalization(train_candidates)
    train_x = normalized_vectors(train_candidates, means, stds)
    train_y = labels(train_candidates)

    eval_all: list[dict[str, Any]] = []
    eval_candidates: list[dict[str, Any]] = []
    eval_x: list[list[float]] = []
    if args.eval_features_jsonl:
        eval_path = Path(args.eval_features_jsonl).expanduser().resolve()
        eval_all = load_rows(eval_path)
        eval_candidates = candidate_rows(eval_all)
        eval_x = normalized_vectors(eval_candidates, means, stds)

    report: dict[str, Any] = {
        "run_name": args.run_name,
        "train_features_jsonl": str(train_path),
        "eval_features_jsonl": str(Path(args.eval_features_jsonl).expanduser().resolve()) if args.eval_features_jsonl else "",
        "train_candidate_rows": len(train_candidates),
        "eval_candidate_rows": len(eval_candidates),
        "targets": {
            "baseline_f1": args.baseline_f1,
            "vp_improvement": args.vp_improvement,
            "stretch_improvement": args.stretch_improvement,
            "vp_target_f1": args.baseline_f1 * (1.0 + args.vp_improvement),
            "stretch_target_f1": args.baseline_f1 * (1.0 + args.stretch_improvement),
        },
        "results": {},
        "leaderboard": [],
    }
    vp_target = float(report["targets"]["vp_target_f1"])
    stretch_target = float(report["targets"]["stretch_target_f1"])

    model_specs: list[tuple[str, dict[str, Any], list[float], list[float]]] = []
    if "logreg" in args.models:
        logreg_model = train_logreg(
            train_x,
            train_y,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            balance=not args.no_class_balance,
        )
        model_specs.append(("logreg", logreg_model, predict_logreg(logreg_model, train_x), predict_logreg(logreg_model, eval_x) if eval_x else []))
    if "mlp" in args.models:
        mlp_model = train_mlp(
            train_candidates,
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
        )
        model_specs.append(("mlp", mlp_model, predict_mlp(mlp_model, train_x), predict_mlp(mlp_model, eval_x) if eval_x else []))

    for model_name, model, train_probabilities, eval_probabilities in model_specs:
        policies = choose_policy_candidates(train_candidates, train_probabilities, train_all)
        for policy in policies:
            key = f"{model_name}:{policy['policy']}:t{str(policy['threshold']).replace('.', 'p')}"
            train_metrics = evaluate_policy_on_split(policy, train_candidates, train_probabilities, train_all)
            result: dict[str, Any] = {
                "model": model_name,
                "policy": policy["policy"],
                "train": train_metrics,
            }
            if eval_candidates:
                result["eval"] = evaluate_policy_on_split(policy, eval_candidates, eval_probabilities, eval_all)
            report["results"][key] = result
            split_name = "eval" if "eval" in result else "train"
            metrics = result[split_name]
            report["leaderboard"].append(
                {
                    "result_key": key,
                    "model": model_name,
                    "policy": policy["policy"],
                    "split": split_name,
                    "threshold": metrics["threshold"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "target_status": target_status(metrics["f1"], vp_target, stretch_target),
                }
            )

        model_json = {
            "model_type": model_name,
            "feature_names": names,
            "normalization": {"mean": means, "std": stds},
            **model,
            "interpretability": logreg_interpretability(model, names) if model_name == "logreg" else {},
        }
        model_path = output_dir / f"{args.run_name}_{model_name}_model.json"
        model_path.write_text(json.dumps(model_json, indent=2), encoding="utf-8")
        report.setdefault("model_paths", {})[model_name] = str(model_path)

    report["leaderboard"].sort(key=lambda item: (item["f1"], item["precision"], item["recall"]), reverse=True)
    report["leaderboard"] = report["leaderboard"][: args.max_report_policies]

    deployable_models: dict[str, str] = {}
    for row in report["leaderboard"]:
        model_name = str(row["model"])
        base_model_path = report.get("model_paths", {}).get(model_name)
        if not isinstance(base_model_path, str):
            continue
        base_model = json.loads(Path(base_model_path).read_text(encoding="utf-8"))
        base_model["threshold"] = float(row["threshold"])
        base_model["selected_policy"] = row["policy"]
        base_model["offline_selection_metrics"] = {
            "split": row["split"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "target_status": row["target_status"],
        }
        safe_policy = str(row["policy"]).replace(".", "p").replace(":", "_")
        safe_threshold = str(row["threshold"]).replace(".", "p")
        deployable_path = output_dir / f"{args.run_name}_{model_name}_{safe_policy}_t{safe_threshold}_deployable.json"
        deployable_path.write_text(json.dumps(base_model, indent=2), encoding="utf-8")
        deployable_models[str(row["result_key"])] = str(deployable_path)
    report["deployable_model_paths"] = deployable_models

    report_path = output_dir / f"{args.run_name}_report.json"
    markdown_path = output_dir / f"{args.run_name}_report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, markdown_path)

    print("=" * 72)
    print("Phase 8 Q3 Invisible Policy Optimization")
    print("=" * 72)
    print(f"train_candidate_rows: {len(train_candidates)}")
    if eval_candidates:
        print(f"eval_candidate_rows: {len(eval_candidates)}")
    print(f"baseline_f1: {args.baseline_f1:.6f}")
    print(f"vp_target_f1: {vp_target:.6f}")
    print(f"stretch_target_f1: {stretch_target:.6f}")
    for row in report["leaderboard"][:5]:
        print(
            f"[{row['split']}] {row['model']} {row['policy']} "
            f"f1={row['f1']:.6f} p={row['precision']:.6f} r={row['recall']:.6f} "
            f"t={row['threshold']:.2f} target={row['target_status']}"
        )
    print(f"saved_json: {report_path}")
    print(f"saved_markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
