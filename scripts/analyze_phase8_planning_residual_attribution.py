#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COORDINATE_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute Q4 planning-awareness residual errors by joining official outputs, "
            "candidate feature rows, and a frozen acceptor model."
        )
    )
    parser.add_argument("--export-manifest", default="")
    parser.add_argument("--official-jsonl", default="")
    parser.add_argument("--candidate-features-jsonl", required=True)
    parser.add_argument("--acceptor-model-json", required=True)
    parser.add_argument("--strict-threshold", type=float, default=0.5)
    parser.add_argument("--loose-threshold", type=float, default=4.0)
    parser.add_argument("--candidate-present-threshold", type=float, default=4.0)
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_jsonl_from_manifest(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for run in manifest.get("runs", []):
        if isinstance(run, dict) and run.get("task_type") == "planning_awareness":
            output_jsonl = run.get("output_jsonl")
            if output_jsonl:
                return resolve_repo_path(str(output_jsonl))
    raise SystemExit(f"No planning_awareness output_jsonl found in {manifest_path}")


def resolve_official_jsonl(args: argparse.Namespace) -> Path:
    if args.official_jsonl:
        return resolve_repo_path(args.official_jsonl)
    if args.export_manifest:
        return resolve_jsonl_from_manifest(resolve_repo_path(args.export_manifest))
    raise SystemExit("Provide either --official-jsonl or --export-manifest.")


def coordinates(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORDINATE_PATTERN.findall(text)]


def point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def reference_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if isinstance(item, dict) and item.get("from") in {"gpt", "assistant"}:
            value = item.get("value", "")
            return value if isinstance(value, str) else ""
    return ""


def sample_id(record: dict[str, Any]) -> str:
    kg_prediction = record.get("kg_prediction", {})
    if isinstance(kg_prediction, dict) and kg_prediction.get("sample_id") is not None:
        return str(kg_prediction["sample_id"])
    return str(record.get("sample_id", record.get("id", "")))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_candidate_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        if row.get("row_type") != "candidate":
            continue
        by_sample[str(row.get("sample_id", ""))].append(row)
    return by_sample


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


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


def annotate_probabilities(
    candidates_by_sample: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
) -> None:
    for rows in candidates_by_sample.values():
        for row in rows:
            row["acceptor_probability"] = candidate_probability(model, row)


def match_predictions(
    predicted_coords: list[tuple[float, float]],
    gt_coords: list[tuple[float, float]],
    threshold: float,
) -> tuple[set[int], set[int], set[int]]:
    remaining_gt = set(range(len(gt_coords)))
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for pred_index, predicted_coord in enumerate(predicted_coords):
        best_gt = -1
        best_distance = float("inf")
        for gt_index in remaining_gt:
            distance = point_distance(predicted_coord, gt_coords[gt_index])
            if distance < best_distance:
                best_distance = distance
                best_gt = gt_index
        if best_gt >= 0 and best_distance <= threshold:
            matched_pred.add(pred_index)
            matched_gt.add(best_gt)
            remaining_gt.remove(best_gt)
    false_positive_pred = set(range(len(predicted_coords))) - matched_pred
    return matched_pred, matched_gt, false_positive_pred


def nearest_candidate(
    coord: tuple[float, float],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float | None]:
    best_row = None
    best_distance = float("inf")
    for row in rows:
        distance = point_distance(coord, (float(row.get("x", 0.0)), float(row.get("y", 0.0))))
        if distance < best_distance:
            best_row = row
            best_distance = distance
    if best_row is None:
        return None, None
    return best_row, best_distance


def probability_bucket(probability: float) -> str:
    if probability < 0.30:
        return "<0.30"
    if probability < 0.45:
        return "0.30-0.45"
    if probability < 0.56:
        return "0.45-0.56"
    if probability < 0.65:
        return "0.56-0.65"
    if probability < 0.80:
        return "0.65-0.80"
    return ">=0.80"


def rank_bucket(rank: float) -> str:
    if rank <= 1:
        return "1"
    if rank <= 2:
        return "2"
    if rank <= 3:
        return "3"
    if rank <= 6:
        return "4-6"
    return "7+"


def distance_bucket(distance: float | None) -> str:
    if distance is None:
        return "missing"
    if distance <= 0.5:
        return "<=0.5m"
    if distance <= 1.0:
        return "0.5-1m"
    if distance <= 2.0:
        return "1-2m"
    if distance <= 4.0:
        return "2-4m"
    return ">4m"


def coord_region(coord: tuple[float, float]) -> str:
    x, y = coord
    if x < -1.0:
        long = "behind"
    elif x > 1.0:
        long = "ahead"
    else:
        long = "near_zero"
    abs_y = abs(y)
    if abs_y < 1.0:
        lat = "<1m"
    elif abs_y < 3.0:
        lat = "1-3m"
    else:
        lat = ">=3m"
    return f"{long}|abs_y={lat}"


def add_candidate_buckets(prefix: str, row: dict[str, Any] | None, buckets: Counter[str]) -> None:
    if row is None:
        buckets[f"{prefix}|candidate=absent"] += 1
        return
    probability = float(row.get("acceptor_probability", 0.0))
    buckets[f"{prefix}|prob={probability_bucket(probability)}"] += 1
    buckets[f"{prefix}|rank={rank_bucket(float(row.get('rank', 999)))}"] += 1
    buckets[f"{prefix}|visibility={row.get('visibility_state', 'unknown')}"] += 1
    buckets[f"{prefix}|status={row.get('status', 'unknown')}"] += 1
    buckets[f"{prefix}|trajectory_distance={distance_bucket(float(row.get('distance_to_trajectory', 999)))}"] += 1


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Phase 8 Q4 Planning Residual Attribution",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key, value in report["counts"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Buckets", "", "| Bucket | Count |", "| --- | ---: |"])
    for key, value in report["buckets"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Examples", ""])
    for example in report["examples"]:
        lines.extend(
            [
                f"### sample `{example['sample_id']}`",
                "",
                f"- type: `{example['type']}`",
                f"- coord: `{example['coord']}`",
                f"- nearest candidate distance: `{example.get('nearest_candidate_distance')}`",
                f"- nearest candidate: `{example.get('nearest_candidate')}`",
                f"- predicted coords: `{example['predicted_coords']}`",
                f"- reference coords: `{example['reference_coords']}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    official_jsonl = resolve_official_jsonl(args)
    candidate_features = resolve_repo_path(args.candidate_features_jsonl)
    acceptor_model_path = resolve_repo_path(args.acceptor_model_json)
    model = json.loads(acceptor_model_path.read_text(encoding="utf-8"))
    threshold = float(model.get("threshold", 0.5))

    candidates_by_sample = load_candidate_rows(candidate_features)
    annotate_probabilities(candidates_by_sample, model)
    records = load_jsonl(official_jsonl)
    counts = Counter()
    buckets = Counter()
    examples: list[dict[str, Any]] = []

    for record in records:
        counts["samples"] += 1
        sid = sample_id(record)
        rows = candidates_by_sample.get(sid, [])
        predicted_coords = coordinates(str(record.get("outputs", "")))
        gt_coords = coordinates(reference_text(record))
        strict_matched_pred, strict_matched_gt, false_positive_pred = match_predictions(
            predicted_coords,
            gt_coords,
            args.strict_threshold,
        )
        loose_matched_pred, loose_matched_gt, _loose_fp = match_predictions(
            predicted_coords,
            gt_coords,
            args.loose_threshold,
        )
        counts["reference_mentions"] += len(gt_coords)
        counts["predicted_mentions"] += len(predicted_coords)
        counts["strict_matched_mentions"] += len(strict_matched_pred)
        counts["loose_matched_mentions"] += len(loose_matched_pred)
        counts["strict_false_positive_mentions"] += len(false_positive_pred)
        counts["strict_false_negative_mentions"] += len(gt_coords) - len(strict_matched_gt)
        counts["strict_to_loose_recoverable_mentions"] += max(0, len(loose_matched_pred) - len(strict_matched_pred))

        for pred_index in false_positive_pred:
            coord = predicted_coords[pred_index]
            candidate, candidate_distance = nearest_candidate(coord, rows)
            buckets[f"false_positive|region={coord_region(coord)}"] += 1
            buckets[f"false_positive|nearest_candidate_distance={distance_bucket(candidate_distance)}"] += 1
            add_candidate_buckets("false_positive", candidate, buckets)
            if len(examples) < args.examples:
                examples.append(
                    {
                        "type": "false_positive",
                        "sample_id": sid,
                        "coord": coord,
                        "nearest_candidate_distance": candidate_distance,
                        "nearest_candidate": None if candidate is None else {
                            "object_id": candidate.get("object_id"),
                            "rank": candidate.get("rank"),
                            "probability": round(float(candidate.get("acceptor_probability", 0.0)), 6),
                            "distance_to_trajectory": candidate.get("distance_to_trajectory"),
                            "visibility_state": candidate.get("visibility_state"),
                        },
                        "predicted_coords": predicted_coords,
                        "reference_coords": gt_coords,
                    }
                )

        for gt_index, coord in enumerate(gt_coords):
            if gt_index in strict_matched_gt:
                continue
            candidate, candidate_distance = nearest_candidate(coord, rows)
            present = candidate_distance is not None and candidate_distance <= args.candidate_present_threshold
            accepted_like = candidate is not None and float(candidate.get("acceptor_probability", 0.0)) >= threshold
            buckets[f"false_negative|region={coord_region(coord)}"] += 1
            buckets[f"false_negative|candidate_present={present}"] += 1
            buckets[f"false_negative|accepted_like={accepted_like}"] += 1
            buckets[f"false_negative|nearest_candidate_distance={distance_bucket(candidate_distance)}"] += 1
            if gt_index in loose_matched_gt:
                buckets["false_negative|strict_miss_loose_match=True"] += 1
            else:
                buckets["false_negative|strict_miss_loose_match=False"] += 1
            add_candidate_buckets("false_negative", candidate if present else None, buckets)
            if len(examples) < args.examples:
                examples.append(
                    {
                        "type": "false_negative",
                        "sample_id": sid,
                        "coord": coord,
                        "nearest_candidate_distance": candidate_distance,
                        "nearest_candidate": None if candidate is None else {
                            "object_id": candidate.get("object_id"),
                            "rank": candidate.get("rank"),
                            "probability": round(float(candidate.get("acceptor_probability", 0.0)), 6),
                            "distance_to_trajectory": candidate.get("distance_to_trajectory"),
                            "visibility_state": candidate.get("visibility_state"),
                        },
                        "predicted_coords": predicted_coords,
                        "reference_coords": gt_coords,
                    }
                )

    report = {
        "official_jsonl": str(official_jsonl),
        "candidate_features_jsonl": str(candidate_features),
        "acceptor_model_json": str(acceptor_model_path),
        "strict_threshold": args.strict_threshold,
        "loose_threshold": args.loose_threshold,
        "candidate_present_threshold": args.candidate_present_threshold,
        "counts": dict(counts),
        "buckets": dict(sorted(buckets.items())),
        "examples": examples,
    }
    output_json = resolve_repo_path(args.output_json)
    output_markdown = resolve_repo_path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_markdown)
    print(f"samples: {counts['samples']}")
    print(f"strict_matched_mentions: {counts['strict_matched_mentions']}")
    print(f"loose_matched_mentions: {counts['loose_matched_mentions']}")
    print(f"strict_to_loose_recoverable_mentions: {counts['strict_to_loose_recoverable_mentions']}")
    print(f"strict_false_positive_mentions: {counts['strict_false_positive_mentions']}")
    print(f"strict_false_negative_mentions: {counts['strict_false_negative_mentions']}")
    print(f"saved_json: {output_json}")
    print(f"saved_markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
