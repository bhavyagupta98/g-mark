#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COORDINATE_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Q4 planning-awareness official exports by matching predicted output "
            "coordinates to reference answer coordinates and summarizing count/localization errors."
        )
    )
    parser.add_argument("--official-jsonl", default="")
    parser.add_argument(
        "--export-manifest",
        default="",
        help="Optional official export manifest. If set, the planning_awareness JSONL is resolved from it.",
    )
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=20)
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
            candidate_distance = point_distance(predicted_coord, gt_coords[gt_index])
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_gt = gt_index
        if best_gt >= 0 and best_distance <= threshold:
            matched_pred.add(pred_index)
            matched_gt.add(best_gt)
            remaining_gt.remove(best_gt)
    false_positive_pred = set(range(len(predicted_coords))) - matched_pred
    return matched_pred, matched_gt, false_positive_pred


def bucket_count(prefix: str, count: int, buckets: Counter[str]) -> None:
    if count == 0:
        label = "0"
    elif count == 1:
        label = "1"
    elif count == 2:
        label = "2"
    elif count == 3:
        label = "3"
    else:
        label = "4+"
    buckets[f"{prefix}={label}"] += 1


def coord_buckets(prefix: str, coords: list[tuple[float, float]], buckets: Counter[str]) -> None:
    for x, y in coords:
        if x < -1.0:
            buckets[f"{prefix}|longitudinal=behind"] += 1
        elif x > 1.0:
            buckets[f"{prefix}|longitudinal=ahead"] += 1
        else:
            buckets[f"{prefix}|longitudinal=near_zero"] += 1

        abs_y = abs(y)
        if abs_y < 1.0:
            buckets[f"{prefix}|abs_y=<1m"] += 1
        elif abs_y < 3.0:
            buckets[f"{prefix}|abs_y=1-3m"] += 1
        else:
            buckets[f"{prefix}|abs_y=>=3m"] += 1


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Phase 8 Q4 Planning-Awareness Official Mismatch Report",
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
                f"- reference count: `{example['reference_count']}`",
                f"- predicted count: `{example['predicted_count']}`",
                f"- matched: `{example['matched_count']}`",
                f"- false positives: `{example['false_positive_count']}`",
                f"- false negatives: `{example['false_negative_count']}`",
                f"- reference coords: `{example['reference_coords']}`",
                f"- predicted coords: `{example['predicted_coords']}`",
                f"- output: {example['output_text']}",
                f"- reference: {example['reference_text']}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    official_jsonl = resolve_official_jsonl(args)
    if not official_jsonl.exists():
        raise SystemExit(f"Official JSONL not found: {official_jsonl}")

    records = load_records(official_jsonl)
    counts = Counter()
    buckets = Counter()
    examples: list[dict[str, Any]] = []

    for record in records:
        counts["samples"] += 1
        sid = sample_id(record)
        predicted_coords = coordinates(str(record.get("outputs", "")))
        gt_text = reference_text(record)
        gt_coords = coordinates(gt_text)
        matched_pred, matched_gt, false_positive_pred = match_predictions(
            predicted_coords,
            gt_coords,
            args.match_threshold,
        )
        false_negative_count = len(gt_coords) - len(matched_gt)
        false_positive_count = len(false_positive_pred)

        counts["reference_mentions"] += len(gt_coords)
        counts["predicted_mentions"] += len(predicted_coords)
        counts["matched_mentions"] += len(matched_pred)
        counts["false_positive_mentions"] += false_positive_count
        counts["false_negative_mentions"] += false_negative_count
        if len(predicted_coords) == len(gt_coords):
            counts["exact_count_matches"] += 1
        elif len(predicted_coords) < len(gt_coords):
            counts["under_predicted_count_rows"] += 1
        else:
            counts["over_predicted_count_rows"] += 1
        if not predicted_coords and gt_coords:
            counts["empty_prediction_positive_reference_rows"] += 1
        if predicted_coords and not gt_coords:
            counts["positive_prediction_empty_reference_rows"] += 1
        if false_positive_count or false_negative_count:
            counts["localization_or_count_error_rows"] += 1

        bucket_count("reference_count", len(gt_coords), buckets)
        bucket_count("predicted_count", len(predicted_coords), buckets)
        coord_buckets("false_positive", [predicted_coords[index] for index in false_positive_pred], buckets)
        coord_buckets(
            "false_negative",
            [coord for index, coord in enumerate(gt_coords) if index not in matched_gt],
            buckets,
        )

        if (false_positive_count or false_negative_count) and len(examples) < args.examples:
            examples.append(
                {
                    "sample_id": sid,
                    "reference_count": len(gt_coords),
                    "predicted_count": len(predicted_coords),
                    "matched_count": len(matched_pred),
                    "false_positive_count": false_positive_count,
                    "false_negative_count": false_negative_count,
                    "reference_coords": gt_coords,
                    "predicted_coords": predicted_coords,
                    "output_text": str(record.get("outputs", "")),
                    "reference_text": gt_text,
                }
            )

    report = {
        "official_jsonl": str(official_jsonl),
        "match_threshold": args.match_threshold,
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
    print(f"reference_mentions: {counts['reference_mentions']}")
    print(f"predicted_mentions: {counts['predicted_mentions']}")
    print(f"matched_mentions: {counts['matched_mentions']}")
    print(f"false_positive_mentions: {counts['false_positive_mentions']}")
    print(f"false_negative_mentions: {counts['false_negative_mentions']}")
    print(f"saved_json: {output_json}")
    print(f"saved_markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
