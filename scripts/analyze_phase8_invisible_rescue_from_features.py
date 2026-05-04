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
            "Fast Q3 rescue analysis from official exports plus a precomputed candidate feature table. "
            "This avoids expensive scene preparation."
        )
    )
    parser.add_argument("--reference-export-manifest", required=True, help="Usually legacy_traj6.")
    parser.add_argument("--candidate-export-manifest", required=True, help="Usually logreg_acceptor_t0p25.")
    parser.add_argument("--reference-features-jsonl", required=True, help="Candidate feature table for the reference policy.")
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


def resolve_invisible_jsonl(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for run in manifest.get("runs", []):
        if isinstance(run, dict) and run.get("task_type") == "invisible_objects":
            output_jsonl = run.get("output_jsonl")
            if output_jsonl:
                return resolve_repo_path(str(output_jsonl))
    raise SystemExit(f"No invisible_objects output_jsonl found in {manifest_path}")


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


def load_official_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kg_prediction = record.get("kg_prediction", {})
            sample_id = str(
                kg_prediction.get("sample_id")
                if isinstance(kg_prediction, dict) and kg_prediction.get("sample_id") is not None
                else record.get("sample_id", record.get("id", ""))
            )
            records[sample_id] = record
    return records


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


def load_candidate_features(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_type") != "candidate":
                continue
            sample_id = str(row.get("sample_id", ""))
            by_sample.setdefault(sample_id, []).append(row)
    return by_sample


def nearest_feature_row(
    gt_coord: tuple[float, float],
    rows: list[dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any] | None, float | None]:
    best_row = None
    best_distance = float("inf")
    for row in rows:
        x = row.get("x")
        y = row.get("y")
        if not isinstance(x, (float, int)) or not isinstance(y, (float, int)):
            continue
        candidate_distance = point_distance(gt_coord, (float(x), float(y)))
        if candidate_distance < best_distance:
            best_distance = candidate_distance
            best_row = row
    if best_row is not None and best_distance <= threshold:
        return best_row, best_distance
    return None, None


def bucket_value(prefix: str, value: Any, buckets: Counter[str]) -> None:
    buckets[f"{prefix}={value}"] += 1


def bucket_numeric(prefix: str, value: Any, buckets: Counter[str]) -> None:
    if not isinstance(value, (float, int)):
        buckets[f"{prefix}=missing"] += 1
        return
    numeric = float(value)
    if prefix.endswith("abs_relative_y"):
        if numeric < 1:
            bucket_value(prefix, "<1m", buckets)
        elif numeric < 3:
            bucket_value(prefix, "1-3m", buckets)
        else:
            bucket_value(prefix, ">=3m", buckets)
    elif prefix.endswith("distance_to_trajectory"):
        if numeric < 2:
            bucket_value(prefix, "<2m", buckets)
        elif numeric < 6:
            bucket_value(prefix, "2-6m", buckets)
        else:
            bucket_value(prefix, ">=6m", buckets)
    elif prefix.endswith("relative_x"):
        if numeric < -1:
            bucket_value(prefix, "behind", buckets)
        elif numeric > 1:
            bucket_value(prefix, "ahead", buckets)
        else:
            bucket_value(prefix, "near_zero", buckets)


def add_feature_buckets(label: str, row: dict[str, Any] | None, buckets: Counter[str]) -> None:
    if row is None:
        buckets[f"{label}|feature_row=missing"] += 1
        return
    buckets[f"{label}|feature_row=matched"] += 1
    bucket_value(f"{label}|status", row.get("status", "missing"), buckets)
    bucket_value(f"{label}|support_count", row.get("support_count", "missing"), buckets)
    bucket_value(f"{label}|selected_by_policy", row.get("selected_by_policy", "missing"), buckets)
    bucket_value(f"{label}|candidate_matches_gt", row.get("candidate_matches_gt", "missing"), buckets)
    bucket_numeric(f"{label}|relative_x", row.get("relative_x"), buckets)
    bucket_numeric(f"{label}|abs_relative_y", row.get("abs_relative_y"), buckets)
    bucket_numeric(f"{label}|distance_to_trajectory", row.get("distance_to_trajectory"), buckets)


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Phase 8 Q3 Rescue Analysis From Features",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key, value in report["counts"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Buckets", "", "| Bucket | Count |", "| --- | ---: |"])
    for key, value in report["feature_buckets"].items():
        lines.append(f"| `{key}` | `{value}` |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    reference_manifest = resolve_repo_path(args.reference_export_manifest)
    candidate_manifest = resolve_repo_path(args.candidate_export_manifest)
    reference_jsonl = resolve_invisible_jsonl(reference_manifest)
    candidate_jsonl = resolve_invisible_jsonl(candidate_manifest)
    features_path = resolve_repo_path(args.reference_features_jsonl)

    print("loading_official_records", flush=True)
    reference_records = load_official_records(reference_jsonl)
    candidate_records = load_official_records(candidate_jsonl)
    print("loading_candidate_features", flush=True)
    features_by_sample = load_candidate_features(features_path)

    counts: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    rescue_examples: list[dict[str, Any]] = []
    suppressed_fp_examples: list[dict[str, Any]] = []

    sample_ids = sorted(
        set(reference_records) & set(candidate_records),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    for sample_id in sample_ids:
        reference_record = reference_records[sample_id]
        candidate_record = candidate_records[sample_id]
        gt_coords = coordinates(reference_text(reference_record))
        reference_coords = coordinates(str(reference_record.get("outputs", "")))
        candidate_coords = coordinates(str(candidate_record.get("outputs", "")))
        _, reference_matched_gt, reference_fp_pred = match_predictions(
            reference_coords, gt_coords, args.match_threshold
        )
        _, candidate_matched_gt, candidate_fp_pred = match_predictions(
            candidate_coords, gt_coords, args.match_threshold
        )

        counts["rows"] += 1
        counts["gt_mentions"] += len(gt_coords)
        counts["reference_tp_mentions"] += len(reference_matched_gt)
        counts["candidate_tp_mentions"] += len(candidate_matched_gt)
        counts["reference_fp_mentions"] += len(reference_fp_pred)
        counts["candidate_fp_mentions"] += len(candidate_fp_pred)

        rescue_gt_indices = reference_matched_gt - candidate_matched_gt
        counts["rescue_targets"] += len(rescue_gt_indices)
        feature_rows = features_by_sample.get(sample_id, [])
        for gt_index in rescue_gt_indices:
            gt_coord = gt_coords[gt_index]
            feature_row, feature_distance = nearest_feature_row(gt_coord, feature_rows, args.match_threshold)
            add_feature_buckets("rescue_target", feature_row, buckets)
            if len(rescue_examples) < args.examples:
                rescue_examples.append(
                    {
                        "sample_id": sample_id,
                        "gt_coord": gt_coord,
                        "feature_distance": feature_distance,
                        "feature_row": feature_row,
                    }
                )

        candidate_fp_coords = {candidate_coords[index] for index in candidate_fp_pred}
        for pred_index in reference_fp_pred:
            reference_coord = reference_coords[pred_index]
            if any(point_distance(reference_coord, candidate_coord) <= args.match_threshold for candidate_coord in candidate_fp_coords):
                continue
            counts["suppressed_reference_fp"] += 1
            feature_row, feature_distance = nearest_feature_row(reference_coord, feature_rows, args.match_threshold)
            add_feature_buckets("suppressed_fp", feature_row, buckets)
            if len(suppressed_fp_examples) < args.examples:
                suppressed_fp_examples.append(
                    {
                        "sample_id": sample_id,
                        "reference_coord": reference_coord,
                        "feature_distance": feature_distance,
                        "feature_row": feature_row,
                    }
                )

    report = {
        "reference_export_manifest": str(reference_manifest),
        "candidate_export_manifest": str(candidate_manifest),
        "reference_features_jsonl": str(features_path),
        "match_threshold": args.match_threshold,
        "counts": dict(sorted(counts.items())),
        "feature_buckets": dict(sorted(buckets.items())),
        "rescue_examples": rescue_examples,
        "suppressed_fp_examples": suppressed_fp_examples,
    }

    output_json = resolve_repo_path(args.output_json)
    output_markdown = resolve_repo_path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_markdown)

    print("=" * 72)
    print("Phase 8 Q3 Rescue Analysis From Features")
    print("=" * 72)
    print(f"rows: {counts['rows']}")
    print(f"reference_tp: {counts['reference_tp_mentions']} candidate_tp: {counts['candidate_tp_mentions']}")
    print(f"reference_fp: {counts['reference_fp_mentions']} candidate_fp: {counts['candidate_fp_mentions']}")
    print(f"rescue_targets: {counts['rescue_targets']}")
    print(f"suppressed_reference_fp: {counts['suppressed_reference_fp']}")
    print(f"saved_json: {output_json}")
    print(f"saved_markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
