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
            "Diagnose Q3 invisible-object false negatives against a precomputed candidate "
            "feature table. Answers whether missed GT objects were present in the candidate "
            "pool/shortlist before the acceptor made its final decision."
        )
    )
    parser.add_argument("--export-manifest", required=True, help="Official export manifest for the Q3 policy under test.")
    parser.add_argument("--features-jsonl", required=True, help="Candidate feature table, usually legacy_traj6 shortlist export.")
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
        if not isinstance(run, dict) or run.get("task_type") != "invisible_objects":
            continue
        output_jsonl = run.get("output_jsonl")
        if output_jsonl:
            return resolve_repo_path(str(output_jsonl))
    raise SystemExit(f"No invisible_objects output_jsonl found in {manifest_path}")


def coordinates(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORDINATE_PATTERN.findall(text)]


def reference_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if isinstance(item, dict) and item.get("from") in {"gpt", "assistant"}:
            value = item.get("value", "")
            return value if isinstance(value, str) else ""
    return ""


def point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


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


def load_candidate_features(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_type") == "candidate":
                by_sample[str(row.get("sample_id", ""))].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda item: int(item.get("rank", 999999)))
    return by_sample


def nearest_candidate(
    gt_coord: tuple[float, float],
    rows: list[dict[str, Any]],
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
    if best_row is None:
        return None, None
    return best_row, best_distance


def add_feature_buckets(prefix: str, row: dict[str, Any] | None, buckets: Counter[str]) -> None:
    if row is None:
        buckets[f"{prefix}|candidate=missing"] += 1
        return
    rank = row.get("rank")
    if isinstance(rank, int):
        if rank <= 1:
            buckets[f"{prefix}|rank=1"] += 1
        elif rank <= 3:
            buckets[f"{prefix}|rank=2-3"] += 1
        elif rank <= 6:
            buckets[f"{prefix}|rank=4-6"] += 1
        else:
            buckets[f"{prefix}|rank=>6"] += 1
    buckets[f"{prefix}|status={row.get('status', 'missing')}"] += 1
    buckets[f"{prefix}|support_count={row.get('support_count', 'missing')}"] += 1
    rel_x = row.get("relative_x")
    if isinstance(rel_x, (float, int)):
        if rel_x < -1:
            buckets[f"{prefix}|relative_x=behind"] += 1
        elif rel_x > 1:
            buckets[f"{prefix}|relative_x=ahead"] += 1
        else:
            buckets[f"{prefix}|relative_x=near_zero"] += 1
    abs_y = row.get("abs_relative_y")
    if isinstance(abs_y, (float, int)):
        if abs_y < 1:
            buckets[f"{prefix}|abs_relative_y=<1m"] += 1
        elif abs_y < 3:
            buckets[f"{prefix}|abs_relative_y=1-3m"] += 1
        else:
            buckets[f"{prefix}|abs_relative_y=>=3m"] += 1
    traj = row.get("distance_to_trajectory")
    if isinstance(traj, (float, int)):
        if traj < 2:
            buckets[f"{prefix}|distance_to_trajectory=<2m"] += 1
        elif traj < 6:
            buckets[f"{prefix}|distance_to_trajectory=2-6m"] += 1
        else:
            buckets[f"{prefix}|distance_to_trajectory=>=6m"] += 1


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Q3 Invisible Shortlist Coverage",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key, value in report["counts"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Coverage", "", "| Bucket | Count |", "| --- | ---: |"])
    for key, value in report["coverage_buckets"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Feature Buckets", "", "| Bucket | Count |", "| --- | ---: |"])
    for key, value in report["feature_buckets"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Examples", ""])
    for example in report["examples"]:
        lines.append(
            f"- sample `{example['sample_id']}` gt={example['gt_coord']} "
            f"nearest_rank={example.get('nearest_rank')} nearest_distance={example.get('nearest_distance')}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = resolve_repo_path(args.export_manifest)
    official_jsonl = resolve_invisible_jsonl(manifest_path)
    features_path = resolve_repo_path(args.features_jsonl)
    records = load_official_records(official_jsonl)
    features_by_sample = load_candidate_features(features_path)

    counts: Counter[str] = Counter()
    coverage_buckets: Counter[str] = Counter()
    feature_buckets: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for sample_id, record in records.items():
        gt_coords = coordinates(reference_text(record))
        pred_coords = coordinates(str(record.get("outputs", "")))
        _, matched_gt, false_positive_pred = match_predictions(pred_coords, gt_coords, args.match_threshold)
        rows = features_by_sample.get(sample_id, [])
        counts["rows"] += 1
        counts["gt_mentions"] += len(gt_coords)
        counts["predicted_mentions"] += len(pred_coords)
        counts["matched_gt_mentions"] += len(matched_gt)
        counts["false_positive_mentions"] += len(false_positive_pred)
        for gt_index, gt_coord in enumerate(gt_coords):
            if gt_index in matched_gt:
                continue
            counts["false_negative_mentions"] += 1
            row, distance = nearest_candidate(gt_coord, rows)
            in_pool = row is not None and distance is not None and distance <= args.match_threshold
            if not in_pool:
                coverage_buckets["fn_gt_absent_from_shortlist"] += 1
                add_feature_buckets("absent_fn_nearest", row, feature_buckets)
            else:
                rank = int(row.get("rank", 999999))
                coverage_buckets["fn_gt_present_in_shortlist"] += 1
                if rank <= 1:
                    coverage_buckets["fn_gt_present_rank_1"] += 1
                if rank <= 3:
                    coverage_buckets["fn_gt_present_rank_le_3"] += 1
                if rank <= 6:
                    coverage_buckets["fn_gt_present_rank_le_6"] += 1
                if rank <= 12:
                    coverage_buckets["fn_gt_present_rank_le_12"] += 1
                add_feature_buckets("present_fn_candidate", row, feature_buckets)
            if len(examples) < args.examples:
                examples.append(
                    {
                        "sample_id": sample_id,
                        "gt_coord": [round(gt_coord[0], 3), round(gt_coord[1], 3)],
                        "present_in_shortlist": bool(in_pool),
                        "nearest_rank": None if row is None else row.get("rank"),
                        "nearest_distance": None if distance is None else round(distance, 6),
                        "nearest_object_id": None if row is None else row.get("object_id"),
                        "nearest_status": None if row is None else row.get("status"),
                        "nearest_support_count": None if row is None else row.get("support_count"),
                    }
                )

    report = {
        "export_manifest": str(manifest_path),
        "official_jsonl": str(official_jsonl),
        "features_jsonl": str(features_path),
        "match_threshold": args.match_threshold,
        "counts": dict(sorted(counts.items())),
        "coverage_buckets": dict(coverage_buckets.most_common()),
        "feature_buckets": dict(feature_buckets.most_common(60)),
        "examples": examples,
    }
    output_json = resolve_repo_path(args.output_json)
    output_markdown = resolve_repo_path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_markdown)

    print("=" * 72)
    print("Phase 8 Q3 Invisible Shortlist Coverage")
    print("=" * 72)
    for key, value in report["counts"].items():
        print(f"{key}: {value}")
    for key, value in report["coverage_buckets"].items():
        print(f"{key}: {value}")
    print(f"saved_json: {output_json}")
    print(f"saved_markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
