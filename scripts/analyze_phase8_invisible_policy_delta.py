#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.scene import ObjectTrack  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402


DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)
COORDINATE_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two official Q3 invisible-object exports and mine recall/precision deltas. "
            "Use this to see which legacy true positives a stricter policy loses, and which "
            "legacy false positives it suppresses."
        )
    )
    parser.add_argument("--reference-export-manifest", required=True, help="Usually legacy_traj6.")
    parser.add_argument("--candidate-export-manifest", required=True, help="Usually logreg_acceptor_t0p25.")
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument(
        "--skip-feature-buckets",
        action="store_true",
        help="Only compute coordinate-level delta counts. Avoids expensive scene preparation.",
    )
    parser.add_argument(
        "--feature-target",
        default="all",
        choices=("all", "rescue", "suppressed_fp"),
        help=(
            "When feature buckets are enabled, choose which policy deltas should trigger scene "
            "preparation. Use `rescue` to inspect legacy true positives lost by the candidate."
        ),
    )
    parser.add_argument(
        "--max-feature-samples",
        type=int,
        default=0,
        help="Maximum number of delta samples to prepare for feature buckets. Use 0 for no cap.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_v2vgot_root(raw_value: str) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


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


def object_ids(record: dict[str, Any]) -> list[str]:
    kg_prediction = record.get("kg_prediction", {})
    values = kg_prediction.get("object_ids", []) if isinstance(kg_prediction, dict) else []
    return [str(item) for item in values] if isinstance(values, list) else []


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


def nearest_track_to_coord(coord: tuple[float, float], tracks: tuple[ObjectTrack, ...]) -> tuple[ObjectTrack | None, float | None]:
    best_track = None
    best_distance = float("inf")
    for track in tracks:
        candidate_distance = point_distance(coord, (track.position.x, track.position.y))
        if candidate_distance < best_distance:
            best_track = track
            best_distance = candidate_distance
    if best_track is None:
        return None, None
    return best_track, best_distance


def track_features(track: ObjectTrack | None, sample: Any) -> dict[str, Any]:
    if track is None:
        return {"found": False}
    asker = next((agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id), None)
    relative_x = track.position.x - asker.pose.position.x if asker is not None else track.position.x
    relative_y = track.position.y - asker.pose.position.y if asker is not None else track.position.y
    distance_to_asker = math.hypot(relative_x, relative_y)
    distance_to_trajectory = None
    if sample.scene.future_trajectory.points:
        distance_to_trajectory = min(
            point_distance((track.position.x, track.position.y), (point.x, point.y))
            for point in sample.scene.future_trajectory.points
        )
    return {
        "found": True,
        "object_id": track.object_id,
        "status": track.status.value,
        "support_count": len(track.provenance.source_agent_ids),
        "confidence": round(float(track.confidence), 6),
        "conflict_score": round(float(track.conflict_score), 6),
        "uncertainty_score": round(float(track.uncertainty_score), 6),
        "relative_x": round(float(relative_x), 6),
        "relative_y": round(float(relative_y), 6),
        "abs_relative_y": round(abs(float(relative_y)), 6),
        "distance_to_asker": round(float(distance_to_asker), 6),
        "distance_to_trajectory": None if distance_to_trajectory is None else round(float(distance_to_trajectory), 6),
    }


def bucket(label: str, features: dict[str, Any], buckets: Counter[str]) -> None:
    buckets[f"{label}|status={features.get('status', 'missing')}"] += 1
    buckets[f"{label}|support={features.get('support_count', 'missing')}"] += 1
    relative_x = features.get("relative_x")
    if isinstance(relative_x, (int, float)):
        if relative_x < -1:
            buckets[f"{label}|longitudinal=behind"] += 1
        elif relative_x > 1:
            buckets[f"{label}|longitudinal=ahead"] += 1
        else:
            buckets[f"{label}|longitudinal=near_zero"] += 1
    abs_y = features.get("abs_relative_y")
    if isinstance(abs_y, (int, float)):
        if abs_y < 1:
            buckets[f"{label}|abs_y=<1m"] += 1
        elif abs_y < 3:
            buckets[f"{label}|abs_y=1-3m"] += 1
        else:
            buckets[f"{label}|abs_y=>=3m"] += 1
    distance_to_trajectory = features.get("distance_to_trajectory")
    if isinstance(distance_to_trajectory, (int, float)):
        if distance_to_trajectory < 2:
            buckets[f"{label}|trajectory=<2m"] += 1
        elif distance_to_trajectory < 6:
            buckets[f"{label}|trajectory=2-6m"] += 1
        else:
            buckets[f"{label}|trajectory=>=6m"] += 1


def append_example(examples: list[dict[str, Any]], limit: int, payload: dict[str, Any]) -> None:
    if len(examples) < limit:
        examples.append(payload)


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    counts = report["counts"]
    lines = [
        "# Phase 8 Q3 Invisible Policy Delta",
        "",
        f"- `reference_manifest`: `{report['reference_export_manifest']}`",
        f"- `candidate_manifest`: `{report['candidate_export_manifest']}`",
        f"- `split`: `{report['split']}`",
        f"- `match_threshold`: `{report['match_threshold']}`",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key in sorted(counts):
        lines.append(f"| `{key}` | `{counts[key]}` |")
    lines.extend(
        [
            "",
            "## Buckets",
            "",
            "| Bucket | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(report["feature_buckets"].items()):
        lines.append(f"| `{key}` | `{value}` |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    reference_manifest = resolve_repo_path(args.reference_export_manifest)
    candidate_manifest = resolve_repo_path(args.candidate_export_manifest)
    reference_jsonl = resolve_invisible_jsonl(reference_manifest)
    candidate_jsonl = resolve_invisible_jsonl(candidate_manifest)
    v2vgot_root = resolve_v2vgot_root(args.v2vgot_root)

    reference_records = load_official_records(reference_jsonl)
    candidate_records = load_official_records(candidate_jsonl)
    sample_ids = sorted(
        set(reference_records) & set(candidate_records),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )

    raw_samples: dict[str, Any] = {}
    evaluator: V2VGoTQAPhase5AEvaluator | None = None
    if not args.skip_feature_buckets:
        adapter = V2VGoTQABenchmarkAdapter(str(v2vgot_root))
        evaluator = V2VGoTQAPhase5AEvaluator(str(v2vgot_root))
        raw_samples = {
            sample.sample_id: sample
            for sample in adapter.load_samples(split_name=args.split, file_name=args.file_name)
            if sample.task_type == BenchmarkTaskType.INVISIBLE_OBJECTS
        }

    counts: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    rescue_examples: list[dict[str, Any]] = []
    suppressed_fp_examples: list[dict[str, Any]] = []
    prepared_feature_samples = 0

    for sample_id in sample_ids:
        reference_record = reference_records[sample_id]
        candidate_record = candidate_records[sample_id]
        gt_coords = coordinates(reference_text(reference_record))
        reference_coords = coordinates(str(reference_record.get("outputs", "")))
        candidate_coords = coordinates(str(candidate_record.get("outputs", "")))
        reference_matched_pred, reference_matched_gt, reference_fp_pred = match_predictions(
            reference_coords, gt_coords, args.match_threshold
        )
        candidate_matched_pred, candidate_matched_gt, candidate_fp_pred = match_predictions(
            candidate_coords, gt_coords, args.match_threshold
        )

        counts["rows"] += 1
        counts["gt_mentions"] += len(gt_coords)
        counts["reference_tp_mentions"] += len(reference_matched_gt)
        counts["candidate_tp_mentions"] += len(candidate_matched_gt)
        counts["reference_fp_mentions"] += len(reference_fp_pred)
        counts["candidate_fp_mentions"] += len(candidate_fp_pred)

        reference_ids = object_ids(reference_record)
        candidate_fp_coords = {candidate_coords[index] for index in candidate_fp_pred}
        rescue_gt_indices = reference_matched_gt - candidate_matched_gt
        suppressed_fp_indices: list[int] = []
        for pred_index in reference_fp_pred:
            reference_coord = reference_coords[pred_index]
            if any(point_distance(reference_coord, candidate_coord) <= args.match_threshold for candidate_coord in candidate_fp_coords):
                continue
            suppressed_fp_indices.append(pred_index)

        counts["reference_tp_candidate_fn_mentions"] += len(rescue_gt_indices)
        counts["reference_fp_suppressed_by_candidate_mentions"] += len(suppressed_fp_indices)

        if args.skip_feature_buckets:
            continue

        sample = raw_samples.get(sample_id)
        if sample is None or evaluator is None:
            continue
        should_prepare_rescue = bool(rescue_gt_indices) and args.feature_target in {"all", "rescue"}
        should_prepare_suppressed = bool(suppressed_fp_indices) and args.feature_target in {"all", "suppressed_fp"}
        if not should_prepare_rescue and not should_prepare_suppressed:
            continue
        if args.max_feature_samples > 0 and prepared_feature_samples >= args.max_feature_samples:
            continue
        prepared_sample = replace(sample, scene=evaluator.prepare_sample(sample, baseline_mode="cooperative"))
        prepared_feature_samples += 1

        for gt_index in rescue_gt_indices if should_prepare_rescue else ():
            gt_coord = gt_coords[gt_index]
            track, track_distance = nearest_track_to_coord(gt_coord, prepared_sample.scene.object_tracks)
            features = track_features(track, prepared_sample)
            bucket("rescue_target", features, buckets)
            append_example(
                rescue_examples,
                args.examples,
                {
                    "sample_id": sample_id,
                    "gt_coord": gt_coord,
                    "nearest_track_distance": None if track_distance is None else round(track_distance, 6),
                    "features": features,
                    "reference_outputs": reference_record.get("outputs", ""),
                    "candidate_outputs": candidate_record.get("outputs", ""),
                },
            )

        for pred_index in suppressed_fp_indices if should_prepare_suppressed else ():
            reference_coord = reference_coords[pred_index]
            object_id = reference_ids[pred_index] if pred_index < len(reference_ids) else ""
            track = prepared_sample.scene.get_object(object_id) if object_id else None
            features = track_features(track, prepared_sample)
            bucket("suppressed_fp", features, buckets)
            append_example(
                suppressed_fp_examples,
                args.examples,
                {
                    "sample_id": sample_id,
                    "object_id": object_id,
                    "reference_coord": reference_coord,
                    "features": features,
                    "reference_outputs": reference_record.get("outputs", ""),
                    "candidate_outputs": candidate_record.get("outputs", ""),
                },
            )

    report = {
        "reference_export_manifest": str(reference_manifest),
        "candidate_export_manifest": str(candidate_manifest),
        "reference_jsonl": str(reference_jsonl),
        "candidate_jsonl": str(candidate_jsonl),
        "v2vgot_root": str(v2vgot_root),
        "split": args.split,
        "match_threshold": args.match_threshold,
        "counts": dict(sorted(counts.items())),
        "prepared_feature_samples": prepared_feature_samples,
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
    print("Phase 8 Q3 Invisible Policy Delta")
    print("=" * 72)
    print(f"split: {args.split}")
    print(f"rows: {counts['rows']}")
    print(f"reference_tp: {counts['reference_tp_mentions']} candidate_tp: {counts['candidate_tp_mentions']}")
    print(f"reference_fp: {counts['reference_fp_mentions']} candidate_fp: {counts['candidate_fp_mentions']}")
    print(f"rescue_targets: {counts['reference_tp_candidate_fn_mentions']}")
    print(f"suppressed_reference_fp: {counts['reference_fp_suppressed_by_candidate_mentions']}")
    print(f"prepared_feature_samples: {prepared_feature_samples}")
    print(f"saved_json: {output_json}")
    print(f"saved_markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
