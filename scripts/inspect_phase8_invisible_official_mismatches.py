#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

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
            "Inspect Q3 invisible-object official exports by matching predicted output "
            "coordinates to reference answer coordinates and summarizing generic feature buckets."
        )
    )
    parser.add_argument("--official-jsonl", default="")
    parser.add_argument(
        "--export-manifest",
        default="",
        help="Optional official export manifest. If set, the Q3 invisible_objects JSONL is resolved from it.",
    )
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument(
        "--skip-feature-buckets",
        action="store_true",
        help="Only compute coordinate/count metrics. Avoids expensive scene preparation.",
    )
    return parser


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (base / path).resolve()


def resolve_official_jsonl(args: argparse.Namespace) -> Path:
    if args.official_jsonl:
        return Path(args.official_jsonl).expanduser().resolve()

    if not args.export_manifest:
        raise SystemExit("Provide either --official-jsonl or --export-manifest.")

    manifest_path = Path(args.export_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for run in manifest.get("runs", []):
        if not isinstance(run, dict):
            continue
        if str(run.get("task_type", "")) != "invisible_objects":
            continue
        output_jsonl = run.get("output_jsonl")
        if not output_jsonl:
            continue
        return resolve_path(manifest_path.parent, str(output_jsonl))

    raise SystemExit(f"No invisible_objects run with output_jsonl found in {manifest_path}")


def resolve_v2vgot_root(raw_value: str) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def coordinates(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORDINATE_PATTERN.findall(text)]


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def reference_text(record: dict[str, object]) -> str:
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


def object_features(track: ObjectTrack | None, sample) -> dict[str, object]:
    if track is None:
        return {"found": False}
    asker = next((agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id), None)
    distance_to_asker = None
    if asker is not None:
        distance_to_asker = point_distance(
            (track.position.x, track.position.y),
            (asker.pose.position.x, asker.pose.position.y),
        )
    distance_to_trajectory = None
    if sample.scene.future_trajectory.points:
        distance_to_trajectory = min(
            point_distance(
                (track.position.x, track.position.y),
                (point.x, point.y),
            )
            for point in sample.scene.future_trajectory.points
        )
    return {
        "found": True,
        "status": track.status.value,
        "confidence": round(float(track.confidence), 4),
        "support_count": len(track.provenance.source_agent_ids),
        "conflict_score": round(float(track.conflict_score), 4),
        "uncertainty_score": round(float(track.uncertainty_score), 4),
        "distance_to_asker": None if distance_to_asker is None else round(distance_to_asker, 4),
        "distance_to_trajectory": None if distance_to_trajectory is None else round(distance_to_trajectory, 4),
    }


def nearest_track_to_coord(
    coord: tuple[float, float],
    scene,
    max_distance: float = 1.0,
) -> tuple[ObjectTrack | None, float | None]:
    if scene is None:
        return None, None
    best_track = None
    best_distance = float("inf")
    object_tracks = getattr(scene, "object_tracks", getattr(scene, "objects", ()))
    for track in object_tracks:
        candidate_distance = point_distance((track.position.x, track.position.y), coord)
        if candidate_distance < best_distance:
            best_track = track
            best_distance = candidate_distance
    if best_distance <= max_distance:
        return best_track, best_distance
    return None, None


def bucket_features(
    *,
    label: str,
    features: dict[str, object],
    feature_buckets: Counter[str],
    coord: tuple[float, float] | None = None,
) -> None:
    feature_buckets[f"{label}|status={features.get('status', 'missing')}"] += 1
    feature_buckets[f"{label}|support={features.get('support_count', 'missing')}"] += 1
    distance_to_asker = features.get("distance_to_asker")
    if isinstance(distance_to_asker, (float, int)):
        if distance_to_asker < 2.0:
            feature_buckets[f"{label}|asker_distance=<2m"] += 1
        elif distance_to_asker < 10.0:
            feature_buckets[f"{label}|asker_distance=2-10m"] += 1
        else:
            feature_buckets[f"{label}|asker_distance=>=10m"] += 1
    distance_to_trajectory = features.get("distance_to_trajectory")
    if isinstance(distance_to_trajectory, (float, int)):
        if distance_to_trajectory < 2.0:
            feature_buckets[f"{label}|trajectory_distance=<2m"] += 1
        elif distance_to_trajectory <= 3.0:
            feature_buckets[f"{label}|trajectory_distance=2-3m"] += 1
        else:
            feature_buckets[f"{label}|trajectory_distance=>3m"] += 1
    if coord is not None:
        x, y = coord
        abs_x = abs(x)
        abs_y = abs(y)
        if x < -1.0:
            feature_buckets[f"{label}|longitudinal=behind"] += 1
        elif x > 1.0:
            feature_buckets[f"{label}|longitudinal=ahead"] += 1
        else:
            feature_buckets[f"{label}|longitudinal=near_zero"] += 1

        if abs_x < 10.0:
            feature_buckets[f"{label}|abs_x=<10m"] += 1
        elif abs_x < 30.0:
            feature_buckets[f"{label}|abs_x=10-30m"] += 1
        else:
            feature_buckets[f"{label}|abs_x=>=30m"] += 1

        if abs_y < 1.0:
            feature_buckets[f"{label}|abs_y=<1m"] += 1
        elif abs_y < 3.0:
            feature_buckets[f"{label}|abs_y=1-3m"] += 1
        else:
            feature_buckets[f"{label}|abs_y=>=3m"] += 1


def main() -> None:
    args = build_parser().parse_args()
    official_path = resolve_official_jsonl(args)
    if not official_path.exists():
        raise SystemExit(
            f"Official JSONL not found: {official_path}\n"
            "Tip: pass the corrected export manifest with --export-manifest, or use the generated "
            "filename pattern `invisible_objects_<scenario_name>_official.jsonl`."
        )
    v2vgot_root = resolve_v2vgot_root(args.v2vgot_root)

    samples = {}
    evaluator = None
    if not args.skip_feature_buckets:
        adapter = V2VGoTQABenchmarkAdapter(str(v2vgot_root))
        evaluator = V2VGoTQAPhase5AEvaluator(str(v2vgot_root))
        samples = {
            sample.sample_id: sample
            for sample in adapter.load_samples(split_name=args.split, file_name=args.file_name)
            if sample.task_type == BenchmarkTaskType.INVISIBLE_OBJECTS
        }

    counters: Counter[str] = Counter()
    feature_buckets: Counter[str] = Counter()
    false_positive_examples: list[dict[str, object]] = []
    false_negative_examples: list[dict[str, object]] = []

    with official_path.open("r", encoding="utf-8") as handle:
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
            object_ids = (
                kg_prediction.get("object_ids", [])
                if isinstance(kg_prediction, dict)
                else []
            )
            if not isinstance(object_ids, list):
                object_ids = []

            predicted_coords = coordinates(str(record.get("outputs", "")))
            gt_coords = coordinates(reference_text(record))
            counters["rows"] += 1
            counters[f"predicted_count_{len(predicted_coords)}"] += 1
            counters[f"gt_count_{len(gt_coords)}"] += 1
            if predicted_coords:
                counters["predicted_positive_rows"] += 1
            if gt_coords:
                counters["gt_positive_rows"] += 1

            remaining_gt = list(gt_coords)
            prepared_scene = None
            sample = samples.get(sample_id)
            if sample is not None and evaluator is not None:
                prepared_scene = evaluator.prepare_sample(sample, baseline_mode="cooperative")

            for index, predicted_coord in enumerate(predicted_coords):
                best_index = -1
                best_distance = float("inf")
                for gt_index, gt_coord in enumerate(remaining_gt):
                    candidate_distance = distance(predicted_coord, gt_coord)
                    if candidate_distance < best_distance:
                        best_distance = candidate_distance
                        best_index = gt_index

                matched = best_index >= 0 and best_distance <= args.match_threshold
                if matched:
                    counters["predicted_mentions_matched"] += 1
                    remaining_gt.pop(best_index)
                else:
                    counters["predicted_mentions_false_positive"] += 1

                object_id = str(object_ids[index]) if index < len(object_ids) else ""
                label = "tp" if matched else "fp"
                features = {"found": False}
                if not args.skip_feature_buckets:
                    track = prepared_scene.get_object(object_id) if prepared_scene is not None and object_id else None
                    features = object_features(track, sample) if sample is not None else object_features(track, None)
                    bucket_features(
                        label=label,
                        features=features,
                        feature_buckets=feature_buckets,
                        coord=predicted_coord,
                    )
                if len(false_positive_examples) < args.examples and not matched:
                    false_positive_examples.append(
                        {
                            "sample_id": sample_id,
                            "object_id": object_id,
                            "predicted_coord": predicted_coord,
                            "nearest_gt_distance": None if best_distance == float("inf") else round(best_distance, 3),
                            "gt_coords": gt_coords,
                            "features": features,
                            "outputs": record.get("outputs", ""),
                            "reference": reference_text(record),
                        }
                    )

            counters["gt_mentions_unmatched"] += len(remaining_gt)
            for gt_coord in remaining_gt:
                track = None
                nearest_track_distance = None
                features = {"found": False}
                if not args.skip_feature_buckets:
                    track, nearest_track_distance = nearest_track_to_coord(gt_coord, prepared_scene)
                    features = object_features(track, sample) if sample is not None else object_features(track, None)
                    bucket_features(
                        label="fn",
                        features=features,
                        feature_buckets=feature_buckets,
                        coord=gt_coord,
                    )
                if len(false_negative_examples) < args.examples:
                    false_negative_examples.append(
                        {
                            "sample_id": sample_id,
                            "gt_coord": gt_coord,
                            "nearest_track_distance": nearest_track_distance,
                            "nearest_track_id": "" if track is None else track.object_id,
                            "features": features,
                            "outputs": record.get("outputs", ""),
                            "reference": reference_text(record),
                        }
                    )

    print(
        json.dumps(
            {
                "official_jsonl": str(official_path),
                "v2vgot_root": str(v2vgot_root),
                "match_threshold": args.match_threshold,
                "counts": dict(sorted(counters.items())),
                "feature_buckets": dict(sorted(feature_buckets.items())),
                "false_positive_examples": false_positive_examples,
                "false_negative_examples": false_negative_examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
