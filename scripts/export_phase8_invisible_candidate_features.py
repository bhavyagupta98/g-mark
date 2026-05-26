#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.qa.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.qa.v2vgotqa_evaluator import GraphAblationMode  # noqa: E402
from kg_coop_drive.application.qa.v2vgotqa_router import InvisibleObjectsHandler, InvisibleSelectionPolicy  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.scene import ObjectTrack  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

COORDINATE_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")
DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export train/validation Q3 invisible-object candidate features for transparent "
            "train-calibrated policy analysis. Rows include hidden candidates and unmatched GT coordinates."
        )
    )
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument(
        "--graph-ablation-mode",
        default=GraphAblationMode.FULL.value,
        choices=tuple(item.value for item in GraphAblationMode),
    )
    parser.add_argument(
        "--invisible-ranker",
        default="legacy",
        choices=("legacy", "risk_adaptive", "road_region", "road_region_strict", "temporal_guard", "backtrack_guard"),
    )
    parser.add_argument("--invisible-max-results", type=int, default=1)
    parser.add_argument("--invisible-max-distance-to-trajectory", type=float, default=6.0)
    parser.add_argument("--invisible-min-risk", type=float, default=0.58)
    parser.add_argument("--invisible-min-relative-to-best", type=float, default=0.75)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for the full split.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    return parser


class _WorkerState:
    def __init__(self, *, evaluator: V2VGoTQAPhase5AEvaluator, handler: InvisibleObjectsHandler, policy: InvisibleSelectionPolicy) -> None:
        self.evaluator = evaluator
        self.handler = handler
        self.policy = policy


_WORKER_STATE: dict[tuple[str, str, str, str, int, float, float], _WorkerState] = {}


def _worker_state(config: dict[str, object]) -> _WorkerState:
    key = (
        str(config["v2vgot_root"]),
        str(config["baseline_mode"]),
        str(config["graph_ablation_mode"]),
        str(config["invisible_ranker"]),
        int(config["shortlist_size"]),
        float(config["min_risk"]),
        float(config["min_relative_to_best"]),
    )
    state = _WORKER_STATE.get(key)
    if state is not None:
        return state
    policy = InvisibleSelectionPolicy(
        max_results=int(config["max_results"]),
        shortlist_size=int(config["shortlist_size"]),
        max_distance_to_trajectory=float(config["max_distance_to_trajectory"]),
        min_risk=float(config["min_risk"]),
        min_relative_to_best=float(config["min_relative_to_best"]),
    )
    handler = InvisibleObjectsHandler(ranker=str(config["invisible_ranker"]), selection_policy=policy)
    evaluator = V2VGoTQAPhase5AEvaluator(
        str(config["v2vgot_root"]),
        graph_ablation=str(config["graph_ablation_mode"]),
    )
    state = _WorkerState(evaluator=evaluator, handler=handler, policy=policy)
    _WORKER_STATE[key] = state
    return state


def resolve_v2vgot_root(raw_value: str) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def coordinates(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORDINATE_PATTERN.findall(text)]


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


def nearest_coord_distance(
    coord: tuple[float, float],
    gt_coords: list[tuple[float, float]],
) -> float | None:
    if not gt_coords:
        return None
    return min(point_distance(coord, gt_coord) for gt_coord in gt_coords)


def nearest_track_to_coord(
    coord: tuple[float, float],
    tracks: tuple[ObjectTrack, ...],
) -> tuple[ObjectTrack | None, float | None]:
    best_track = None
    best_distance = float("inf")
    for track in tracks:
        distance = point_distance(coord, (track.position.x, track.position.y))
        if distance < best_distance:
            best_track = track
            best_distance = distance
    if best_track is None:
        return None, None
    return best_track, best_distance


def agent_relative_xy(sample, track: ObjectTrack) -> tuple[float, float]:
    asker = next((agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id), None)
    if asker is None:
        return track.position.x, track.position.y
    return (
        track.position.x - asker.pose.position.x,
        track.position.y - asker.pose.position.y,
    )


def track_feature_payload(sample, handler: InvisibleObjectsHandler, track: ObjectTrack) -> dict[str, object]:
    relative_x, relative_y = agent_relative_xy(sample, track)
    return {
        "object_id": track.object_id,
        "object_type": track.object_type,
        "x": round(float(track.position.x), 6),
        "y": round(float(track.position.y), 6),
        "relative_x": round(float(relative_x), 6),
        "relative_y": round(float(relative_y), 6),
        "abs_relative_x": round(abs(float(relative_x)), 6),
        "abs_relative_y": round(abs(float(relative_y)), 6),
        "distance_to_asker": round(handler._distance_to_asker(sample.scene, track), 6),  # noqa: SLF001
        "distance_to_trajectory": round(handler._distance_to_trajectory(sample.scene, track), 6),  # noqa: SLF001
        "support_count": len(track.provenance.source_agent_ids),
        "confidence": round(float(track.confidence), 6),
        "conflict_score": round(float(track.conflict_score), 6),
        "uncertainty_score": round(float(track.uncertainty_score), 6),
        "status": track.status.value,
        "age_frames": int(track.age_frames),
        "miss_count": int(track.miss_count),
        "source_agent_ids": list(track.provenance.source_agent_ids),
    }


def _sample_rows(sample, config: dict[str, object]) -> tuple[list[dict[str, object]], int, int]:
    state = _worker_state(config)
    prepared_scene = state.evaluator.prepare_sample(sample, baseline_mode=str(config["baseline_mode"]))
    prepared_sample = replace(sample, scene=prepared_scene)
    gt_coords = coordinates(reference_text(sample.raw_record))
    answer = state.handler.answer(prepared_sample)
    selected_ids = set(answer.object_ids)
    ranked = state.handler._ranked_role_scores(  # noqa: SLF001
        prepared_sample,
        role="hidden_relevant",
        max_results=int(config["shortlist_size"]),
        min_distance_to_asker=state.policy.min_distance_to_asker,
        max_distance_to_trajectory=state.policy.max_distance_to_trajectory,
    )
    ranked_by_id = {item.object_track.object_id: index + 1 for index, item in enumerate(ranked)}
    candidate_coords = [
        (item.object_track.position.x, item.object_track.position.y)
        for item in ranked
    ]
    selected_coords = [
        (item.object_track.position.x, item.object_track.position.y)
        for item in ranked
        if item.object_track.object_id in selected_ids
    ]

    rows: list[dict[str, object]] = []
    candidate_rows = 0
    unmatched_gt_rows = 0
    for rank, item in enumerate(ranked, start=1):
        track = item.object_track
        coord = (track.position.x, track.position.y)
        nearest_gt = nearest_coord_distance(coord, gt_coords)
        row = {
            "row_type": "candidate",
            "sample_id": sample.sample_id,
            "split": str(config["split"]),
            "rank": rank,
            "ranker": str(config["invisible_ranker"]),
            "selected_by_policy": track.object_id in selected_ids,
            "gt_positive_row": bool(gt_coords),
            "gt_count": len(gt_coords),
            "candidate_matches_gt": nearest_gt is not None and nearest_gt <= float(config["match_threshold"]),
            "nearest_gt_distance": None if nearest_gt is None else round(nearest_gt, 6),
            "role_score": round(float(item.score), 6),
            "visibility_state": item.visibility_state.value if item.visibility_state is not None else None,
            **track_feature_payload(prepared_sample, state.handler, track),
        }
        rows.append(row)
        candidate_rows += 1

    for gt_index, gt_coord in enumerate(gt_coords):
        selected_match_distance = nearest_coord_distance(gt_coord, selected_coords)
        if selected_match_distance is not None and selected_match_distance <= float(config["match_threshold"]):
            continue
        nearest_candidate_distance = nearest_coord_distance(gt_coord, candidate_coords)
        nearest_track, nearest_track_distance = nearest_track_to_coord(gt_coord, prepared_scene.object_tracks)
        row = {
            "row_type": "unmatched_gt",
            "sample_id": sample.sample_id,
            "split": str(config["split"]),
            "gt_index": gt_index,
            "gt_x": round(float(gt_coord[0]), 6),
            "gt_y": round(float(gt_coord[1]), 6),
            "ranker": str(config["invisible_ranker"]),
            "gt_count": len(gt_coords),
            "nearest_candidate_distance": (
                None if nearest_candidate_distance is None else round(nearest_candidate_distance, 6)
            ),
            "nearest_track_distance": None if nearest_track_distance is None else round(nearest_track_distance, 6),
            "nearest_track_rank": (
                None if nearest_track is None else ranked_by_id.get(nearest_track.object_id)
            ),
        }
        if nearest_track is not None:
            row.update(
                {
                    f"nearest_track_{key}": value
                    for key, value in track_feature_payload(prepared_sample, state.handler, nearest_track).items()
                }
            )
        rows.append(row)
        unmatched_gt_rows += 1
    return rows, candidate_rows, unmatched_gt_rows


def main() -> None:
    args = build_parser().parse_args()
    v2vgot_root = resolve_v2vgot_root(args.v2vgot_root)
    output_path = Path(args.output_jsonl).expanduser()
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = V2VGoTQABenchmarkAdapter(str(v2vgot_root))
    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name=args.split, file_name=args.file_name)
        if sample.task_type == BenchmarkTaskType.INVISIBLE_OBJECTS
    )
    if args.limit > 0:
        samples = samples[: args.limit]

    worker_config: dict[str, object] = {
        "v2vgot_root": str(v2vgot_root),
        "split": args.split,
        "baseline_mode": args.baseline_mode,
        "graph_ablation_mode": args.graph_ablation_mode,
        "invisible_ranker": args.invisible_ranker,
        "max_results": args.invisible_max_results,
        "shortlist_size": args.shortlist_size,
        "max_distance_to_trajectory": args.invisible_max_distance_to_trajectory,
        "min_risk": args.invisible_min_risk,
        "min_relative_to_best": args.invisible_min_relative_to_best,
        "match_threshold": args.match_threshold,
    }

    candidate_rows = 0
    unmatched_gt_rows = 0
    with output_path.open("w", encoding="utf-8") as handle:
        if args.workers > 1:
            chunksize = max(1, len(samples) // max(args.workers * 8, 1))
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                iterator = executor.map(
                    _sample_rows,
                    samples,
                    itertools.repeat(worker_config),
                    chunksize=chunksize,
                )
                for index, (rows, sample_candidate_rows, sample_unmatched_gt_rows) in enumerate(iterator, start=1):
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                    candidate_rows += sample_candidate_rows
                    unmatched_gt_rows += sample_unmatched_gt_rows
                    if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(samples)):
                        print(f"export_progress: {index}/{len(samples)} samples")
        else:
            for index, sample in enumerate(samples, start=1):
                rows, sample_candidate_rows, sample_unmatched_gt_rows = _sample_rows(sample, worker_config)
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
                candidate_rows += sample_candidate_rows
                unmatched_gt_rows += sample_unmatched_gt_rows
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(samples)):
                    print(f"export_progress: {index}/{len(samples)} samples")

    print("=" * 72)
    print("Phase 8 Invisible Candidate Feature Export")
    print("=" * 72)
    print(f"v2vgot_root: {v2vgot_root}")
    print(f"split: {args.split}")
    print(f"ranker: {args.invisible_ranker}")
    print(f"workers: {args.workers}")
    print(f"samples: {len(samples)}")
    print(f"candidate_rows: {candidate_rows}")
    print(f"unmatched_gt_rows: {unmatched_gt_rows}")
    print(f"saved_jsonl: {output_path}")


if __name__ == "__main__":
    main()
