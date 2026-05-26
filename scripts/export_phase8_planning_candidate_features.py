#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.qa.planning_awareness import (  # noqa: E402
    PLANNING_LOGREG_FEATURE_NAMES,
    planning_logreg_feature_values,
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.application.qa.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.qa.v2vgotqa_evaluator import GraphAblationMode  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

COORDINATE_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")
DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export Q4 planning-awareness candidate features for train-frozen "
            "logistic acceptance. Labels are created only from the selected split's "
            "reference coordinates and are never used at inference."
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
    parser.add_argument("--planning-ranker", default="relational_importance")
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for the full split.")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    return parser


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


def main() -> None:
    args = build_parser().parse_args()
    v2vgot_root = resolve_v2vgot_root(args.v2vgot_root)
    output_path = Path(args.output_jsonl).expanduser()
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = V2VGoTQABenchmarkAdapter(str(v2vgot_root))
    evaluator = V2VGoTQAPhase5AEvaluator(
        str(v2vgot_root),
        graph_ablation=args.graph_ablation_mode,
    )
    orchestrator = build_planning_awareness_orchestrator(
        ranker=args.planning_ranker,
        selection_policy="default",
    )
    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name=args.split, file_name=args.file_name)
        if sample.task_type == BenchmarkTaskType.PLANNING_AWARENESS
    )
    if args.limit > 0:
        samples = samples[: args.limit]

    candidate_rows = 0
    positive_rows = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            prepared_scene = evaluator.prepare_sample(sample, baseline_mode=args.baseline_mode)
            prepared_sample = replace(sample, scene=prepared_scene)
            gt_coords = coordinates(reference_text(sample.raw_record))
            decision = orchestrator.select(prepared_sample.scene)
            ordered = decision.considered_candidates
            for rank, candidate in enumerate(ordered, start=1):
                coord = (candidate.object_track.position.x, candidate.object_track.position.y)
                nearest_gt = nearest_coord_distance(coord, gt_coords)
                matched = nearest_gt is not None and nearest_gt <= args.match_threshold
                row = {
                    "row_type": "candidate",
                    "sample_id": sample.sample_id,
                    "split": args.split,
                    "ranker": args.planning_ranker,
                    "rank": rank,
                    "object_id": candidate.object_track.object_id,
                    "object_type": candidate.object_track.object_type,
                    "x": round(float(coord[0]), 6),
                    "y": round(float(coord[1]), 6),
                    "gt_coords": [[round(x, 6), round(y, 6)] for x, y in gt_coords],
                    "gt_count": len(gt_coords),
                    "candidate_matches_gt": matched,
                    "nearest_gt_distance": None if nearest_gt is None else round(nearest_gt, 6),
                    "visibility_state": candidate.visibility_state.value,
                    "status": candidate.object_track.status.value,
                    "source_agent_ids": list(candidate.object_track.provenance.source_agent_ids),
                }
                row.update(
                    planning_logreg_feature_values(
                        prepared_sample.scene,
                        candidate,
                        rank=rank,
                        ordered_candidates=ordered,
                        feature_names=PLANNING_LOGREG_FEATURE_NAMES,
                    )
                )
                handle.write(json.dumps(row) + "\n")
                candidate_rows += 1
                if matched:
                    positive_rows += 1
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(samples)):
                print(f"export_progress: {index}/{len(samples)} samples")

    print("=" * 72)
    print("Phase 8 Q4 Planning Candidate Feature Export")
    print("=" * 72)
    print(f"v2vgot_root: {v2vgot_root}")
    print(f"split: {args.split}")
    print(f"ranker: {args.planning_ranker}")
    print(f"samples: {len(samples)}")
    print(f"candidate_rows: {candidate_rows}")
    print(f"positive_candidate_rows: {positive_rows}")
    print(f"saved_jsonl: {output_path}")


if __name__ == "__main__":
    main()
