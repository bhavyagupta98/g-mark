#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.v2vgotqa_router import OccludingObjectsHandler  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect ranked occluding candidates and score features for selected samples."
    )
    parser.add_argument("--sample-id", action="append", dest="sample_ids", default=[])
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument(
        "--ranker",
        default="risk_adaptive",
        choices=("heuristic", "top3_open", "top3_far_supported", "top3_hybrid", "risk_adaptive"),
    )
    parser.add_argument("--output-json", default="")
    return parser


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def main() -> None:
    args = build_parser().parse_args()
    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    evaluator = V2VGoTQAPhase5AEvaluator(str(repository_root))
    handler = OccludingObjectsHandler(ranker=args.ranker)

    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name=args.split, file_name=args.file_name)
        if sample.task_type == BenchmarkTaskType.OCCLUDING_OBJECTS
    )[: args.limit]
    wanted_ids = set(args.sample_ids)
    if wanted_ids:
        samples = tuple(sample for sample in samples if sample.sample_id in wanted_ids)

    report_rows: list[dict[str, object]] = []
    for sample in samples:
        prepared_scene = evaluator.prepare_sample(sample, baseline_mode="cooperative")
        prepared_sample = type(sample)(
            sample_id=sample.sample_id,
            dataset_name=sample.dataset_name,
            split_name=sample.split_name,
            file_name=sample.file_name,
            task_type=sample.task_type,
            scene=prepared_scene,
            raw_record=sample.raw_record,
            qa_type_id=sample.qa_type_id,
        )
        scores = handler._ranked_role_scores(  # noqa: SLF001
            prepared_sample,
            role="blocker",
            max_results=args.max_candidates,
        )
        selected_heuristic = handler._top_occluding_objects(prepared_sample).objects  # noqa: SLF001
        selected_open = handler._top_occluding_objects_open_top3(prepared_sample).objects  # noqa: SLF001
        selected_active = handler.answer(prepared_sample).object_ids
        row = {
            "sample_id": sample.sample_id,
            "question": sample.scene.raw_question,
            "reference_answer": sample.scene.raw_answer,
            "ranker": args.ranker,
            "selected_active": list(selected_active),
            "selected_heuristic": [item.object_id for item in selected_heuristic],
            "selected_top3_open": [item.object_id for item in selected_open],
            "candidates": [
                {
                    "rank": index + 1,
                    "object_id": score.object_track.object_id,
                    "object_type": score.object_track.object_type,
                    "status": score.object_track.status.value,
                    "score": round(score.score, 6),
                    "distance_to_trajectory": round(score.distance_to_trajectory, 6),
                    "distance_to_asker": round(score.distance_to_asker, 6),
                    "support_count": score.support_count,
                    "best_alignment_radians": round(score.best_alignment_radians, 6),
                    "aligned_hidden_object_ids": list(score.aligned_hidden_object_ids[:5]),
                    "aligned_hidden_distances_to_trajectory": [
                        round(value, 6)
                        for value in score.aligned_hidden_distances_to_trajectory[:5]
                    ],
                }
                for index, score in enumerate(scores)
            ],
        }
        report_rows.append(row)

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")
        print(f"saved_json: {output_path}")

    for row in report_rows:
        print("=" * 72)
        print(f"sample_id: {row['sample_id']}")
        print(f"selected_active[{row['ranker']}]: {row['selected_active']}")
        print(f"selected_heuristic: {row['selected_heuristic']}")
        print(f"selected_top3_open: {row['selected_top3_open']}")
        print(f"reference: {row['reference_answer']}")
        for candidate in row["candidates"]:
            print(
                "  "
                f"rank={candidate['rank']} id={candidate['object_id']} "
                f"score={candidate['score']} dtraj={candidate['distance_to_trajectory']} "
                f"dasker={candidate['distance_to_asker']} align={candidate['best_alignment_radians']} "
                f"status={candidate['status']} support={candidate['support_count']}"
            )


if __name__ == "__main__":
    main()
