#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.v2vgotqa_evaluator import GraphAblationMode, V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.v2vgotqa_router import V2VGoTQARouter  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.scene import CooperativeScene, VisibilityState  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402


@dataclass(frozen=True)
class SceneStats:
    object_count: int
    relation_count: int
    visibility_fact_count: int
    visible_to_asker_count: int
    occluded_to_asker_count: int
    uncertain_to_asker_count: int
    candidate_count: int
    supported_count: int
    confirmed_count: int
    observation_count: int
    avg_support_count: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect whether cooperative and ego-only G-MARK prepared scenes actually differ "
            "before final QA scoring."
        )
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit-per-task", type=int, default=100)
    parser.add_argument("--qa-type-id", action="append", type=int, default=[])
    parser.add_argument("--task-type", action="append", default=[])
    parser.add_argument("--include-answers", action="store_true")
    parser.add_argument(
        "--graph-ablation-mode",
        action="append",
        choices=tuple(mode.value for mode in GraphAblationMode),
        default=[],
        help=(
            "Optionally inspect cooperative graph construction under one or more ablation modes. "
            "This is additive to the default cooperative-vs-ego-only comparison."
        ),
    )
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--output-json", default="")
    return parser


def scene_stats(scene: CooperativeScene) -> SceneStats:
    visibility_for_asker = [
        fact for fact in scene.visibility_facts if fact.agent_id == scene.asker_agent_id
    ]
    status_counts = Counter(track.status.value for track in scene.object_tracks)
    support_counts = [len(track.provenance.source_agent_ids) for track in scene.object_tracks]
    avg_support_count = sum(support_counts) / len(support_counts) if support_counts else 0.0
    return SceneStats(
        object_count=len(scene.object_tracks),
        relation_count=len(scene.relations),
        visibility_fact_count=len(scene.visibility_facts),
        visible_to_asker_count=sum(
            1 for fact in visibility_for_asker if fact.state == VisibilityState.VISIBLE
        ),
        occluded_to_asker_count=sum(
            1 for fact in visibility_for_asker if fact.state == VisibilityState.OCCLUDED
        ),
        uncertain_to_asker_count=sum(
            1 for fact in visibility_for_asker if fact.state == VisibilityState.UNCERTAIN
        ),
        candidate_count=status_counts["candidate"],
        supported_count=status_counts["supported"],
        confirmed_count=status_counts["confirmed"],
        observation_count=len(scene.observations),
        avg_support_count=avg_support_count,
    )


def task_key(sample: BenchmarkSample) -> str:
    return f"qa{sample.qa_type_id}_{sample.task_type.value}"


def task_allowed(sample: BenchmarkSample, task_types: set[str], qa_type_ids: set[int]) -> bool:
    if task_types and sample.task_type.value not in task_types:
        return False
    if qa_type_ids and int(sample.qa_type_id or -1) not in qa_type_ids:
        return False
    return True


def answer_payload(router: V2VGoTQARouter, sample: BenchmarkSample, scene: CooperativeScene) -> dict[str, object]:
    prepared = BenchmarkSample(
        sample_id=sample.sample_id,
        dataset_name=sample.dataset_name,
        split_name=sample.split_name,
        file_name=sample.file_name,
        task_type=sample.task_type,
        scene=scene,
        raw_record=sample.raw_record,
        qa_type_id=sample.qa_type_id,
    )
    answer = router.answer(prepared)
    return {
        "answer_text": answer.answer_text,
        "object_ids": list(answer.object_ids),
        "supported": answer.supported,
    }


def inspect_sample(
    *,
    evaluator: V2VGoTQAPhase5AEvaluator,
    graph_ablation_evaluators: dict[str, V2VGoTQAPhase5AEvaluator],
    router: V2VGoTQARouter,
    sample: BenchmarkSample,
    include_answers: bool,
) -> dict[str, object]:
    cooperative_scene = evaluator.prepare_sample(sample, baseline_mode="cooperative")
    ego_only_scene = evaluator.prepare_sample(sample, baseline_mode="ego_only")
    coop_ids = {track.object_id for track in cooperative_scene.object_tracks}
    ego_ids = {track.object_id for track in ego_only_scene.object_tracks}
    payload: dict[str, object] = {
        "sample_id": sample.sample_id,
        "qa_type_id": sample.qa_type_id,
        "task_type": sample.task_type.value,
        "timestamp": sample.scene.global_timestamp_index,
        "cooperative": asdict(scene_stats(cooperative_scene)),
        "ego_only": asdict(scene_stats(ego_only_scene)),
        "coop_only_object_count": len(coop_ids - ego_ids),
        "ego_only_object_count": len(ego_ids - coop_ids),
        "shared_object_count": len(coop_ids & ego_ids),
        "coop_only_object_ids_sample": sorted(coop_ids - ego_ids)[:10],
        "ego_only_object_ids_sample": sorted(ego_ids - coop_ids)[:10],
    }
    if graph_ablation_evaluators:
        mode_payloads: dict[str, object] = {}
        for mode, mode_evaluator in graph_ablation_evaluators.items():
            mode_scene = mode_evaluator.prepare_sample(sample, baseline_mode="cooperative")
            mode_ids = {track.object_id for track in mode_scene.object_tracks}
            mode_payload: dict[str, object] = {
                "stats": asdict(scene_stats(mode_scene)),
                "missing_full_object_count": len(coop_ids - mode_ids),
                "extra_object_count": len(mode_ids - coop_ids),
                "object_ids_sample": sorted(mode_ids)[:10],
                "missing_full_object_ids_sample": sorted(coop_ids - mode_ids)[:10],
            }
            if include_answers:
                mode_answer = answer_payload(router, sample, mode_scene)
                mode_payload.update(
                    {
                        "answer": mode_answer["answer_text"],
                        "answer_object_ids": mode_answer["object_ids"],
                        "answer_equal_to_full": mode_answer["answer_text"]
                        == answer_payload(router, sample, cooperative_scene)["answer_text"],
                    }
                )
            mode_payloads[mode] = mode_payload
        payload["graph_ablation_modes"] = mode_payloads
    if include_answers:
        cooperative_answer = answer_payload(router, sample, cooperative_scene)
        ego_only_answer = answer_payload(router, sample, ego_only_scene)
        cooperative_object_ids = set(str(value) for value in cooperative_answer["object_ids"])  # type: ignore[index]
        payload.update(
            {
                "cooperative_answer": cooperative_answer["answer_text"],
                "cooperative_answer_object_ids": cooperative_answer["object_ids"],
                "cooperative_answer_uses_coop_only_object": bool(cooperative_object_ids & (coop_ids - ego_ids)),
                "ego_only_answer": ego_only_answer["answer_text"],
                "ego_only_answer_object_ids": ego_only_answer["object_ids"],
                "answers_equal": cooperative_answer["answer_text"] == ego_only_answer["answer_text"],
            }
        )
    return payload


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = build_parser().parse_args()
    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    evaluator = V2VGoTQAPhase5AEvaluator(args.v2vgot_root)
    graph_ablation_evaluators = {
        mode: V2VGoTQAPhase5AEvaluator(args.v2vgot_root, graph_ablation=mode)
        for mode in dict.fromkeys(args.graph_ablation_mode)
    }
    router = V2VGoTQARouter()
    task_types = set(args.task_type)
    qa_type_ids = set(args.qa_type_id)
    samples = [
        sample
        for sample in adapter.load_samples(split_name=args.split)
        if task_allowed(sample, task_types, qa_type_ids)
        and sample.task_type != BenchmarkTaskType.UNKNOWN
    ]

    inspected_by_task: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        key = task_key(sample)
        if len(inspected_by_task[key]) >= args.limit_per_task:
            continue
        inspected_by_task[key].append(
            inspect_sample(
                evaluator=evaluator,
                graph_ablation_evaluators=graph_ablation_evaluators,
                router=router,
                sample=sample,
                include_answers=args.include_answers,
            )
        )

    summary: dict[str, object] = {
        "v2vgot_root": str(Path(args.v2vgot_root).expanduser()),
        "split": args.split,
        "limit_per_task": args.limit_per_task,
        "tasks": {},
    }
    for key, rows in sorted(inspected_by_task.items()):
        coop_objects = [float(row["cooperative"]["object_count"]) for row in rows]  # type: ignore[index]
        ego_objects = [float(row["ego_only"]["object_count"]) for row in rows]  # type: ignore[index]
        coop_occluded = [float(row["cooperative"]["occluded_to_asker_count"]) for row in rows]  # type: ignore[index]
        ego_occluded = [float(row["ego_only"]["occluded_to_asker_count"]) for row in rows]  # type: ignore[index]
        coop_only = [float(row["coop_only_object_count"]) for row in rows]
        answer_diffs = [
            row
            for row in rows
            if bool(row.get("answers_equal")) is False
        ]
        coop_answer_uses_coop_only = [
            row
            for row in rows
            if bool(row.get("cooperative_answer_uses_coop_only_object")) is True
        ]
        task_summary = {
            "samples": len(rows),
            "avg_cooperative_objects": average(coop_objects),
            "avg_ego_only_objects": average(ego_objects),
            "avg_coop_minus_ego_objects": average(
                [left - right for left, right in zip(coop_objects, ego_objects)]
            ),
            "avg_coop_only_objects": average(coop_only),
            "avg_cooperative_occluded_to_asker": average(coop_occluded),
            "avg_ego_only_occluded_to_asker": average(ego_occluded),
            "empty_cooperative_scene_count": sum(1 for value in coop_objects if value == 0),
            "empty_ego_only_scene_count": sum(1 for value in ego_objects if value == 0),
            "frames_with_coop_only_objects": sum(1 for value in coop_only if value > 0),
            "answer_difference_count": len(answer_diffs) if args.include_answers else None,
            "cooperative_answer_uses_coop_only_count": (
                len(coop_answer_uses_coop_only) if args.include_answers else None
            ),
            "examples": rows[: args.examples],
        }
        if graph_ablation_evaluators:
            mode_summaries = {}
            for mode in graph_ablation_evaluators:
                mode_stats = [
                    row["graph_ablation_modes"][mode]["stats"]  # type: ignore[index]
                    for row in rows
                ]
                missing_full_objects = [
                    float(row["graph_ablation_modes"][mode]["missing_full_object_count"])  # type: ignore[index]
                    for row in rows
                ]
                answer_diffs_to_full = [
                    row
                    for row in rows
                    if bool(
                        row["graph_ablation_modes"][mode].get("answer_equal_to_full", True)  # type: ignore[index]
                    )
                    is False
                ]
                mode_summaries[mode] = {
                    "avg_objects": average(
                        [float(stats["object_count"]) for stats in mode_stats]  # type: ignore[index]
                    ),
                    "avg_relations": average(
                        [float(stats["relation_count"]) for stats in mode_stats]  # type: ignore[index]
                    ),
                    "avg_candidates": average(
                        [float(stats["candidate_count"]) for stats in mode_stats]  # type: ignore[index]
                    ),
                    "avg_missing_full_objects": average(missing_full_objects),
                    "answer_difference_from_full_count": (
                        len(answer_diffs_to_full) if args.include_answers else None
                    ),
                }
            task_summary["graph_ablation_mode_summaries"] = mode_summaries
        summary["tasks"][key] = task_summary  # type: ignore[index]

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_json: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
