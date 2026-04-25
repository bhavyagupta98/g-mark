#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator
from kg_coop_drive.application.planning_awareness import (
    PlanningAwarenessRanker,
    PlanningAwarenessSelectionPolicy,
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.application.v2vgotqa_router import (
    NotableObjectsHandler,
    OccludingObjectsHandler,
    PlanningAwarenessHandler,
    V2VGoTQARouter,
)
from kg_coop_drive.domain.benchmark import BenchmarkTaskType
from kg_coop_drive.domain.scene import TrackStatus, VisibilityState
from kg_coop_drive.infrastructure.local_llm_client import (
    LocalOpenAICompatibleLLMClient,
    LocalOpenAICompatibleLLMConfig,
)
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one Phase 5A V2V-GoT-QA sample in detail.")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--task-type",
        required=True,
        help="BenchmarkTaskType value for narrowing the lookup.",
    )
    parser.add_argument(
        "--file-name",
        default="v2v4real_3d_grounding_qa_dataset_v2vgot.json",
    )
    parser.add_argument(
        "--planning-ranker",
        default=PlanningAwarenessRanker.HEURISTIC.value,
        choices=tuple(item.value for item in PlanningAwarenessRanker),
        help="Planning-awareness ranker used for qa_type_id 14.",
    )
    parser.add_argument(
        "--planning-selection-policy",
        default=PlanningAwarenessSelectionPolicy.DEFAULT.value,
        choices=tuple(item.value for item in PlanningAwarenessSelectionPolicy),
        help="Final selection policy used for planning-awareness outputs.",
    )
    parser.add_argument(
        "--notable-ranker",
        default="heuristic",
        choices=("heuristic", "energy", "llm"),
        help="Notable-objects ranker. `energy` uses an interaction-energy style scorer; `llm` reranks a visible shortlist with the local model.",
    )
    parser.add_argument(
        "--occluding-ranker",
        default="heuristic",
        choices=("heuristic", "llm"),
        help="Occluding-objects ranker. `llm` reranks geometric blocker candidates with the local model.",
    )
    parser.add_argument("--llm-base-url", default=os.environ.get("KG_LOCAL_LLM_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.environ.get("KG_LOCAL_LLM_MODEL", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("KG_LOCAL_LLM_API_KEY", "local-token"))
    parser.add_argument("--llm-timeout-seconds", type=float, default=float(os.environ.get("KG_LOCAL_LLM_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.environ.get("KG_LOCAL_LLM_MAX_TOKENS", "192")))
    return parser


def build_optional_llm_client(args: argparse.Namespace) -> LocalOpenAICompatibleLLMClient | None:
    if (
        args.notable_ranker != "llm"
        and
        args.planning_ranker != PlanningAwarenessRanker.LLM.value
        and args.occluding_ranker != "llm"
    ):
        return None
    if not args.llm_base_url or not args.llm_model:
        raise SystemExit(
            "LLM-backed notable, planning, or occluding rankers require --llm-base-url and "
            "--llm-model (or KG_LOCAL_LLM_BASE_URL / KG_LOCAL_LLM_MODEL)."
        )
    return LocalOpenAICompatibleLLMClient(
        LocalOpenAICompatibleLLMConfig(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
            timeout_seconds=args.llm_timeout_seconds,
            max_tokens=args.llm_max_tokens,
        )
    )


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def filter_sample_visibility(scene, object_id: str, agent_id: str) -> tuple[str, ...]:
    states = []
    for fact in scene.visibility_facts:
        if fact.object_id == object_id and fact.agent_id == agent_id:
            states.append(fact.state.value)
    return tuple(states)


def main() -> None:
    args = build_parser().parse_args()
    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    llm_client = build_optional_llm_client(args)
    planning_orchestrator = build_planning_awareness_orchestrator(
        args.planning_ranker,
        llm_client=llm_client,
        selection_policy=args.planning_selection_policy,
    )
    evaluator = V2VGoTQAPhase5AEvaluator(
        str(repository_root),
        router=V2VGoTQARouter(
            handlers=(
                NotableObjectsHandler(
                    ranker=args.notable_ranker,
                    llm_client=llm_client if args.notable_ranker == "llm" else None,
                ),
                OccludingObjectsHandler(llm_client=llm_client if args.occluding_ranker == "llm" else None),
                PlanningAwarenessHandler(orchestrator=planning_orchestrator),
            )
        ),
    )
    task_type = BenchmarkTaskType(args.task_type)

    samples = adapter.load_samples(split_name=args.split, file_name=args.file_name)
    selected = None
    for sample in samples:
        if sample.sample_id == args.sample_id and sample.task_type == task_type:
            selected = sample
            break

    if selected is None:
        raise SystemExit(
            f"Could not find sample_id={args.sample_id} with task_type={task_type.value} in split={args.split}."
        )

    print_section("Sample Metadata")
    print(f"repository_root: {repository_root}")
    print(f"sample_id: {selected.sample_id}")
    print(f"task_type: {selected.task_type.value}")
    print(f"qa_type_id: {selected.qa_type_id}")
    print(f"scene_id: {selected.scene.scene_id}")
    print(f"global_timestamp_index: {selected.scene.global_timestamp_index}")
    print(f"asker_agent_id: {selected.scene.asker_agent_id}")
    print(f"question: {selected.scene.raw_question}")
    print(f"reference_answer: {selected.scene.raw_answer}")
    print(f"planning_ranker: {args.planning_ranker}")
    print(f"planning_selection_policy: {args.planning_selection_policy}")
    print(f"notable_ranker: {args.notable_ranker}")
    print(f"occluding_ranker: {args.occluding_ranker}")
    if (
        args.notable_ranker == "llm"
        or args.planning_ranker == PlanningAwarenessRanker.LLM.value
        or args.occluding_ranker == "llm"
    ):
        print(f"llm_base_url: {args.llm_base_url}")
        print(f"llm_model: {args.llm_model}")
        print(f"llm_timeout_seconds: {args.llm_timeout_seconds}")
        print(f"llm_max_tokens: {args.llm_max_tokens}")

    for baseline_mode in ("cooperative", "ego_only"):
        prepared_scene = evaluator.prepare_sample(selected, baseline_mode=baseline_mode)
        prepared_sample = replace(selected, scene=prepared_scene)
        prediction = evaluator._router.answer(prepared_sample)  # noqa: SLF001

        print_section(f"Mode: {baseline_mode}")
        print(f"prediction_supported: {prediction.supported}")
        print(f"prediction_object_ids: {list(prediction.object_ids)}")
        print(f"prediction_answer: {prediction.answer_text}")
        print(f"observation_count: {len(prepared_scene.observations)}")
        print(f"track_count: {len(prepared_scene.object_tracks)}")
        print(f"visibility_fact_count: {len(prepared_scene.visibility_facts)}")
        status_counts = {
            status.value: sum(1 for track in prepared_scene.object_tracks if track.status == status)
            for status in TrackStatus
        }
        print(f"track_status_counts: {status_counts}")

        if task_type == BenchmarkTaskType.PLANNING_AWARENESS:
            decision = planning_orchestrator.select(prepared_scene)
            print("Planning-Awareness Scores")
            if not decision.considered_candidates:
                print("- none")
            else:
                for candidate in decision.considered_candidates:
                    print(
                        f"- object_id={candidate.object_track.object_id}, "
                        f"score={candidate.score:.3f}, "
                        f"visibility={candidate.visibility_state.value}, "
                        f"distance_to_trajectory={candidate.distance_to_trajectory:.2f}, "
                        f"status={candidate.object_track.status.value}, "
                        f"rationale={list(candidate.rationale)}"
                    )

        print("Selected Object Details")
        selected_tracks = {
            track.object_id: track
            for track in prepared_scene.object_tracks
            if track.object_id in prediction.object_ids
        }
        if not selected_tracks:
            print("- none")
        else:
            for object_id in prediction.object_ids:
                track = selected_tracks.get(object_id)
                if track is None:
                    print(f"- {object_id}: not found in prepared scene")
                    continue
                visibility_states = filter_sample_visibility(
                    prepared_scene,
                    object_id=object_id,
                    agent_id=prepared_scene.asker_agent_id,
                )
                print(
                    f"- object_id={track.object_id}, status={track.status.value}, type={track.object_type}, "
                    f"position=({track.position.x:.2f}, {track.position.y:.2f}), "
                    f"confidence={track.confidence:.2f}, support_count={len(track.observations)}, "
                    f"uncertainty={track.uncertainty_score:.2f}, conflict={track.conflict_score:.2f}, "
                    f"provenance={list(track.provenance.source_agent_ids)}, "
                    f"visibility={list(visibility_states)}"
                )

        print("Top Candidate Tracks")
        candidate_tracks = [
            track for track in prepared_scene.object_tracks if track.status == TrackStatus.CANDIDATE
        ]
        candidate_tracks.sort(key=lambda track: (-track.confidence, track.object_id))
        if not candidate_tracks:
            print("- none")
        else:
            for track in candidate_tracks[:8]:
                visibility_states = filter_sample_visibility(
                    prepared_scene,
                    object_id=track.object_id,
                    agent_id=prepared_scene.asker_agent_id,
                )
                print(
                    f"- object_id={track.object_id}, position=({track.position.x:.2f}, {track.position.y:.2f}), "
                    f"confidence={track.confidence:.2f}, support_count={len(track.observations)}, "
                    f"provenance={list(track.provenance.source_agent_ids)}, visibility={list(visibility_states)}"
                )

        print("Visible / Occluded Summary For Asker")
        visible_ids = sorted(
            {
                fact.object_id
                for fact in prepared_scene.visibility_facts
                if fact.agent_id == prepared_scene.asker_agent_id and fact.state == VisibilityState.VISIBLE
            }
        )
        occluded_ids = sorted(
            {
                fact.object_id
                for fact in prepared_scene.visibility_facts
                if fact.agent_id == prepared_scene.asker_agent_id and fact.state == VisibilityState.OCCLUDED
            }
        )
        print(f"- visible_ids: {visible_ids[:20]}")
        print(f"- occluded_ids: {occluded_ids[:20]}")


if __name__ == "__main__":
    main()
