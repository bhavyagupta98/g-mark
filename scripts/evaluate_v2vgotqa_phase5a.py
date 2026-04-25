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
    parser = argparse.ArgumentParser(description="Evaluate Phase 5A V2V-GoT-QA tasks.")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        default=[],
        help="Optional BenchmarkTaskType value to evaluate. Repeatable.",
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
    parser.add_argument("--output-jsonl", default="")
    return parser


def build_optional_llm_client(args: argparse.Namespace) -> LocalOpenAICompatibleLLMClient | None:
    if (
        args.notable_ranker != "llm"
        and args.planning_ranker != PlanningAwarenessRanker.LLM.value
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


def parse_task_types(raw_task_types: list[str]) -> tuple[BenchmarkTaskType, ...]:
    if not raw_task_types:
        return ()
    return tuple(BenchmarkTaskType(value) for value in raw_task_types)


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
    router = V2VGoTQARouter(
        handlers=(
            NotableObjectsHandler(
                ranker=args.notable_ranker,
                llm_client=llm_client if args.notable_ranker == "llm" else None,
            ),
            OccludingObjectsHandler(llm_client=llm_client if args.occluding_ranker == "llm" else None),
            PlanningAwarenessHandler(orchestrator=planning_orchestrator),
        )
    )
    evaluator = V2VGoTQAPhase5AEvaluator(str(repository_root), router=router)

    samples = adapter.load_samples(split_name=args.split, file_name=args.file_name)
    selected_task_types = parse_task_types(args.task_types)
    if selected_task_types:
        samples = tuple(sample for sample in samples if sample.task_type in selected_task_types)
    samples = samples[: args.limit]
    predictions = evaluator.evaluate_samples(samples, baseline_mode=args.baseline_mode)
    summary = evaluator.summarize(predictions)

    print("=" * 72)
    print("Phase 5A V2V-GoT-QA Evaluation")
    print("=" * 72)
    print(f"repository_root: {repository_root}")
    print(f"split: {args.split}")
    print(f"baseline_mode: {args.baseline_mode}")
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
    print(f"sample_count: {summary.total_samples}")
    print(f"supported_predictions: {summary.supported_predictions}")
    print(f"unsupported_predictions: {summary.unsupported_predictions}")
    if selected_task_types:
        print(f"task_types: {[item.value for item in selected_task_types]}")
    else:
        print(f"task_types: {[item.value for item in router.supported_task_types()]}")

    print()
    print("Predictions")
    print("-" * 72)
    for prediction in predictions[:10]:
        print(
            f"[{prediction.task_type.value}] sample_id={prediction.sample_id} "
            f"supported={prediction.supported} objects={list(prediction.object_ids)}"
        )
        print(f"answer: {prediction.answer_text}")
        print()

    if args.output_jsonl:
        output_path = Path(args.output_jsonl).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": prediction.sample_id,
                            "dataset_name": prediction.dataset_name,
                            "split_name": prediction.split_name,
                            "task_type": prediction.task_type.value,
                            "qa_type_id": prediction.qa_type_id,
                            "supported": prediction.supported,
                            "answer_text": prediction.answer_text,
                            "object_ids": list(prediction.object_ids),
                            "baseline_mode": prediction.baseline_mode,
                            "planning_ranker": args.planning_ranker,
                            "planning_selection_policy": args.planning_selection_policy,
                            "notable_ranker": args.notable_ranker,
                            "occluding_ranker": args.occluding_ranker,
                            "llm_base_url": args.llm_base_url if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else "",
                            "llm_model": args.llm_model if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else "",
                            "llm_timeout_seconds": args.llm_timeout_seconds if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else 0,
                            "llm_max_tokens": args.llm_max_tokens if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else 0,
                        }
                    )
                    + "\n"
                )
        print(f"saved_predictions: {output_path}")


if __name__ == "__main__":
    main()
