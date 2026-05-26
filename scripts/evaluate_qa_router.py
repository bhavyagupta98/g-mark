#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.qa.v2vgotqa_evaluator import (
    GraphAblationMode,
    TemporalExecutionMode,
    V2VGoTQAPhase5AEvaluator,
)
from kg_coop_drive.application.qa.latency_profiling import SampleLatencyRecord, EvaluationLatencyCollector
from kg_coop_drive.application.qa.planning_awareness import (
    PlanningAwarenessRanker,
    PlanningAwarenessSelectionPolicy,
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.application.qa.v2vgotqa_router import (
    AgentMotionPredictionHandler,
    ControlSettingsHandler,
    FutureTrajectoryHandler,
    InvisibleObjectsHandler,
    InvisibleSelectionPolicy,
    NotableObjectsHandler,
    ObjectMotionPredictionHandler,
    OccludingObjectsHandler,
    PlanningAwarenessHandler,
    V2VGoTQARouter,
)
from kg_coop_drive.application.planning.future_trajectory_planner import (
    ControlConditionedFutureTrajectoryPlanner,
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
    parser = argparse.ArgumentParser(description="Evaluate V2V-GoT-QA tasks with the local KG router.")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=25, help="Maximum samples to evaluate. Use 0 for the full split.")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument(
        "--temporal-execution-mode",
        default=TemporalExecutionMode.SERIAL.value,
        choices=tuple(item.value for item in TemporalExecutionMode),
        help="Temporal execution mode. Default keeps existing serial behavior.",
    )
    parser.add_argument(
        "--graph-ablation-mode",
        default=GraphAblationMode.FULL.value,
        choices=tuple(item.value for item in GraphAblationMode),
        help="Optional G-MARK graph ablation. Defaults to full graph behavior.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        default=[],
        help="Optional BenchmarkTaskType value to evaluate. Repeatable.",
    )
    parser.add_argument(
        "--qa-type-id",
        action="append",
        dest="qa_type_ids",
        type=int,
        default=[],
        help=(
            "Optional raw V2V-GoT qa_type_id filter. Repeatable. "
            "Useful for splitting Q5 and Q7, which share object_motion_prediction."
        ),
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
        "--planning-selection-source",
        default="composition",
        choices=("composition", "orchestrator"),
        help=(
            "Q4 selection source. `composition` preserves the current benchmark checkpoint "
            "path; `orchestrator` evaluates the pluggable planning-awareness ranker/policy."
        ),
    )
    parser.add_argument(
        "--planning-acceptor-model-json",
        default="",
        help="Frozen Q4 planning-awareness acceptor model for acceptor-based planning policies.",
    )
    parser.add_argument(
        "--future-trajectory-model-json",
        default="",
        help="Frozen Q9 future-trajectory control-delta model.",
    )
    parser.add_argument(
        "--object-motion-model-json",
        default="",
        help="Frozen Q5 object-motion endpoint model.",
    )
    parser.add_argument(
        "--agent-motion-model-json",
        default="",
        help="Frozen Q6 agent-motion notability model.",
    )
    parser.add_argument(
        "--control-selection-policy",
        default="rule",
        choices=("rule", "linear_classifier"),
        help="Q8 control-settings policy.",
    )
    parser.add_argument(
        "--control-model-json",
        default="",
        help="Frozen Q8 control-settings model JSON.",
    )
    parser.add_argument(
        "--notable-ranker",
        default="heuristic",
        choices=("heuristic", "energy", "llm"),
        help="Notable-objects ranker. `energy` uses an interaction-energy style scorer; `llm` reranks a visible shortlist with the local model.",
    )
    parser.add_argument(
        "--occluding-ranker",
        default="risk_adaptive",
        choices=("heuristic", "top3_open", "top3_far_supported", "top3_hybrid", "risk_adaptive", "llm"),
        help="Occluding-objects ranker. `llm` reranks geometric blocker candidates with the local model.",
    )
    parser.add_argument(
        "--invisible-ranker",
        default="legacy",
        choices=(
            "legacy",
            "risk_adaptive",
            "road_region",
            "road_region_strict",
            "temporal_guard",
            "backtrack_guard",
            "logreg_acceptor",
            "mlp_acceptor",
            "logreg_legacy_fallback",
            "logreg_lateral_rescue",
        ),
        help=(
            "Invisible-objects ranker. `legacy` keeps the broad Phase 6 selector; "
            "`risk_adaptive` applies generic precision gates; `road_region` adds "
            "generic lateral-road-region scoring; `road_region_strict` suppresses "
            "far centerline clutter more aggressively; `temporal_guard` suppresses "
            "repeated far-behind centerline clutter on top of legacy ranking; "
            "`backtrack_guard` suppresses behind-centerline candidates that sit very near the trajectory; "
            "`logreg_acceptor` loads a train-calibrated logistic model; "
            "`mlp_acceptor` loads a compact train-calibrated MLP model; "
            "`logreg_legacy_fallback` uses the calibrated model when accepted, otherwise falls back to legacy; "
            "`logreg_lateral_rescue` rescues train-mined ahead/lateral supported candidates when logreg is empty."
        ),
    )
    parser.add_argument("--invisible-acceptor-model-json", default="")
    parser.add_argument("--invisible-max-results", type=int, default=1)
    parser.add_argument("--invisible-shortlist-size", type=int, default=6)
    parser.add_argument("--invisible-max-distance-to-trajectory", type=float, default=5.0)
    parser.add_argument("--invisible-min-risk", type=float, default=0.58)
    parser.add_argument("--invisible-min-relative-to-best", type=float, default=0.75)
    parser.add_argument("--llm-base-url", default=os.environ.get("KG_LOCAL_LLM_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.environ.get("KG_LOCAL_LLM_MODEL", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("KG_LOCAL_LLM_API_KEY", "local-token"))
    parser.add_argument("--llm-timeout-seconds", type=float, default=float(os.environ.get("KG_LOCAL_LLM_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.environ.get("KG_LOCAL_LLM_MAX_TOKENS", "192")))
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument(
        "--latency-jsonl",
        default="",
        help="Optional per-sample latency breakdown output path.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
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


def load_invisible_acceptor_model(path_value: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def load_planning_acceptor_model(path_value: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def load_future_trajectory_model(path_value: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def load_object_motion_model(path_value: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def load_control_model(path_value: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def load_agent_motion_model(path_value: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def parse_task_types(raw_task_types: list[str]) -> tuple[BenchmarkTaskType, ...]:
    if not raw_task_types:
        return ()
    return tuple(BenchmarkTaskType(value) for value in raw_task_types)


def apply_sample_limit(samples: tuple[object, ...], limit: int) -> tuple[object, ...]:
    if limit <= 0:
        return samples
    return samples[:limit]


def router_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "planning_ranker": args.planning_ranker,
        "planning_selection_policy": args.planning_selection_policy,
        "planning_selection_source": args.planning_selection_source,
        "planning_acceptor_model": load_planning_acceptor_model(args.planning_acceptor_model_json),
        "future_trajectory_model": load_future_trajectory_model(args.future_trajectory_model_json),
        "object_motion_model": load_object_motion_model(args.object_motion_model_json),
        "agent_motion_model": load_agent_motion_model(args.agent_motion_model_json),
        "control_selection_policy": args.control_selection_policy,
        "control_model": load_control_model(args.control_model_json),
        "notable_ranker": args.notable_ranker,
        "occluding_ranker": args.occluding_ranker,
        "invisible_ranker": args.invisible_ranker,
        "invisible_max_results": args.invisible_max_results,
        "invisible_shortlist_size": args.invisible_shortlist_size,
        "invisible_max_distance_to_trajectory": args.invisible_max_distance_to_trajectory,
        "invisible_min_risk": args.invisible_min_risk,
        "invisible_min_relative_to_best": args.invisible_min_relative_to_best,
        "invisible_acceptor_model": load_invisible_acceptor_model(args.invisible_acceptor_model_json),
    }


def build_router(config: dict[str, Any], llm_client: LocalOpenAICompatibleLLMClient | None = None) -> V2VGoTQARouter:
    planning_orchestrator = build_planning_awareness_orchestrator(
        str(config["planning_ranker"]),
        llm_client=llm_client,
        selection_policy=str(config["planning_selection_policy"]),
        acceptor_model=dict(config["planning_acceptor_model"]),
    )
    return V2VGoTQARouter(
        handlers=(
            NotableObjectsHandler(
                ranker=str(config["notable_ranker"]),
                llm_client=llm_client if config["notable_ranker"] == "llm" else None,
            ),
            OccludingObjectsHandler(
                ranker=str(config["occluding_ranker"]),
                llm_client=llm_client if config["occluding_ranker"] == "llm" else None,
            ),
            InvisibleObjectsHandler(
                ranker=str(config["invisible_ranker"]),
                selection_policy=InvisibleSelectionPolicy(
                    max_results=int(config["invisible_max_results"]),
                    shortlist_size=int(config["invisible_shortlist_size"]),
                    max_distance_to_trajectory=float(config["invisible_max_distance_to_trajectory"]),
                    min_risk=float(config["invisible_min_risk"]),
                    min_relative_to_best=float(config["invisible_min_relative_to_best"]),
                ),
                acceptor_model=dict(config["invisible_acceptor_model"]),
            ),
            PlanningAwarenessHandler(
                orchestrator=planning_orchestrator,
                selection_source=str(config["planning_selection_source"]),
            ),
            ObjectMotionPredictionHandler(
                model=dict(config["object_motion_model"]),
            ),
            AgentMotionPredictionHandler(
                model=dict(config["agent_motion_model"]),
            ),
            FutureTrajectoryHandler(
                planner=ControlConditionedFutureTrajectoryPlanner(
                    model=dict(config["future_trajectory_model"])
                ),
            ),
            ControlSettingsHandler(
                selection_policy=str(config["control_selection_policy"]),
                model=dict(config["control_model"]),
            ),
        )
    )


def chunk_samples(samples: tuple[object, ...], workers: int) -> list[tuple[object, ...]]:
    if workers <= 1 or len(samples) <= 1:
        return [samples]
    chunk_count = min(workers, len(samples))
    chunk_size = (len(samples) + chunk_count - 1) // chunk_count
    return [
        samples[start : start + chunk_size]
        for start in range(0, len(samples), chunk_size)
    ]


def evaluate_chunk_worker(payload: tuple[int, tuple[object, ...], str, str, str, str, dict[str, Any], int, bool]):
    (
        chunk_index,
        samples,
        repository_root,
        baseline_mode,
        graph_ablation_mode,
        temporal_execution_mode,
        router_config,
        progress_every,
        profile_latency,
    ) = payload
    router = build_router(router_config, llm_client=None)
    latency_collector = EvaluationLatencyCollector() if profile_latency else None
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root,
        router=router,
        graph_ablation=graph_ablation_mode,
        temporal_execution_mode=temporal_execution_mode,
        latency_collector=latency_collector,
    )
    predictions = evaluator.evaluate_samples(
        samples,
        baseline_mode=baseline_mode,
        progress_every=progress_every,
    )
    latency_records = latency_collector.records() if latency_collector is not None else tuple()
    return chunk_index, predictions, latency_records


def evaluate_samples_parallel(
    *,
    repository_root: Path,
    samples: tuple[object, ...],
    baseline_mode: str,
    graph_ablation_mode: str,
    temporal_execution_mode: str,
    router_config: dict[str, Any],
    workers: int,
    progress_every: int,
    profile_latency: bool = False,
):
    chunks = chunk_samples(samples, workers)
    if len(chunks) == 1:
        router = build_router(router_config)
        latency_collector = EvaluationLatencyCollector() if profile_latency else None
        evaluator = V2VGoTQAPhase5AEvaluator(
            str(repository_root),
            router=router,
            graph_ablation=graph_ablation_mode,
            temporal_execution_mode=temporal_execution_mode,
            latency_collector=latency_collector,
        )
        predictions = evaluator.evaluate_samples(
            samples,
            baseline_mode=baseline_mode,
            progress_every=progress_every,
        )
        latency_records = latency_collector.records() if latency_collector is not None else tuple()
        return predictions, latency_records

    results = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                evaluate_chunk_worker,
                (
                    index,
                    chunk,
                    str(repository_root),
                    baseline_mode,
                    graph_ablation_mode,
                    temporal_execution_mode,
                    router_config,
                    progress_every,
                    profile_latency,
                ),
            )
            for index, chunk in enumerate(chunks)
        ]
        completed_predictions = 0
        for future in as_completed(futures):
            chunk_index, chunk_predictions, chunk_latency_records = future.result()
            results.append((chunk_index, chunk_predictions, chunk_latency_records))
            completed_predictions += len(chunk_predictions)
            print(
                f"parallel_progress: {completed_predictions}/{len(samples)} "
                f"chunks={len(results)}/{len(chunks)}",
                flush=True,
            )

    predictions = tuple(
        prediction
        for _, chunk_predictions, _ in sorted(results, key=lambda item: item[0])
        for prediction in chunk_predictions
    )
    latency_records = tuple(
        record
        for _, _, chunk_latency_records in sorted(results, key=lambda item: item[0])
        for record in chunk_latency_records
    )
    return predictions, latency_records


def write_latency_jsonl(path: Path, records: tuple[SampleLatencyRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "split_name": record.split_name,
                        "task_type": record.task_type,
                        "qa_type_id": record.qa_type_id,
                        "baseline_mode": record.baseline_mode,
                        "timings_ms": record.timings_ms,
                    }
                )
                + "\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    llm_client = build_optional_llm_client(args)
    if args.workers > 1 and llm_client is not None:
        raise SystemExit("--workers > 1 is only supported for non-LLM rankers.")
    router_config = router_config_from_args(args)

    samples = adapter.load_samples(split_name=args.split, file_name=args.file_name)
    selected_task_types = parse_task_types(args.task_types)
    if selected_task_types:
        samples = tuple(sample for sample in samples if sample.task_type in selected_task_types)
    if args.qa_type_ids:
        selected_qa_type_ids = set(int(value) for value in args.qa_type_ids)
        samples = tuple(sample for sample in samples if sample.qa_type_id in selected_qa_type_ids)
    samples = apply_sample_limit(samples, args.limit)
    predictions, latency_records = evaluate_samples_parallel(
        repository_root=repository_root,
        samples=samples,
        baseline_mode=args.baseline_mode,
        graph_ablation_mode=args.graph_ablation_mode,
        temporal_execution_mode=args.temporal_execution_mode,
        router_config=router_config,
        workers=args.workers,
        progress_every=args.progress_every,
        profile_latency=bool(args.latency_jsonl),
    )
    summary = V2VGoTQAPhase5AEvaluator.summarize(predictions)

    print("=" * 72)
    print("KG QA Task Evaluation")
    print("=" * 72)
    print(f"repository_root: {repository_root}")
    print(f"split: {args.split}")
    print(f"baseline_mode: {args.baseline_mode}")
    print(f"graph_ablation_mode: {args.graph_ablation_mode}")
    print(f"temporal_execution_mode: {args.temporal_execution_mode}")
    print(f"planning_ranker: {args.planning_ranker}")
    print(f"planning_selection_policy: {args.planning_selection_policy}")
    print(f"planning_selection_source: {args.planning_selection_source}")
    print(f"planning_acceptor_model_json: {args.planning_acceptor_model_json}")
    print(f"future_trajectory_model_json: {args.future_trajectory_model_json}")
    print(f"object_motion_model_json: {args.object_motion_model_json}")
    print(f"agent_motion_model_json: {args.agent_motion_model_json}")
    print(f"control_selection_policy: {args.control_selection_policy}")
    print(f"control_model_json: {args.control_model_json}")
    print(f"notable_ranker: {args.notable_ranker}")
    print(f"occluding_ranker: {args.occluding_ranker}")
    print(f"invisible_ranker: {args.invisible_ranker}")
    print(f"invisible_max_results: {args.invisible_max_results}")
    print(f"invisible_shortlist_size: {args.invisible_shortlist_size}")
    print(f"invisible_max_distance_to_trajectory: {args.invisible_max_distance_to_trajectory}")
    print(f"invisible_min_risk: {args.invisible_min_risk}")
    print(f"invisible_min_relative_to_best: {args.invisible_min_relative_to_best}")
    print(f"invisible_acceptor_model_json: {args.invisible_acceptor_model_json}")
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
    if args.qa_type_ids:
        print(f"qa_type_ids: {args.qa_type_ids}")

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
                            "graph_ablation_mode": args.graph_ablation_mode,
                            "planning_ranker": args.planning_ranker,
                            "planning_selection_policy": args.planning_selection_policy,
                            "planning_selection_source": args.planning_selection_source,
                            "planning_acceptor_model_json": args.planning_acceptor_model_json,
                            "future_trajectory_model_json": args.future_trajectory_model_json,
                            "object_motion_model_json": args.object_motion_model_json,
                            "agent_motion_model_json": args.agent_motion_model_json,
                            "control_selection_policy": args.control_selection_policy,
                            "control_model_json": args.control_model_json,
                            "notable_ranker": args.notable_ranker,
                            "occluding_ranker": args.occluding_ranker,
                            "invisible_ranker": args.invisible_ranker,
                            "invisible_max_results": args.invisible_max_results,
                            "invisible_max_distance_to_trajectory": args.invisible_max_distance_to_trajectory,
                            "invisible_min_risk": args.invisible_min_risk,
                            "invisible_min_relative_to_best": args.invisible_min_relative_to_best,
                            "invisible_acceptor_model_json": args.invisible_acceptor_model_json,
                            "llm_base_url": args.llm_base_url if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else "",
                            "llm_model": args.llm_model if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else "",
                            "llm_timeout_seconds": args.llm_timeout_seconds if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else 0,
                            "llm_max_tokens": args.llm_max_tokens if args.notable_ranker == "llm" or args.planning_ranker == PlanningAwarenessRanker.LLM.value or args.occluding_ranker == "llm" else 0,
                        }
                    )
                    + "\n"
                )
        print(f"saved_predictions: {output_path}")
    if args.latency_jsonl:
        latency_path = Path(args.latency_jsonl).expanduser().resolve()
        write_latency_jsonl(latency_path, latency_records)
        print(f"saved_latency: {latency_path}")


if __name__ == "__main__":
    main()
