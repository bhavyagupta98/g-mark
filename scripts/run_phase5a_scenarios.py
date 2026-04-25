#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.planning_awareness import (  # noqa: E402
    PlanningAwarenessRanker,
    PlanningAwarenessSelectionPolicy,
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.v2vgotqa_router import PlanningAwarenessHandler, V2VGoTQARouter  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.local_llm_client import (  # noqa: E402
    LocalOpenAICompatibleLLMClient,
    LocalOpenAICompatibleLLMConfig,
)
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    baseline_mode: str
    planning_ranker: str
    planning_selection_policy: str
    enabled_by_default: bool = True


CLASSICAL_SCENARIOS = (
    ScenarioSpec("heuristic_cooperative", "cooperative", "heuristic", "default"),
    ScenarioSpec("heuristic_ego_only", "ego_only", "heuristic", "default"),
    ScenarioSpec("risk_default_cooperative", "cooperative", "risk_aware", "default"),
    ScenarioSpec("risk_default_ego_only", "ego_only", "risk_aware", "default"),
    ScenarioSpec("risk_top2_cooperative", "cooperative", "risk_aware", "top2"),
    ScenarioSpec("risk_top2_ego_only", "ego_only", "risk_aware", "top2"),
    ScenarioSpec("risk_diverse_top2_cooperative", "cooperative", "risk_aware", "diverse_top2"),
    ScenarioSpec("risk_diverse_top2_ego_only", "ego_only", "risk_aware", "diverse_top2"),
    ScenarioSpec("energy_cooperative", "cooperative", "energy_based", "default"),
    ScenarioSpec("relational_cooperative", "cooperative", "relational_importance", "default"),
)

LLM_SCENARIOS = (
    ScenarioSpec("llm_top2_cooperative", "cooperative", "llm", "top2", enabled_by_default=False),
    ScenarioSpec("llm_top2_ego_only", "ego_only", "llm", "top2", enabled_by_default=False),
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
    parser = argparse.ArgumentParser(description="Run a Phase 5A scenario matrix.")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--task-type",
        default=BenchmarkTaskType.PLANNING_AWARENESS.value,
        choices=tuple(item.value for item in BenchmarkTaskType),
    )
    parser.add_argument(
        "--scenario-set",
        default="classical",
        choices=("classical", "llm", "all", "phase5_closeout"),
        help="Which predefined scenario family to run.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_names",
        default=[],
        help="Run only the named scenarios. Repeatable.",
    )
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--output-dir", default="outputs/phase5a_matrix")
    parser.add_argument("--manifest-name", default="scenario_manifest.json")
    parser.add_argument("--llm-base-url", default=os.environ.get("KG_LOCAL_LLM_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.environ.get("KG_LOCAL_LLM_MODEL", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("KG_LOCAL_LLM_API_KEY", "local-token"))
    return parser


def available_scenarios() -> tuple[ScenarioSpec, ...]:
    return CLASSICAL_SCENARIOS + LLM_SCENARIOS


def select_scenarios(args: argparse.Namespace) -> tuple[ScenarioSpec, ...]:
    all_specs = available_scenarios()
    if args.scenario_names:
        wanted = set(args.scenario_names)
        chosen = tuple(spec for spec in all_specs if spec.name in wanted)
        missing = sorted(wanted - {spec.name for spec in chosen})
        if missing:
            raise SystemExit(f"Unknown scenario names: {', '.join(missing)}")
        return chosen
    if args.scenario_set == "classical":
        return CLASSICAL_SCENARIOS
    if args.scenario_set == "llm":
        return LLM_SCENARIOS
    if args.scenario_set == "phase5_closeout":
        wanted = {
            "heuristic_cooperative",
            "heuristic_ego_only",
            "risk_default_cooperative",
            "risk_default_ego_only",
            "risk_top2_cooperative",
            "risk_top2_ego_only",
            "risk_diverse_top2_cooperative",
            "risk_diverse_top2_ego_only",
            "energy_cooperative",
            "relational_cooperative",
            "llm_top2_cooperative",
            "llm_top2_ego_only",
        }
        return tuple(spec for spec in all_specs if spec.name in wanted)
    return all_specs


def build_optional_llm_client(args: argparse.Namespace) -> LocalOpenAICompatibleLLMClient | None:
    if not args.llm_base_url or not args.llm_model:
        return None
    return LocalOpenAICompatibleLLMClient(
        LocalOpenAICompatibleLLMConfig(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    repository_root = resolve_v2vgot_root()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    samples = adapter.load_samples(split_name=args.split, file_name=args.file_name)
    task_type = BenchmarkTaskType(args.task_type)
    samples = tuple(sample for sample in samples if sample.task_type == task_type)[: args.limit]
    llm_client = build_optional_llm_client(args)

    if not samples:
        raise SystemExit(f"No samples found for task_type={task_type.value} split={args.split}.")

    selected_scenarios = select_scenarios(args)
    manifest_records: list[dict[str, object]] = []

    print("=" * 72)
    print("Phase 5A Scenario Matrix")
    print("=" * 72)
    print(f"repository_root: {repository_root}")
    print(f"split: {args.split}")
    print(f"task_type: {task_type.value}")
    print(f"sample_count: {len(samples)}")
    print(f"output_dir: {output_dir}")
    print(f"scenario_count: {len(selected_scenarios)}")

    for scenario in selected_scenarios:
        if scenario.planning_ranker == PlanningAwarenessRanker.LLM.value and llm_client is None:
            print(f"[SKIP] {scenario.name}: missing --llm-base-url / --llm-model")
            continue

        orchestrator = build_planning_awareness_orchestrator(
            ranker=scenario.planning_ranker,
            llm_client=llm_client,
            selection_policy=scenario.planning_selection_policy,
        )
        router = V2VGoTQARouter(handlers=(PlanningAwarenessHandler(orchestrator=orchestrator),))
        evaluator = V2VGoTQAPhase5AEvaluator(str(repository_root), router=router)
        predictions = evaluator.evaluate_samples(samples, baseline_mode=scenario.baseline_mode)
        summary = evaluator.summarize(predictions)

        output_path = output_dir / f"{scenario.name}.jsonl"
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
                            "planning_ranker": scenario.planning_ranker,
                            "planning_selection_policy": scenario.planning_selection_policy,
                            "scenario_name": scenario.name,
                            "llm_base_url": args.llm_base_url if scenario.planning_ranker == PlanningAwarenessRanker.LLM.value else "",
                            "llm_model": args.llm_model if scenario.planning_ranker == PlanningAwarenessRanker.LLM.value else "",
                        }
                    )
                    + "\n"
                )

        manifest_records.append(
            {
                **asdict(scenario),
                "output_jsonl": str(output_path),
                "supported_predictions": summary.supported_predictions,
                "unsupported_predictions": summary.unsupported_predictions,
                "total_samples": summary.total_samples,
            }
        )
        print(
            f"[DONE] {scenario.name}: "
            f"baseline={scenario.baseline_mode}, "
            f"ranker={scenario.planning_ranker}, "
            f"policy={scenario.planning_selection_policy}, "
            f"output={output_path.name}"
        )

    manifest_path = output_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "repository_root": str(repository_root),
                "split": args.split,
                "task_type": task_type.value,
                "sample_count": len(samples),
                "scenarios": manifest_records,
            },
            handle,
            indent=2,
        )
    print()
    print(f"saved_manifest: {manifest_path}")


if __name__ == "__main__":
    main()
