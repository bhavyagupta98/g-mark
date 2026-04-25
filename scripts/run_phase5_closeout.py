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
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.v2vgotqa_router import (  # noqa: E402
    NotableObjectsHandler,
    OccludingObjectsHandler,
    PlanningAwarenessHandler,
    V2VGoTQARouter,
)
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.benchmark_references import references_for_task  # noqa: E402
from kg_coop_drive.infrastructure.local_llm_client import (  # noqa: E402
    LocalOpenAICompatibleLLMClient,
    LocalOpenAICompatibleLLMConfig,
)
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402
from run_phase5a_scenarios import CLASSICAL_SCENARIOS, LLM_SCENARIOS, ScenarioSpec  # noqa: E402

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)

SUPPORTED_PHASE5_TASKS = (
    BenchmarkTaskType.NOTABLE_OBJECTS,
    BenchmarkTaskType.OCCLUDING_OBJECTS,
    BenchmarkTaskType.INVISIBLE_OBJECTS,
    BenchmarkTaskType.PLANNING_AWARENESS,
)

NON_PLANNING_SCENARIOS = (
    ScenarioSpec("cooperative_router", "cooperative", PlanningAwarenessRanker.HEURISTIC.value, "default"),
    ScenarioSpec("ego_only_router", "ego_only", PlanningAwarenessRanker.HEURISTIC.value, "default"),
)

PLANNING_CLOSEOUT_SCENARIOS = tuple(CLASSICAL_SCENARIOS) + tuple(LLM_SCENARIOS)


@dataclass(frozen=True)
class TaskScenarioRun:
    task_type: str
    scenario_name: str
    baseline_mode: str
    planning_ranker: str
    planning_selection_policy: str
    output_jsonl: str
    supported_predictions: int
    unsupported_predictions: int
    total_samples: int


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Phase 5 closeout matrix across supported tasks.")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        default=[],
        help="Optional BenchmarkTaskType value to run. Repeatable. Defaults to all supported Phase 5 tasks.",
    )
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--output-dir", default="outputs/phase5_closeout")
    parser.add_argument("--manifest-name", default="phase5_closeout_manifest.json")
    parser.add_argument("--report-name", default="phase5_closeout_report.json")
    parser.add_argument("--markdown-name", default="phase5_closeout_report.md")
    parser.add_argument("--llm-base-url", default=os.environ.get("KG_LOCAL_LLM_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.environ.get("KG_LOCAL_LLM_MODEL", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("KG_LOCAL_LLM_API_KEY", "local-token"))
    parser.add_argument("--llm-timeout-seconds", type=float, default=float(os.environ.get("KG_LOCAL_LLM_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.environ.get("KG_LOCAL_LLM_MAX_TOKENS", "128")))
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
    parser.add_argument(
        "--full-sweep-all-tasks",
        action="store_true",
        help=(
            "Run the full classical+LLM planning-awareness scenario family for every supported "
            "Phase 5 task. For non-planning tasks, planning ranker/policy settings act as control "
            "conditions and should not change outputs under the current router."
        ),
    )
    return parser


def select_task_types(raw_task_types: list[str]) -> tuple[BenchmarkTaskType, ...]:
    if not raw_task_types:
        return SUPPORTED_PHASE5_TASKS
    return tuple(BenchmarkTaskType(value) for value in raw_task_types)


def build_optional_llm_client(args: argparse.Namespace) -> LocalOpenAICompatibleLLMClient | None:
    if not args.llm_base_url or not args.llm_model:
        return None
    return LocalOpenAICompatibleLLMClient(
        LocalOpenAICompatibleLLMConfig(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
            timeout_seconds=args.llm_timeout_seconds,
            max_tokens=args.llm_max_tokens,
        )
    )


def load_jsonl(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record["sample_id"])] = record
    return records


def normalize_ids(record: dict[str, object]) -> tuple[str, ...]:
    values = record.get("object_ids", [])
    if not isinstance(values, list):
        return ()
    return tuple(str(item) for item in values)


def summarize_against_baseline(
    baseline_records: dict[str, dict[str, object]],
    candidate_records: dict[str, dict[str, object]],
) -> dict[str, int]:
    common_ids = sorted(set(baseline_records) & set(candidate_records))
    exact_matches = 0
    set_matches = 0
    semantic_differences = 0
    for sample_id in common_ids:
        left = baseline_records[sample_id]
        right = candidate_records[sample_id]
        left_ids = normalize_ids(left)
        right_ids = normalize_ids(right)
        if left.get("answer_text") == right.get("answer_text") and left_ids == right_ids:
            exact_matches += 1
        if tuple(sorted(set(left_ids))) == tuple(sorted(set(right_ids))):
            set_matches += 1
        else:
            semantic_differences += 1
    return {
        "common_samples": len(common_ids),
        "exact_matches": exact_matches,
        "unordered_set_matches": set_matches,
        "semantic_differences": semantic_differences,
    }


def baseline_name_for_task(task_type: BenchmarkTaskType, full_sweep_all_tasks: bool = False) -> str:
    if full_sweep_all_tasks:
        return "risk_diverse_top2_cooperative"
    if task_type == BenchmarkTaskType.PLANNING_AWARENESS:
        return "risk_diverse_top2_cooperative"
    return "cooperative_router"


def scenarios_for_task(task_type: BenchmarkTaskType, full_sweep_all_tasks: bool = False) -> tuple[ScenarioSpec, ...]:
    if full_sweep_all_tasks:
        return PLANNING_CLOSEOUT_SCENARIOS
    if task_type == BenchmarkTaskType.PLANNING_AWARENESS:
        return PLANNING_CLOSEOUT_SCENARIOS
    return NON_PLANNING_SCENARIOS


def task_sweep_note(task_type: BenchmarkTaskType, full_sweep_all_tasks: bool) -> str:
    if full_sweep_all_tasks and task_type != BenchmarkTaskType.PLANNING_AWARENESS:
        return (
            "Full scenario sweep requested. Under the current router, this task does not consume "
            "the planning-awareness ranker/policy, so scenario-to-scenario differences should "
            "mainly reflect `cooperative` vs `ego_only` preparation rather than the ranking method."
        )
    if task_type != BenchmarkTaskType.PLANNING_AWARENESS:
        return (
            "This task currently uses deterministic router logic. Planning ranker/policy settings "
            "do not affect its outputs under the current Phase 5 implementation."
        )
    return "This task uses the planning-awareness orchestrator, so ranker and selection policy are active variables."


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def main() -> None:
    args = build_parser().parse_args()
    repository_root = resolve_v2vgot_root()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    all_samples = adapter.load_samples(split_name=args.split, file_name=args.file_name)
    task_types = select_task_types(args.task_types)
    llm_client = build_optional_llm_client(args)

    manifest_runs: list[TaskScenarioRun] = []
    report: dict[str, object] = {
        "repository_root": str(repository_root),
        "split": args.split,
        "limit": args.limit,
        "full_sweep_all_tasks": args.full_sweep_all_tasks,
        "notable_ranker": args.notable_ranker,
        "occluding_ranker": args.occluding_ranker,
        "task_types": [task_type.value for task_type in task_types],
        "tasks": {},
    }

    markdown_sections: list[str] = [
        "# Phase 5 Closeout",
        "",
        f"- `repository_root`: `{repository_root}`",
        f"- `split`: `{args.split}`",
        f"- `limit`: `{args.limit}`",
        f"- `full_sweep_all_tasks`: `{args.full_sweep_all_tasks}`",
        f"- `notable_ranker`: `{args.notable_ranker}`",
        f"- `occluding_ranker`: `{args.occluding_ranker}`",
        f"- `tasks`: `{', '.join(task_type.value for task_type in task_types)}`",
        "",
        "Published references below are target context from the V2V-GoT paper. Our current outputs are structural prediction comparisons, not reproduced benchmark F1/L2 scores.",
        "",
    ]

    print("=" * 72)
    print("Phase 5 Closeout")
    print("=" * 72)
    print(f"repository_root: {repository_root}")
    print(f"split: {args.split}")
    print(f"limit: {args.limit}")
    print(f"notable_ranker: {args.notable_ranker}")
    print(f"occluding_ranker: {args.occluding_ranker}")
    print(f"tasks: {[task_type.value for task_type in task_types]}")

    for task_type in task_types:
        task_samples = tuple(sample for sample in all_samples if sample.task_type == task_type)[: args.limit]
        if not task_samples:
            print(f"[SKIP] task={task_type.value}: no samples")
            continue

        task_dir = output_dir / task_type.value
        task_dir.mkdir(parents=True, exist_ok=True)
        task_runs: list[dict[str, object]] = []

        print()
        print(f"[TASK] {task_type.value}: sample_count={len(task_samples)}")
        print(f"  [NOTE] {task_sweep_note(task_type, args.full_sweep_all_tasks)}")

        for scenario in scenarios_for_task(task_type, args.full_sweep_all_tasks):
            if scenario.planning_ranker == PlanningAwarenessRanker.LLM.value and llm_client is None:
                print(f"[SKIP] {task_type.value}/{scenario.name}: missing llm endpoint/model")
                continue

            orchestrator = build_planning_awareness_orchestrator(
                ranker=scenario.planning_ranker,
                llm_client=llm_client,
                selection_policy=scenario.planning_selection_policy,
            )
            router = V2VGoTQARouter(
                handlers=(
                    NotableObjectsHandler(
                        ranker=args.notable_ranker,
                        llm_client=llm_client if args.notable_ranker == "llm" else None,
                    ),
                    OccludingObjectsHandler(llm_client=llm_client if args.occluding_ranker == "llm" else None),
                    PlanningAwarenessHandler(orchestrator=orchestrator),
                )
            )
            evaluator = V2VGoTQAPhase5AEvaluator(str(repository_root), router=router)
            predictions = evaluator.evaluate_samples(task_samples, baseline_mode=scenario.baseline_mode)
            summary = evaluator.summarize(predictions)

            output_path = task_dir / f"{scenario.name}.jsonl"
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
                            }
                        )
                        + "\n"
                    )

            run_record = TaskScenarioRun(
                task_type=task_type.value,
                scenario_name=scenario.name,
                baseline_mode=scenario.baseline_mode,
                planning_ranker=scenario.planning_ranker,
                planning_selection_policy=scenario.planning_selection_policy,
                output_jsonl=str(output_path),
                supported_predictions=summary.supported_predictions,
                unsupported_predictions=summary.unsupported_predictions,
                total_samples=summary.total_samples,
            )
            manifest_runs.append(run_record)
            task_runs.append(asdict(run_record))

            print(
                f"  [DONE] scenario={scenario.name}, "
                f"baseline={scenario.baseline_mode}, "
                f"ranker={scenario.planning_ranker}, "
                f"policy={scenario.planning_selection_policy}"
            )

        baseline_name = baseline_name_for_task(task_type, args.full_sweep_all_tasks)
        by_name = {str(run["scenario_name"]): run for run in task_runs}
        baseline_run = by_name.get(baseline_name)
        task_report: dict[str, object] = {
            "sample_count": len(task_samples),
            "baseline_scenario": baseline_name,
            "sweep_note": task_sweep_note(task_type, args.full_sweep_all_tasks),
            "notable_ranker": args.notable_ranker,
            "occluding_ranker": args.occluding_ranker,
            "paper_references": [asdict(reference) for reference in references_for_task(task_type)],
            "scenarios": [],
        }

        table_rows = [[
            "Scenario",
            "Mode",
            "Ranker",
            "Policy",
            "Supported",
            "Exact vs Baseline",
            "Set vs Baseline",
            "Semantic Diff",
        ]]

        if baseline_run is None:
            raise SystemExit(f"Baseline scenario `{baseline_name}` missing for task `{task_type.value}`.")
        baseline_records = load_jsonl(Path(str(baseline_run["output_jsonl"])))

        for run in sorted(task_runs, key=lambda item: (item["scenario_name"] != baseline_name, str(item["scenario_name"]))):
            scenario_name = str(run["scenario_name"])
            scenario_stats: dict[str, object] = {
                "scenario_name": scenario_name,
                "baseline_mode": run["baseline_mode"],
                "planning_ranker": run["planning_ranker"],
                "planning_selection_policy": run["planning_selection_policy"],
                "notable_ranker": args.notable_ranker,
                "occluding_ranker": args.occluding_ranker,
                "supported_predictions": run["supported_predictions"],
                "unsupported_predictions": run["unsupported_predictions"],
                "total_samples": run["total_samples"],
            }
            if scenario_name == baseline_name:
                scenario_stats["comparison_to_baseline"] = {
                    "common_samples": int(run["total_samples"]),
                    "exact_matches": int(run["total_samples"]),
                    "unordered_set_matches": int(run["total_samples"]),
                    "semantic_differences": 0,
                }
                exact_text = f"{run['total_samples']}/{run['total_samples']}"
                set_text = exact_text
                semantic_text = "0"
            else:
                candidate_records = load_jsonl(Path(str(run["output_jsonl"])))
                comparison = summarize_against_baseline(baseline_records, candidate_records)
                scenario_stats["comparison_to_baseline"] = comparison
                exact_text = f"{comparison['exact_matches']}/{comparison['common_samples']}"
                set_text = f"{comparison['unordered_set_matches']}/{comparison['common_samples']}"
                semantic_text = str(comparison["semantic_differences"])
            task_report["scenarios"].append(scenario_stats)
            table_rows.append([
                scenario_name,
                str(run["baseline_mode"]),
                str(run["planning_ranker"]),
                str(run["planning_selection_policy"]),
                f"{run['supported_predictions']}/{run['total_samples']}",
                exact_text,
                set_text,
                semantic_text,
            ])

        cast_tasks = report["tasks"]
        assert isinstance(cast_tasks, dict)
        cast_tasks[task_type.value] = task_report

        markdown_sections.append(f"## {task_type.value}")
        markdown_sections.append("")
        markdown_sections.append(task_sweep_note(task_type, args.full_sweep_all_tasks))
        markdown_sections.append("")
        paper_refs = references_for_task(task_type)
        if paper_refs:
            markdown_sections.append("Published target context:")
            for reference in paper_refs:
                direction = "higher is better" if reference.higher_is_better else "lower is better"
                markdown_sections.append(
                    f"- `{reference.method_name}` `{reference.metric_name}` = `{reference.metric_value}` "
                    f"({direction}, {reference.source_table})"
                )
                if reference.notes:
                    markdown_sections.append(f"  {reference.notes}")
            markdown_sections.append("")
        markdown_sections.append(f"Baseline scenario: `{baseline_name}`")
        markdown_sections.append("")
        markdown_sections.append(markdown_table(table_rows))
        markdown_sections.append("")

    manifest_path = output_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "repository_root": str(repository_root),
                "split": args.split,
                "limit": args.limit,
                "notable_ranker": args.notable_ranker,
                "occluding_ranker": args.occluding_ranker,
                "task_types": [task_type.value for task_type in task_types],
                "runs": [asdict(run) for run in manifest_runs],
            },
            handle,
            indent=2,
        )

    report_path = output_dir / args.report_name
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    markdown_path = output_dir / args.markdown_name
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    print()
    print(f"saved_manifest: {manifest_path}")
    print(f"saved_report: {report_path}")
    print(f"saved_markdown: {markdown_path}")


if __name__ == "__main__":
    main()
