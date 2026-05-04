#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.benchmark_references import references_for_task  # noqa: E402
from kg_coop_drive.domain.scene import CooperativeScene, ObjectTrack  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)

REFERENCE_COORDINATE_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")


@dataclass(frozen=True)
class ScenarioScore:
    task_type: str
    scenario_name: str
    baseline_mode: str
    planning_ranker: str
    planning_selection_policy: str
    total_samples: int
    exact_match_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    unresolved_reference_mentions: int
    resolved_reference_mentions: int


def resolve_v2vgot_root() -> Path:
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Phase 5 closeout outputs against benchmark reference answers.")
    parser.add_argument("--manifest", required=True, help="Path to phase5_closeout_manifest.json")
    parser.add_argument("--distance-threshold", type=float, default=3.0)
    parser.add_argument("--json-name", default="phase5_scored_report.json")
    parser.add_argument("--markdown-name", default="phase5_scored_report.md")
    return parser


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


def resolve_manifest_path(manifest_path: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (manifest_path.parent / path).resolve()


def infer_task_types(manifest: dict[str, object], runs: list[object]) -> tuple[BenchmarkTaskType, ...]:
    inferred: list[BenchmarkTaskType] = []
    seen: set[BenchmarkTaskType] = set()

    raw_task_types = manifest.get("task_types", [])
    if isinstance(raw_task_types, list):
        for value in raw_task_types:
            task_type = BenchmarkTaskType(str(value))
            if task_type in seen:
                continue
            inferred.append(task_type)
            seen.add(task_type)

    for run in runs:
        if not isinstance(run, dict) or "task_type" not in run:
            continue
        task_type = BenchmarkTaskType(str(run["task_type"]))
        if task_type in seen:
            continue
        inferred.append(task_type)
        seen.add(task_type)
    return tuple(inferred)


def run_value(run: dict[str, object], manifest: dict[str, object], key: str, default: str) -> str:
    value = run.get(key, manifest.get(key, default))
    return str(value)


def run_total_samples(run: dict[str, object], prediction_records: dict[str, dict[str, object]]) -> int:
    value = run.get("total_samples")
    if value is not None:
        return int(value)
    return len(prediction_records)


def normalize_ids(record: dict[str, object]) -> tuple[str, ...]:
    values = record.get("object_ids", [])
    if not isinstance(values, list):
        return ()
    return tuple(str(item) for item in values)


def extract_reference_coordinates(answer_text: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in REFERENCE_COORDINATE_PATTERN.findall(answer_text))


def euclidean_distance(track: ObjectTrack, point: tuple[float, float]) -> float:
    return math.hypot(track.position.x - point[0], track.position.y - point[1])


def resolve_reference_object_ids(
    scene: CooperativeScene,
    answer_text: str,
    distance_threshold: float,
) -> tuple[tuple[str, ...], int]:
    coordinates = extract_reference_coordinates(answer_text)
    if not coordinates:
        return (), 0

    remaining_tracks = list(scene.object_tracks)
    resolved_ids: list[str] = []
    unresolved = 0

    for coordinate in coordinates:
        ranked_tracks = sorted(
            remaining_tracks,
            key=lambda track: (
                euclidean_distance(track, coordinate),
                track.object_id,
            ),
        )
        if not ranked_tracks:
            unresolved += 1
            continue
        best_track = ranked_tracks[0]
        best_distance = euclidean_distance(best_track, coordinate)
        if best_distance > distance_threshold:
            unresolved += 1
            continue
        resolved_ids.append(best_track.object_id)
        remaining_tracks = [track for track in remaining_tracks if track.object_id != best_track.object_id]

    return tuple(resolved_ids), unresolved


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Closeout manifest contains no runs.")

    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    evaluator = V2VGoTQAPhase5AEvaluator(str(repository_root))

    task_types = infer_task_types(manifest, runs)
    if not task_types:
        raise SystemExit("Closeout manifest contains no task_types and no task_type values in runs.")
    split = str(manifest.get("split", "val"))
    all_samples = adapter.load_samples(split_name=split)
    samples_by_task = {
        task_type: {
            sample.sample_id: sample
            for sample in all_samples
            if sample.task_type == task_type
        }
        for task_type in task_types
    }

    reference_ids_by_task_and_sample: dict[tuple[str, str], tuple[str, ...]] = {}
    unresolved_by_task_and_sample: dict[tuple[str, str], int] = {}

    for task_type in task_types:
        task_samples = samples_by_task.get(task_type, {})
        for sample_id, sample in task_samples.items():
            prepared_scene = evaluator.prepare_sample(sample, baseline_mode="cooperative")
            reference_ids, unresolved = resolve_reference_object_ids(
                prepared_scene,
                sample.scene.raw_answer,
                distance_threshold=args.distance_threshold,
            )
            reference_ids_by_task_and_sample[(task_type.value, sample_id)] = tuple(sorted(set(reference_ids)))
            unresolved_by_task_and_sample[(task_type.value, sample_id)] = unresolved

    scores_by_task: dict[str, list[ScenarioScore]] = {task_type.value: [] for task_type in task_types}

    for run in runs:
        if not isinstance(run, dict):
            continue
        task_type_value = str(run["task_type"])
        task_type = BenchmarkTaskType(task_type_value)
        prediction_records = load_jsonl(resolve_manifest_path(manifest_path, str(run["output_jsonl"])))
        tp = fp = fn = exact_match_count = 0
        unresolved_reference_mentions = 0
        resolved_reference_mentions = 0

        for sample_id, prediction in prediction_records.items():
            predicted_ids = tuple(sorted(set(normalize_ids(prediction))))
            reference_ids = reference_ids_by_task_and_sample.get((task_type_value, sample_id), ())
            unresolved_reference_mentions += unresolved_by_task_and_sample.get((task_type_value, sample_id), 0)
            resolved_reference_mentions += len(reference_ids)

            predicted_set = set(predicted_ids)
            reference_set = set(reference_ids)
            tp += len(predicted_set & reference_set)
            fp += len(predicted_set - reference_set)
            fn += len(reference_set - predicted_set)
            if predicted_ids == reference_ids:
                exact_match_count += 1

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * tp, (2 * tp) + fp + fn)

        scenario_score = ScenarioScore(
            task_type=task_type_value,
            scenario_name=run_value(run, manifest, "scenario_name", manifest_path.stem),
            baseline_mode=run_value(run, manifest, "baseline_mode", "cooperative"),
            planning_ranker=run_value(run, manifest, "planning_ranker", "heuristic"),
            planning_selection_policy=run_value(run, manifest, "planning_selection_policy", "default"),
            total_samples=run_total_samples(run, prediction_records),
            exact_match_count=exact_match_count,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            unresolved_reference_mentions=unresolved_reference_mentions,
            resolved_reference_mentions=resolved_reference_mentions,
        )
        scores_by_task[task_type_value].append(scenario_score)

    report = {
        "repository_root": str(repository_root),
        "manifest": str(manifest_path),
        "distance_threshold": args.distance_threshold,
        "tasks": {},
    }
    markdown_sections = [
        "# Phase 5 Scored Report",
        "",
        f"- `manifest`: `{manifest_path}`",
        f"- `repository_root`: `{repository_root}`",
        f"- `distance_threshold`: `{args.distance_threshold}`",
        "",
        "Scores below are object-level metrics derived by resolving benchmark reference answer coordinates back to cooperative-scene object IDs. They are closer to the paper's QA F1 framing than the earlier structural diff summaries, but they are still our local reproduction layer rather than an official benchmark script.",
        "",
    ]

    print("=" * 72)
    print("Phase 5 Scored Report")
    print("=" * 72)
    print(f"manifest: {manifest_path}")
    print(f"repository_root: {repository_root}")
    print(f"distance_threshold: {args.distance_threshold}")

    for task_type in task_types:
        task_scores = sorted(
            scores_by_task[task_type.value],
            key=lambda score: (-score.f1, -score.exact_match_count, score.scenario_name),
        )
        paper_refs = references_for_task(task_type)

        rows = [[
            "Scenario",
            "Mode",
            "Ranker",
            "Policy",
            "Exact",
            "Precision",
            "Recall",
            "F1",
            "TP",
            "FP",
            "FN",
            "Resolved Ref",
            "Unresolved Ref",
        ]]
        best_scenario = task_scores[0].scenario_name if task_scores else ""

        print()
        print(f"[TASK] {task_type.value}")
        for score in task_scores:
            rows.append([
                score.scenario_name,
                score.baseline_mode,
                score.planning_ranker,
                score.planning_selection_policy,
                f"{score.exact_match_count}/{score.total_samples}",
                f"{score.precision:.3f}",
                f"{score.recall:.3f}",
                f"{score.f1:.3f}",
                str(score.tp),
                str(score.fp),
                str(score.fn),
                str(score.resolved_reference_mentions),
                str(score.unresolved_reference_mentions),
            ])
            print(
                f"  - {score.scenario_name}: "
                f"F1={score.f1:.3f}, P={score.precision:.3f}, R={score.recall:.3f}, "
                f"exact={score.exact_match_count}/{score.total_samples}"
            )

        report["tasks"][task_type.value] = {
            "best_scenario": best_scenario,
            "paper_references": [asdict(reference) for reference in paper_refs],
            "scenario_scores": [asdict(score) for score in task_scores],
        }

        markdown_sections.append(f"## {task_type.value}")
        markdown_sections.append("")
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
        markdown_sections.append(f"Best local scenario by object-level F1: `{best_scenario}`")
        markdown_sections.append("")
        markdown_sections.append(markdown_table(rows))
        markdown_sections.append("")

    json_path = manifest_path.with_name(args.json_name)
    markdown_path = manifest_path.with_name(args.markdown_name)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    print()
    print(f"saved_json: {json_path}")
    print(f"saved_markdown: {markdown_path}")


if __name__ == "__main__":
    main()
