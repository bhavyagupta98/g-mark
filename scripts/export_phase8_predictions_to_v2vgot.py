#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.scene import CooperativeScene  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)

OFFICIAL_QA_TYPE_BY_TASK = {
    BenchmarkTaskType.NOTABLE_OBJECTS: 11,
    BenchmarkTaskType.OCCLUDING_OBJECTS: 12,
    BenchmarkTaskType.INVISIBLE_OBJECTS: 13,
    BenchmarkTaskType.PLANNING_AWARENESS: 14,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export normalized Phase 8 QA predictions to V2V-GoT/LLaVA official-style "
            "JSONL records with an `outputs` field."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/phase8_official_exports")
    parser.add_argument("--split", default="")
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        default=[],
        help="Optional task type to export. Repeatable. Defaults to supported Q1-Q4 tasks in manifest.",
    )
    parser.add_argument("--scenario-name", default="")
    return parser


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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


def parse_task_types(raw_values: list[str]) -> tuple[BenchmarkTaskType, ...]:
    if not raw_values:
        return ()
    return tuple(BenchmarkTaskType(value) for value in raw_values)


def output_text_for_task(
    task_type: BenchmarkTaskType,
    scene: CooperativeScene,
    object_ids: tuple[str, ...],
) -> str:
    if not object_ids:
        if task_type == BenchmarkTaskType.OCCLUDING_OBJECTS:
            return "There is no object obstructing your view."
        if task_type == BenchmarkTaskType.INVISIBLE_OBJECTS:
            return "There is no notable object invisible to you."
        if task_type == BenchmarkTaskType.PLANNING_AWARENESS:
            return "There is no notable object."
        return "There is no notable object visible to you."

    phrases: list[str] = []
    for object_id in object_ids:
        object_track = scene.get_object(object_id)
        if object_track is None:
            continue
        location = f"({object_track.position.x:.1f},{object_track.position.y:.1f})"
        if task_type == BenchmarkTaskType.OCCLUDING_OBJECTS:
            phrases.append(f"There is a car at {location} obstructing your view.")
        elif task_type == BenchmarkTaskType.INVISIBLE_OBJECTS:
            phrases.append(f"There is a car at {location} invisible to you.")
        elif task_type == BenchmarkTaskType.PLANNING_AWARENESS:
            phrases.append(f"There is a car at {location} close to your planned future trajectory.")
        else:
            phrases.append(f"There is a car at {location} visible to you.")

    if not phrases:
        return output_text_for_task(task_type, scene, ())
    return " ".join(phrases)


def export_task(
    task_type: BenchmarkTaskType,
    run: dict[str, object],
    samples_by_task_and_id: dict[tuple[BenchmarkTaskType, str], BenchmarkSample],
    evaluator: V2VGoTQAPhase5AEvaluator,
    output_dir: Path,
    baseline_mode: str,
    scenario_name: str,
) -> dict[str, object]:
    prediction_path = resolve_repo_path(str(run["output_jsonl"]))
    predictions = load_jsonl(prediction_path)
    output_path = output_dir / f"{task_type.value}_{scenario_name}_official.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    missing_samples = 0
    missing_objects = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id, prediction in sorted(
            predictions.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        ):
            sample = samples_by_task_and_id.get((task_type, sample_id))
            if sample is None:
                missing_samples += 1
                continue
            prepared_scene = evaluator.prepare_sample(sample, baseline_mode=baseline_mode)
            object_ids = normalize_ids(prediction)
            missing_objects += sum(1 for object_id in object_ids if prepared_scene.get_object(object_id) is None)
            output_record = dict(sample.raw_record)
            output_record["outputs"] = output_text_for_task(task_type, prepared_scene, object_ids)
            output_record["kg_prediction"] = {
                "sample_id": sample_id,
                "task_type": task_type.value,
                "object_ids": list(object_ids),
                "source_jsonl": str(prediction_path),
                "scenario_name": scenario_name,
            }
            handle.write(json.dumps(output_record) + "\n")
            exported_count += 1

    return {
        "task_type": task_type.value,
        "qa_type_id": OFFICIAL_QA_TYPE_BY_TASK[task_type],
        "source_jsonl": str(prediction_path),
        "output_jsonl": str(output_path),
        "exported_count": exported_count,
        "missing_samples": missing_samples,
        "missing_objects": missing_objects,
    }


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = args.split or str(manifest.get("split", "val"))
    scenario_name = args.scenario_name or str(manifest.get("scenario_name", manifest_path.stem))
    output_dir = resolve_repo_path(args.output_dir)

    selected_task_types = parse_task_types(args.task_types)
    selected_task_set = set(selected_task_types)

    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    evaluator = V2VGoTQAPhase5AEvaluator(str(repository_root))
    samples = adapter.load_samples(split_name=split, file_name=args.file_name)
    samples_by_task_and_id = {
        (sample.task_type, sample.sample_id): sample
        for sample in samples
    }

    exported_runs: list[dict[str, object]] = []
    for run in manifest.get("runs", []):
        if not isinstance(run, dict):
            continue
        task_type = BenchmarkTaskType(str(run["task_type"]))
        if task_type not in OFFICIAL_QA_TYPE_BY_TASK:
            continue
        if selected_task_set and task_type not in selected_task_set:
            continue
        exported_runs.append(
            export_task(
                task_type=task_type,
                run=run,
                samples_by_task_and_id=samples_by_task_and_id,
                evaluator=evaluator,
                output_dir=output_dir,
                baseline_mode=args.baseline_mode,
                scenario_name=scenario_name,
            )
        )

    export_manifest = {
        "source_manifest": str(manifest_path),
        "repository_root": str(repository_root),
        "split": split,
        "baseline_mode": args.baseline_mode,
        "scenario_name": scenario_name,
        "runs": exported_runs,
    }
    export_manifest_path = output_dir / f"{scenario_name}_official_export_manifest.json"
    export_manifest_path.write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Phase 8 Official Export")
    print("=" * 72)
    print(f"source_manifest: {manifest_path}")
    print(f"repository_root: {repository_root}")
    print(f"output_dir: {output_dir}")
    for run in exported_runs:
        print(
            f"[{run['task_type']}] qa_type_id={run['qa_type_id']} "
            f"exported={run['exported_count']} missing_samples={run['missing_samples']} "
            f"missing_objects={run['missing_objects']}"
        )
        print(f"  output_jsonl: {run['output_jsonl']}")
    print(f"saved_manifest: {export_manifest_path}")


if __name__ == "__main__":
    main()
