#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
import re

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
    BenchmarkTaskType.OBJECT_MOTION_PREDICTION: 15,
    BenchmarkTaskType.AGENT_MOTION_PREDICTION: 16,
    BenchmarkTaskType.CONTROL_SETTINGS: 18,
    BenchmarkTaskType.FUTURE_TRAJECTORY: 19,
}

OBJECT_GROUNDING_TASKS = {
    BenchmarkTaskType.NOTABLE_OBJECTS,
    BenchmarkTaskType.OCCLUDING_OBJECTS,
    BenchmarkTaskType.INVISIBLE_OBJECTS,
    BenchmarkTaskType.PLANNING_AWARENESS,
}


def _q5_motion_action_label(vx: float, vy: float) -> str:
    speed = (vx * vx + vy * vy) ** 0.5
    if speed < 0.1:
        return "staying at the same location"
    if abs(vy) > abs(vx):
        return "turning right" if vy >= 0.0 else "turning left"
    if vx >= 0.0:
        return "moving forward"
    return "turning right" if vy >= 0.0 else "turning left"


_Q5_FROM_TO_RE = re.compile(
    r"(?P<object_id>[A-Za-z0-9_\-]+)\s*=\s*"
    r"(?P<action>[^;,.]+?)\s+from\s+"
    r"\((?P<sx>-?\d+(?:\.\d+)?),\s*(?P<sy>-?\d+(?:\.\d+)?)\)\s+to\s+"
    r"\((?P<tx>-?\d+(?:\.\d+)?),\s*(?P<ty>-?\d+(?:\.\d+)?)\)",
    re.IGNORECASE,
)
_Q5_TRAJECTORY_RE = re.compile(
    r"(?P<object_id>[A-Za-z0-9_\-]+)\s*=\s*"
    r"(?P<action>[^;,.]+?)\s+from\s+"
    r"\((?P<sx>-?\d+(?:\.\d+)?),\s*(?P<sy>-?\d+(?:\.\d+)?)\)\s+"
    r"trajectory\s+\[(?P<trajectory>[^\]]+)\]",
    re.IGNORECASE | re.DOTALL,
)
_POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


def _parse_q5_router_answer(answer_text: str) -> list[tuple[str, str, float, float, list[tuple[float, float]]]]:
    rows: list[tuple[str, str, float, float, list[tuple[float, float]]]] = []
    for match in _Q5_TRAJECTORY_RE.finditer(answer_text):
        points = [
            (float(raw_x), float(raw_y))
            for raw_x, raw_y in _POINT_RE.findall(str(match.group("trajectory")))
        ]
        if not points:
            continue
        rows.append(
            (
                str(match.group("object_id")),
                str(match.group("action")).strip().lower(),
                float(match.group("sx")),
                float(match.group("sy")),
                points,
            )
        )
    if rows:
        return rows
    for match in _Q5_FROM_TO_RE.finditer(answer_text):
        rows.append(
            (
                str(match.group("object_id")),
                str(match.group("action")).strip().lower(),
                float(match.group("sx")),
                float(match.group("sy")),
                [(float(match.group("tx")), float(match.group("ty")))],
            )
        )
    return rows


def _render_q5_motion_output(
    scene: CooperativeScene,
    object_ids: tuple[str, ...],
    answer_text: str,
) -> str:
    parsed_answer_rows = _parse_q5_router_answer(answer_text)
    if parsed_answer_rows:
        phrases = []
        for _object_id, action, sx, sy, points in parsed_answer_rows[:3]:
            normalized_action = action
            if normalized_action not in {
                "moving forward",
                "turning left",
                "turning right",
                "staying at the same location",
            }:
                dx = points[-1][0] - sx
                dy = points[-1][1] - sy
                normalized_action = _q5_motion_action_label(dx, dy)
            rendered_points = ",".join(f"({x:.1f},{y:.1f})" for x, y in points)
            phrases.append(
                "There is a car at "
                f"({sx:.1f},{sy:.1f}) {normalized_action}. "
                "The predicted future trajectory is "
                f"[{rendered_points}]."
            )
        return " ".join(phrases)

    selected_tracks = []
    if object_ids:
        for object_id in object_ids:
            track = scene.get_object(object_id)
            if track is not None:
                selected_tracks.append(track)
    if not selected_tracks and scene.object_tracks:
        selected_tracks = list(scene.object_tracks[:1])

    if not selected_tracks:
        return "There is no notable object."

    phrases: list[str] = []
    for track in selected_tracks[:3]:
        start_x = track.position.x
        start_y = track.position.y
        if track.velocity is None:
            vx = 0.0
            vy = 0.0
        else:
            vx = track.velocity.x
            vy = track.velocity.y
        end_x = start_x + vx
        end_y = start_y + vy
        action = _q5_motion_action_label(vx, vy)
        phrases.append(
            "There is a car at "
            f"({start_x:.1f},{start_y:.1f}) {action}. "
            "The predicted future trajectory is "
            f"[({end_x:.1f},{end_y:.1f})]."
        )
    return " ".join(phrases)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export normalized QA predictions to V2V-GoT/LLaVA official-style "
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
    parser.add_argument(
        "--qa-type-id",
        action="append",
        dest="qa_type_ids",
        type=int,
        default=[],
        help=(
            "Optional raw V2V-GoT qa_type_id export filter/override. Repeatable. "
            "Use 15 for Q5 and 17 for Q7 object_motion_prediction splits."
        ),
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


def prediction_answer_text(record: dict[str, object]) -> str:
    value = record.get("answer_text", "")
    return value if isinstance(value, str) else ""


def parse_task_types(raw_values: list[str]) -> tuple[BenchmarkTaskType, ...]:
    if not raw_values:
        return ()
    return tuple(BenchmarkTaskType(value) for value in raw_values)


def output_text_for_task(
    task_type: BenchmarkTaskType,
    scene: CooperativeScene,
    object_ids: tuple[str, ...],
    answer_text: str = "",
) -> str:
    if task_type == BenchmarkTaskType.OBJECT_MOTION_PREDICTION:
        return _render_q5_motion_output(scene, object_ids, answer_text)

    if task_type not in OBJECT_GROUNDING_TASKS:
        return answer_text

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
        return output_text_for_task(task_type, scene, (), answer_text)
    return " ".join(phrases)


def export_task(
    task_type: BenchmarkTaskType,
    run: dict[str, object],
    samples_by_task_qa_and_id: dict[tuple[BenchmarkTaskType, int | None, str], BenchmarkSample],
    evaluator: V2VGoTQAPhase5AEvaluator,
    output_dir: Path,
    baseline_mode: str,
    scenario_name: str,
    progress_every: int,
) -> dict[str, object]:
    prediction_path = resolve_repo_path(str(run["output_jsonl"]))
    predictions = load_jsonl(prediction_path)
    raw_qa_type_id = run.get("qa_type_id", OFFICIAL_QA_TYPE_BY_TASK[task_type])
    qa_type_id = int(raw_qa_type_id) if raw_qa_type_id is not None else OFFICIAL_QA_TYPE_BY_TASK[task_type]
    output_path = output_dir / f"{task_type.value}_qa_type_{qa_type_id}_{scenario_name}_official.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    missing_samples = 0
    missing_objects = 0
    with output_path.open("w", encoding="utf-8") as handle:
        total_predictions = len(predictions)
        for sample_id, prediction in sorted(
            predictions.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        ):
            prediction_qa_type_id = prediction.get("qa_type_id", qa_type_id)
            try:
                lookup_qa_type_id = int(prediction_qa_type_id) if prediction_qa_type_id is not None else qa_type_id
            except (TypeError, ValueError):
                lookup_qa_type_id = qa_type_id
            if lookup_qa_type_id != qa_type_id:
                continue
            sample = samples_by_task_qa_and_id.get((task_type, qa_type_id, sample_id))
            if sample is None and raw_qa_type_id is None:
                sample = samples_by_task_qa_and_id.get((task_type, None, sample_id))
            if sample is None:
                missing_samples += 1
                continue
            object_ids = normalize_ids(prediction)
            output_record = dict(sample.raw_record)
            if task_type in OBJECT_GROUNDING_TASKS or task_type == BenchmarkTaskType.OBJECT_MOTION_PREDICTION:
                prepared_scene = evaluator.prepare_sample(sample, baseline_mode=baseline_mode)
                missing_objects += sum(1 for object_id in object_ids if prepared_scene.get_object(object_id) is None)
                output_record["outputs"] = output_text_for_task(
                    task_type,
                    prepared_scene,
                    object_ids,
                    prediction_answer_text(prediction),
                )
            else:
                # Q6/Q8/Q9 style tasks do not require graph/object grounding export text synthesis.
                output_record["outputs"] = prediction_answer_text(prediction)
            output_record["kg_prediction"] = {
                "sample_id": sample_id,
                "task_type": task_type.value,
                "object_ids": list(object_ids),
                "source_jsonl": str(prediction_path),
                "scenario_name": scenario_name,
            }
            handle.write(json.dumps(output_record) + "\n")
            exported_count += 1
            if progress_every > 0 and (exported_count == 1 or exported_count % progress_every == 0 or exported_count == total_predictions):
                print(
                    f"[INFO] export_progress task={task_type.value} "
                    f"{exported_count}/{total_predictions} "
                    f"missing_samples={missing_samples} missing_objects={missing_objects}",
                    flush=True,
                )

    return {
        "task_type": task_type.value,
        "qa_type_id": qa_type_id,
        "source_jsonl": str(prediction_path),
        "output_jsonl": str(output_path),
        "exported_count": exported_count,
        "missing_samples": missing_samples,
        "missing_objects": missing_objects,
    }


def main() -> None:
    args = build_parser().parse_args()
    print("[INFO] export_qa_predictions: start", flush=True)
    manifest_path = resolve_repo_path(args.manifest)
    print(f"[INFO] manifest_path: {manifest_path}", flush=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = args.split or str(manifest.get("split", "val"))
    scenario_name = args.scenario_name or str(manifest.get("scenario_name", manifest_path.stem))
    output_dir = resolve_repo_path(args.output_dir)
    print(f"[INFO] split={split} scenario_name={scenario_name}", flush=True)
    print(f"[INFO] output_dir={output_dir}", flush=True)

    selected_task_types = parse_task_types(args.task_types)
    selected_task_set = set(selected_task_types)
    selected_qa_type_ids = set(int(value) for value in args.qa_type_ids)
    print(
        "[INFO] filters: "
        f"task_types={sorted(value.value for value in selected_task_set) if selected_task_set else 'all'} "
        f"qa_type_ids={sorted(selected_qa_type_ids) if selected_qa_type_ids else 'all'}",
        flush=True,
    )

    repository_root = resolve_v2vgot_root()
    print(f"[INFO] repository_root={repository_root}", flush=True)
    print("[INFO] adapter_init", flush=True)
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    print("[INFO] evaluator_init", flush=True)
    evaluator = V2VGoTQAPhase5AEvaluator(str(repository_root))
    print(f"[INFO] loading_samples split={split} file_name={args.file_name}", flush=True)
    samples = adapter.load_samples(split_name=split, file_name=args.file_name)
    print(f"[INFO] loaded_samples={len(samples)}", flush=True)
    print("[INFO] building_sample_index_maps", flush=True)
    samples_by_task_qa_and_id = {
        (sample.task_type, sample.qa_type_id, sample.sample_id): sample
        for sample in samples
    }
    print(f"[INFO] exact_index_size={len(samples_by_task_qa_and_id)}", flush=True)
    samples_by_task_qa_and_id.update(
        {
            (sample.task_type, None, sample.sample_id): sample
            for sample in samples
        }
    )
    print(f"[INFO] merged_index_size={len(samples_by_task_qa_and_id)}", flush=True)

    exported_runs: list[dict[str, object]] = []
    run_count = 0
    for run in manifest.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_count += 1
        task_type = BenchmarkTaskType(str(run["task_type"]))
        print(
            f"[INFO] considering_run#{run_count} "
            f"task={task_type.value} qa_type_id={run.get('qa_type_id')}",
            flush=True,
        )
        if task_type not in OFFICIAL_QA_TYPE_BY_TASK:
            print(f"[INFO] skip_run#{run_count}: unsupported task for official export", flush=True)
            continue
        if selected_task_set and task_type not in selected_task_set:
            print(f"[INFO] skip_run#{run_count}: filtered by task_type", flush=True)
            continue
        run_qa_type_id = run.get("qa_type_id", OFFICIAL_QA_TYPE_BY_TASK[task_type])
        if selected_qa_type_ids:
            if run_qa_type_id is None and len(selected_qa_type_ids) == 1:
                run = dict(run)
                run_qa_type_id = next(iter(selected_qa_type_ids))
                run["qa_type_id"] = run_qa_type_id
            if run_qa_type_id is None or int(run_qa_type_id) not in selected_qa_type_ids:
                print(f"[INFO] skip_run#{run_count}: filtered by qa_type_id", flush=True)
                continue
        print(
            f"[INFO] exporting_run#{run_count} task={task_type.value} qa_type_id={run_qa_type_id}",
            flush=True,
        )
        exported_runs.append(
            export_task(
                task_type=task_type,
                run=run,
                samples_by_task_qa_and_id=samples_by_task_qa_and_id,
                evaluator=evaluator,
                output_dir=output_dir,
                baseline_mode=args.baseline_mode,
                scenario_name=scenario_name,
                progress_every=250,
            )
        )
        last = exported_runs[-1]
        print(
            f"[INFO] exported_run#{run_count} task={last['task_type']} "
            f"exported={last['exported_count']} missing_samples={last['missing_samples']} "
            f"missing_objects={last['missing_objects']}",
            flush=True,
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
    print(f"[INFO] export_manifest_written={export_manifest_path}", flush=True)

    task_scope = ",".join(sorted(str(run["task_type"]) for run in exported_runs)) if exported_runs else "none"
    print("=" * 72)
    print("Official QA Export")
    print("=" * 72)
    print(f"task_scope: {task_scope}")
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
