#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.e2e.run_e2e_validation_report import (  # noqa: E402
    build_val_task_configs,
    extract_primary_metric,
    latest_manifest,
    read_json,
)


@dataclass(frozen=True)
class AblationModeConfig:
    label: str
    baseline_mode: str
    graph_ablation_mode: str
    description: str


ALL_ABLATION_MODES: tuple[AblationModeConfig, ...] = (
    AblationModeConfig("full", "cooperative", "full", "Full cooperative G-MARK."),
    AblationModeConfig("no_provenance", "cooperative", "no_provenance", "Remove source-agent provenance/support traces."),
    AblationModeConfig(
        "no_candidate_retention",
        "cooperative",
        "no_candidate_retention",
        "Drop retained candidate tracks.",
    ),
    AblationModeConfig(
        "no_uncertainty_conflict",
        "cooperative",
        "no_uncertainty_conflict",
        "Neutralize uncertainty and conflict scores.",
    ),
    AblationModeConfig(
        "no_graph_relations",
        "cooperative",
        "no_graph_relations",
        "Remove derived relation edges.",
    ),
    AblationModeConfig("ego_only_graph", "ego_only", "full", "Use only asking-agent evidence."),
    AblationModeConfig(
        "flat_non_graph_readout",
        "cooperative",
        "flat_non_graph_readout",
        "Keep a flat object list while removing graph-specific signals.",
    ),
)

DEFAULT_ABLATION_MODE_LABELS: tuple[str, ...] = (
    "full",
    "no_provenance",
    "no_candidate_retention",
    "no_uncertainty_conflict",
    "ego_only_graph",
    "flat_non_graph_readout",
)

DEFAULT_COMPONENT_TASK_NAMES: tuple[str, ...] = (
    "q1_notable_objects",
    "q2_occluding_objects",
    "q5_object_motion_prediction",
    "q7_object_motion_prediction",
    "q3_invisible_objects",
    "q4_planning_awareness",
    "q8_control_settings",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated G-MARK ablations. Produces validation-only metrics from an existing "
            "manifest and optional train+validation metrics from ablation-specific train runs."
        )
    )
    parser.add_argument("--run-name", default=f"gmark_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--manifest-json",
        default="",
        help="Existing full G-MARK e2e_model_manifest.json for validation-only ablations.",
    )
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        default=[],
        choices=tuple(mode.label for mode in ALL_ABLATION_MODES),
        help=(
            "Ablation mode to run. Repeatable. Defaults to the component-ablation set "
            "without no_graph_relations."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        default=[],
        choices=(
            "q1_notable_objects",
            "q2_occluding_objects",
            "q5_object_motion_prediction",
            "q6_agent_motion_prediction",
            "q7_object_motion_prediction",
            "q3_invisible_objects",
            "q4_planning_awareness",
            "q8_control_settings",
            "q9_future_trajectory",
        ),
        help=(
            "Task metric to include. Repeatable. Defaults to KG component-sensitive "
            "tasks: Q1/Q2/Q3/Q4/Q5/Q7/Q8."
        ),
    )
    parser.add_argument("--skip-validation-only", action="store_true")
    parser.add_argument(
        "--run-trained-validation",
        action="store_false",
        dest="skip_trained_validation",
        help="Opt in to ablation-specific training plus validation. Disabled by default.",
    )
    parser.add_argument("--skip-trained-validation", action="store_true")
    parser.set_defaults(skip_trained_validation=True)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run(command: Sequence[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def qa_record_key(record: dict[str, object]) -> tuple[str, str, str, str]:
    conversations = record.get("conversations", [])
    question = ""
    if isinstance(conversations, list) and conversations:
        first = conversations[0]
        if isinstance(first, dict):
            question = str(first.get("value", ""))
    return (
        str(record.get("scenario_index", "")),
        str(record.get("global_timestamp_index", record.get("local_timestamp_index", ""))),
        str(record.get("qa_type_id", "")),
        question,
    )


def split_qa_json_path(v2vgot_root: str, split: str) -> Path:
    split_dir = "train_no_fusion_keep_all" if split == "train" else "no_fusion_keep_all"
    return (
        Path(v2vgot_root).expanduser().resolve()
        / "DMSTrack"
        / "V2V4Real"
        / "official_models"
        / split_dir
        / "npy"
        / "co_llm"
        / "v2v4real_3d_grounding_qa_dataset_v2vgot.json"
    )


def write_val_filtered_train_json(*, v2vgot_root: str, run_root: Path) -> tuple[Path, dict[str, object]]:
    train_json = split_qa_json_path(v2vgot_root, "train")
    val_json = split_qa_json_path(v2vgot_root, "val")
    with train_json.open("r", encoding="utf-8") as handle:
        train_records = json.load(handle)
    with val_json.open("r", encoding="utf-8") as handle:
        val_records = json.load(handle)
    if not isinstance(train_records, list) or not isinstance(val_records, list):
        raise ValueError("Expected train and val QA JSON files to contain lists.")

    val_keys = {
        qa_record_key(record)
        for record in val_records
        if isinstance(record, dict)
    }
    filtered_records = [
        record
        for record in train_records
        if isinstance(record, dict) and qa_record_key(record) not in val_keys
    ]
    removed_count = len(train_records) - len(filtered_records)
    output_path = run_root / "filtered_train" / "v2v4real_3d_grounding_qa_dataset_v2vgot_train_minus_val_overlap.json"
    report = {
        "source_train_json": str(train_json),
        "source_val_json": str(val_json),
        "filtered_train_json": str(output_path),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "removed_overlap_records": removed_count,
        "filtered_train_records": len(filtered_records),
        "overlap_key": "scenario_index + timestamp + qa_type_id + exact question text",
    }
    write_json(output_path, filtered_records)
    write_json(run_root / "filtered_train" / "train_minus_val_overlap_report.json", report)
    print("[INFO] Wrote val-filtered train QA JSON")
    print(f"[INFO]   removed_overlap_records: {removed_count}")
    print(f"[INFO]   filtered_train_records: {len(filtered_records)}")
    print(f"[INFO]   filtered_train_json: {output_path}")
    return output_path, report


def selected_modes(raw_modes: list[str]) -> tuple[AblationModeConfig, ...]:
    selected_labels = tuple(raw_modes) if raw_modes else DEFAULT_ABLATION_MODE_LABELS
    requested = set(selected_labels)
    return tuple(mode for mode in ALL_ABLATION_MODES if mode.label in requested)


def selected_task_configs(task_configs: tuple[object, ...], raw_tasks: list[str]) -> tuple[object, ...]:
    selected_names = tuple(raw_tasks) if raw_tasks else DEFAULT_COMPONENT_TASK_NAMES
    requested = set(selected_names)
    return tuple(task for task in task_configs if str(task.name) in requested)  # type: ignore[attr-defined]


def task_metric_labels(task_configs: tuple[object, ...]) -> dict[str, str]:
    return {str(task.name): str(task.metric_label) for task in task_configs}  # type: ignore[attr-defined]


def run_validation_for_manifest(
    *,
    manifest_path: Path,
    mode: AblationModeConfig,
    output_root: Path,
    v2vgot_root: str,
    workers: int,
    progress_every: int,
    task_names: set[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = read_json(manifest_path)
    model_paths = manifest.get("model_paths")
    if not isinstance(model_paths, dict):
        raise ValueError(f"Invalid model_paths in manifest: {manifest_path}")

    task_configs = tuple(
        task
        for task in build_val_task_configs({key: str(value) for key, value in model_paths.items()})
        if str(task.name) in task_names
    )
    results: dict[str, object] = {}
    runs: list[dict[str, object]] = []

    for task in task_configs:
        scenario = f"{mode.label}_val_{task.name}"  # type: ignore[attr-defined]
        command = [
            sys.executable,
            "scripts/run_qa_split_pipeline.py",
            "--purpose",
            "val_report",
            "--split",
            "val",
            "--task-type",
            str(task.task_type),  # type: ignore[attr-defined]
            "--scenario-name",
            scenario,
            "--baseline-mode",
            mode.baseline_mode,
            "--graph-ablation-mode",
            mode.graph_ablation_mode,
            "--output-root",
            str(output_root),
            "--v2vgot-root",
            v2vgot_root,
            "--workers",
            str(workers),
            "--progress-every",
            str(progress_every),
            *task.extra_args,  # type: ignore[attr-defined]
        ]
        run(command)
        summary_path = (
            output_root
            / "official_eval_reports"
            / f"{scenario}_official_export_manifest_official_qa_eval_summary.json"
        )
        summary = read_json(summary_path)
        metric = extract_primary_metric(str(task.name), summary)  # type: ignore[attr-defined]
        results[str(task.name)] = metric  # type: ignore[attr-defined]
        runs.append(
            {
                "task": str(task.name),  # type: ignore[attr-defined]
                "scenario": scenario,
                "summary_json": str(summary_path),
            }
        )

    return results, runs


def run_training_for_mode(
    *,
    mode: AblationModeConfig,
    run_root: Path,
    v2vgot_root: str,
    train_file_name: str,
    workers: int,
    progress_every: int,
) -> Path:
    trained_base = run_root / "trained_e2e_runs"
    trained_run_name = f"{mode.label}_trained"
    command = [
        sys.executable,
        "scripts/e2e/run_gmark_ablation_train_pipeline.py",
        "--run-name",
        trained_run_name,
        "--output-base",
        str(trained_base),
        "--v2vgot-root",
        v2vgot_root,
        "--train-file-name",
        train_file_name,
        "--workers",
        str(workers),
        "--progress-every",
        str(progress_every),
        "--baseline-mode",
        mode.baseline_mode,
        "--graph-ablation-mode",
        mode.graph_ablation_mode,
    ]
    run(command)
    return trained_base / trained_run_name / "e2e_model_manifest.json"


def metric_table_markdown(
    *,
    title: str,
    results_by_mode: dict[str, dict[str, object]],
    metric_labels: dict[str, str],
    modes: tuple[AblationModeConfig, ...],
) -> str:
    tasks = list(metric_labels)
    lines = [f"## {title}", ""]
    lines.append("| Ablation | " + " | ".join(f"`{task}`" for task in tasks) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in tasks) + " |")
    for mode in modes:
        if mode.label not in results_by_mode:
            continue
        row = results_by_mode[mode.label]
        cells = []
        for task in tasks:
            value = row.get(task)
            if isinstance(value, (float, int)):
                cells.append(f"`{float(value):.6f}`")
            else:
                cells.append(f"`{value}`")
        lines.append(f"| `{mode.label}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Metric labels:")
    for task, label in metric_labels.items():
        lines.append(f"- `{task}`: `{label}`")
    return "\n".join(lines)


def interpretation_notes_markdown() -> str:
    return "\n".join(
        [
            "## Interpretation Notes",
            "",
            "- `ego_only_graph` is a strict asking-agent evidence baseline: only object tracks visible to the asking agent are retained.",
            "- `no_provenance` removes source-agent/observation traces and provenance-derived relation facts.",
            "- `no_uncertainty_conflict` removes direct uncertainty/conflict scores and conflict-derived relation facts.",
            "- `no_candidate_retention` removes retained/promoted candidate tracks before downstream task heads see the scene.",
            "- Q6 and Q9 are intentionally excluded from the default component-ablation table because their current heads are not primarily graph-evidence driven.",
            "- `no_graph_relations` is intentionally excluded from the default run until we evaluate relation-aware heads separately.",
        ]
    )


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest_json).expanduser().resolve() if args.manifest_json else latest_manifest()
    source_manifest = read_json(manifest_path)
    v2vgot_root = args.v2vgot_root or str(source_manifest.get("v2vgot_root", "/workspace/repos/V2V-GoT"))
    modes = selected_modes(args.modes)

    run_root = REPO_ROOT / "outputs" / "gmark_ablations" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    all_task_configs = build_val_task_configs({key: str(value) for key, value in source_manifest["model_paths"].items()})  # type: ignore[index]
    task_configs = selected_task_configs(all_task_configs, args.tasks)
    metric_labels = task_metric_labels(task_configs)
    task_names = set(metric_labels)

    print("=" * 72)
    print("G-MARK Ablation Report")
    print("=" * 72)
    print(f"run_name: {args.run_name}")
    print(f"run_root: {run_root}")
    print(f"source_manifest: {manifest_path}")
    print(f"v2vgot_root: {v2vgot_root}")
    print(f"modes: {[mode.label for mode in modes]}")
    print(f"tasks: {list(metric_labels)}")
    print(f"run_trained_validation: {not args.skip_trained_validation}")

    filtered_train_json = ""
    filtered_train_report: dict[str, object] = {}
    if not args.skip_trained_validation:
        filtered_train_path, filtered_train_report = write_val_filtered_train_json(
            v2vgot_root=v2vgot_root,
            run_root=run_root,
        )
        filtered_train_json = str(filtered_train_path)

    validation_only: dict[str, dict[str, object]] = {}
    trained_validation: dict[str, dict[str, object]] = {}
    run_records: dict[str, object] = {
        "validation_only": {},
        "trained_validation": {},
    }

    if not args.skip_validation_only:
        for mode in modes:
            try:
                mode_root = run_root / "validation_only" / mode.label
                results, runs = run_validation_for_manifest(
                    manifest_path=manifest_path,
                    mode=mode,
                    output_root=mode_root,
                    v2vgot_root=v2vgot_root,
                    workers=args.workers,
                    progress_every=args.progress_every,
                    task_names=task_names,
                )
                validation_only[mode.label] = results
                run_records["validation_only"][mode.label] = runs  # type: ignore[index]
            except Exception as exc:
                if args.fail_fast:
                    raise
                validation_only[mode.label] = {task: "ERR" for task in metric_labels}
                run_records["validation_only"][mode.label] = {"error": repr(exc)}  # type: ignore[index]
                print(f"[ERROR] validation-only mode={mode.label}: {exc}")

    if not args.skip_trained_validation:
        for mode in modes:
            try:
                trained_manifest = run_training_for_mode(
                    mode=mode,
                    run_root=run_root,
                    v2vgot_root=v2vgot_root,
                    train_file_name=filtered_train_json,
                    workers=args.workers,
                    progress_every=args.progress_every,
                    task_names=task_names,
                )
                mode_root = run_root / "trained_validation" / mode.label
                results, runs = run_validation_for_manifest(
                    manifest_path=trained_manifest,
                    mode=mode,
                    output_root=mode_root,
                    v2vgot_root=v2vgot_root,
                    workers=args.workers,
                    progress_every=args.progress_every,
                )
                trained_validation[mode.label] = results
                run_records["trained_validation"][mode.label] = {  # type: ignore[index]
                    "manifest_json": str(trained_manifest),
                    "runs": runs,
                }
            except Exception as exc:
                if args.fail_fast:
                    raise
                trained_validation[mode.label] = {task: "ERR" for task in metric_labels}
                run_records["trained_validation"][mode.label] = {"error": repr(exc)}  # type: ignore[index]
                print(f"[ERROR] trained-validation mode={mode.label}: {exc}")

    markdown_sections = [
        f"# G-MARK Ablation Report `{args.run_name}`",
        "",
        f"- source manifest: `{manifest_path}`",
        f"- v2vgot_root: `{v2vgot_root}`",
        f"- created_at_utc: `{datetime.utcnow().isoformat(timespec='seconds')}Z`",
    ]
    if filtered_train_report:
        markdown_sections.extend(
            [
                f"- filtered train JSON: `{filtered_train_report.get('filtered_train_json', '')}`",
                f"- train records removed as val overlap: `{filtered_train_report.get('removed_overlap_records', 0)}`",
            ]
        )
    markdown_sections.extend(["", "## Modes", ""])
    for mode in modes:
        markdown_sections.append(
            f"- `{mode.label}`: baseline=`{mode.baseline_mode}`, graph=`{mode.graph_ablation_mode}`. {mode.description}"
        )
    markdown_sections.append("")
    if validation_only:
        markdown_sections.append(
            metric_table_markdown(
                title="Validation-Only Ablations",
                results_by_mode=validation_only,
                metric_labels=metric_labels,
                modes=modes,
            )
        )
    if trained_validation:
        markdown_sections.append(
            metric_table_markdown(
                title="Train+Validation Ablations",
                results_by_mode=trained_validation,
                metric_labels=metric_labels,
                modes=modes,
            )
        )
    markdown_sections.append(interpretation_notes_markdown())
    markdown = "\n".join(markdown_sections) + "\n"

    summary = {
        "run_name": args.run_name,
        "source_manifest": str(manifest_path),
        "v2vgot_root": v2vgot_root,
        "modes": [mode.__dict__ for mode in modes],
        "metric_labels": metric_labels,
        "filtered_train": filtered_train_report,
        "validation_only": validation_only,
        "trained_validation": trained_validation,
        "runs": run_records,
    }
    write_json(run_root / "ablation_summary.json", summary)
    (run_root / "ablation_summary.md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"saved_json: {run_root / 'ablation_summary.json'}")
    print(f"saved_markdown: {run_root / 'ablation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
