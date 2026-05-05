#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

Q_BASELINES = {
    "q1_notable_objects": 0.5250,
    "q2_occluding_objects": 0.3010,
    "q5_object_motion_prediction": 8.0500,
    "q3_invisible_objects": 0.4400,
    "q4_planning_awareness": 0.6080,
    "q8_control_settings": 0.0876,
    "q9_future_trajectory": 2.6200,
}


@dataclass(frozen=True)
class TaskRunConfig:
    name: str
    task_type: str
    metric_label: str
    higher_is_better: bool
    extra_args: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run val-split official QA evaluations for frozen Q1/Q2/Q3/Q4/Q5/Q8/Q9 "
            "and print report-ready baseline comparison tables."
        )
    )
    parser.add_argument(
        "--manifest-json",
        default="",
        help="Path to e2e_model_manifest.json from the train pipeline. Defaults to latest under outputs/e2e_runs.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--v2vgot-root", default="")
    return parser


def run(command: Sequence[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def summary_json_path(purpose: str, scenario_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / f"phase8_{purpose}"
        / "official_eval_reports"
        / f"{scenario_name}_official_export_manifest_official_qa_eval_summary.json"
    )


def summary_markdown_path(purpose: str, scenario_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / f"phase8_{purpose}"
        / "official_eval_reports"
        / f"{scenario_name}_official_export_manifest_official_qa_eval_summary.md"
    )


def export_manifest_path(purpose: str, scenario_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / f"phase8_{purpose}"
        / "official_exports"
        / f"{scenario_name}_official_export_manifest.json"
    )


def latest_manifest() -> Path:
    base = REPO_ROOT / "outputs" / "e2e_runs"
    candidates = sorted(base.glob("*/e2e_model_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No e2e model manifest found under outputs/e2e_runs.")
    return candidates[0]


def build_val_task_configs(model_paths: dict[str, str]) -> tuple[TaskRunConfig, ...]:
    return (
        TaskRunConfig(
            name="q1_notable_objects",
            task_type="notable_objects",
            metric_label="F1@0.5m",
            higher_is_better=True,
            extra_args=("--notable-ranker", "heuristic"),
        ),
        TaskRunConfig(
            name="q2_occluding_objects",
            task_type="occluding_objects",
            metric_label="F1@0.5m",
            higher_is_better=True,
            extra_args=("--occluding-ranker", "risk_adaptive"),
        ),
        TaskRunConfig(
            name="q5_object_motion_prediction",
            task_type="object_motion_prediction",
            metric_label="L2 Avg All (m)",
            higher_is_better=False,
        ),
        TaskRunConfig(
            name="q3_invisible_objects",
            task_type="invisible_objects",
            metric_label="F1@0.5m",
            higher_is_better=True,
            extra_args=(
                "--invisible-ranker",
                "logreg_acceptor",
                "--invisible-acceptor-model-json",
                model_paths["q3_model_json"],
                "--invisible-max-results",
                "1",
                "--invisible-shortlist-size",
                "64",
                "--invisible-max-distance-to-trajectory",
                "8.0",
                "--invisible-min-risk",
                "0.58",
                "--invisible-min-relative-to-best",
                "0.75",
            ),
        ),
        TaskRunConfig(
            name="q4_planning_awareness",
            task_type="planning_awareness",
            metric_label="F1@0.5m",
            higher_is_better=True,
            extra_args=(
                "--planning-ranker",
                "relational_importance",
                "--planning-selection-policy",
                "trajectory_calibrated_acceptor",
                "--planning-selection-source",
                "orchestrator",
                "--planning-acceptor-model-json",
                model_paths["q4_model_json"],
            ),
        ),
        TaskRunConfig(
            name="q8_control_settings",
            task_type="control_settings",
            metric_label="Action L1 (edit_dist/8)",
            higher_is_better=False,
            extra_args=(
                "--control-selection-policy",
                "linear_classifier",
                "--control-model-json",
                model_paths["q8_model_json"],
            ),
        ),
        TaskRunConfig(
            name="q9_future_trajectory",
            task_type="future_trajectory",
            metric_label="L2 Avg All (m)",
            higher_is_better=False,
            extra_args=(
                "--future-trajectory-model-json",
                model_paths["q9_model_json"],
            ),
        ),
    )


def extract_primary_metric(task_name: str, summary_payload: dict[str, object]) -> float:
    runs = summary_payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Missing runs in summary for {task_name}")
    run0 = runs[0]
    if not isinstance(run0, dict):
        raise ValueError(f"Invalid run payload for {task_name}")
    metrics = run0.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics in summary for {task_name}")

    if task_name in {"q1_notable_objects", "q2_occluding_objects", "q3_invisible_objects", "q4_planning_awareness"}:
        localization = metrics.get("localization")
        if not isinstance(localization, dict):
            raise ValueError(f"Missing localization metrics for {task_name}")
        at_0p5 = localization.get("0.5")
        if not isinstance(at_0p5, dict):
            raise ValueError(f"Missing 0.5m localization metric for {task_name}")
        f1_value = at_0p5.get("f1")
        if not isinstance(f1_value, (float, int)):
            raise ValueError(f"Missing localization F1@0.5m for {task_name}")
        return float(f1_value)

    if task_name == "q8_control_settings":
        action_edit_dist = metrics.get("action_edit_dist")
        if not isinstance(action_edit_dist, (float, int)):
            raise ValueError("Missing action_edit_dist for Q8")
        return float(action_edit_dist) / 8.0

    if task_name in {"q5_object_motion_prediction", "q9_future_trajectory"}:
        l2_all = metrics.get("l2_error_avg_all")
        if not isinstance(l2_all, (float, int)):
            raise ValueError(f"Missing l2_error_avg_all for {task_name}")
        return float(l2_all)

    raise ValueError(f"Unknown task metric extraction: {task_name}")


def relative_improvement(ours: float, baseline: float, higher_is_better: bool) -> float:
    if baseline == 0.0:
        return 0.0
    if higher_is_better:
        return ((ours - baseline) / baseline) * 100.0
    return ((baseline - ours) / baseline) * 100.0


def table_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Task | Metric | Ours | Baseline | Relative Improvement |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"`{row['task']}` | `{row['metric']}` | `{row['ours']:.6f}` | `{row['baseline']:.6f}` | `{row['relative_improvement_pct']:+.2f}%` |"
        )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest_json).expanduser().resolve() if args.manifest_json else latest_manifest()
    manifest = read_json(manifest_path)
    run_name = str(manifest.get("run_name", "unknown_e2e_run"))
    model_paths = manifest.get("model_paths")
    if not isinstance(model_paths, dict):
        raise ValueError(f"Invalid model_paths in manifest: {manifest_path}")

    v2vgot_root = args.v2vgot_root or str(manifest.get("v2vgot_root", "/workspace/repos/V2V-GoT"))
    e2e_root = REPO_ROOT / "outputs" / "e2e_runs" / run_name
    archive_dir = e2e_root / "val_eval"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("E2E Validation Report")
    print("=" * 72)
    print(f"manifest: {manifest_path}")
    print(f"run_name: {run_name}")
    print(f"v2vgot_root: {v2vgot_root}")
    print(f"workers: {args.workers}")

    task_configs = build_val_task_configs({k: str(v) for k, v in model_paths.items()})
    results: list[dict[str, object]] = []
    run_records: list[dict[str, str]] = []
    for task in task_configs:
        scenario = f"e2e_{run_name}_val_{task.name}"
        command = [
            sys.executable,
            "scripts/run_qa_split_pipeline.py",
            "--purpose",
            "val_report",
            "--split",
            "val",
            "--task-type",
            task.task_type,
            "--scenario-name",
            scenario,
            "--baseline-mode",
            "cooperative",
            "--v2vgot-root",
            v2vgot_root,
            "--workers",
            str(args.workers),
            "--progress-every",
            str(args.progress_every),
            *task.extra_args,
        ]
        run(command)

        src_json = summary_json_path("val_report", scenario)
        src_md = summary_markdown_path("val_report", scenario)
        src_export_manifest = export_manifest_path("val_report", scenario)
        copy_if_exists(src_json, archive_dir / src_json.name)
        copy_if_exists(src_md, archive_dir / src_md.name)
        copy_if_exists(src_export_manifest, archive_dir / src_export_manifest.name)
        run_records.append(
            {
                "task_name": task.name,
                "task_type": task.task_type,
                "scenario_name": scenario,
                "summary_json": str(src_json),
                "summary_markdown": str(src_md),
                "official_export_manifest": str(src_export_manifest),
            }
        )

        summary = read_json(src_json)
        ours = extract_primary_metric(task.name, summary)
        baseline = Q_BASELINES[task.name]
        rel = relative_improvement(ours, baseline, task.higher_is_better)
        results.append(
            {
                "task": task.name,
                "metric": task.metric_label,
                "ours": ours,
                "baseline": baseline,
                "relative_improvement_pct": rel,
            }
        )

    markdown = table_markdown(results)
    print()
    print(markdown)

    report = {
        "run_name": run_name,
        "manifest_json": str(manifest_path),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "results": results,
        "runs": run_records,
        "baselines": Q_BASELINES,
    }
    report_json = archive_dir / "e2e_validation_summary.json"
    report_md = archive_dir / "e2e_validation_summary.md"
    write_json(report_json, report)
    report_md.write_text(markdown + "\n", encoding="utf-8")
    print()
    print(f"saved_json: {report_json}")
    print(f"saved_markdown: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
