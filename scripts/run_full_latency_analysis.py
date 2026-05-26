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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


@dataclass(frozen=True)
class TaskConfig:
    name: str
    task_type: str
    qa_type_ids: tuple[int, ...] = ()
    extra_args: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full Q1-Q9 latency analysis and build one exhaustive per-component table."
        )
    )
    parser.add_argument("--run-name", default=f"latency_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--purpose", default="train_dev", choices=("train_dev", "val_report"))
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--temporal-execution-mode",
        default="serial",
        choices=("serial", "parallel_prefetch"),
        help="Temporal execution mode passed to latency runs.",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--manifest-json",
        default="",
        help="Optional e2e_model_manifest.json to auto-load Q3/Q4/Q5/Q6/Q7/Q8/Q9 model paths.",
    )
    parser.add_argument("--q3-model-json", default="")
    parser.add_argument("--q4-model-json", default="")
    parser.add_argument("--q5-model-json", default="")
    parser.add_argument("--q6-model-json", default="")
    parser.add_argument("--q7-model-json", default="")
    parser.add_argument("--q8-model-json", default="")
    parser.add_argument("--skip-official-eval", action="store_true", default=True)
    return parser


def run(command: Sequence[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_paths(args: argparse.Namespace) -> dict[str, str]:
    model_paths = {
        "q3_model_json": args.q3_model_json,
        "q4_model_json": args.q4_model_json,
        "q5_model_json": args.q5_model_json,
        "q6_model_json": args.q6_model_json,
        "q7_model_json": args.q7_model_json,
        "q8_model_json": args.q8_model_json,
        "q9_model_json": "",
    }
    if not args.manifest_json:
        return model_paths
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    manifest = read_json(manifest_path)
    manifest_model_paths = manifest.get("model_paths", {})
    if not isinstance(manifest_model_paths, dict):
        return model_paths
    for key in tuple(model_paths.keys()):
        if not model_paths[key]:
            value = manifest_model_paths.get(key, "")
            model_paths[key] = str(value) if value is not None else ""
    return model_paths


def resolve_v2vgot_root(raw: str) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def ensure_clean_q9_model(*, run_root: Path, v2vgot_root: Path, workers: int, progress_every: int) -> Path:
    clean_dir = run_root / "q9_clean_model"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_run_name = f"{run_root.name}_q9_clean_model"
    command = [
        sys.executable,
        "scripts/run_gmark_table1_q9_clean_eval.py",
        "--run-name",
        clean_run_name,
        "--v2vgot-root",
        str(v2vgot_root),
        "--output-root",
        str(clean_dir),
        "--workers",
        str(workers),
        "--progress-every",
        str(progress_every),
        "--skip-official-eval",
        "--skip-reported-rows",
    ]
    run(command)
    model_path = clean_dir / f"{clean_run_name}_clean_q9_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected clean Q9 model not found: {model_path}")
    return model_path


def build_tasks(model_paths: dict[str, str]) -> tuple[TaskConfig, ...]:
    q5_args: tuple[str, ...] = ("--object-motion-model-json", model_paths["q5_model_json"]) if model_paths["q5_model_json"] else ()
    q7_model = model_paths["q7_model_json"] or model_paths["q5_model_json"]
    q7_args: tuple[str, ...] = ("--object-motion-model-json", q7_model) if q7_model else ()
    q6_args: tuple[str, ...] = ("--agent-motion-model-json", model_paths["q6_model_json"]) if model_paths["q6_model_json"] else ()
    q3_args: tuple[str, ...] = ()
    if model_paths["q3_model_json"]:
        q3_args = (
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
        )
    q4_args: tuple[str, ...] = ()
    if model_paths["q4_model_json"]:
        q4_args = (
            "--planning-ranker",
            "relational_importance",
            "--planning-selection-policy",
            "trajectory_calibrated_acceptor",
            "--planning-selection-source",
            "orchestrator",
            "--planning-acceptor-model-json",
            model_paths["q4_model_json"],
        )
    q8_args: tuple[str, ...] = ()
    if model_paths["q8_model_json"]:
        q8_args = (
            "--control-selection-policy",
            "linear_classifier",
            "--control-model-json",
            model_paths["q8_model_json"],
        )
    q9_args: tuple[str, ...] = ("--future-trajectory-model-json", model_paths["q9_model_json"])

    return (
        TaskConfig(name="q1_notable_objects", task_type="notable_objects", extra_args=("--notable-ranker", "heuristic")),
        TaskConfig(name="q2_occluding_objects", task_type="occluding_objects", extra_args=("--occluding-ranker", "risk_adaptive")),
        TaskConfig(name="q3_invisible_objects", task_type="invisible_objects", qa_type_ids=(13,), extra_args=q3_args),
        TaskConfig(name="q4_planning_awareness", task_type="planning_awareness", qa_type_ids=(14,), extra_args=q4_args),
        TaskConfig(name="q5_object_motion_prediction", task_type="object_motion_prediction", qa_type_ids=(15,), extra_args=q5_args),
        TaskConfig(name="q6_agent_motion_prediction", task_type="agent_motion_prediction", qa_type_ids=(16,), extra_args=q6_args),
        TaskConfig(name="q7_object_motion_prediction", task_type="object_motion_prediction", qa_type_ids=(17,), extra_args=q7_args),
        TaskConfig(name="q8_control_settings", task_type="control_settings", qa_type_ids=(18,), extra_args=q8_args),
        TaskConfig(name="q9_future_trajectory", task_type="future_trajectory", qa_type_ids=(19,), extra_args=q9_args),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = int(round((len(sorted_values) - 1) * q))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int | None, str], list[float]] = {}
    samples_by_task: dict[tuple[str, int | None], int] = {}
    for row in rows:
        task_type = str(row.get("task_type", "unknown"))
        qa_type_id = row.get("qa_type_id")
        task_key = (task_type, int(qa_type_id) if isinstance(qa_type_id, int) else None)
        samples_by_task[task_key] = samples_by_task.get(task_key, 0) + 1
        timings = row.get("timings_ms")
        if not isinstance(timings, dict):
            continue
        for component, value in timings.items():
            if not isinstance(component, str) or not isinstance(value, (float, int)):
                continue
            key = (task_key[0], task_key[1], component)
            grouped.setdefault(key, []).append(float(value))

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1] if item[1] is not None else -1, item[2])):
        task_type, qa_type_id, component = key
        values = grouped[key]
        values_sorted = sorted(values)
        summary_rows.append(
            {
                "task_type": task_type,
                "qa_type_id": qa_type_id,
                "sample_count": samples_by_task.get((task_type, qa_type_id), 0),
                "component": component,
                "avg_ms": sum(values) / float(len(values)) if values else 0.0,
                "p50_ms": percentile(values_sorted, 0.50),
                "p90_ms": percentile(values_sorted, 0.90),
            }
        )
    return summary_rows


def to_markdown(summary_rows: list[dict[str, object]], run_name: str, combined_jsonl: Path) -> str:
    lines = [
        f"# Full Latency Analysis `{run_name}`",
        "",
        f"- combined_latency_jsonl: `{combined_jsonl}`",
        "",
        "| task_type | qa_type_id | samples | component | avg_ms | p50_ms | p90_ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task_type"]),
                    str(row["qa_type_id"]),
                    str(row["sample_count"]),
                    str(row["component"]),
                    f"{float(row['avg_ms']):.3f}",
                    f"{float(row['p50_ms']):.3f}",
                    f"{float(row['p90_ms']):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser() if args.output_root else (REPO_ROOT / "outputs" / "latency_analysis")
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    run_root = output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    v2vgot_root = resolve_v2vgot_root(args.v2vgot_root)
    model_paths = resolve_model_paths(args)
    clean_q9_model_path = ensure_clean_q9_model(
        run_root=run_root,
        v2vgot_root=v2vgot_root,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    model_paths["q9_model_json"] = str(clean_q9_model_path)
    tasks = build_tasks(model_paths)
    all_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []

    for task in tasks:
        scenario = f"{args.run_name}_{task.name}"
        latency_jsonl = run_root / f"{scenario}_latency.jsonl"
        command = [
            sys.executable,
            "scripts/run_qa_split_pipeline.py",
            "--purpose",
            args.purpose,
            "--split",
            args.split,
            "--task-type",
            task.task_type,
            "--scenario-name",
            scenario,
            "--baseline-mode",
            "cooperative",
            "--temporal-execution-mode",
            args.temporal_execution_mode,
            "--workers",
            str(args.workers),
            "--progress-every",
            str(args.progress_every),
            "--latency-jsonl",
            str(latency_jsonl),
        ]
        command.extend(["--v2vgot-root", str(v2vgot_root)])
        for qa_type_id in task.qa_type_ids:
            command.extend(["--qa-type-id", str(qa_type_id)])
        command.extend(task.extra_args)
        if args.skip_official_eval:
            command.append("--skip-official-eval")
        run(command)

        task_rows = read_jsonl(latency_jsonl)
        all_rows.extend(task_rows)
        run_records.append(
            {
                "task_name": task.name,
                "task_type": task.task_type,
                "qa_type_ids": list(task.qa_type_ids),
                "scenario_name": scenario,
                "latency_jsonl": str(latency_jsonl),
                "sample_count": len(task_rows),
            }
        )

    combined_jsonl = run_root / f"{args.run_name}_combined_latency.jsonl"
    with combined_jsonl.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    summary_rows = summarize_rows(all_rows)
    summary_json = {
        "run_name": args.run_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "purpose": args.purpose,
        "split": args.split,
        "workers": args.workers,
        "progress_every": args.progress_every,
        "v2vgot_root": args.v2vgot_root,
        "resolved_v2vgot_root": str(v2vgot_root),
        "manifest_json": args.manifest_json,
        "q9_clean_model_json": str(clean_q9_model_path),
        "combined_latency_jsonl": str(combined_jsonl),
        "run_records": run_records,
        "rows": summary_rows,
    }
    summary_json_path = run_root / f"{args.run_name}_latency_summary.json"
    summary_md_path = run_root / f"{args.run_name}_latency_summary.md"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    summary_md_path.write_text(to_markdown(summary_rows, args.run_name, combined_jsonl), encoding="utf-8")

    print(f"saved_combined_latency: {combined_jsonl}")
    print(f"saved_summary_json: {summary_json_path}")
    print(f"saved_summary_markdown: {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
