#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402
from scripts.build_v2vgot_table1_reproduction import REPORTED_TABLE_I_ROWS  # noqa: E402

TABLE1_Q9_FILE_NAME = "v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json"
DEFAULT_TRAIN_FILE_NAME = "v2v4real_3d_grounding_qa_dataset_v2vgot.json"
COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
POSITION_RE = re.compile(
    r"I am\s+(?P<agent>[A-Za-z0-9_]+)\s+at\s+"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
)
_WORKER_BASELINE_MODE = "cooperative"
_WORKER_GRAPH_ABLATION_MODE = "full"
_WORKER_MODEL: CleanQ9Model | None = None


@dataclass(frozen=True)
class PlanningMetrics:
    l2_1s: float | None
    l2_2s: float | None
    l2_3s: float | None
    l2_avg: float | None
    cr_1s: float | None
    cr_2s: float | None
    cr_3s: float | None
    cr_avg: float | None


@dataclass(frozen=True)
class CleanQ9Model:
    feature_names: tuple[str, ...]
    feature_mean: list[float]
    feature_scale: list[float]
    coefficients: list[list[float]]
    ridge_alpha: float
    train_rows: int
    usable_rows: int
    filtered_overlap_rows: int


def init_worker(
    v2vgot_root: str,
    graph_ablation_mode: str,
    baseline_mode: str,
    model: CleanQ9Model | None = None,
) -> None:
    global _WORKER_BASELINE_MODE, _WORKER_GRAPH_ABLATION_MODE, _WORKER_MODEL
    _WORKER_BASELINE_MODE = baseline_mode
    _WORKER_GRAPH_ABLATION_MODE = graph_ablation_mode
    _WORKER_MODEL = model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a clean G-MARK Table-I/Q9 future-trajectory row. "
            "This runner intentionally excludes target-derived Q9 metadata "
            "(`dist`, `angle`, `suggested_*`, `future_trajectory_str_*`) from "
            "the model inputs."
        )
    )
    parser.add_argument("--run-name", default=f"gmark_table1_q9_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--output-root", default="outputs/v2vgot_table1_reproduction/gmark_q9_clean")
    parser.add_argument("--train-file-name", default=DEFAULT_TRAIN_FILE_NAME)
    parser.add_argument("--val-file-name", default=TABLE1_Q9_FILE_NAME)
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument(
        "--graph-ablation-mode",
        default="full",
        choices=(
            "full",
            "no_provenance",
            "no_candidate_retention",
            "no_uncertainty_conflict",
            "no_graph_relations",
            "flat_non_graph_readout",
        ),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument(
        "--skip-train-start-index",
        type=int,
        default=0,
        help=(
            "Zero-based index in filtered train samples where a temporary skip window starts. "
            "Use with --skip-train-count."
        ),
    )
    parser.add_argument(
        "--skip-train-count",
        type=int,
        default=0,
        help="Number of filtered train samples to skip from --skip-train-start-index.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for graph feature extraction and prediction. Use 1 for serial execution.",
    )
    parser.add_argument(
        "--max-inflight-per-worker",
        type=int,
        default=2,
        help=(
            "Maximum queued multiprocessing tasks per worker. Lower values reduce "
            "memory and make progress less vulnerable to one slow sample."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--allow-train-val-overlap",
        action="store_true",
        help="By default, exact QA-key overlaps with the released validation Q9 file are removed from train.",
    )
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--skip-reported-rows", action="store_true")
    return parser


def run(command: Sequence[str], *, v2vgot_root: str) -> None:
    print()
    print("$ " + " ".join(command))
    env = dict(os.environ)
    env["V2VGOT_ROOT"] = v2vgot_root
    subprocess.run(command, cwd=str(REPO_ROOT), check=True, env=env)


def resolve_output_root(raw_output_root: str) -> Path:
    output_root = Path(raw_output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def parse_current_position(sample: BenchmarkSample) -> tuple[float, float]:
    match = POSITION_RE.search(sample.scene.raw_question)
    if match:
        return float(match.group("x")), float(match.group("y"))
    asker = next(
        (agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id),
        None,
    )
    if asker is None:
        return (0.0, 0.0)
    return (asker.pose.position.x, asker.pose.position.y)


def parse_waypoints(text: str, limit: int = 6) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in COORD_RE.findall(text)[:limit])


def raw_answer(sample: BenchmarkSample) -> str:
    conversations = sample.raw_record.get("conversations", [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        return ""
    second = conversations[1]
    if not isinstance(second, dict):
        return ""
    value = second.get("value", "")
    return value if isinstance(value, str) else ""


def qa_overlap_key(sample: BenchmarkSample) -> tuple[str, str, str, str]:
    return (
        str(sample.raw_record.get("scenario_index", "")),
        str(sample.raw_record.get("local_timestamp_index", "")),
        str(sample.raw_record.get("qa_type_id", sample.qa_type_id)),
        sample.scene.raw_question.strip(),
    )


def load_q9_samples(
    *,
    adapter: V2VGoTQABenchmarkAdapter,
    split_name: str,
    file_name: str,
    limit: int,
) -> tuple[BenchmarkSample, ...]:
    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name=split_name, file_name=file_name)
        if sample.task_type == BenchmarkTaskType.FUTURE_TRAJECTORY
    )
    if limit > 0:
        samples = samples[:limit]
    return samples


def clean_features(sample: BenchmarkSample) -> list[float]:
    current_x, current_y = parse_current_position(sample)
    asker_raw = str(sample.raw_record.get("asker_cav_id", "")).strip()
    asker_from_q = "1" if "I am CAV_1" in sample.scene.raw_question else ""
    asker_is_cav1 = 1.0 if (asker_raw == "1" or asker_from_q == "1") else 0.0
    return [
        1.0,
        current_x,
        current_y,
        asker_is_cav1,
    ]


def train_feature_target_worker(sample: BenchmarkSample) -> tuple[list[float], list[float]] | None:
    waypoints = parse_waypoints(raw_answer(sample))
    if len(waypoints) != 6:
        return None
    return clean_features(sample), [value for point in waypoints for value in point]


def unordered_parallel_map(
    *,
    function,
    items: Sequence[BenchmarkSample],
    workers: int,
    max_inflight_per_worker: int,
    initializer_args: tuple[object, ...],
    progress_every: int,
    progress_label: str,
):
    max_inflight = max(1, workers * max(1, max_inflight_per_worker))
    item_iter = iter(enumerate(items, start=1))
    pending = {}
    submitted = 0
    completed = 0

    def submit_next(pool: ProcessPoolExecutor) -> bool:
        nonlocal submitted
        try:
            original_index, item = next(item_iter)
        except StopIteration:
            return False
        pending[pool.submit(function, item)] = original_index
        submitted += 1
        return True

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=initializer_args,
    ) as pool:
        for _ in range(min(max_inflight, len(items))):
            submit_next(pool)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                original_index = pending.pop(future)
                completed += 1
                result = future.result()
                if progress_every > 0 and (
                    completed == 1 or completed % progress_every == 0 or completed == len(items)
                ):
                    print(
                        f"{progress_label}: completed={completed}/{len(items)} "
                        f"submitted={submitted}/{len(items)} last_input_index={original_index} "
                        f"workers={workers}",
                        flush=True,
                    )
                yield original_index, result
                submit_next(pool)


def feature_names() -> tuple[str, ...]:
    return (
        "bias",
        "current_x",
        "current_y",
        "asker_is_cav1",
    )


def train_clean_model(
    *,
    train_samples: tuple[BenchmarkSample, ...],
    val_samples: tuple[BenchmarkSample, ...],
    v2vgot_root: str,
    graph_ablation_mode: str,
    baseline_mode: str,
    ridge_alpha: float,
    allow_train_val_overlap: bool,
    progress_every: int,
    workers: int,
    max_inflight_per_worker: int,
    skip_train_start_index: int,
    skip_train_count: int,
) -> tuple[CleanQ9Model, dict[str, object]]:
    val_keys = {qa_overlap_key(sample) for sample in val_samples}
    filtered_samples: list[BenchmarkSample] = []
    filtered_overlap_rows = 0
    for sample in train_samples:
        if not allow_train_val_overlap and qa_overlap_key(sample) in val_keys:
            filtered_overlap_rows += 1
            continue
        filtered_samples.append(sample)

    skip_start = max(0, int(skip_train_start_index))
    skip_count = max(0, int(skip_train_count))
    skip_end = min(len(filtered_samples), skip_start + skip_count)
    skipped_train_rows = max(0, skip_end - skip_start)
    if skipped_train_rows > 0:
        filtered_samples = filtered_samples[:skip_start] + filtered_samples[skip_end:]
        print(
            f"train_skip_window: start={skip_start} count={skip_count} applied={skipped_train_rows}",
            flush=True,
        )

    rows: list[list[float]] = []
    targets: list[list[float]] = []
    if workers <= 1:
        for index, sample in enumerate(filtered_samples, start=1):
            result = train_feature_target_worker(sample)
            if result is not None:
                features, target = result
                rows.append(features)
                targets.append(target)
            if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(filtered_samples)):
                print(f"train_feature_progress: {index}/{len(filtered_samples)} usable={len(rows)}", flush=True)
    else:
        for _, result in unordered_parallel_map(
            function=train_feature_target_worker,
            items=filtered_samples,
            workers=workers,
            max_inflight_per_worker=max_inflight_per_worker,
            initializer_args=(v2vgot_root, graph_ablation_mode, baseline_mode, None),
            progress_every=progress_every,
            progress_label="train_feature_progress",
        ):
            if result is not None:
                features, target = result
                rows.append(features)
                targets.append(target)

    if not rows:
        raise ValueError("No usable clean Q9 train rows.")

    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    mean[0] = 0.0
    scale[0] = 1.0
    x_norm = (x - mean) / scale
    regularizer = np.eye(x_norm.shape[1], dtype=float) * float(ridge_alpha)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(x_norm.T @ x_norm + regularizer, x_norm.T @ y).T
    predictions = x_norm @ coefficients.T
    errors = np.linalg.norm(
        predictions.reshape(-1, 6, 2) - y.reshape(-1, 6, 2),
        axis=2,
    )
    model = CleanQ9Model(
        feature_names=feature_names(),
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        coefficients=coefficients.tolist(),
        ridge_alpha=float(ridge_alpha),
        train_rows=len(train_samples),
        usable_rows=len(rows),
        filtered_overlap_rows=filtered_overlap_rows,
    )
    report = {
        "train_rows": len(train_samples),
        "usable_rows": len(rows),
        "filtered_overlap_rows": filtered_overlap_rows,
        "skip_train_start_index": skip_start,
        "skip_train_count": skip_count,
        "skipped_train_rows": skipped_train_rows,
        "timed_out_rows": 0,
        "ridge_alpha": float(ridge_alpha),
        "feature_names": list(model.feature_names),
        "leakage_policy": {
            "excluded_fields": [
                "dist",
                "angle",
                "suggested_speed_idx",
                "suggested_steering_idx",
                "future_trajectory_str_in_ego",
                "future_trajectory_str_in_self",
            ],
            "uses_question_visible_current_position": True,
            "uses_non_target_scene_graph_aggregates": True,
            "uses_scene_future_trajectory": False,
        },
        "train_metrics_local": {
            "l2_error_avg_1s": float(np.mean(np.mean(errors[:, :2], axis=1))),
            "l2_error_avg_2s": float(np.mean(np.mean(errors[:, :4], axis=1))),
            "l2_error_avg_3s": float(np.mean(np.mean(errors, axis=1))),
            "l2_error_avg_all": float(np.mean(np.mean(errors, axis=1))),
        },
    }
    return model, report


def predict_waypoints(model: CleanQ9Model, features: list[float]) -> tuple[tuple[float, float], ...]:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model.feature_mean, dtype=float)
    scale = np.asarray(model.feature_scale, dtype=float)
    coefficients = np.asarray(model.coefficients, dtype=float)
    prediction = ((x - mean) / scale) @ coefficients.T
    return tuple((float(prediction[index * 2]), float(prediction[index * 2 + 1])) for index in range(6))


def render_trajectory(points: tuple[tuple[float, float], ...]) -> str:
    rendered = ",".join(f"({x:.1f},{y:.1f})" for x, y in points)
    return f"The suggested future trajectory is [{rendered}]."


def prediction_record_worker(sample: BenchmarkSample) -> dict[str, object]:
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model was not initialized.")
    points = predict_waypoints(_WORKER_MODEL, clean_features(sample))
    return {
        "sample_id": sample.sample_id,
        "dataset_name": sample.dataset_name,
        "split_name": sample.split_name,
        "task_type": BenchmarkTaskType.FUTURE_TRAJECTORY.value,
        "qa_type_id": sample.qa_type_id,
        "supported": True,
        "answer_text": render_trajectory(points),
        "object_ids": [],
        "baseline_mode": _WORKER_BASELINE_MODE,
        "graph_ablation_mode": _WORKER_GRAPH_ABLATION_MODE,
        "future_trajectory_model_json": "clean_q9_no_oracle_metadata_inline",
    }


def write_predictions(
    *,
    model: CleanQ9Model,
    val_samples: tuple[BenchmarkSample, ...],
    v2vgot_root: str,
    baseline_mode: str,
    output_jsonl: Path,
    graph_ablation_mode: str,
    progress_every: int,
    workers: int,
    max_inflight_per_worker: int,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        if workers <= 1:
            for index, sample in enumerate(val_samples, start=1):
                points = predict_waypoints(model, clean_features(sample))
                record = {
                    "sample_id": sample.sample_id,
                    "dataset_name": sample.dataset_name,
                    "split_name": sample.split_name,
                    "task_type": BenchmarkTaskType.FUTURE_TRAJECTORY.value,
                    "qa_type_id": sample.qa_type_id,
                    "supported": True,
                    "answer_text": render_trajectory(points),
                    "object_ids": [],
                    "baseline_mode": baseline_mode,
                    "graph_ablation_mode": graph_ablation_mode,
                    "future_trajectory_model_json": "clean_q9_no_oracle_metadata_inline",
                }
                handle.write(json.dumps(record) + "\n")
                if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(val_samples)):
                    print(f"prediction_progress: {index}/{len(val_samples)} sample_id={sample.sample_id}", flush=True)
        else:
            records_by_index: dict[int, dict[str, object]] = {}
            for original_index, record in unordered_parallel_map(
                function=prediction_record_worker,
                items=val_samples,
                workers=workers,
                max_inflight_per_worker=max_inflight_per_worker,
                initializer_args=(v2vgot_root, graph_ablation_mode, baseline_mode, model),
                progress_every=progress_every,
                progress_label="prediction_progress",
            ):
                records_by_index[original_index] = record
            for original_index in sorted(records_by_index):
                handle.write(json.dumps(records_by_index[original_index]) + "\n")
    return len(val_samples)


def write_manifest(
    *,
    manifest_path: Path,
    v2vgot_root: str,
    file_name: str,
    scenario_name: str,
    output_jsonl: Path,
    total_samples: int,
    baseline_mode: str,
    graph_ablation_mode: str,
) -> None:
    payload = {
        "repository_root": v2vgot_root,
        "split": "val",
        "file_name": file_name,
        "scenario_name": scenario_name,
        "task_types": [BenchmarkTaskType.FUTURE_TRAJECTORY.value],
        "runs": [
            {
                "task_type": BenchmarkTaskType.FUTURE_TRAJECTORY.value,
                "scenario_name": scenario_name,
                "file_name": file_name,
                "baseline_mode": baseline_mode,
                "graph_ablation_mode": graph_ablation_mode,
                "future_trajectory_model_json": "clean_q9_no_oracle_metadata_inline",
                "output_jsonl": str(output_jsonl),
                "supported_predictions": total_samples,
                "unsupported_predictions": 0,
                "total_samples": total_samples,
                "qa_type_ids": [19],
                "qa_type_id": 19,
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_metrics(summary_path: Path) -> PlanningMetrics:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary.get("runs", [])
    if not isinstance(runs, list) or not runs or not isinstance(runs[0], dict):
        raise ValueError(f"Official summary has no usable runs: {summary_path}")
    metrics = runs[0].get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Official summary has no metrics object: {summary_path}")

    def value(name: str) -> float | None:
        raw = metrics.get(name)
        return float(raw) if isinstance(raw, (float, int)) else None

    return PlanningMetrics(
        l2_1s=value("l2_error_avg_1s"),
        l2_2s=value("l2_error_avg_2s"),
        l2_3s=value("l2_error_avg_3s"),
        l2_avg=value("l2_error_avg_all"),
        cr_1s=value("collision_rate_1s"),
        cr_2s=value("collision_rate_2s"),
        cr_3s=value("collision_rate_3s"),
        cr_avg=value("collision_rate_avg_all"),
    )


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def table_markdown(method_name: str, metrics: PlanningMetrics, source: str, include_reported_rows: bool) -> str:
    lines = [
        "| Method | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | L2 Avg ↓ | CR 1s ↓ | CR 2s ↓ | CR 3s ↓ | CR Avg ↓ | Comm MB ↓ | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if include_reported_rows:
        for row in REPORTED_TABLE_I_ROWS:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.method}`",
                        fmt(row.l2_1s),
                        fmt(row.l2_2s),
                        fmt(row.l2_3s),
                        fmt(row.l2_avg),
                        fmt(row.cr_1s),
                        fmt(row.cr_2s),
                        fmt(row.cr_3s),
                        fmt(row.cr_avg),
                        fmt(row.comm_mb, 4),
                        f"`{row.source}`",
                    ]
                )
                + " |"
            )
    lines.append(
        "| "
        + " | ".join(
            [
                f"`{method_name}`",
                fmt(metrics.l2_1s),
                fmt(metrics.l2_2s),
                fmt(metrics.l2_3s),
                fmt(metrics.l2_avg),
                fmt(metrics.cr_1s),
                fmt(metrics.cr_2s),
                fmt(metrics.cr_3s),
                fmt(metrics.cr_avg),
                "-",
                f"`{source}`",
            ]
        )
        + " |"
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    output_root = resolve_output_root(args.output_root)
    v2vgot_root = str(Path(args.v2vgot_root).expanduser().resolve())
    adapter = V2VGoTQABenchmarkAdapter(v2vgot_root)

    train_samples = load_q9_samples(
        adapter=adapter,
        split_name="train",
        file_name=args.train_file_name,
        limit=args.limit_train,
    )
    val_samples = load_q9_samples(
        adapter=adapter,
        split_name="val",
        file_name=args.val_file_name,
        limit=args.limit_val,
    )
    if not train_samples:
        raise ValueError(f"No train Q9 samples found in {args.train_file_name}.")
    if not val_samples:
        raise ValueError(f"No validation Q9 samples found in {args.val_file_name}.")

    model, train_report = train_clean_model(
        train_samples=train_samples,
        val_samples=val_samples,
        v2vgot_root=v2vgot_root,
        graph_ablation_mode=args.graph_ablation_mode,
        baseline_mode=args.baseline_mode,
        ridge_alpha=args.ridge_alpha,
        allow_train_val_overlap=args.allow_train_val_overlap,
        progress_every=args.progress_every,
        workers=max(1, args.workers),
        max_inflight_per_worker=args.max_inflight_per_worker,
        skip_train_start_index=args.skip_train_start_index,
        skip_train_count=args.skip_train_count,
    )

    model_path = output_root / f"{args.run_name}_clean_q9_model.json"
    model_path.write_text(json.dumps(model.__dict__, indent=2), encoding="utf-8")
    train_report_path = output_root / f"{args.run_name}_clean_q9_train_report.json"
    train_report_path.write_text(json.dumps(train_report, indent=2), encoding="utf-8")

    prediction_path = output_root / f"{args.run_name}.jsonl"
    total_predictions = write_predictions(
        model=model,
        val_samples=val_samples,
        v2vgot_root=v2vgot_root,
        baseline_mode=args.baseline_mode,
        output_jsonl=prediction_path,
        graph_ablation_mode=args.graph_ablation_mode,
        progress_every=args.progress_every,
        workers=max(1, args.workers),
        max_inflight_per_worker=args.max_inflight_per_worker,
    )
    manifest_path = output_root / f"{args.run_name}_manifest.json"
    write_manifest(
        manifest_path=manifest_path,
        v2vgot_root=v2vgot_root,
        file_name=args.val_file_name,
        scenario_name=args.run_name,
        output_jsonl=prediction_path,
        total_samples=total_predictions,
        baseline_mode=args.baseline_mode,
        graph_ablation_mode=args.graph_ablation_mode,
    )

    official_dir = output_root / "official_exports"
    reports_dir = output_root / "official_eval_reports"
    tools_dir = official_dir / "tools"
    export_manifest = official_dir / f"{args.run_name}_official_export_manifest.json"
    summary_path = reports_dir / f"{args.run_name}_official_export_manifest_official_qa_eval_summary.json"

    run(
        [
            args.python,
            "scripts/export_qa_predictions.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(official_dir),
            "--split",
            "val",
            "--scenario-name",
            args.run_name,
            "--task-type",
            BenchmarkTaskType.FUTURE_TRAJECTORY.value,
            "--qa-type-id",
            "19",
            "--file-name",
            args.val_file_name,
        ],
        v2vgot_root=v2vgot_root,
    )
    if not args.skip_official_eval:
        run(
            [
                args.python,
                "scripts/run_v2vgot_official_qa_eval.py",
                "--export-manifest",
                str(export_manifest),
                "--output-dir",
                str(reports_dir),
                "--tools-dir",
                str(tools_dir),
                "--task-type",
                BenchmarkTaskType.FUTURE_TRAJECTORY.value,
                "--num-future-waypoints",
                "6",
                "--npy-save-path",
                str(Path(v2vgot_root) / "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy"),
                "--v2vgot-root",
                v2vgot_root,
            ],
            v2vgot_root=v2vgot_root,
        )

    metrics = extract_metrics(summary_path) if summary_path.exists() else PlanningMetrics(None, None, None, None, None, None, None, None)
    report = {
        "run_name": args.run_name,
        "method": "G-MARK clean Q9",
        "v2vgot_root": v2vgot_root,
        "train_file_name": args.train_file_name,
        "val_file_name": args.val_file_name,
        "baseline_mode": args.baseline_mode,
        "graph_ablation_mode": args.graph_ablation_mode,
        "model_json": str(model_path),
        "train_report_json": str(train_report_path),
        "prediction_jsonl": str(prediction_path),
        "manifest_json": str(manifest_path),
        "official_summary_json": str(summary_path),
        "metrics": metrics.__dict__,
        "comparison_baseline": "V2V-GoT Table I reported row",
        "leakage_policy": train_report["leakage_policy"],
    }
    report_path = output_root / f"{args.run_name}_clean_q9_table1_row.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = "\n".join(
        [
            f"# Clean G-MARK Table-I/Q9 Row `{args.run_name}`",
            "",
            "- method: `G-MARK clean Q9`",
            f"- train split file: `{args.train_file_name}`",
            f"- released validation Q9 file: `{args.val_file_name}`",
            f"- train rows: `{train_report['train_rows']}`",
            f"- usable train rows: `{train_report['usable_rows']}`",
            f"- filtered train/validation overlap rows: `{train_report['filtered_overlap_rows']}`",
            f"- skipped train rows (temporary window): `{train_report['skipped_train_rows']}`",
            f"- model: `{model_path}`",
            f"- official summary: `{summary_path}`",
            "",
            "Leakage policy: this row excludes `dist`, `angle`, `suggested_speed_idx`, "
            "`suggested_steering_idx`, `future_trajectory_str_in_ego`, and "
            "`future_trajectory_str_in_self` from the model inputs. It uses current "
            "position plus non-target scene/KG aggregates only.",
            "",
            table_markdown(
                "G-MARK clean Q9",
                metrics,
                str(summary_path),
                include_reported_rows=not args.skip_reported_rows,
            ),
            "",
        ]
    )
    markdown_path = output_root / f"{args.run_name}_clean_q9_table1_row.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"saved_model: {model_path}")
    print(f"saved_report: {report_path}")
    print(f"saved_markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
