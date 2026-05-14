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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_coop_drive.application.qa.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.application.scene_graph.local_graph_serializer import LocalGraphSerializer  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402
from scripts.build_v2vgot_table1_reproduction import REPORTED_TABLE_I_ROWS  # noqa: E402
from scripts.e2e.run_e2e_validation_report import latest_manifest, read_json  # noqa: E402

TABLE1_Q9_FILE_NAME = "v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json"
TABLE1_Q9_RELATIVE_PATH = (
    "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/"
    + TABLE1_Q9_FILE_NAME
)


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
    comm_mb: float | None = None


@dataclass(frozen=True)
class CommunicationReport:
    sample_count: int
    total_bytes: int
    average_bytes: float
    average_mb_decimal: float
    average_mb_mib: float
    serialization: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run G-MARK on the released V2V-GoT Table-I/Q9 planning evaluation file "
            "and emit report-ready L2/CR metrics."
        )
    )
    parser.add_argument("--run-name", default=f"gmark_table1_q9_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--manifest-json",
        default="",
        help="G-MARK e2e_model_manifest.json containing q9_model_json. Defaults to latest outputs/e2e_runs manifest.",
    )
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--output-root", default="outputs/v2vgot_table1_reproduction/gmark_q9")
    parser.add_argument("--method-name", default="G-MARK")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for the full released Q9 split.")
    parser.add_argument(
        "--comm-limit",
        type=int,
        default=0,
        help="Use 0 to compute communication on all released Q9 samples; positive values are for smoke checks.",
    )
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
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only parse existing outputs for this run-name/output-root.",
    )
    parser.add_argument(
        "--skip-comm",
        action="store_true",
        help="Skip serialized KG communication accounting.",
    )
    parser.add_argument(
        "--skip-reported-rows",
        action="store_true",
        help="Only write the G-MARK row instead of a Table-I-style table with reported V2V-GoT rows.",
    )
    return parser


def run(command: Sequence[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_root(raw_output_root: str) -> Path:
    output_root = Path(raw_output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    return output_root


def resolve_manifest(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve() if path_value else latest_manifest()


def resolve_v2vgot_root(manifest: dict[str, object], raw_v2vgot_root: str) -> str:
    if raw_v2vgot_root:
        return str(Path(raw_v2vgot_root).expanduser().resolve())
    return str(manifest.get("v2vgot_root", "/workspace/repos/V2V-GoT"))


def table1_q9_json_path(v2vgot_root: str) -> Path:
    return Path(v2vgot_root).expanduser().resolve() / TABLE1_Q9_RELATIVE_PATH


def count_table1_q9_samples(v2vgot_root: str) -> int:
    adapter = V2VGoTQABenchmarkAdapter(v2vgot_root)
    return sum(
        1
        for sample in adapter.load_samples(split_name="val", file_name=TABLE1_Q9_FILE_NAME)
        if sample.task_type == BenchmarkTaskType.FUTURE_TRAJECTORY
    )


def summary_path(output_root: Path, run_name: str) -> Path:
    return (
        output_root
        / "official_eval_reports"
        / f"{run_name}_official_export_manifest_official_qa_eval_summary.json"
    )


def extract_metrics(summary: dict[str, object]) -> PlanningMetrics:
    runs = summary.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Official summary has no runs.")
    run_payload = runs[0]
    if not isinstance(run_payload, dict):
        raise ValueError("Official summary run payload is invalid.")
    metrics = run_payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Official summary has no metrics object.")

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


def compact_json_size_bytes(serialized_json: str) -> int:
    payload = json.loads(serialized_json)
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return len(compact.encode("utf-8"))


def compute_communication_report(
    *,
    v2vgot_root: str,
    file_name: str,
    baseline_mode: str,
    graph_ablation_mode: str,
    limit: int,
    progress_every: int,
) -> CommunicationReport:
    adapter = V2VGoTQABenchmarkAdapter(v2vgot_root)
    evaluator = V2VGoTQAPhase5AEvaluator(v2vgot_root, graph_ablation=graph_ablation_mode)
    serializer = LocalGraphSerializer()

    samples = tuple(
        sample
        for sample in adapter.load_samples(split_name="val", file_name=file_name)
        if sample.task_type == BenchmarkTaskType.FUTURE_TRAJECTORY
    )
    if limit > 0:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No future_trajectory samples found in {file_name}.")

    total_bytes = 0
    for index, sample in enumerate(samples, start=1):
        scene = evaluator.prepare_sample(sample, baseline_mode=baseline_mode)
        total_bytes += compact_json_size_bytes(serializer.to_json(scene))
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(samples)):
            print(f"comm_progress: {index}/{len(samples)} sample_id={sample.sample_id}", flush=True)

    average_bytes = total_bytes / float(len(samples))
    return CommunicationReport(
        sample_count=len(samples),
        total_bytes=total_bytes,
        average_bytes=average_bytes,
        average_mb_decimal=average_bytes / 1_000_000.0,
        average_mb_mib=average_bytes / (1024.0 * 1024.0),
        serialization="compact_json(LocalGraphSerializer(prepared_gmark_scene)) average bytes per Q9 sample",
    )


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def row_markdown(method_name: str, metrics: PlanningMetrics, source: str) -> str:
    return "\n".join(
        [
            "| Method | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | L2 Avg ↓ | CR 1s ↓ | CR 2s ↓ | CR 3s ↓ | CR Avg ↓ | Comm MB ↓ | Source |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                    fmt(metrics.comm_mb, 4),
                    f"`{source}`",
                ]
            )
            + " |",
        ]
    )


def table1_markdown(method_name: str, metrics: PlanningMetrics, source: str, include_reported_rows: bool) -> str:
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
    lines.append(row_markdown(method_name, metrics, source).splitlines()[-1])
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = resolve_manifest(args.manifest_json)
    manifest = read_json(manifest_path)
    model_paths = manifest.get("model_paths", {})
    if not isinstance(model_paths, dict):
        raise ValueError(f"Invalid model_paths in manifest: {manifest_path}")
    q9_model_json = str(model_paths.get("q9_model_json", "")).strip()
    if not q9_model_json:
        raise ValueError(f"Manifest does not contain q9_model_json: {manifest_path}")

    v2vgot_root = resolve_v2vgot_root(manifest, args.v2vgot_root)
    q9_json = table1_q9_json_path(v2vgot_root)
    if not q9_json.exists():
        raise FileNotFoundError(f"Released Table-I/Q9 QA file not found: {q9_json}")
    q9_sample_count = count_table1_q9_samples(v2vgot_root)
    if q9_sample_count == 0:
        raise ValueError(
            "Released Table-I/Q9 QA file loaded zero future_trajectory samples. "
            "Check V2VGoTQABenchmarkAdapter Q9 classification for this file."
        )

    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        command = [
            args.python,
            "scripts/run_qa_split_pipeline.py",
            "--purpose",
            "val_report",
            "--split",
            "val",
            "--task-type",
            "future_trajectory",
            "--scenario-name",
            args.run_name,
            "--baseline-mode",
            args.baseline_mode,
            "--graph-ablation-mode",
            args.graph_ablation_mode,
            "--output-root",
            str(output_root),
            "--v2vgot-root",
            v2vgot_root,
            "--file-name",
            TABLE1_Q9_FILE_NAME,
            "--future-trajectory-model-json",
            q9_model_json,
            "--workers",
            str(args.workers),
            "--progress-every",
            str(args.progress_every),
            "--limit",
            str(args.limit),
        ]
        run(command)

    official_summary_path = summary_path(output_root, args.run_name)
    summary = read_json(official_summary_path)
    metrics = extract_metrics(summary)
    communication_report = None
    if not args.skip_comm:
        communication_report = compute_communication_report(
            v2vgot_root=v2vgot_root,
            file_name=TABLE1_Q9_FILE_NAME,
            baseline_mode=args.baseline_mode,
            graph_ablation_mode=args.graph_ablation_mode,
            limit=args.comm_limit,
            progress_every=args.progress_every,
        )
        metrics = PlanningMetrics(
            l2_1s=metrics.l2_1s,
            l2_2s=metrics.l2_2s,
            l2_3s=metrics.l2_3s,
            l2_avg=metrics.l2_avg,
            cr_1s=metrics.cr_1s,
            cr_2s=metrics.cr_2s,
            cr_3s=metrics.cr_3s,
            cr_avg=metrics.cr_avg,
            comm_mb=communication_report.average_mb_decimal,
        )
    source = str(official_summary_path)

    report = {
        "method": args.method_name,
        "run_name": args.run_name,
        "manifest_json": str(manifest_path),
        "q9_model_json": q9_model_json,
        "v2vgot_root": v2vgot_root,
        "table1_q9_file": str(q9_json),
        "table1_q9_sample_count": q9_sample_count,
        "official_summary_json": source,
        "baseline_mode": args.baseline_mode,
        "graph_ablation_mode": args.graph_ablation_mode,
        "metrics": metrics.__dict__,
        "communication": None if communication_report is None else communication_report.__dict__,
    }
    write_json(output_root / f"{args.run_name}_table1_row.json", report)

    markdown = "\n".join(
        [
            f"# G-MARK Table-I/Q9 Row `{args.run_name}`",
            "",
            f"- method: `{args.method_name}`",
            f"- released Q9 file: `{q9_json}`",
            f"- released Q9 sample count: `{q9_sample_count}`",
            f"- source manifest: `{manifest_path}`",
            f"- official summary: `{official_summary_path}`",
            f"- communication accounting: `{'skipped' if communication_report is None else communication_report.serialization}`",
            "",
            table1_markdown(
                args.method_name,
                metrics,
                source,
                include_reported_rows=not args.skip_reported_rows,
            ),
            "",
        ]
    )
    markdown_path = output_root / f"{args.run_name}_table1_row.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"saved_json: {output_root / f'{args.run_name}_table1_row.json'}")
    print(f"saved_markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
