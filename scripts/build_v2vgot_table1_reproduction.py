#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

METRIC_PATTERNS = {
    "l2_1s": re.compile(r"l2_error_avg_1s:\s+([0-9.eE+-]+)"),
    "l2_2s": re.compile(r"l2_error_avg_2s:\s+([0-9.eE+-]+)"),
    "l2_3s": re.compile(r"l2_error_avg_3s:\s+([0-9.eE+-]+)"),
    "l2_avg": re.compile(r"l2_error_avg_all:\s+([0-9.eE+-]+)"),
    "cr_1s": re.compile(r"collision_rate_1s:\s+([0-9.eE+-]+)"),
    "cr_2s": re.compile(r"collision_rate_2s:\s+([0-9.eE+-]+)"),
    "cr_3s": re.compile(r"collision_rate_3s:\s+([0-9.eE+-]+)"),
    "cr_avg": re.compile(r"collision_rate_avg_all:\s+([0-9.eE+-]+)"),
    "ttc": re.compile(r"TTC:\s+([0-9.eE+-]+)"),
    "comfort": re.compile(r"C:\s+([0-9.eE+-]+)"),
    "pdms": re.compile(r"PDMS:\s+([0-9.eE+-]+)"),
    "pdms_sample_average": re.compile(r"PDMS_sample_average:\s+([0-9.eE+-]+)"),
}


@dataclass(frozen=True)
class PlanningRow:
    method: str
    family: str
    l2_1s: float | None = None
    l2_2s: float | None = None
    l2_3s: float | None = None
    l2_avg: float | None = None
    cr_1s: float | None = None
    cr_2s: float | None = None
    cr_3s: float | None = None
    cr_avg: float | None = None
    comm_mb: float | None = None
    source: str = ""


@dataclass(frozen=True)
class BaselineRun:
    method: str
    model_name: str
    checkpoint_id: str
    inference_script: str
    eval_script: str
    result_file: str
    required_feature_sources: tuple[str, ...]


REPORTED_TABLE_I_ROWS: tuple[PlanningRow, ...] = (
    PlanningRow("No Fusion", "reported_v2vgot_table_i", 3.47, 5.79, 8.26, 5.84, 1.48, 4.24, 7.72, 4.48, 0.0, "V2V-GoT Table I"),
    PlanningRow("Early Fusion", "reported_v2vgot_table_i", 3.48, 5.61, 7.82, 5.63, 1.16, 3.51, 5.66, 3.44, 1.9208, "V2V-GoT Table I"),
    PlanningRow("AttFuse", "reported_v2vgot_table_i", 3.65, 6.21, 8.75, 6.20, 1.19, 4.41, 6.38, 3.99, 0.4008, "V2V-GoT Table I"),
    PlanningRow("V2X-ViT", "reported_v2vgot_table_i", 3.46, 5.80, 8.19, 5.81, 1.45, 4.24, 6.59, 4.09, 0.4008, "V2V-GoT Table I"),
    PlanningRow("CoBEVT", "reported_v2vgot_table_i", 3.38, 5.42, 7.46, 5.42, 1.31, 4.41, 5.75, 3.82, 0.4008, "V2V-GoT Table I"),
    PlanningRow("V2V-LLM", "reported_v2vgot_table_i", 2.90, 4.91, 6.98, 4.93, 0.75, 2.87, 4.93, 2.85, 0.4068, "V2V-GoT Table I"),
    PlanningRow("V2V-GoT", "reported_v2vgot_table_i", 1.65, 2.63, 3.59, 2.62, 0.12, 1.92, 3.45, 1.83, 0.4068, "V2V-GoT Table I"),
)

BASELINE_RUNS: tuple[BaselineRun, ...] = (
    BaselineRun(
        method="No Fusion",
        model_name="v2vllmq5_10ep_both_shallow_f2_ego_only",
        checkpoint_id="490",
        inference_script="LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_ego_only.sh",
        eval_script="LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh",
        result_file="LLaVA/results/v2vllmq5_10ep_both_shallow_f2_ego_only.txt",
        required_feature_sources=("no_fusion_keep_all",),
    ),
    BaselineRun(
        method="Early Fusion",
        model_name="v2vllmq5_10ep_both_shallow_f2_early",
        checkpoint_id="490",
        inference_script="LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_early.sh",
        eval_script="LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh",
        result_file="LLaVA/results/v2vllmq5_10ep_both_shallow_f2_early.txt",
        required_feature_sources=("early",),
    ),
    BaselineRun(
        method="AttFuse",
        model_name="v2vllmq5_10ep_both_shallow_f2_attfuse",
        checkpoint_id="490",
        inference_script="LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_attfuse.sh",
        eval_script="LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh",
        result_file="LLaVA/results/v2vllmq5_10ep_both_shallow_f2_attfuse.txt",
        required_feature_sources=("attfuse",),
    ),
    BaselineRun(
        method="V2X-ViT",
        model_name="v2vllmq5_10ep_both_shallow_f2_v2xvit",
        checkpoint_id="490",
        inference_script="LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_v2xvit.sh",
        eval_script="LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh",
        result_file="LLaVA/results/v2vllmq5_10ep_both_shallow_f2_v2xvit.txt",
        required_feature_sources=("v2xvit",),
    ),
    BaselineRun(
        method="CoBEVT",
        model_name="v2vllmq5_10ep_both_shallow_f2_cobevt",
        checkpoint_id="490",
        inference_script="LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2_cobevt.sh",
        eval_script="LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh",
        result_file="LLaVA/results/v2vllmq5_10ep_both_shallow_f2_cobevt.txt",
        required_feature_sources=("cobevt",),
    ),
    BaselineRun(
        method="V2V-LLM",
        model_name="v2vllmq5_10ep_both_shallow_f2",
        checkpoint_id="490",
        inference_script="LLaVA/scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2.sh",
        eval_script="LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh",
        result_file="LLaVA/results/v2vllmq5_10ep_both_shallow_f2.txt",
        required_feature_sources=("no_fusion_keep_all",),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit/reproduce V2V-GoT Table I planning metrics and combine them with "
            "G-MARK Q9 metrics. Audit/report modes do not require GPU; running V2V "
            "inference does."
        )
    )
    parser.add_argument("--v2vgot-root", default=str(REPO_ROOT.parent / "V2V-GoT"))
    parser.add_argument("--output-dir", default="outputs/v2vgot_table1_reproduction")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--run-v2v-baselines", action="store_true", help="Run V2V-LLM baseline inference/eval scripts. Requires GPU + llava conda env.")
    parser.add_argument("--run-v2vgot", action="store_true", help="Run V2V-GoT graph inference/eval scripts. Requires GPU + llava conda env.")
    parser.add_argument(
        "--only-method",
        action="append",
        default=[],
        help=(
            "Limit runnable upstream methods. Can be repeated. Examples: "
            "`--only-method V2V-LLM`, `--only-method V2V-GoT`."
        ),
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip runnable rows whose scripts/checkpoints/features are missing.")
    parser.add_argument("--conda-init", default="/opt/conda/etc/profile.d/conda.sh", help="Conda shell hook sourced before upstream V2V-GoT scripts if present.")
    parser.add_argument("--skip-reported-baselines", action="store_true")
    parser.add_argument(
        "--gmark-log",
        action="append",
        default=[],
        help=(
            "Optional G-MARK official-evaluator log. Format can be PATH or METHOD=PATH. "
            "Defaults to latest phase8 val Q9 log if present."
        ),
    )
    return parser


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def row_from_metrics(method: str, family: str, metrics: dict[str, float], source: str, comm_mb: float | None = None) -> PlanningRow | None:
    required = ("l2_1s", "l2_2s", "l2_3s", "l2_avg", "cr_1s", "cr_2s", "cr_3s", "cr_avg")
    if not any(key in metrics for key in required):
        return None
    return PlanningRow(
        method=method,
        family=family,
        l2_1s=metrics.get("l2_1s"),
        l2_2s=metrics.get("l2_2s"),
        l2_3s=metrics.get("l2_3s"),
        l2_avg=metrics.get("l2_avg"),
        cr_1s=metrics.get("cr_1s"),
        cr_2s=metrics.get("cr_2s"),
        cr_3s=metrics.get("cr_3s"),
        cr_avg=metrics.get("cr_avg"),
        comm_mb=comm_mb,
        source=source,
    )


def run_command(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    print("$ " + " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)


def shell_source_command(script: str, *args: str, conda_init: str = "") -> list[str]:
    parts = []
    if conda_init:
        parts.append(f"if [ -f {conda_init} ]; then source {conda_init}; fi")
    parts.append("source " + " ".join([script, *args]))
    return ["bash", "-lc", " && ".join(parts)]


def checkpoint_dir(v2vgot_root: Path, model_name: str, checkpoint_id: str) -> Path:
    return (
        v2vgot_root
        / "LLaVA"
        / "checkpoints"
        / "llava-v1.5-7b-task-lora"
        / f"llava-v1.5-7b-task-lora_v2v4real_3d_grounding_{model_name}"
        / f"checkpoint-{checkpoint_id}"
    )


def audit_assets(v2vgot_root: Path) -> tuple[list[dict[str, object]], bool]:
    checks: list[dict[str, object]] = []

    def add(label: str, path: Path) -> None:
        checks.append({"label": label, "path": str(path), "exists": path.exists()})

    add("V2V-GoT root", v2vgot_root)
    add("V2V-LLM Q5 QA JSON", v2vgot_root / "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vllmq5.json")
    add("V2V-GoT Q9 QA JSON", v2vgot_root / "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json")
    add("Q9 collision GT root", v2vgot_root / "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy")
    add("LLaVA model_vqa_loader", v2vgot_root / "LLaVA/llava/eval/model_vqa_loader.py")
    add("V2V-LLM eval script", v2vgot_root / "LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_v2vllmq5.sh")
    add("V2V-GoT Q9 eval script", v2vgot_root / "LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_nq9.sh")

    for run in BASELINE_RUNS:
        add(f"{run.method} inference script", v2vgot_root / run.inference_script)
        add(f"{run.method} checkpoint", checkpoint_dir(v2vgot_root, run.model_name, run.checkpoint_id))
        for feature_source in run.required_feature_sources:
            add(f"{run.method} feature source `{feature_source}`", v2vgot_root / "DMSTrack/V2V4Real/official_models" / feature_source / "npy/co_llm")

    add(
        "V2V-GoT checkpoint",
        checkpoint_dir(v2vgot_root, "v2vgot_10ep_both_shallow_f2", "4330"),
    )

    ok = all(bool(item["exists"]) for item in checks)
    return checks, ok


def parse_existing_v2v_rows(v2vgot_root: Path) -> list[PlanningRow]:
    rows: list[PlanningRow] = []
    reported_comm = {row.method: row.comm_mb for row in REPORTED_TABLE_I_ROWS}
    for run in BASELINE_RUNS:
        result_path = v2vgot_root / run.result_file
        if not result_path.exists():
            continue
        row = row_from_metrics(
            run.method,
            "reproduced_v2vllm",
            parse_metrics(result_path.read_text(encoding="utf-8", errors="replace")),
            str(result_path),
            comm_mb=reported_comm.get(run.method),
        )
        if row is not None:
            rows.append(row)

    v2vgot_result_candidates = [
        v2vgot_root / "LLaVA/results/v2vgot_10ep_both_shallow_f2_4330_full_nq9sm3w6dc.txt",
        v2vgot_root / "LLaVA/results/v2vgot_10ep_both_shallow_f2_4330_full_nq9sm3w6d.txt",
    ]
    for result_path in v2vgot_result_candidates:
        if not result_path.exists():
            continue
        row = row_from_metrics(
            "V2V-GoT",
            "reproduced_v2vgot",
            parse_metrics(result_path.read_text(encoding="utf-8", errors="replace")),
            str(result_path),
            comm_mb=reported_comm.get("V2V-GoT"),
        )
        if row is not None:
            rows.append(row)
            break
    return rows


def default_gmark_logs() -> list[tuple[str, Path]]:
    path = REPO_ROOT / "outputs/phase8_val_report/official_eval_reports/future_trajectory_qa_type_19_official_eval.log"
    return [("G-MARK full", path)] if path.exists() else []


def parse_gmark_log_specs(specs: Sequence[str]) -> list[tuple[str, Path]]:
    if not specs:
        return default_gmark_logs()
    parsed: list[tuple[str, Path]] = []
    for index, spec in enumerate(specs, start=1):
        if "=" in spec:
            method, raw_path = spec.split("=", 1)
        else:
            method, raw_path = f"G-MARK {index}", spec
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        parsed.append((method.strip(), path))
    return parsed


def parse_gmark_rows(specs: Sequence[str]) -> list[PlanningRow]:
    rows: list[PlanningRow] = []
    for method, path in parse_gmark_log_specs(specs):
        if not path.exists():
            continue
        row = row_from_metrics(
            method,
            "gmark_local",
            parse_metrics(path.read_text(encoding="utf-8", errors="replace")),
            str(path),
            comm_mb=None,
        )
        if row is not None:
            rows.append(row)
    return rows


def table_markdown(title: str, rows: Sequence[PlanningRow]) -> str:
    lines = [
        f"## {title}",
        "",
        "| Method | Family | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | L2 Avg ↓ | CR 1s ↓ | CR 2s ↓ | CR 3s ↓ | CR Avg ↓ | Comm MB ↓ | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.method}`",
                    f"`{row.family}`",
                    fmt(row.l2_1s),
                    fmt(row.l2_2s),
                    fmt(row.l2_3s),
                    fmt(row.l2_avg),
                    fmt(row.cr_1s),
                    fmt(row.cr_2s),
                    fmt(row.cr_3s),
                    fmt(row.cr_avg),
                    fmt(row.comm_mb, 4) if row.comm_mb is not None else "-",
                    f"`{row.source}`" if row.source else "",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def audit_markdown(checks: Sequence[dict[str, object]]) -> str:
    lines = [
        "## Asset Audit",
        "",
        "| Status | Asset | Path |",
        "| --- | --- | --- |",
    ]
    for item in checks:
        lines.append(
            "| "
            + ("OK" if item["exists"] else "MISSING")
            + f" | `{item['label']}` | `{item['path']}` |"
        )
    return "\n".join(lines)


def write_outputs(output_dir: Path, checks: Sequence[dict[str, object]], rows: Sequence[PlanningRow]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "asset_audit": list(checks),
        "rows": [row.__dict__ for row in rows],
    }
    (output_dir / "table1_reproduction_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    sections = [
        "# V2V-GoT Table I Reproduction Report",
        "",
        audit_markdown(checks),
        "",
        table_markdown("Planning Table", rows),
        "",
        "## Notes",
        "",
        "- `reported_v2vgot_table_i` rows are literature values from V2V-GoT Table I.",
        "- `reproduced_*` rows are parsed from local V2V-GoT evaluator logs when available.",
        "- `gmark_local` rows are parsed from local G-MARK Q9 official-evaluator logs.",
        "- Running V2V-LLM/V2V-GoT inference requires GPU, the released LoRA checkpoints, processed feature folders, and the `llava` conda environment.",
        "- Communication values for reproduced baselines use V2V-GoT Table I accounting unless a separate communication estimator is added.",
    ]
    (output_dir / "table1_reproduction_report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def normalized_methods(methods: Sequence[str]) -> set[str]:
    return {method.strip().casefold() for method in methods if method.strip()}


def method_selected(method: str, selected: set[str]) -> bool:
    return not selected or method.casefold() in selected


def missing_run_assets(v2vgot_root: Path, run: BaselineRun) -> list[str]:
    missing: list[str] = []
    for label, path in (
        ("inference script", v2vgot_root / run.inference_script),
        ("eval script", v2vgot_root / run.eval_script),
        ("checkpoint", checkpoint_dir(v2vgot_root, run.model_name, run.checkpoint_id)),
    ):
        if not path.exists():
            missing.append(f"{label}: {path}")
    for feature_source in run.required_feature_sources:
        path = v2vgot_root / "DMSTrack/V2V4Real/official_models" / feature_source / "npy/co_llm"
        if not path.exists():
            missing.append(f"feature source `{feature_source}`: {path}")
    return missing


def run_v2v_baselines(v2vgot_root: Path, output_dir: Path, selected_methods: set[str], skip_missing: bool, conda_init: str) -> None:
    for run in BASELINE_RUNS:
        if not method_selected(run.method, selected_methods):
            continue
        missing = missing_run_assets(v2vgot_root, run)
        if missing:
            message = f"{run.method} cannot run; missing " + "; ".join(missing)
            if skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise FileNotFoundError(message)
        log_prefix = output_dir / "run_logs" / run.model_name
        run_command(shell_source_command(run.inference_script, conda_init=conda_init), v2vgot_root, log_prefix.with_suffix(".inference.log"))
        run_command(shell_source_command(run.eval_script, run.model_name, conda_init=conda_init), v2vgot_root, log_prefix.with_suffix(".eval.log"))


def missing_v2vgot_assets(v2vgot_root: Path) -> list[str]:
    missing: list[str] = []
    for label, path in (
        ("inference script", v2vgot_root / "LLaVA/scripts/v1_5/inference_v2vgot.sh"),
        ("Q9 eval script", v2vgot_root / "LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_nq9.sh"),
        ("checkpoint", checkpoint_dir(v2vgot_root, "v2vgot_10ep_both_shallow_f2", "4330")),
    ):
        if not path.exists():
            missing.append(f"{label}: {path}")
    return missing


def run_v2vgot(v2vgot_root: Path, output_dir: Path, selected_methods: set[str], skip_missing: bool, conda_init: str) -> None:
    if not method_selected("V2V-GoT", selected_methods):
        return
    missing = missing_v2vgot_assets(v2vgot_root)
    if missing:
        message = "V2V-GoT cannot run; missing " + "; ".join(missing)
        if skip_missing:
            print(f"[SKIP] {message}")
            return
        raise FileNotFoundError(message)
    run_command(shell_source_command("LLaVA/scripts/v1_5/inference_v2vgot.sh", conda_init=conda_init), v2vgot_root, output_dir / "run_logs/v2vgot.inference.log")
    run_command(shell_source_command("LLaVA/scripts/v1_5/eval/eval_v2v4real_3d_grounding_nq9.sh", "v2vgot_10ep_both_shallow_f2", "nq9sm3w6dc", "full", "4330", conda_init=conda_init), v2vgot_root, output_dir / "run_logs/v2vgot.nq9.eval.log")


def main() -> int:
    args = build_parser().parse_args()
    v2vgot_root = Path(args.v2vgot_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    checks, ok = audit_assets(v2vgot_root)
    for item in checks:
        print(("OK      " if item["exists"] else "MISSING ") + f"{item['label']}: {item['path']}")
    print(f"asset_audit_ok: {ok}")

    if args.audit_only:
        write_outputs(output_dir, checks, [])
        print(f"saved_markdown: {output_dir / 'table1_reproduction_report.md'}")
        return 0 if ok else 2

    selected_methods = normalized_methods(args.only_method)

    if args.run_v2v_baselines:
        run_v2v_baselines(v2vgot_root, output_dir, selected_methods, args.skip_missing, args.conda_init)
    if args.run_v2vgot:
        run_v2vgot(v2vgot_root, output_dir, selected_methods, args.skip_missing, args.conda_init)

    rows: list[PlanningRow] = []
    if not args.skip_reported_baselines:
        rows.extend(REPORTED_TABLE_I_ROWS)
    rows.extend(parse_existing_v2v_rows(v2vgot_root))
    rows.extend(parse_gmark_rows(args.gmark_log))

    write_outputs(output_dir, checks, rows)
    print()
    print(table_markdown("Planning Table", rows))
    print()
    print(f"saved_json: {output_dir / 'table1_reproduction_summary.json'}")
    print(f"saved_markdown: {output_dir / 'table1_reproduction_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
