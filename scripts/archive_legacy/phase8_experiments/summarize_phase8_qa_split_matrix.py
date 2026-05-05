#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_ORDER = {
    "notable_objects": 1,
    "occluding_objects": 2,
    "invisible_objects": 3,
    "planning_awareness": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 8 official QA train/validation reports into one table."
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        default=[],
        help="Official QA eval summary JSON. Repeat for each train/val task report.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def split_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "phase8_train_dev" in parts:
        return "train"
    if "phase8_val_report" in parts:
        return "val"
    text = str(path)
    if "_train_" in text:
        return "train"
    if "_val_" in text:
        return "val"
    return ""


def scenario_from_export_manifest(export_manifest: str) -> str:
    stem = Path(export_manifest).stem
    suffix = "_official_export_manifest"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def metric(metrics: dict[str, Any], name: str) -> Any:
    localization = metrics.get("localization", {})
    if not isinstance(localization, dict):
        return ""
    threshold_metrics = localization.get("0.5", {})
    if not isinstance(threshold_metrics, dict):
        return ""
    return threshold_metrics.get(name, "")


def rounded(value: Any) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def load_rows(summary_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in summary_paths:
        path = resolve_path(raw_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        scenario = scenario_from_export_manifest(str(summary.get("export_manifest", "")))
        split = split_from_path(path)
        for run in summary.get("runs", []):
            if not isinstance(run, dict):
                continue
            metrics = run.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            rows.append(
                {
                    "split": split,
                    "task_type": run.get("task_type", ""),
                    "qa_type_id": run.get("qa_type_id", ""),
                    "scenario": scenario,
                    "returncode": run.get("returncode", ""),
                    "f1_0p5": metric(metrics, "f1"),
                    "precision_0p5": metric(metrics, "precision"),
                    "recall_0p5": metric(metrics, "recall"),
                    "binary_f1": metrics.get("binary_f1", ""),
                    "output_parse_error_rate": metrics.get("output_parse_error_rate", ""),
                    "summary_json": str(path),
                    "export_manifest": summary.get("export_manifest", ""),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            TASK_ORDER.get(str(row["task_type"]), 99),
            str(row["scenario"]),
        ),
    )


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Phase 8 QA Train/Validation Matrix",
        "",
        "| Split | Task | Scenario | F1 @ 0.5m | Precision @ 0.5m | Recall @ 0.5m | Binary F1 | Parse Error | Return Code |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"`{row['split']}` | "
            + f"`{row['task_type']}` | "
            + f"`{row['scenario']}` | "
            + f"`{rounded(row['f1_0p5'])}` | "
            + f"`{rounded(row['precision_0p5'])}` | "
            + f"`{rounded(row['recall_0p5'])}` | "
            + f"`{rounded(row['binary_f1'])}` | "
            + f"`{rounded(row['output_parse_error_rate'])}` | "
            + f"`{row['returncode']}` |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if not args.summary_json:
        raise SystemExit("Provide at least one --summary-json path.")
    rows = load_rows(args.summary_json)

    output_json = resolve_path(args.output_json)
    output_markdown = resolve_path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    write_markdown(rows, output_markdown)

    print(f"saved_json: {output_json}")
    print(f"saved_markdown: {output_markdown}")
    for row in rows:
        print(
            f"[{row['split']}] {row['task_type']} "
            f"f1={rounded(row['f1_0p5'])} p={rounded(row['precision_0p5'])} r={rounded(row['recall_0p5'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
