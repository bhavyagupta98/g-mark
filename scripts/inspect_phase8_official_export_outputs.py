#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Phase 8 official-style export outputs for empty/no-object text, "
            "coordinate mentions, and kg_prediction object IDs."
        )
    )
    parser.add_argument("--export-manifest", required=True)
    parser.add_argument("--task-type", default="")
    parser.add_argument("--examples", type=int, default=5)
    return parser


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (base / path).resolve()


def is_no_object_output(text: str) -> bool:
    lowered = text.lower()
    return "no notable object" in lowered or "no object" in lowered


def has_coordinate_output(text: str) -> bool:
    return "(" in text and ")" in text


def inspect_export(path: Path, examples: int) -> dict[str, object]:
    counters: Counter[str] = Counter()
    example_rows: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            outputs = str(record.get("outputs", ""))
            kg_prediction = record.get("kg_prediction", {})
            object_ids = (
                kg_prediction.get("object_ids", [])
                if isinstance(kg_prediction, dict)
                else []
            )
            if not isinstance(object_ids, list):
                object_ids = []

            counters["rows"] += 1
            counters[f"kg_object_count_{len(object_ids)}"] += 1
            if object_ids:
                counters["kg_positive_rows"] += 1
            if is_no_object_output(outputs):
                counters["no_object_text_rows"] += 1
            if has_coordinate_output(outputs):
                counters["coordinate_text_rows"] += 1
            if object_ids and is_no_object_output(outputs):
                counters["kg_positive_but_no_object_text_rows"] += 1

            if len(example_rows) < examples and (
                object_ids or is_no_object_output(outputs)
            ):
                example_rows.append(
                    {
                        "sample_id": record.get("id", record.get("sample_id")),
                        "kg_object_ids": object_ids,
                        "outputs": outputs,
                    }
                )

    return {
        "path": str(path),
        "counts": dict(sorted(counters.items())),
        "examples": example_rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.export_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent

    selected_task = args.task_type
    reports = []
    for run in manifest.get("runs", []):
        if not isinstance(run, dict):
            continue
        task_type = str(run.get("task_type", ""))
        if selected_task and task_type != selected_task:
            continue
        output_jsonl = run.get("output_jsonl")
        if not output_jsonl:
            continue
        reports.append(
            {
                "task_type": task_type,
                **inspect_export(resolve_path(base, str(output_jsonl)), args.examples),
            }
        )

    print(json.dumps({"export_manifest": str(manifest_path), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
