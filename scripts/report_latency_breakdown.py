#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-sample latency JSONL into exhaustive per-component tables "
            "grouped by task_type and qa_type_id."
        )
    )
    parser.add_argument("--latency-jsonl", required=True, help="Input latency JSONL from evaluate_qa_router.")
    parser.add_argument("--output-json", default="", help="Optional output summary JSON path.")
    parser.add_argument("--output-markdown", default="", help="Optional output markdown table path.")
    return parser


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def format_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Latency Breakdown Summary",
        "",
        f"- input: `{summary['input_path']}`",
        f"- sample_count: `{summary['sample_count']}`",
        "",
        "## Per-Task Component Latency (ms)",
        "",
        "| task_type | qa_type_id | samples | component | avg_ms | p50_ms | p90_ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    rows = summary["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
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
    return "\n".join(lines) + "\n"


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = int(round((len(sorted_values) - 1) * q))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def summarize(rows: list[dict[str, object]], input_path: Path) -> dict[str, object]:
    grouped: dict[tuple[str, int | None], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    components: set[str] = set()
    for row in rows:
        task_type = str(row.get("task_type", "unknown"))
        qa_type_id = row.get("qa_type_id")
        timings = row.get("timings_ms")
        if not isinstance(timings, dict):
            continue
        for component, value in timings.items():
            if not isinstance(component, str):
                continue
            if not isinstance(value, (float, int)):
                continue
            grouped[(task_type, int(qa_type_id) if isinstance(qa_type_id, int) else None)][component].append(float(value))
            components.add(component)

    out_rows: list[dict[str, object]] = []
    for task_key in sorted(grouped.keys(), key=lambda item: (item[0], item[1] if item[1] is not None else -1)):
        task_type, qa_type_id = task_key
        component_values = grouped[task_key]
        sample_count = 0
        if "sample_total_ms" in component_values:
            sample_count = len(component_values["sample_total_ms"])
        else:
            for values in component_values.values():
                sample_count = max(sample_count, len(values))
        for component in sorted(components):
            values = component_values.get(component, [])
            sorted_values = sorted(values)
            out_rows.append(
                {
                    "task_type": task_type,
                    "qa_type_id": qa_type_id,
                    "sample_count": sample_count,
                    "component": component,
                    "avg_ms": mean(values),
                    "p50_ms": percentile(sorted_values, 0.50),
                    "p90_ms": percentile(sorted_values, 0.90),
                }
            )
    return {
        "input_path": str(input_path),
        "sample_count": len(rows),
        "rows": out_rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.latency_jsonl).expanduser().resolve()
    rows = read_jsonl(input_path)
    summary = summarize(rows, input_path=input_path)

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_json: {output_json}")

    if args.output_markdown:
        output_markdown = Path(args.output_markdown).expanduser().resolve()
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(format_markdown(summary), encoding="utf-8")
        print(f"saved_markdown: {output_markdown}")

    print(f"input_path: {input_path}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"components_reported: {len({row['component'] for row in summary['rows']})}")
    print(f"table_rows: {len(summary['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
