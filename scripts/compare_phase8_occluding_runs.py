#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)

REFERENCE_COORDINATE_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two Phase 8 occluding-object prediction files by reference-count "
            "alignment. This is a diagnostic tool, not an official scorer."
        )
    )
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--show-samples", type=int, default=10)
    parser.add_argument("--json-name", default="phase8_occluding_run_comparison.json")
    parser.add_argument("--markdown-name", default="phase8_occluding_run_comparison.md")
    return parser


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def resolve_path(path_value: str) -> Path:
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


def normalize_ids(record: dict[str, object] | None) -> tuple[str, ...]:
    if record is None:
        return ()
    values = record.get("object_ids", [])
    if not isinstance(values, list):
        return ()
    return tuple(str(item) for item in values)


def extract_reference_coordinates(answer_text: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in REFERENCE_COORDINATE_PATTERN.findall(answer_text))


def count_error(predicted_count: int, reference_count: int) -> int:
    return abs(predicted_count - reference_count)


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def main() -> None:
    args = build_parser().parse_args()
    baseline_path = resolve_path(args.baseline_jsonl)
    candidate_path = resolve_path(args.candidate_jsonl)
    baseline_records = load_jsonl(baseline_path)
    candidate_records = load_jsonl(candidate_path)

    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    samples = adapter.load_samples(split_name=args.split, file_name=args.file_name)
    samples_by_id = {
        sample.sample_id: sample
        for sample in samples
        if sample.task_type == BenchmarkTaskType.OCCLUDING_OBJECTS
    }

    improved: list[dict[str, object]] = []
    worsened: list[dict[str, object]] = []
    unchanged: list[dict[str, object]] = []
    candidate_added_ids = 0
    candidate_removed_ids = 0

    for sample_id in sorted(set(baseline_records) | set(candidate_records), key=lambda item: int(item) if item.isdigit() else item):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue

        reference_count = len(extract_reference_coordinates(sample.scene.raw_answer))
        baseline_ids = normalize_ids(baseline_records.get(sample_id))
        candidate_ids = normalize_ids(candidate_records.get(sample_id))
        baseline_error = count_error(len(baseline_ids), reference_count)
        candidate_error = count_error(len(candidate_ids), reference_count)
        added_ids = tuple(object_id for object_id in candidate_ids if object_id not in baseline_ids)
        removed_ids = tuple(object_id for object_id in baseline_ids if object_id not in candidate_ids)
        candidate_added_ids += len(added_ids)
        candidate_removed_ids += len(removed_ids)

        row = {
            "sample_id": sample_id,
            "question": sample.scene.raw_question,
            "reference_answer": sample.scene.raw_answer,
            "reference_count": reference_count,
            "baseline_ids": baseline_ids,
            "candidate_ids": candidate_ids,
            "added_ids": added_ids,
            "removed_ids": removed_ids,
            "baseline_error": baseline_error,
            "candidate_error": candidate_error,
        }
        if candidate_error < baseline_error:
            improved.append(row)
        elif candidate_error > baseline_error:
            worsened.append(row)
        else:
            unchanged.append(row)

    report = {
        "repository_root": str(repository_root),
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "split": args.split,
        "sample_count": len(improved) + len(worsened) + len(unchanged),
        "improved_count_alignment": len(improved),
        "worsened_count_alignment": len(worsened),
        "unchanged_count_alignment": len(unchanged),
        "candidate_added_ids": candidate_added_ids,
        "candidate_removed_ids": candidate_removed_ids,
        "examples": {
            "improved": improved[: args.show_samples],
            "worsened": worsened[: args.show_samples],
        },
    }

    output_dir = candidate_path.parent
    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_sections = [
        "# Phase 8 Occluding Run Comparison",
        "",
        f"- `baseline`: `{args.baseline_name}`",
        f"- `baseline_path`: `{baseline_path}`",
        f"- `candidate`: `{args.candidate_name}`",
        f"- `candidate_path`: `{candidate_path}`",
        f"- `repository_root`: `{repository_root}`",
        "",
        markdown_table(
            [
                ["Metric", "Value"],
                ["Samples", str(report["sample_count"])],
                ["Improved count alignment", str(len(improved))],
                ["Worsened count alignment", str(len(worsened))],
                ["Unchanged count alignment", str(len(unchanged))],
                ["Candidate added IDs", str(candidate_added_ids)],
                ["Candidate removed IDs", str(candidate_removed_ids)],
            ]
        ),
        "",
    ]

    for title, rows in (("Improved Examples", improved), ("Worsened Examples", worsened)):
        markdown_sections.extend([f"## {title}", ""])
        if not rows:
            markdown_sections.extend(["No examples captured.", ""])
            continue
        for row in rows[: args.show_samples]:
            markdown_sections.extend(
                [
                    (
                        f"- `sample_id={row['sample_id']}` ref_count=`{row['reference_count']}` "
                        f"baseline_ids=`{list(row['baseline_ids'])}` candidate_ids=`{list(row['candidate_ids'])}` "
                        f"added=`{list(row['added_ids'])}` removed=`{list(row['removed_ids'])}`"
                    ),
                    f"  - reference: {row['reference_answer']}",
                ]
            )
        markdown_sections.append("")

    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    print("=" * 72)
    print("Phase 8 Occluding Run Comparison")
    print("=" * 72)
    print(f"samples: {report['sample_count']}")
    print(f"improved_count_alignment: {len(improved)}")
    print(f"worsened_count_alignment: {len(worsened)}")
    print(f"unchanged_count_alignment: {len(unchanged)}")
    print(f"candidate_added_ids: {candidate_added_ids}")
    print(f"candidate_removed_ids: {candidate_removed_ids}")
    print(f"saved_json: {json_path}")
    print(f"saved_markdown: {markdown_path}")


if __name__ == "__main__":
    main()
