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


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Phase 8 occluding-object archived predictions against raw "
            "reference answer structure. This is a failure-bucketing tool, not an "
            "official benchmark scorer."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--show-samples", type=int, default=8)
    parser.add_argument("--json-name", default="phase8_occluding_mismatch_report.json")
    parser.add_argument("--markdown-name", default="phase8_occluding_mismatch_report.md")
    return parser


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


def normalize_ids(record: dict[str, object]) -> tuple[str, ...]:
    values = record.get("object_ids", [])
    if not isinstance(values, list):
        return ()
    return tuple(str(item) for item in values)


def extract_reference_coordinates(answer_text: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in REFERENCE_COORDINATE_PATTERN.findall(answer_text))


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def find_occluding_run(manifest: dict[str, object]) -> dict[str, object]:
    for run in manifest.get("runs", []):
        if isinstance(run, dict) and run.get("task_type") == BenchmarkTaskType.OCCLUDING_OBJECTS.value:
            return run
    raise SystemExit("Manifest contains no occluding_objects run.")


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = find_occluding_run(manifest)

    prediction_path = Path(str(run["output_jsonl"])).expanduser()
    if not prediction_path.is_absolute():
        prediction_path = (REPO_ROOT / prediction_path).resolve()
    prediction_records = load_jsonl(prediction_path)

    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    split = str(manifest.get("split", "val"))
    samples = adapter.load_samples(split_name=split, file_name=args.file_name)
    samples_by_id = {
        sample.sample_id: sample
        for sample in samples
        if sample.task_type == BenchmarkTaskType.OCCLUDING_OBJECTS
    }

    buckets: dict[str, int] = {
        "exact_count": 0,
        "under_predicted_count": 0,
        "over_predicted_count": 0,
        "empty_prediction_with_reference": 0,
        "prediction_without_reference": 0,
    }
    examples: dict[str, list[dict[str, object]]] = {key: [] for key in buckets}
    reference_count_histogram: dict[str, int] = {}
    predicted_count_histogram: dict[str, int] = {}

    inspected = 0
    total_reference_mentions = 0
    total_predicted_mentions = 0

    for sample_id, prediction in sorted(prediction_records.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue

        inspected += 1
        reference_coordinates = extract_reference_coordinates(sample.scene.raw_answer)
        predicted_ids = normalize_ids(prediction)
        reference_count = len(reference_coordinates)
        predicted_count = len(predicted_ids)
        total_reference_mentions += reference_count
        total_predicted_mentions += predicted_count
        reference_count_histogram[str(reference_count)] = reference_count_histogram.get(str(reference_count), 0) + 1
        predicted_count_histogram[str(predicted_count)] = predicted_count_histogram.get(str(predicted_count), 0) + 1

        if reference_count == predicted_count:
            bucket = "exact_count"
        elif predicted_count < reference_count:
            bucket = "under_predicted_count"
        else:
            bucket = "over_predicted_count"
        buckets[bucket] += 1

        if reference_count > 0 and predicted_count == 0:
            buckets["empty_prediction_with_reference"] += 1
            extra_bucket = "empty_prediction_with_reference"
        elif reference_count == 0 and predicted_count > 0:
            buckets["prediction_without_reference"] += 1
            extra_bucket = "prediction_without_reference"
        else:
            extra_bucket = ""

        example = {
            "sample_id": sample_id,
            "question": sample.scene.raw_question,
            "reference_answer": sample.scene.raw_answer,
            "reference_coordinates": reference_coordinates,
            "predicted_ids": predicted_ids,
            "predicted_answer": str(prediction.get("answer_text", "")),
            "reference_count": reference_count,
            "predicted_count": predicted_count,
        }
        if len(examples[bucket]) < args.show_samples:
            examples[bucket].append(example)
        if extra_bucket and len(examples[extra_bucket]) < args.show_samples:
            examples[extra_bucket].append(example)

    report = {
        "repository_root": str(repository_root),
        "manifest": str(manifest_path),
        "prediction_path": str(prediction_path),
        "split": split,
        "inspected_samples": inspected,
        "total_reference_mentions": total_reference_mentions,
        "total_predicted_mentions": total_predicted_mentions,
        "buckets": buckets,
        "reference_count_histogram": dict(sorted(reference_count_histogram.items(), key=lambda item: int(item[0]))),
        "predicted_count_histogram": dict(sorted(predicted_count_histogram.items(), key=lambda item: int(item[0]))),
        "examples": examples,
    }

    output_dir = manifest_path.parent
    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_sections = [
        "# Phase 8 Occluding Mismatch Inspection",
        "",
        f"- `manifest`: `{manifest_path}`",
        f"- `prediction_path`: `{prediction_path}`",
        f"- `repository_root`: `{repository_root}`",
        f"- `split`: `{split}`",
        "",
        "This report buckets archived occluding-object predictions by reference/predicted answer count. It is an inspection aid for Phase 8 selector work, not an official benchmark score.",
        "",
        "## Summary",
        "",
        markdown_table(
            [
                ["Metric", "Value"],
                ["Inspected samples", str(inspected)],
                ["Reference coordinate mentions", str(total_reference_mentions)],
                ["Predicted object mentions", str(total_predicted_mentions)],
                ["Exact count matches", str(buckets["exact_count"])],
                ["Under-predicted counts", str(buckets["under_predicted_count"])],
                ["Over-predicted counts", str(buckets["over_predicted_count"])],
                ["Empty predictions with reference", str(buckets["empty_prediction_with_reference"])],
                ["Predictions without reference", str(buckets["prediction_without_reference"])],
            ]
        ),
        "",
        "## Count Histograms",
        "",
        markdown_table(
            [["Reference Count", "Samples"]]
            + [[count, str(value)] for count, value in report["reference_count_histogram"].items()]
        ),
        "",
        markdown_table(
            [["Predicted Count", "Samples"]]
            + [[count, str(value)] for count, value in report["predicted_count_histogram"].items()]
        ),
        "",
    ]

    for bucket_name in (
        "under_predicted_count",
        "over_predicted_count",
        "empty_prediction_with_reference",
        "prediction_without_reference",
    ):
        markdown_sections.extend([f"## {bucket_name}", ""])
        bucket_examples = examples[bucket_name]
        if not bucket_examples:
            markdown_sections.extend(["No examples captured.", ""])
            continue
        for example in bucket_examples:
            markdown_sections.extend(
                [
                    f"- `sample_id={example['sample_id']}` ref_count=`{example['reference_count']}` pred_count=`{example['predicted_count']}` pred_ids=`{list(example['predicted_ids'])}`",
                    f"  - reference: {example['reference_answer']}",
                    f"  - predicted: {example['predicted_answer']}",
                ]
            )
        markdown_sections.append("")

    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    print("=" * 72)
    print("Phase 8 Occluding Mismatch Inspection")
    print("=" * 72)
    print(f"inspected_samples: {inspected}")
    print(f"reference_mentions: {total_reference_mentions}")
    print(f"predicted_mentions: {total_predicted_mentions}")
    print(f"exact_count: {buckets['exact_count']}")
    print(f"under_predicted_count: {buckets['under_predicted_count']}")
    print(f"over_predicted_count: {buckets['over_predicted_count']}")
    print(f"empty_prediction_with_reference: {buckets['empty_prediction_with_reference']}")
    print(f"prediction_without_reference: {buckets['prediction_without_reference']}")
    print(f"saved_json: {json_path}")
    print(f"saved_markdown: {markdown_path}")


if __name__ == "__main__":
    main()
