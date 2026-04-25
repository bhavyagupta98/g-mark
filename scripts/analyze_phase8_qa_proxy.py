#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProxySummary:
    task_type: str
    total_samples: int
    reference_positive: int
    predicted_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    exact_presence_matches: int
    exact_count_matches: int
    reference_coordinate_mentions: int
    predicted_object_mentions: int
    ambiguous_reference_samples: int


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
            "Analyze archived Phase 8 QA outputs with a local proxy scorer based on "
            "benchmark answer coordinates and empty-vs-positive matching."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--show-samples", type=int, default=5)
    parser.add_argument("--json-name", default="phase8_qa_proxy_report.json")
    parser.add_argument("--markdown-name", default="phase8_qa_proxy_report.md")
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


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    split = str(manifest.get("split", "val"))
    samples = adapter.load_samples(split_name=split, file_name=args.file_name)
    samples_by_task_and_id = {
        (sample.task_type.value, sample.sample_id): sample
        for sample in samples
    }

    report: dict[str, object] = {
        "repository_root": str(repository_root),
        "manifest": str(manifest_path),
        "split": split,
        "tasks": {},
    }
    markdown_sections = [
        "# Phase 8 QA Proxy Analysis",
        "",
        f"- `manifest`: `{manifest_path}`",
        f"- `repository_root`: `{repository_root}`",
        f"- `split`: `{split}`",
        "",
        "This is a local proxy analysis, not an official benchmark scorer. It uses coordinate mentions in benchmark reference answers plus empty-vs-positive matching to estimate which QA tasks are most promising for improvement.",
        "",
    ]

    print("=" * 72)
    print("Phase 8 QA Proxy Analysis")
    print("=" * 72)
    print(f"manifest: {manifest_path}")
    print(f"repository_root: {repository_root}")
    print(f"split: {split}")

    for run in manifest.get("runs", []):
        if not isinstance(run, dict):
            continue
        task_type_value = str(run["task_type"])
        task_type = BenchmarkTaskType(task_type_value)
        prediction_records = load_jsonl(Path(str(run["output_jsonl"])))

        tp = fp = fn = tn = 0
        exact_presence_matches = 0
        exact_count_matches = 0
        reference_coordinate_mentions = 0
        predicted_object_mentions = 0
        ambiguous_reference_samples = 0
        false_negative_examples: list[dict[str, object]] = []
        false_positive_examples: list[dict[str, object]] = []
        count_mismatch_examples: list[dict[str, object]] = []

        for sample_id, prediction in prediction_records.items():
            sample = samples_by_task_and_id.get((task_type_value, sample_id))
            if sample is None:
                continue

            reference_answer = sample.scene.raw_answer
            reference_coords = extract_reference_coordinates(reference_answer)
            reference_count = len(reference_coords)
            predicted_ids = normalize_ids(prediction)
            predicted_count = len(predicted_ids)

            reference_positive = reference_count > 0
            predicted_positive = predicted_count > 0

            if not reference_positive and "no notable object" not in reference_answer.lower():
                ambiguous_reference_samples += 1

            reference_coordinate_mentions += reference_count
            predicted_object_mentions += predicted_count

            if reference_positive and predicted_positive:
                tp += 1
            elif not reference_positive and predicted_positive:
                fp += 1
            elif reference_positive and not predicted_positive:
                fn += 1
            else:
                tn += 1

            if reference_positive == predicted_positive:
                exact_presence_matches += 1

            if reference_count == predicted_count:
                exact_count_matches += 1
            elif len(count_mismatch_examples) < args.show_samples:
                count_mismatch_examples.append(
                    {
                        "sample_id": sample_id,
                        "question": sample.scene.raw_question,
                        "reference_answer": reference_answer,
                        "reference_count": reference_count,
                        "predicted_count": predicted_count,
                        "predicted_ids": list(predicted_ids),
                    }
                )

            if reference_positive and not predicted_positive and len(false_negative_examples) < args.show_samples:
                false_negative_examples.append(
                    {
                        "sample_id": sample_id,
                        "question": sample.scene.raw_question,
                        "reference_answer": reference_answer,
                        "predicted_answer": str(prediction.get("answer_text", "")),
                    }
                )
            if not reference_positive and predicted_positive and len(false_positive_examples) < args.show_samples:
                false_positive_examples.append(
                    {
                        "sample_id": sample_id,
                        "question": sample.scene.raw_question,
                        "reference_answer": reference_answer,
                        "predicted_answer": str(prediction.get("answer_text", "")),
                        "predicted_ids": list(predicted_ids),
                    }
                )

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * tp, (2 * tp) + fp + fn)
        summary = ProxySummary(
            task_type=task_type.value,
            total_samples=len(prediction_records),
            reference_positive=tp + fn,
            predicted_positive=tp + fp,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            true_negative=tn,
            precision=precision,
            recall=recall,
            f1=f1,
            exact_presence_matches=exact_presence_matches,
            exact_count_matches=exact_count_matches,
            reference_coordinate_mentions=reference_coordinate_mentions,
            predicted_object_mentions=predicted_object_mentions,
            ambiguous_reference_samples=ambiguous_reference_samples,
        )

        report["tasks"][task_type.value] = {
            "summary": summary.__dict__,
            "false_negative_examples": false_negative_examples,
            "false_positive_examples": false_positive_examples,
            "count_mismatch_examples": count_mismatch_examples,
        }

        print()
        print(f"[TASK] {task_type.value}")
        print(
            f"  proxy_presence_f1={summary.f1:.3f}, "
            f"precision={summary.precision:.3f}, "
            f"recall={summary.recall:.3f}, "
            f"ref_positive={summary.reference_positive}, "
            f"pred_positive={summary.predicted_positive}"
        )

        rows = [[
            "Task",
            "Presence F1",
            "Precision",
            "Recall",
            "Ref Positive",
            "Pred Positive",
            "FN",
            "FP",
            "Presence Match",
            "Count Match",
        ], [
            task_type.value,
            f"{summary.f1:.3f}",
            f"{summary.precision:.3f}",
            f"{summary.recall:.3f}",
            str(summary.reference_positive),
            str(summary.predicted_positive),
            str(summary.false_negative),
            str(summary.false_positive),
            f"{summary.exact_presence_matches}/{summary.total_samples}",
            f"{summary.exact_count_matches}/{summary.total_samples}",
        ]]

        markdown_sections.append(f"## {task_type.value}")
        markdown_sections.append("")
        markdown_sections.append(markdown_table(rows))
        markdown_sections.append("")
        if false_negative_examples:
            markdown_sections.append("Representative false negatives:")
            for example in false_negative_examples:
                markdown_sections.append(
                    f"- `sample_id={example['sample_id']}` reference=`{example['reference_answer']}`"
                )
            markdown_sections.append("")
        if false_positive_examples:
            markdown_sections.append("Representative false positives:")
            for example in false_positive_examples:
                markdown_sections.append(
                    f"- `sample_id={example['sample_id']}` predicted_ids=`{example['predicted_ids']}`"
                )
            markdown_sections.append("")

    json_path = manifest_path.with_name(args.json_name)
    markdown_path = manifest_path.with_name(args.markdown_name)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    print()
    print(f"saved_json: {json_path}")
    print(f"saved_markdown: {markdown_path}")


if __name__ == "__main__":
    main()
