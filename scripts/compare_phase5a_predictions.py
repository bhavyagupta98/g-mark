#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two Phase 5A V2V-GoT-QA prediction JSONL files."
    )
    parser.add_argument("--left", required=True, help="Path to the first JSONL file.")
    parser.add_argument("--right", required=True, help="Path to the second JSONL file.")
    parser.add_argument(
        "--show-differences",
        type=int,
        default=20,
        help="Maximum number of differing samples to print.",
    )
    return parser


def load_predictions(path: Path) -> dict[str, dict[str, object]]:
    predictions: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in predictions:
                raise ValueError(f"Duplicate sample_id `{sample_id}` in {path} at line {line_number}.")
            predictions[sample_id] = record
    return predictions


def normalize_object_ids(record: dict[str, object]) -> tuple[str, ...]:
    object_ids = record.get("object_ids", [])
    if not isinstance(object_ids, list):
        return ()
    return tuple(str(item) for item in object_ids)


def normalize_object_id_set(record: dict[str, object]) -> tuple[str, ...]:
    return tuple(sorted(set(normalize_object_ids(record))))


def main() -> None:
    args = build_parser().parse_args()
    left_path = Path(args.left).expanduser().resolve()
    right_path = Path(args.right).expanduser().resolve()

    left_predictions = load_predictions(left_path)
    right_predictions = load_predictions(right_path)

    left_ids = set(left_predictions)
    right_ids = set(right_predictions)
    common_ids = tuple(sorted(left_ids & right_ids))
    left_only = tuple(sorted(left_ids - right_ids))
    right_only = tuple(sorted(right_ids - left_ids))

    exact_answer_matches = 0
    ordered_object_id_matches = 0
    unordered_object_id_matches = 0
    support_matches = 0
    ordering_only_differences = 0
    semantic_differences = 0
    ordering_only_samples: list[tuple[str, dict[str, object], dict[str, object]]] = []
    semantic_differing_samples: list[tuple[str, dict[str, object], dict[str, object], list[str]]] = []
    task_counter: Counter[str] = Counter()
    differing_task_counter: Counter[str] = Counter()
    semantic_differing_task_counter: Counter[str] = Counter()

    for sample_id in common_ids:
        left = left_predictions[sample_id]
        right = right_predictions[sample_id]
        task_type = str(left.get("task_type", "unknown"))
        task_counter[task_type] += 1

        difference_reasons: list[str] = []
        if left.get("answer_text") == right.get("answer_text"):
            exact_answer_matches += 1
        else:
            difference_reasons.append("answer_text")

        if normalize_object_ids(left) == normalize_object_ids(right):
            ordered_object_id_matches += 1
        else:
            difference_reasons.append("object_ids")

        if normalize_object_id_set(left) == normalize_object_id_set(right):
            unordered_object_id_matches += 1
            if "object_ids" in difference_reasons:
                ordering_only_differences += 1
        else:
            if "object_ids" not in difference_reasons:
                difference_reasons.append("object_id_set")

        if bool(left.get("supported")) == bool(right.get("supported")):
            support_matches += 1
        else:
            difference_reasons.append("supported")

        if difference_reasons:
            differing_task_counter[task_type] += 1

            ordered_ids_match = normalize_object_ids(left) == normalize_object_ids(right)
            unordered_ids_match = normalize_object_id_set(left) == normalize_object_id_set(right)
            support_match = bool(left.get("supported")) == bool(right.get("supported"))

            if (
                not ordered_ids_match
                and unordered_ids_match
                and support_match
            ):
                ordering_only_samples.append((sample_id, left, right))
            else:
                semantic_differences += 1
                semantic_differing_samples.append((sample_id, left, right, difference_reasons))
                semantic_differing_task_counter[task_type] += 1

    print("=" * 72)
    print("Phase 5A Prediction Comparison")
    print("=" * 72)
    print(f"left_file: {left_path}")
    print(f"right_file: {right_path}")
    print(f"left_count: {len(left_predictions)}")
    print(f"right_count: {len(right_predictions)}")
    print(f"common_count: {len(common_ids)}")
    print(f"left_only_count: {len(left_only)}")
    print(f"right_only_count: {len(right_only)}")
    print(f"exact_answer_matches: {exact_answer_matches}")
    print(f"ordered_object_id_matches: {ordered_object_id_matches}")
    print(f"unordered_object_id_matches: {unordered_object_id_matches}")
    print(f"support_matches: {support_matches}")
    print(f"ordering_only_differences: {ordering_only_differences}")
    print(f"raw_differing_samples: {sum(differing_task_counter.values())}")
    print(f"semantic_differences: {semantic_differences}")

    if task_counter:
        print()
        print("Per-Task Coverage")
        print("-" * 72)
        for task_type in sorted(task_counter):
            print(
                f"- {task_type}: total={task_counter[task_type]}, "
                f"raw_differing={differing_task_counter[task_type]}, "
                f"semantic_differing={semantic_differing_task_counter[task_type]}"
            )

    if left_only:
        print()
        print("Left-Only Sample IDs")
        print("-" * 72)
        print(", ".join(left_only[:20]))

    if right_only:
        print()
        print("Right-Only Sample IDs")
        print("-" * 72)
        print(", ".join(right_only[:20]))

    if ordering_only_samples:
        print()
        print("Ordering-Only Differences")
        print("-" * 72)
        for sample_id, left, right in ordering_only_samples[: args.show_differences]:
            print(f"[{left.get('task_type', 'unknown')}] sample_id={sample_id}")
            print(f"left objects: {left.get('object_ids')}")
            print(f"right objects: {right.get('object_ids')}")
            print()

    if semantic_differing_samples:
        print()
        print("Semantic Differences")
        print("-" * 72)
        for sample_id, left, right, reasons in semantic_differing_samples[: args.show_differences]:
            print(
                f"[{left.get('task_type', 'unknown')}] sample_id={sample_id} "
                f"reasons={reasons}"
            )
            print(
                f"left ({left.get('baseline_mode', 'unknown')}): "
                f"supported={left.get('supported')} objects={left.get('object_ids')}"
            )
            print(f"answer: {left.get('answer_text')}")
            print(
                f"right ({right.get('baseline_mode', 'unknown')}): "
                f"supported={right.get('supported')} objects={right.get('object_ids')}"
            )
            print(f"answer: {right.get('answer_text')}")
            print()


if __name__ == "__main__":
    main()
