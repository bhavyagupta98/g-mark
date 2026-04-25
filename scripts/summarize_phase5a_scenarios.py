#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.domain.benchmark_references import references_for_task  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a Phase 5A scenario matrix manifest.")
    parser.add_argument("--manifest", required=True, help="Path to scenario_manifest.json")
    parser.add_argument(
        "--baseline",
        default="risk_diverse_top2_cooperative",
        help="Scenario name to treat as the comparison baseline.",
    )
    return parser


def load_jsonl(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            records[sample_id] = record
    return records


def normalize_ids(record: dict[str, object]) -> tuple[str, ...]:
    values = record.get("object_ids", [])
    if not isinstance(values, list):
        return ()
    return tuple(str(item) for item in values)


def summarize_against_baseline(
    baseline_records: dict[str, dict[str, object]],
    candidate_records: dict[str, dict[str, object]],
) -> dict[str, int]:
    common_ids = sorted(set(baseline_records) & set(candidate_records))
    exact_matches = 0
    set_matches = 0
    semantic_differences = 0
    for sample_id in common_ids:
        left = baseline_records[sample_id]
        right = candidate_records[sample_id]
        left_ids = normalize_ids(left)
        right_ids = normalize_ids(right)
        left_set = tuple(sorted(set(left_ids)))
        right_set = tuple(sorted(set(right_ids)))
        if left.get("answer_text") == right.get("answer_text") and left_ids == right_ids:
            exact_matches += 1
        if left_set == right_set:
            set_matches += 1
        else:
            semantic_differences += 1
    return {
        "common_samples": len(common_ids),
        "exact_matches": exact_matches,
        "unordered_set_matches": set_matches,
        "semantic_differences": semantic_differences,
    }


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = manifest.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise SystemExit("Manifest contains no scenarios.")

    by_name = {str(item["name"]): item for item in scenarios if isinstance(item, dict)}
    if args.baseline not in by_name:
        raise SystemExit(f"Baseline scenario `{args.baseline}` not found in manifest.")

    baseline_info = by_name[args.baseline]
    baseline_path = Path(str(baseline_info["output_jsonl"])).expanduser().resolve()
    baseline_records = load_jsonl(baseline_path)

    print("=" * 72)
    print("Phase 5A Scenario Summary")
    print("=" * 72)
    print(f"manifest: {manifest_path}")
    print(f"split: {manifest.get('split', 'unknown')}")
    print(f"task_type: {manifest.get('task_type', 'unknown')}")
    print(f"sample_count: {manifest.get('sample_count', 'unknown')}")
    print(f"baseline: {args.baseline}")

    task_type = BenchmarkTaskType(str(manifest.get("task_type", "unknown")))
    paper_references = references_for_task(task_type)
    if paper_references:
        print()
        print("Published References")
        print("-" * 72)
        print(
            "These are paper-reported benchmark references from V2V-GoT and are "
            "shown as target context only. The current Phase 5 scenario outputs "
            "are structural predictions/diff summaries, not official reproduced "
            "F1/L2 benchmark scores yet."
        )
        for reference in paper_references:
            direction = "higher is better" if reference.higher_is_better else "lower is better"
            print(
                f"- {reference.method_name} | {reference.metric_name} = {reference.metric_value} "
                f"({direction}) [{reference.source_table}]"
            )
            if reference.notes:
                print(f"  note: {reference.notes}")
    print()
    print("Scenarios")
    print("-" * 72)

    ordered = sorted(
        by_name.values(),
        key=lambda item: (
            item["name"] != args.baseline,
            str(item["name"]),
        ),
    )
    for item in ordered:
        name = str(item["name"])
        output_path = Path(str(item["output_jsonl"])).expanduser().resolve()
        if name == args.baseline:
            print(
                f"- {name}: baseline, "
                f"baseline_mode={item['baseline_mode']}, "
                f"ranker={item['planning_ranker']}, "
                f"policy={item['planning_selection_policy']}"
            )
            continue
        stats = summarize_against_baseline(baseline_records, load_jsonl(output_path))
        print(
            f"- {name}: "
            f"baseline_mode={item['baseline_mode']}, "
            f"ranker={item['planning_ranker']}, "
            f"policy={item['planning_selection_policy']}, "
            f"exact_matches={stats['exact_matches']}/{stats['common_samples']}, "
            f"set_matches={stats['unordered_set_matches']}/{stats['common_samples']}, "
            f"semantic_differences={stats['semantic_differences']}"
        )


if __name__ == "__main__":
    main()
