#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List sample IDs from a Phase 8 occluding mismatch report bucket."
    )
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--bucket",
        default="under_predicted_count",
        choices=(
            "under_predicted_count",
            "over_predicted_count",
            "empty_prediction_with_reference",
            "prediction_without_reference",
        ),
    )
    parser.add_argument("--limit", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    examples = report.get("examples", {}).get(args.bucket, [])
    if not isinstance(examples, list):
        return
    for example in examples[: args.limit]:
        if isinstance(example, dict) and "sample_id" in example:
            print(example["sample_id"])


if __name__ == "__main__":
    main()
