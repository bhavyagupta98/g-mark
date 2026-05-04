#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


NUMERIC_FEATURES = (
    "rank",
    "role_score",
    "relative_x",
    "relative_y",
    "abs_relative_x",
    "abs_relative_y",
    "distance_to_asker",
    "distance_to_trajectory",
    "support_count",
    "confidence",
    "conflict_score",
    "uncertainty_score",
    "age_frames",
    "miss_count",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Q3 invisible-object candidate feature tables by TP/FP/FN row groups."
    )
    parser.add_argument("--features-jsonl", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    return parser


def row_group(row: dict[str, object]) -> str:
    if row.get("row_type") == "unmatched_gt":
        return "fn_unmatched_gt"
    if row.get("candidate_matches_gt") is True:
        return "tp_candidate"
    if row.get("selected_by_policy") is True:
        return "fp_selected_candidate"
    return "tn_unselected_candidate"


def bucket_row(row: dict[str, object]) -> list[str]:
    buckets: list[str] = []
    for key in ("status", "support_count", "selected_by_policy", "candidate_matches_gt"):
        if key in row:
            buckets.append(f"{key}={row[key]}")
    abs_y = row.get("abs_relative_y")
    if isinstance(abs_y, (float, int)):
        if abs_y < 1.0:
            buckets.append("abs_y=<1m")
        elif abs_y < 3.0:
            buckets.append("abs_y=1-3m")
        else:
            buckets.append("abs_y=>=3m")
    relative_x = row.get("relative_x")
    if isinstance(relative_x, (float, int)):
        if relative_x < -1.0:
            buckets.append("longitudinal=behind")
        elif relative_x > 1.0:
            buckets.append("longitudinal=ahead")
        else:
            buckets.append("longitudinal=near_zero")
    distance_to_trajectory = row.get("distance_to_trajectory")
    if isinstance(distance_to_trajectory, (float, int)):
        if distance_to_trajectory < 2.0:
            buckets.append("trajectory=<2m")
        elif distance_to_trajectory <= 3.0:
            buckets.append("trajectory=2-3m")
        elif distance_to_trajectory <= 6.0:
            buckets.append("trajectory=3-6m")
        else:
            buckets.append("trajectory=>6m")
    return buckets


def summarize_numeric(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for feature in NUMERIC_FEATURES:
        values = [float(row[feature]) for row in rows if isinstance(row.get(feature), (float, int))]
        if not values:
            continue
        values_sorted = sorted(values)
        summary[feature] = {
            "mean": round(mean(values), 6),
            "p10": round(values_sorted[int(0.10 * (len(values_sorted) - 1))], 6),
            "p50": round(values_sorted[int(0.50 * (len(values_sorted) - 1))], 6),
            "p90": round(values_sorted[int(0.90 * (len(values_sorted) - 1))], 6),
        }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    feature_path = Path(args.features_jsonl).expanduser().resolve()
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_rows = 0

    with feature_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            group = row_group(row)
            grouped_rows[group].append(row)
            for bucket in bucket_row(row):
                bucket_counts[group][bucket] += 1
            total_rows += 1

    groups = {
        group: {
            "count": len(rows),
            "numeric": summarize_numeric(rows),
            "buckets": dict(bucket_counts[group].most_common()),
        }
        for group, rows in sorted(grouped_rows.items())
    }
    payload = {
        "features_jsonl": str(feature_path),
        "total_rows": total_rows,
        "groups": groups,
    }

    if args.output_json:
        json_path = Path(args.output_json).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_json: {json_path}")

    if args.output_markdown:
        markdown_path = Path(args.output_markdown).expanduser()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Phase 8 Invisible Candidate Feature Analysis",
            "",
            f"- `features_jsonl`: `{feature_path}`",
            f"- `total_rows`: `{total_rows}`",
            "",
            "| Group | Count | Role Score Mean | D Traj Mean | D Asker Mean | Abs Y Mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for group, summary in groups.items():
            numeric = summary["numeric"]
            lines.append(
                "| "
                + f"`{group}` | `{summary['count']}` | "
                + f"`{numeric.get('role_score', {}).get('mean', '')}` | "
                + f"`{numeric.get('distance_to_trajectory', {}).get('mean', '')}` | "
                + f"`{numeric.get('distance_to_asker', {}).get('mean', '')}` | "
                + f"`{numeric.get('abs_relative_y', {}).get('mean', '')}` |"
            )
        lines.append("")
        lines.append("## Top Buckets")
        lines.append("")
        for group, summary in groups.items():
            lines.append(f"### `{group}`")
            for bucket, count in list(summary["buckets"].items())[:20]:
                lines.append(f"- `{bucket}`: `{count}`")
            lines.append("")
        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"saved_markdown: {markdown_path}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
