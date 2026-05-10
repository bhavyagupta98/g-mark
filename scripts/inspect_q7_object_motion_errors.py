#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

OBJECT_RE = re.compile(
    r"There is a\s+(?:car|object)\s+at\s*"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
    r"(?P<body>.*?)(?=There is a|$)",
    re.IGNORECASE | re.DOTALL,
)
POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Q7 object-motion official exports and split errors into "
            "presence, localization, action, and future-trajectory buckets."
        )
    )
    parser.add_argument("--official-jsonl", required=True)
    parser.add_argument("--match-threshold", type=float, default=4.0)
    parser.add_argument("--worst", type=int, default=12)
    return parser


def _action_label(text: str) -> str:
    for label in ("moving forward", "turning left", "turning right", "staying at the same location"):
        if label in text.lower():
            return label
    return "unknown"


def parse_objects(answer: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for match in OBJECT_RE.finditer(answer):
        body = str(match.group("body"))
        points = [(float(x), float(y)) for x, y in POINT_RE.findall(body)]
        rows.append(
            {
                "x": float(match.group("x")),
                "y": float(match.group("y")),
                "action": _action_label(body),
                "future": points,
            }
        )
    return rows


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def greedy_match(
    gt_rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_rows):
        gt_xy = (float(gt["x"]), float(gt["y"]))
        for pi, pred in enumerate(pred_rows):
            pred_xy = (float(pred["x"]), float(pred["y"]))
            candidates.append((_dist(gt_xy, pred_xy), gi, pi))
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for distance, gi, pi in sorted(candidates):
        if distance >= threshold or gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, distance))
    return matches


def future_l2(gt: dict[str, object], pred: dict[str, object]) -> float | None:
    gt_points = gt.get("future")
    pred_points = pred.get("future")
    if not isinstance(gt_points, list) or not isinstance(pred_points, list):
        return None
    count = min(len(gt_points), len(pred_points))
    if count == 0:
        return None
    errors = [
        _dist(
            (float(gt_points[index][0]), float(gt_points[index][1])),
            (float(pred_points[index][0]), float(pred_points[index][1])),
        )
        for index in range(count)
    ]
    return sum(errors) / len(errors)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def main() -> int:
    args = build_parser().parse_args()
    path = Path(args.official_jsonl).expanduser().resolve()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    binary_tp = binary_fp = binary_tn = binary_fn = 0
    gt_counts: list[int] = []
    pred_counts: list[int] = []
    gt_waypoint_counts: list[int] = []
    pred_waypoint_counts: list[int] = []
    localization_errors: list[float] = []
    future_errors: list[float] = []
    action_correct = 0
    action_total = 0
    worst_rows: list[tuple[float, str, int, int, int]] = []

    for idx, record in enumerate(records):
        conversations = record.get("conversations", [])
        gt_answer = ""
        if isinstance(conversations, list) and len(conversations) > 1 and isinstance(conversations[1], dict):
            gt_answer = str(conversations[1].get("value", ""))
        pred_answer = str(record.get("outputs", ""))
        gt_rows = parse_objects(gt_answer)
        pred_rows = parse_objects(pred_answer)
        gt_counts.append(len(gt_rows))
        pred_counts.append(len(pred_rows))
        gt_waypoint_counts.extend(len(row.get("future", [])) for row in gt_rows)
        pred_waypoint_counts.extend(len(row.get("future", [])) for row in pred_rows)

        if gt_rows and pred_rows:
            binary_tp += 1
        elif gt_rows:
            binary_fn += 1
        elif pred_rows:
            binary_fp += 1
        else:
            binary_tn += 1

        matches = greedy_match(gt_rows, pred_rows, args.match_threshold)
        for gi, pi, loc_error in matches:
            localization_errors.append(loc_error)
            gt_action = str(gt_rows[gi].get("action"))
            pred_action = str(pred_rows[pi].get("action"))
            action_total += 1
            if gt_action == pred_action:
                action_correct += 1
            traj_error = future_l2(gt_rows[gi], pred_rows[pi])
            if traj_error is not None:
                future_errors.append(traj_error)
                sample_id = str(record.get("id", record.get("sample_id", idx)))
                worst_rows.append((traj_error, sample_id, idx, len(gt_rows), len(pred_rows)))

    precision = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) else 0.0
    recall = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) else 0.0
    action_accuracy = action_correct / action_total if action_total else 0.0

    print(f"official_jsonl: {path}")
    print(f"records: {len(records)}")
    print(f"binary_tp={binary_tp} binary_fp={binary_fp} binary_fn={binary_fn} binary_tn={binary_tn}")
    print(f"binary_precision={precision:.6f} binary_recall={recall:.6f}")
    print(f"avg_gt_objects={sum(gt_counts) / len(gt_counts):.6f}")
    print(f"avg_pred_objects={sum(pred_counts) / len(pred_counts):.6f}")
    print(f"gt_waypoint_count_hist={_histogram(gt_waypoint_counts)}")
    print(f"pred_waypoint_count_hist={_histogram(pred_waypoint_counts)}")
    print(f"matched_objects={len(localization_errors)}")
    print(f"localization_l2_avg={_avg(localization_errors):.6f} p90={percentile(localization_errors, 0.9):.6f}")
    print(f"future_l2_avg={_avg(future_errors):.6f} p90={percentile(future_errors, 0.9):.6f}")
    print(f"action_accuracy_on_matches={action_accuracy:.6f}")
    print("worst_future_l2:")
    for error, sample_id, idx, gt_count, pred_count in sorted(worst_rows, reverse=True)[: args.worst]:
        print(f"  error={error:.6f} sample_id={sample_id} row={idx} gt_objects={gt_count} pred_objects={pred_count}")
    return 0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _histogram(values: list[int]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for value in values:
        hist[int(value)] = hist.get(int(value), 0) + 1
    return dict(sorted(hist.items()))


if __name__ == "__main__":
    raise SystemExit(main())
