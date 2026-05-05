#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.control_settings_policy import (  # noqa: E402
    SPEED_CLASSES,
    STEERING_CLASSES,
    build_control_feature_vector,
    parse_speed_steering_idx,
)
from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Q8 control_settings official exports and summarize confusion/error buckets "
            "for speed/steering/action."
        )
    )
    parser.add_argument("--official-jsonl", default="")
    parser.add_argument(
        "--export-manifest",
        default="",
        help="Optional official export manifest. If set, control_settings JSONL is resolved from it.",
    )
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument(
        "--skip-feature-buckets",
        action="store_true",
        help="Only compute label-level stats. Skip scene feature extraction.",
    )
    return parser


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_jsonl_from_manifest(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for run in manifest.get("runs", []):
        if isinstance(run, dict) and run.get("task_type") == "control_settings":
            output_jsonl = run.get("output_jsonl")
            if output_jsonl:
                return resolve_repo_path(str(output_jsonl))
    raise SystemExit(f"No control_settings output_jsonl found in {manifest_path}")


def resolve_official_jsonl(args: argparse.Namespace) -> Path:
    if args.official_jsonl:
        return resolve_repo_path(args.official_jsonl)
    if args.export_manifest:
        return resolve_jsonl_from_manifest(resolve_repo_path(args.export_manifest))
    raise SystemExit("Provide either --official-jsonl or --export-manifest.")


def resolve_v2vgot_root(raw_value: str) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def reference_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if isinstance(item, dict) and item.get("from") in {"gpt", "assistant"}:
            value = item.get("value", "")
            return value if isinstance(value, str) else ""
    return ""


def record_sample_id(record: dict[str, Any]) -> str:
    kg_prediction = record.get("kg_prediction", {})
    if isinstance(kg_prediction, dict) and kg_prediction.get("sample_id") is not None:
        return str(kg_prediction["sample_id"])
    return str(record.get("sample_id", record.get("id", "")))


def parse_prediction(record: dict[str, Any]) -> tuple[int, int]:
    return parse_speed_steering_idx(str(record.get("outputs", "")))


def update_bucket_counts(
    buckets: Counter[str],
    gt_speed: int,
    pred_speed: int,
    gt_steer: int,
    pred_steer: int,
) -> None:
    speed_delta = pred_speed - gt_speed
    steering_delta = pred_steer - gt_steer
    if speed_delta == 0:
        buckets["speed|exact"] += 1
    elif speed_delta < 0:
        buckets["speed|predicted_faster_than_gt"] += 1
    else:
        buckets["speed|predicted_slower_than_gt"] += 1
    if abs(speed_delta) >= 2:
        buckets["speed|large_error_abs>=2"] += 1
    elif abs(speed_delta) == 1:
        buckets["speed|small_error_abs=1"] += 1

    if steering_delta == 0:
        buckets["steering|exact"] += 1
    elif steering_delta < 0:
        buckets["steering|more_left_than_gt"] += 1
    else:
        buckets["steering|more_right_than_gt"] += 1
    if abs(steering_delta) >= 2:
        buckets["steering|large_error_abs>=2"] += 1
    elif abs(steering_delta) == 1:
        buckets["steering|small_error_abs=1"] += 1


def speed_error_bucket(delta: int) -> str:
    if delta == 0:
        return "exact"
    if delta < 0:
        if delta <= -2:
            return "faster_by_2plus"
        return "faster_by_1"
    if delta >= 2:
        return "slower_by_2plus"
    return "slower_by_1"


def steer_error_bucket(delta: int) -> str:
    if delta == 0:
        return "exact"
    if delta < 0:
        if delta <= -2:
            return "left_by_2plus"
        return "left_by_1"
    if delta >= 2:
        return "right_by_2plus"
    return "right_by_1"


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Phase 9 Q8 Control-Settings Mismatch Report",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["counts"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## Speed Confusion (gt -> pred)", ""])
    lines.extend(["| gt \\\\ pred | " + " | ".join(SPEED_CLASSES) + " |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    speed_conf = report["speed_confusion"]
    for gt_name in SPEED_CLASSES:
        row = speed_conf.get(gt_name, {})
        lines.append("| " + gt_name + " | " + " | ".join(str(row.get(pred_name, 0)) for pred_name in SPEED_CLASSES) + " |")

    lines.extend(["", "## Steering Confusion (gt -> pred)", ""])
    lines.extend(
        [
            "| gt \\\\ pred | " + " | ".join(STEERING_CLASSES) + " |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    steering_conf = report["steering_confusion"]
    for gt_name in STEERING_CLASSES:
        row = steering_conf.get(gt_name, {})
        lines.append(
            "| " + gt_name + " | " + " | ".join(str(row.get(pred_name, 0)) for pred_name in STEERING_CLASSES) + " |"
        )

    lines.extend(["", "## Buckets", "", "| Bucket | Count |", "| --- | ---: |"])
    for key, value in report["buckets"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## Examples", ""])
    for item in report["examples"]:
        lines.extend(
            [
                f"### sample `{item['sample_id']}`",
                "",
                f"- gt speed/steering: `{item['gt_speed']}` / `{item['gt_steering']}`",
                f"- pred speed/steering: `{item['pred_speed']}` / `{item['pred_steering']}`",
                f"- speed delta: `{item['speed_delta']}`",
                f"- steering delta: `{item['steering_delta']}`",
                f"- output: {item['output_text']}",
                f"- reference: {item['reference_text']}",
                f"- feature_snapshot: `{item.get('feature_snapshot', {})}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    official_jsonl = resolve_official_jsonl(args)
    if not official_jsonl.exists():
        raise SystemExit(f"Official JSONL not found: {official_jsonl}")

    feature_by_sample: dict[str, dict[str, Any]] = {}
    if not args.skip_feature_buckets:
        v2vgot_root = resolve_v2vgot_root(args.v2vgot_root)
        adapter = V2VGoTQABenchmarkAdapter(str(v2vgot_root))
        evaluator = V2VGoTQAPhase5AEvaluator(str(v2vgot_root))
        for sample in adapter.load_samples(split_name=args.split, file_name=args.file_name):
            if sample.task_type != BenchmarkTaskType.CONTROL_SETTINGS:
                continue
            prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode=args.baseline_mode)
            features = build_control_feature_vector(prepared_scene)
            if len(features) < 13:
                continue
            sid = str(sample.sample_id)
            feature_by_sample[sid] = {
                "top1_risk": round(float(features[0]), 4),
                "top1_distance_to_trajectory": round(float(features[3]), 4),
                "top1_distance_to_asker": round(float(features[4]), 4),
                "top1_confidence": round(float(features[5]), 4),
                "top1_conflict": round(float(features[7]), 4),
                "top1_support_count": int(round(float(features[8]))),
                "visibility_visible": int(round(float(features[9]))),
                "visibility_uncertain": int(round(float(features[10]))),
                "visibility_occluded": int(round(float(features[11]))),
            }

    records = load_records(official_jsonl)
    counts: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    speed_confusion: dict[str, Counter[str]] = {label: Counter() for label in SPEED_CLASSES}
    steering_confusion: dict[str, Counter[str]] = {label: Counter() for label in STEERING_CLASSES}
    examples: list[dict[str, Any]] = []

    for record in records:
        sid = record_sample_id(record)
        gt_text = reference_text(record)
        gt_speed, gt_steer = parse_speed_steering_idx(gt_text)
        pred_speed, pred_steer = parse_prediction(record)
        speed_delta = pred_speed - gt_speed
        steering_delta = pred_steer - gt_steer

        counts["samples"] += 1
        counts["speed_correct"] += int(pred_speed == gt_speed)
        counts["steering_correct"] += int(pred_steer == gt_steer)
        counts["action_correct"] += int((pred_speed == gt_speed) and (pred_steer == gt_steer))
        counts["speed_edit_dist_sum"] += abs(speed_delta)
        counts["steering_edit_dist_sum"] += abs(steering_delta)
        counts["action_edit_dist_sum"] += abs(speed_delta) + abs(steering_delta)

        speed_confusion[SPEED_CLASSES[gt_speed]][SPEED_CLASSES[pred_speed]] += 1
        steering_confusion[STEERING_CLASSES[gt_steer]][STEERING_CLASSES[pred_steer]] += 1

        update_bucket_counts(buckets, gt_speed, pred_speed, gt_steer, pred_steer)
        buckets[f"speed_gt={SPEED_CLASSES[gt_speed]}"] += 1
        buckets[f"speed_pred={SPEED_CLASSES[pred_speed]}"] += 1
        buckets[f"steering_gt={STEERING_CLASSES[gt_steer]}"] += 1
        buckets[f"steering_pred={STEERING_CLASSES[pred_steer]}"] += 1
        buckets[f"speed_error_bucket={speed_error_bucket(speed_delta)}"] += 1
        buckets[f"steering_error_bucket={steer_error_bucket(steering_delta)}"] += 1

        feature_snapshot = feature_by_sample.get(sid)
        if feature_snapshot:
            top1_risk = float(feature_snapshot.get("top1_risk", 0.0))
            dist_traj = float(feature_snapshot.get("top1_distance_to_trajectory", 999.0))
            if top1_risk < 0.2:
                buckets["ctx|top1_risk=<0.2"] += 1
            elif top1_risk < 0.5:
                buckets["ctx|top1_risk=0.2-0.5"] += 1
            else:
                buckets["ctx|top1_risk>=0.5"] += 1

            if dist_traj < 2.0:
                buckets["ctx|traj_dist=<2m"] += 1
            elif dist_traj < 6.0:
                buckets["ctx|traj_dist=2-6m"] += 1
            else:
                buckets["ctx|traj_dist>=6m"] += 1

            if speed_delta < 0:
                if top1_risk < 0.2:
                    buckets["overspeed_ctx|risk_low"] += 1
                elif top1_risk < 0.5:
                    buckets["overspeed_ctx|risk_mid"] += 1
                else:
                    buckets["overspeed_ctx|risk_high"] += 1
            elif speed_delta > 0:
                if top1_risk < 0.2:
                    buckets["underspeed_ctx|risk_low"] += 1
                elif top1_risk < 0.5:
                    buckets["underspeed_ctx|risk_mid"] += 1
                else:
                    buckets["underspeed_ctx|risk_high"] += 1

        if (speed_delta != 0 or steering_delta != 0) and len(examples) < args.examples:
            examples.append(
                {
                    "sample_id": sid,
                    "gt_speed": SPEED_CLASSES[gt_speed],
                    "gt_steering": STEERING_CLASSES[gt_steer],
                    "pred_speed": SPEED_CLASSES[pred_speed],
                    "pred_steering": STEERING_CLASSES[pred_steer],
                    "speed_delta": speed_delta,
                    "steering_delta": steering_delta,
                    "output_text": str(record.get("outputs", "")),
                    "reference_text": gt_text,
                    "feature_snapshot": feature_snapshot or {},
                }
            )

    samples = max(int(counts["samples"]), 1)
    counts["speed_accuracy"] = round(float(counts["speed_correct"]) / samples, 6)
    counts["steering_accuracy"] = round(float(counts["steering_correct"]) / samples, 6)
    counts["action_accuracy"] = round(float(counts["action_correct"]) / samples, 6)
    counts["speed_edit_dist_avg"] = round(float(counts["speed_edit_dist_sum"]) / samples, 6)
    counts["steering_edit_dist_avg"] = round(float(counts["steering_edit_dist_sum"]) / samples, 6)
    counts["action_edit_dist_avg"] = round(float(counts["action_edit_dist_sum"]) / samples, 6)
    counts["action_edit_dist_normalized_by_8"] = round(float(counts["action_edit_dist_avg"]) / 8.0, 6)

    report = {
        "official_jsonl": str(official_jsonl),
        "split": args.split,
        "baseline_mode": args.baseline_mode,
        "counts": dict(counts),
        "speed_confusion": {key: dict(value) for key, value in speed_confusion.items()},
        "steering_confusion": {key: dict(value) for key, value in steering_confusion.items()},
        "buckets": dict(sorted(buckets.items())),
        "examples": examples,
    }

    output_json = resolve_repo_path(args.output_json)
    output_markdown = resolve_repo_path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_markdown)

    print(f"samples: {counts['samples']}")
    print(f"speed_accuracy: {counts['speed_accuracy']}")
    print(f"steering_accuracy: {counts['steering_accuracy']}")
    print(f"action_accuracy: {counts['action_accuracy']}")
    print(f"action_edit_dist_avg: {counts['action_edit_dist_avg']}")
    print(f"action_edit_dist_normalized_by_8: {counts['action_edit_dist_normalized_by_8']}")
    print(f"saved_json: {output_json}")
    print(f"saved_markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
