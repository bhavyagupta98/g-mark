#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.planning.agent_motion_notability_predictor import (  # noqa: E402
    LearnedAgentMotionNotabilityPredictor,
)
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

NOTABLE_RE = re.compile(r"\bis a notable object\b", re.IGNORECASE)
NOT_NOTABLE_RE = re.compile(r"\bis a not notable object\b", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Q6 feature quality and error slices.")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def _extract_label(raw_record: dict[str, object]) -> int | None:
    conversations = raw_record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None
    answer = conversations[1].get("value", "")
    if not isinstance(answer, str):
        return None
    if NOT_NOTABLE_RE.search(answer):
        return 0
    if NOTABLE_RE.search(answer):
        return 1
    return None


def _closest_other_agent(scene):
    asker = next((a for a in scene.agents if a.agent_id == scene.asker_agent_id), None)
    if asker is None:
        return None
    others = [a for a in scene.agents if a.agent_id != scene.asker_agent_id]
    if not others:
        return None
    return min(
        others,
        key=lambda a: (((a.pose.position.x - asker.pose.position.x) ** 2) + ((a.pose.position.y - asker.pose.position.y) ** 2))
        ** 0.5,
    )


def _bin_distance(v: float) -> str:
    if v < 5.0:
        return "d<5"
    if v < 10.0:
        return "5<=d<10"
    if v < 20.0:
        return "10<=d<20"
    return "d>=20"


def _bin_speed(v: float) -> str:
    if v < 0.2:
        return "s<0.2"
    if v < 1.0:
        return "0.2<=s<1.0"
    if v < 3.0:
        return "1.0<=s<3.0"
    return "s>=3.0"


def _bin_density(v: float) -> str:
    iv = int(round(v))
    if iv <= 1:
        return "n<=1"
    if iv <= 3:
        return "2<=n<=3"
    if iv <= 6:
        return "4<=n<=6"
    return "n>=7"


def _feature_summary(rows: list[dict[str, Any]], feature_name: str) -> dict[str, float]:
    pos = [float(r["features"].get(feature_name, 0.0)) for r in rows if r["label"] == 1]
    neg = [float(r["features"].get(feature_name, 0.0)) for r in rows if r["label"] == 0]
    pos_mean = sum(pos) / len(pos) if pos else 0.0
    neg_mean = sum(neg) / len(neg) if neg else 0.0
    gap = pos_mean - neg_mean
    return {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "mean_gap": gap,
        "abs_mean_gap": abs(gap),
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Q6 Feature Quality Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for k, v in report["summary"].items():
        lines.append(f"| `{k}` | `{v}` |")

    lines.extend(["", "## Confusion", "", "| Bucket | Count |", "| --- | ---: |"])
    for k, v in report["confusion"].items():
        lines.append(f"| `{k}` | `{v}` |")

    lines.extend(["", "## Error Slices", "", "| Slice | Count |", "| --- | ---: |"])
    for k, v in report["error_slices"].items():
        lines.append(f"| `{k}` | `{v}` |")

    lines.extend(
        [
            "",
            "## Feature Signal",
            "",
            "| Feature | Pos Mean | Neg Mean | Gap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in report["feature_signal"]:
        lines.append(
            f"| `{item['feature']}` | `{item['pos_mean']:.4f}` | `{item['neg_mean']:.4f}` | `{item['mean_gap']:.4f}` |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    model = json.loads(Path(args.model_json).expanduser().read_text(encoding="utf-8"))
    predictor = LearnedAgentMotionNotabilityPredictor(model=model)

    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    samples = adapter.load_samples(split_name=args.split)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if int(sample.qa_type_id or -1) != 16:
            continue
        label = _extract_label(sample.raw_record)
        if label is None:
            continue
        other = _closest_other_agent(sample.scene)
        if other is None:
            continue
        pred_bool = predictor.predict_is_notable(scene=sample.scene, other_agent=other)
        if pred_bool is None:
            pred = 0
        else:
            pred = 1 if pred_bool else 0
        features = predictor._feature_map(scene=sample.scene, other_agent=other)  # noqa: SLF001
        rows.append(
            {
                "sample_id": sample.sample_id,
                "label": int(label),
                "pred": int(pred),
                "features": features,
            }
        )
        if args.limit > 0 and len(rows) >= args.limit:
            break

    total = len(rows)
    tp = sum(1 for r in rows if r["label"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in rows if r["label"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in rows if r["label"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in rows if r["label"] == 1 and r["pred"] == 0)
    acc = (tp + tn) / total if total else 0.0

    slices: Counter[str] = Counter()
    for r in rows:
        if r["label"] == r["pred"]:
            continue
        f = r["features"]
        slices[f"dist:{_bin_distance(float(f.get('other_min_distance_to_asker_path', 999.0)))}"] += 1
        slices[f"speed:{_bin_speed(float(f.get('other_speed', 0.0)))}"] += 1
        slices[f"density:{_bin_density(float(f.get('asker_nearby_object_count_10m', 0.0)))}"] += 1
        err = "FP" if (r["label"] == 0 and r["pred"] == 1) else "FN"
        slices[f"type:{err}"] += 1

    feature_names = [
        "other_distance_to_asker",
        "other_speed",
        "other_planned_final_dist",
        "other_min_distance_to_asker_path",
        "asker_path_length",
        "asker_nearby_object_count_10m",
        "asker_nearby_dynamic_count_10m",
    ]
    feature_signal = []
    for name in feature_names:
        s = _feature_summary(rows, name)
        feature_signal.append({"feature": name, **s})
    feature_signal.sort(key=lambda item: item["abs_mean_gap"], reverse=True)

    report = {
        "summary": {
            "split": args.split,
            "sample_count": total,
            "accuracy": round(acc, 6),
        },
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "error_slices": dict(sorted(slices.items(), key=lambda kv: (-kv[1], kv[0]))),
        "feature_signal": feature_signal,
    }

    out_json = Path(args.output_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md = Path(args.output_markdown).expanduser()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report, out_md)

    print(f"q6_inspect_done split={args.split} samples={total} acc={acc:.6f}")
    print(f"saved_json={out_json}")
    print(f"saved_markdown={out_md}")
    top = report["feature_signal"][:5]
    print("top_feature_gaps=" + ",".join(f"{i['feature']}:{i['mean_gap']:.3f}" for i in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

