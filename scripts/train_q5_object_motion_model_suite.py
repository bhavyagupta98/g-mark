#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train multiple Q5 model families in one run and write a compact comparison report."
        )
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="q5_motion_suite")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--l2-regularization", type=float, default=1e-3)
    parser.add_argument("--max-match-distance", type=float, default=2.0)
    parser.add_argument("--max-abs-delta", type=float, default=120.0)
    parser.add_argument("--piecewise-min-rows", type=int, default=128)
    parser.add_argument("--tree-max-depth", type=int, default=6)
    parser.add_argument("--tree-min-leaf", type=int, default=64)
    parser.add_argument("--tree-min-gain", type=float, default=0.01)
    return parser


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    families = ("piecewise_linear", "regression_tree")
    records: list[dict[str, object]] = []
    for family in families:
        model_path = output_dir / f"{args.run_name}_{family}_deployable.json"
        report_path = output_dir / f"{args.run_name}_{family}_report.json"
        command = [
            sys.executable,
            "scripts/train_q5_object_motion_predictor.py",
            "--v2vgot-root",
            args.v2vgot_root,
            "--split",
            args.split,
            "--baseline-mode",
            args.baseline_mode,
            "--output-json",
            str(model_path),
            "--output-report",
            str(report_path),
            "--model-family",
            family,
            "--l2-regularization",
            str(args.l2_regularization),
            "--max-match-distance",
            str(args.max_match_distance),
            "--max-abs-delta",
            str(args.max_abs_delta),
            "--piecewise-min-rows",
            str(args.piecewise_min_rows),
            "--tree-max-depth",
            str(args.tree_max_depth),
            "--tree-min-leaf",
            str(args.tree_min_leaf),
            "--tree-min-gain",
            str(args.tree_min_gain),
        ]
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        run(command)

        report = read_json(report_path)
        train_metrics = report.get("train_metrics", {})
        endpoint_l2_avg = float(train_metrics.get("endpoint_l2_avg", 0.0)) if isinstance(train_metrics, dict) else 0.0
        records.append(
            {
                "family": family,
                "model_json": str(model_path),
                "report_json": str(report_path),
                "endpoint_l2_avg": endpoint_l2_avg,
            }
        )

    records_sorted = sorted(records, key=lambda item: float(item["endpoint_l2_avg"]))
    best = records_sorted[0] if records_sorted else {}
    summary = {
        "run_name": args.run_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "split": args.split,
        "baseline_mode": args.baseline_mode,
        "records": records_sorted,
        "best_family": best.get("family", ""),
        "best_model_json": best.get("model_json", ""),
    }
    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("Q5 model suite complete")
    print("=" * 72)
    for item in records_sorted:
        print(
            f"family={item['family']} endpoint_l2_avg={item['endpoint_l2_avg']:.6f} "
            f"model={item['model_json']}"
        )
    print(f"best_family: {summary['best_family']}")
    print(f"saved_summary: {summary_path}")


if __name__ == "__main__":
    main()
