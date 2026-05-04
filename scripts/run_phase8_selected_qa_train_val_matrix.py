#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TASKS = (
    "notable_objects",
    "occluding_objects",
    "invisible_objects",
    "planning_awareness",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the selected Phase 8 Q1-Q4 QA policies on train and validation, "
            "then write one official metrics matrix."
        )
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for full train/validation splits.")
    parser.add_argument(
        "--invisible-acceptor-model-json",
        default=(
            "outputs/phase8_train_dev/q3_policy_optimization_traj8_short64/"
            "q3_logreg_mlp_legacy_traj8_short64_logreg_max_recall_p0p5_t0p33_deployable.json"
        ),
        help="Selected Q3 broad-pool logistic acceptor model.",
    )
    parser.add_argument(
        "--base-invisible-acceptor-model-json",
        default="outputs/phase8_train_dev/invisible_candidate_acceptor_logreg_model.json",
        help="Base Q3 logistic acceptor model used to create the selected t0p25 model if needed.",
    )
    parser.add_argument("--invisible-threshold", type=float, default=0.33)
    parser.add_argument("--notable-ranker", default="heuristic")
    parser.add_argument("--occluding-ranker", default="risk_adaptive")
    parser.add_argument("--invisible-ranker", default="logreg_acceptor")
    parser.add_argument("--invisible-max-results", type=int, default=1)
    parser.add_argument("--invisible-shortlist-size", type=int, default=64)
    parser.add_argument("--invisible-max-distance-to-trajectory", type=float, default=8.0)
    parser.add_argument("--output-json", default="outputs/phase8_selected_qa_train_val_matrix.json")
    parser.add_argument("--output-markdown", default="outputs/phase8_selected_qa_train_val_matrix.md")
    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="Do not rerun evaluation; only summarize already-existing official summary JSON files.",
    )
    return parser


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_selected_invisible_model(args: argparse.Namespace) -> None:
    selected_path = resolve_repo_path(args.invisible_acceptor_model_json)
    if selected_path.exists():
        return

    base_path = resolve_repo_path(args.base_invisible_acceptor_model_json)
    if not base_path.exists():
        raise SystemExit(
            "Missing selected Q3 model and base model.\n"
            f"selected: {selected_path}\n"
            f"base: {base_path}\n"
            "Run scripts/train_phase8_invisible_candidate_acceptor.py first, or pass "
            "--invisible-acceptor-model-json to an existing model."
        )

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    model = json.loads(base_path.read_text(encoding="utf-8"))
    model["threshold"] = args.invisible_threshold
    selected_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"created_selected_invisible_model: {selected_path}")


def protocol_command(args: argparse.Namespace, purpose: str, split: str, task: str) -> list[str]:
    scenario = f"phase8_{purpose}_{split}_{task}_selected"
    return [
        args.python,
        "scripts/run_phase8_qa_split_protocol.py",
        "--purpose",
        purpose,
        "--split",
        split,
        "--task-type",
        task,
        "--scenario-name",
        scenario,
        "--limit",
        str(args.limit),
        "--notable-ranker",
        args.notable_ranker,
        "--occluding-ranker",
        args.occluding_ranker,
        "--invisible-ranker",
        args.invisible_ranker,
        "--invisible-acceptor-model-json",
        args.invisible_acceptor_model_json,
        "--invisible-max-results",
        str(args.invisible_max_results),
        "--invisible-shortlist-size",
        str(args.invisible_shortlist_size),
        "--invisible-max-distance-to-trajectory",
        str(args.invisible_max_distance_to_trajectory),
        "--v2vgot-root",
        args.v2vgot_root,
    ]


def summary_path(purpose: str, split: str, task: str) -> str:
    scenario = f"phase8_{purpose}_{split}_{task}_selected"
    return (
        f"outputs/phase8_{purpose}/official_eval_reports/"
        f"{scenario}_official_export_manifest_official_qa_eval_summary.json"
    )


def summarize_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "scripts/summarize_phase8_qa_split_matrix.py",
    ]
    for purpose, split in (("train_dev", "train"), ("val_report", "val")):
        for task in TASKS:
            command.extend(["--summary-json", summary_path(purpose, split, task)])
    command.extend(
        [
            "--output-json",
            args.output_json,
            "--output-markdown",
            args.output_markdown,
        ]
    )
    return command


def main() -> int:
    args = build_parser().parse_args()
    ensure_selected_invisible_model(args)

    if not args.only_summary:
        for purpose, split in (("train_dev", "train"), ("val_report", "val")):
            for task in TASKS:
                run(protocol_command(args, purpose, split, task))

    run(summarize_command(args))

    print()
    print("=" * 72)
    print("Phase 8 selected QA train/validation matrix complete")
    print("=" * 72)
    print(f"output_json: {resolve_repo_path(args.output_json)}")
    print(f"output_markdown: {resolve_repo_path(args.output_markdown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
