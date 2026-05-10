#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

QA_TASKS = {
    "notable_objects",
    "occluding_objects",
    "invisible_objects",
    "planning_awareness",
}

DEFERRED_TASKS = {
    "object_motion_prediction",
    "agent_motion_prediction",
    "control_settings",
    "future_trajectory",
}

DEFERRED_NUM_FUTURE_WAYPOINTS = {
    "object_motion_prediction": 1,
    "agent_motion_prediction": 1,
    "control_settings": 6,
    "future_trajectory": 6,
}

ALL_TASKS = QA_TASKS | DEFERRED_TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the QA split protocol. Use purpose=train_dev for "
            "policy development and purpose=val_report for held-out reporting."
        )
    )
    parser.add_argument("--purpose", required=True, choices=("train_dev", "val_report"))
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--task-type", required=True, choices=tuple(sorted(ALL_TASKS)))
    parser.add_argument(
        "--qa-type-id",
        action="append",
        dest="qa_type_ids",
        type=int,
        default=[],
        help=(
            "Optional raw V2V-GoT qa_type_id filter. Repeatable. "
            "Use 15 for Q5 and 17 for Q7 object_motion_prediction splits."
        ),
    )
    parser.add_argument("--scenario-name", default="")
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for the full split.")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--notable-ranker", default="heuristic", choices=("heuristic", "energy", "llm"))
    parser.add_argument(
        "--planning-ranker",
        default="heuristic",
        choices=("heuristic", "relational_importance", "risk_aware", "energy_based", "llm"),
    )
    parser.add_argument(
        "--planning-selection-policy",
        default="default",
        choices=(
            "default",
            "top2",
            "diverse_top2",
            "count_adaptive",
            "logreg_acceptor",
            "mlp_acceptor",
            "trajectory_calibrated_acceptor",
            "count_gated_acceptor",
            "soft_count_gated_acceptor",
        ),
    )
    parser.add_argument(
        "--planning-selection-source",
        default="composition",
        choices=("composition", "orchestrator"),
    )
    parser.add_argument("--planning-acceptor-model-json", default="")
    parser.add_argument("--future-trajectory-model-json", default="")
    parser.add_argument("--object-motion-model-json", default="")
    parser.add_argument("--agent-motion-model-json", default="")
    parser.add_argument(
        "--control-selection-policy",
        default="rule",
        choices=("rule", "linear_classifier"),
    )
    parser.add_argument("--control-model-json", default="")
    parser.add_argument(
        "--occluding-ranker",
        default="risk_adaptive",
        choices=("heuristic", "top3_open", "top3_far_supported", "top3_hybrid", "risk_adaptive", "llm"),
    )
    parser.add_argument(
        "--invisible-ranker",
        default="legacy",
        choices=(
            "legacy",
            "risk_adaptive",
            "road_region",
            "road_region_strict",
            "temporal_guard",
            "backtrack_guard",
            "logreg_acceptor",
            "mlp_acceptor",
            "logreg_legacy_fallback",
            "logreg_lateral_rescue",
        ),
    )
    parser.add_argument("--invisible-acceptor-model-json", default="")
    parser.add_argument("--invisible-max-results", type=int, default=1)
    parser.add_argument("--invisible-shortlist-size", type=int, default=6)
    parser.add_argument("--invisible-max-distance-to-trajectory", type=float, default=5.0)
    parser.add_argument("--invisible-min-risk", type=float, default=0.58)
    parser.add_argument("--invisible-min-relative-to-best", type=float, default=0.75)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-official-eval", action="store_true")
    return parser


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def scenario_name(args: argparse.Namespace) -> str:
    if args.scenario_name:
        return args.scenario_name
    return f"phase8_{args.purpose}_{args.split}_{args.task_type}_{args.baseline_mode}"


def write_manifest(
    *,
    path: Path,
    repository_root: str,
    split: str,
    task_type: str,
    scenario_name: str,
    baseline_mode: str,
    planning_ranker: str,
    planning_selection_policy: str,
    planning_selection_source: str,
    planning_acceptor_model_json: str,
    future_trajectory_model_json: str,
    object_motion_model_json: str,
    agent_motion_model_json: str,
    control_selection_policy: str,
    control_model_json: str,
    output_jsonl: Path,
    total_samples: int,
    qa_type_ids: list[int],
) -> None:
    manifest = {
        "repository_root": repository_root,
        "split": split,
        "scenario_name": scenario_name,
        "task_types": [task_type],
        "runs": [
            {
                "task_type": task_type,
                "scenario_name": scenario_name,
                "baseline_mode": baseline_mode,
                "planning_ranker": planning_ranker,
                "planning_selection_policy": planning_selection_policy,
                "planning_selection_source": planning_selection_source,
                "planning_acceptor_model_json": planning_acceptor_model_json,
                "future_trajectory_model_json": future_trajectory_model_json,
                "object_motion_model_json": object_motion_model_json,
                "agent_motion_model_json": agent_motion_model_json,
                "control_selection_policy": control_selection_policy,
                "control_model_json": control_model_json,
                "output_jsonl": str(output_jsonl),
                "supported_predictions": total_samples,
                "unsupported_predictions": 0,
                "total_samples": total_samples,
                "qa_type_ids": qa_type_ids,
                "qa_type_id": qa_type_ids[0] if len(qa_type_ids) == 1 else None,
            }
        ],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def num_future_waypoints_for_official_eval(task_type: str, qa_type_ids: list[int]) -> int:
    return DEFERRED_NUM_FUTURE_WAYPOINTS.get(task_type, 0)


def main() -> None:
    args = build_parser().parse_args()
    if args.purpose == "train_dev" and args.split != "train":
        raise SystemExit("purpose=train_dev must use --split train.")
    if args.purpose == "val_report" and args.split != "val":
        raise SystemExit("purpose=val_report must use --split val.")

    scenario = scenario_name(args)
    output_root = REPO_ROOT / "outputs" / f"phase8_{args.purpose}"
    prediction_path = output_root / f"{scenario}.jsonl"
    manifest_path = output_root / f"{scenario}_manifest.json"
    official_dir = output_root / "official_exports"
    reports_dir = output_root / "official_eval_reports"
    tools_dir = official_dir / "tools"
    output_root.mkdir(parents=True, exist_ok=True)

    eval_command = [
        args.python,
        "scripts/evaluate_qa_router.py",
        "--split",
        args.split,
        "--limit",
        str(args.limit),
        "--task-type",
        args.task_type,
        "--baseline-mode",
        args.baseline_mode,
        "--planning-ranker",
        args.planning_ranker,
        "--planning-selection-policy",
        args.planning_selection_policy,
        "--planning-selection-source",
        args.planning_selection_source,
        "--control-selection-policy",
        args.control_selection_policy,
        "--notable-ranker",
        args.notable_ranker,
        "--occluding-ranker",
        args.occluding_ranker,
        "--invisible-ranker",
        args.invisible_ranker,
        "--invisible-max-results",
        str(args.invisible_max_results),
        "--invisible-shortlist-size",
        str(args.invisible_shortlist_size),
        "--invisible-max-distance-to-trajectory",
        str(args.invisible_max_distance_to_trajectory),
        "--invisible-min-risk",
        str(args.invisible_min_risk),
        "--invisible-min-relative-to-best",
        str(args.invisible_min_relative_to_best),
        "--output-jsonl",
        str(prediction_path),
    ]
    for qa_type_id in args.qa_type_ids:
        eval_command.extend(["--qa-type-id", str(qa_type_id)])
    if args.progress_every > 0:
        eval_command.extend(["--progress-every", str(args.progress_every)])
    if args.workers > 1:
        eval_command.extend(["--workers", str(args.workers)])
    if args.invisible_acceptor_model_json:
        eval_command.extend(["--invisible-acceptor-model-json", args.invisible_acceptor_model_json])
    if args.planning_acceptor_model_json:
        eval_command.extend(["--planning-acceptor-model-json", args.planning_acceptor_model_json])
    if args.future_trajectory_model_json:
        eval_command.extend(["--future-trajectory-model-json", args.future_trajectory_model_json])
    if args.object_motion_model_json:
        eval_command.extend(["--object-motion-model-json", args.object_motion_model_json])
    if args.agent_motion_model_json:
        eval_command.extend(["--agent-motion-model-json", args.agent_motion_model_json])
    if args.control_model_json:
        eval_command.extend(["--control-model-json", args.control_model_json])
    run(eval_command)

    total_samples = count_jsonl(prediction_path)
    write_manifest(
        path=manifest_path,
        repository_root=args.v2vgot_root or "/workspace/repos/V2V-GoT",
        split=args.split,
        task_type=args.task_type,
        scenario_name=scenario,
        baseline_mode=args.baseline_mode,
        planning_ranker=args.planning_ranker,
        planning_selection_policy=args.planning_selection_policy,
        planning_selection_source=args.planning_selection_source,
        planning_acceptor_model_json=args.planning_acceptor_model_json,
        future_trajectory_model_json=args.future_trajectory_model_json,
        object_motion_model_json=args.object_motion_model_json,
        agent_motion_model_json=args.agent_motion_model_json,
        control_selection_policy=args.control_selection_policy,
        control_model_json=args.control_model_json,
        output_jsonl=prediction_path,
        total_samples=total_samples,
        qa_type_ids=[int(value) for value in args.qa_type_ids],
    )
    print(f"saved_manifest: {manifest_path}")

    export_command = [
        args.python,
        "scripts/export_qa_predictions.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(official_dir),
        "--split",
        args.split,
        "--scenario-name",
        scenario,
        "--task-type",
        args.task_type,
    ]
    for qa_type_id in args.qa_type_ids:
        export_command.extend(["--qa-type-id", str(qa_type_id)])
    run(export_command)

    if args.skip_official_eval:
        return

    official_manifest = official_dir / f"{scenario}_official_export_manifest.json"
    official_command = [
        args.python,
        "scripts/run_v2vgot_official_qa_eval.py",
        "--export-manifest",
        str(official_manifest),
        "--output-dir",
        str(reports_dir),
        "--tools-dir",
        str(tools_dir),
        "--task-type",
        args.task_type,
    ]
    if args.task_type in DEFERRED_TASKS:
        num_future_waypoints = num_future_waypoints_for_official_eval(
            args.task_type,
            [int(value) for value in args.qa_type_ids],
        )
        official_command.extend(["--num-future-waypoints", str(num_future_waypoints)])
        if args.v2vgot_root:
            official_command.extend(
                [
                    "--npy-save-path",
                    str(Path(args.v2vgot_root) / "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy"),
                ]
            )
    if args.v2vgot_root:
        official_command.extend(["--v2vgot-root", args.v2vgot_root])
    run(official_command)


if __name__ == "__main__":
    main()
