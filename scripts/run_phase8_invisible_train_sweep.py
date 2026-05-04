#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InvisibleSweepConfig:
    name: str
    ranker: str
    max_results: int
    max_distance_to_trajectory: float
    min_risk: float
    min_relative_to_best: float


DEFAULT_CONFIGS = (
    InvisibleSweepConfig(
        name="legacy_traj5",
        ranker="legacy",
        max_results=1,
        max_distance_to_trajectory=5.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="legacy_traj4",
        ranker="legacy",
        max_results=1,
        max_distance_to_trajectory=4.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="legacy_traj6",
        ranker="legacy",
        max_results=1,
        max_distance_to_trajectory=6.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="risk_balanced",
        ranker="risk_adaptive",
        max_results=1,
        max_distance_to_trajectory=5.0,
        min_risk=0.50,
        min_relative_to_best=0.70,
    ),
    InvisibleSweepConfig(
        name="risk_precision",
        ranker="risk_adaptive",
        max_results=1,
        max_distance_to_trajectory=5.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="road_region_traj6",
        ranker="road_region",
        max_results=1,
        max_distance_to_trajectory=6.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="road_region_traj8",
        ranker="road_region",
        max_results=1,
        max_distance_to_trajectory=8.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="road_region_strict_traj8",
        ranker="road_region_strict",
        max_results=1,
        max_distance_to_trajectory=8.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="temporal_guard_traj6",
        ranker="temporal_guard",
        max_results=1,
        max_distance_to_trajectory=6.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
    InvisibleSweepConfig(
        name="backtrack_guard_traj6",
        ranker="backtrack_guard",
        max_results=1,
        max_distance_to_trajectory=6.0,
        min_risk=0.58,
        min_relative_to_best=0.75,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a train-only Phase 8 Q3 invisible-object policy sweep and aggregate "
            "official-style metrics. Use the winner for one later val_report run."
        )
    )
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for the full train split.")
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(config.name for config in DEFAULT_CONFIGS),
        default=[],
        help="Optional config name to run. Repeatable. Defaults to all configs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing official summary instead of rerunning that config.",
    )
    return parser


def selected_configs(names: list[str]) -> tuple[InvisibleSweepConfig, ...]:
    if not names:
        return DEFAULT_CONFIGS
    selected = set(names)
    return tuple(config for config in DEFAULT_CONFIGS if config.name in selected)


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def summary_path(scenario_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / "phase8_train_dev"
        / "official_eval_reports"
        / f"{scenario_name}_official_export_manifest_official_qa_eval_summary.json"
    )


def metric_value(metrics: dict[str, object], metric_name: str) -> float | None:
    localization = metrics.get("localization", {})
    if not isinstance(localization, dict):
        return None
    threshold_metrics = localization.get("0.5", {})
    if not isinstance(threshold_metrics, dict):
        return None
    value = threshold_metrics.get(metric_name)
    return float(value) if isinstance(value, (float, int)) else None


def load_result(config: InvisibleSweepConfig, scenario_name: str) -> dict[str, object]:
    path = summary_path(scenario_name)
    summary = json.loads(path.read_text(encoding="utf-8"))
    runs = summary.get("runs", [])
    run_summary = runs[0] if isinstance(runs, list) and runs else {}
    metrics = run_summary.get("metrics", {}) if isinstance(run_summary, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "config": config.name,
        "scenario_name": scenario_name,
        "ranker": config.ranker,
        "max_results": config.max_results,
        "max_distance_to_trajectory": config.max_distance_to_trajectory,
        "min_risk": config.min_risk,
        "min_relative_to_best": config.min_relative_to_best,
        "returncode": run_summary.get("returncode") if isinstance(run_summary, dict) else None,
        "localization_f1_0_5": metric_value(metrics, "f1"),
        "localization_precision_0_5": metric_value(metrics, "precision"),
        "localization_recall_0_5": metric_value(metrics, "recall"),
        "binary_f1": metrics.get("binary_f1"),
        "summary_json": str(path),
    }


def write_outputs(results: list[dict[str, object]]) -> None:
    output_dir = REPO_ROOT / "outputs" / "phase8_train_dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase8_invisible_train_sweep_summary.json"
    markdown_path = output_dir / "phase8_invisible_train_sweep_summary.md"

    sorted_results = sorted(
        results,
        key=lambda item: (
            item.get("localization_f1_0_5") is not None,
            item.get("localization_f1_0_5") or -1.0,
        ),
        reverse=True,
    )
    payload = {
        "purpose": "train_dev",
        "split": "train",
        "task_type": "invisible_objects",
        "selection_rule": "choose policy on train, then run the selected generic policy once on val_report",
        "results": sorted_results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8 Invisible Objects Train Sweep",
        "",
        "- `purpose`: `train_dev`",
        "- `split`: `train`",
        "- `task_type`: `invisible_objects`",
        "- selection rule: choose on train first, then rerun the selected generic policy once on validation",
        "",
        "| Config | Ranker | Max Traj | Min Risk | Rel Best | F1 @ 0.5m | Precision | Recall | Binary F1 | Summary |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in sorted_results:
        lines.append(
            "| "
            + f"`{item['config']}` | `{item['ranker']}` | "
            + f"`{item['max_distance_to_trajectory']}` | `{item['min_risk']}` | "
            + f"`{item['min_relative_to_best']}` | `{item.get('localization_f1_0_5', '')}` | "
            + f"`{item.get('localization_precision_0_5', '')}` | "
            + f"`{item.get('localization_recall_0_5', '')}` | "
            + f"`{item.get('binary_f1', '')}` | `{item['summary_json']}` |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved_json: {json_path}")
    print(f"saved_markdown: {markdown_path}")


def main() -> None:
    args = build_parser().parse_args()
    results: list[dict[str, object]] = []
    for config in selected_configs(args.config):
        scenario_name = f"phase8_train_dev_train_invisible_objects_{config.name}"
        path = summary_path(scenario_name)
        if not args.skip_existing or not path.exists():
            command = [
                args.python,
                "scripts/run_phase8_qa_split_protocol.py",
                "--purpose",
                "train_dev",
                "--split",
                "train",
                "--task-type",
                "invisible_objects",
                "--scenario-name",
                scenario_name,
                "--limit",
                str(args.limit),
                "--invisible-ranker",
                config.ranker,
                "--invisible-max-results",
                str(config.max_results),
                "--invisible-max-distance-to-trajectory",
                str(config.max_distance_to_trajectory),
                "--invisible-min-risk",
                str(config.min_risk),
                "--invisible-min-relative-to-best",
                str(config.min_relative_to_best),
            ]
            if args.v2vgot_root:
                command.extend(["--v2vgot-root", args.v2vgot_root])
            run(command)
        results.append(load_result(config, scenario_name))
    write_outputs(results)


if __name__ == "__main__":
    main()
