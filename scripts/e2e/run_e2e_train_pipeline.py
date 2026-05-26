#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskRunConfig:
    name: str
    task_type: str
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Q6E2ETrainingConfig:
    gbdt_backend: str
    gbdt_n_estimators: int
    gbdt_learning_rate: float
    gbdt_max_depth: int
    gbdt_min_samples_leaf: int
    gbdt_subsample: float
    decision_threshold: float


Q6_E2E_TRAINING_DEFAULTS = Q6E2ETrainingConfig(
    gbdt_backend="sklearn",
    gbdt_n_estimators=280,
    gbdt_learning_rate=0.04,
    gbdt_max_depth=2,
    gbdt_min_samples_leaf=96,
    gbdt_subsample=0.7,
    decision_threshold=0.38,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end train pipeline for frozen Q1/Q2/Q3/Q4/Q5/Q6/Q7/Q8/Q9 setup: "
            "feature export, model training, train-split QA evaluation, and artifact archival."
        )
    )
    parser.add_argument(
        "--run-name",
        default=f"phase9_e2e_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Unique run folder name under outputs/e2e_runs.",
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--q5-model-family",
        default="regression_tree",
        choices=("linear", "piecewise_linear", "regression_tree", "gradient_boosting", "mlp"),
    )
    parser.add_argument("--q5-l2-regularization", type=float, default=1e-3)
    parser.add_argument("--q5-max-match-distance", type=float, default=2.0)
    parser.add_argument("--q5-max-abs-delta", type=float, default=120.0)
    parser.add_argument("--q5-piecewise-min-rows", type=int, default=128)
    parser.add_argument("--q5-tree-max-depth", type=int, default=9)
    parser.add_argument("--q5-tree-min-leaf", type=int, default=64)
    parser.add_argument("--q5-tree-min-gain", type=float, default=0.01)
    parser.add_argument(
        "--q5-feature-set",
        default="path_relative",
        choices=("auto", "base", "path_relative"),
        help="Q5 object-motion feature set. E2E promotes path_relative for the Phase 9 tree model.",
    )
    parser.add_argument(
        "--q7-model-family",
        default="regression_tree",
        choices=("linear", "piecewise_linear", "regression_tree", "gradient_boosting", "mlp"),
    )
    parser.add_argument(
        "--q7-feature-set",
        default="path_relative",
        choices=("auto", "base", "path_relative"),
    )
    parser.add_argument("--q7-l2-regularization", type=float, default=1e-3)
    parser.add_argument("--q7-max-match-distance", type=float, default=2.0)
    parser.add_argument("--q7-max-abs-delta", type=float, default=120.0)
    parser.add_argument("--q7-piecewise-min-rows", type=int, default=128)
    parser.add_argument("--q7-tree-max-depth", type=int, default=9)
    parser.add_argument("--q7-tree-min-leaf", type=int, default=64)
    parser.add_argument("--q7-tree-min-gain", type=float, default=0.01)
    parser.add_argument("--q6-gbdt-n-estimators", type=int, default=Q6_E2E_TRAINING_DEFAULTS.gbdt_n_estimators)
    parser.add_argument("--q6-gbdt-learning-rate", type=float, default=Q6_E2E_TRAINING_DEFAULTS.gbdt_learning_rate)
    parser.add_argument("--q6-gbdt-max-depth", type=int, default=Q6_E2E_TRAINING_DEFAULTS.gbdt_max_depth)
    parser.add_argument("--q6-gbdt-min-samples-leaf", type=int, default=Q6_E2E_TRAINING_DEFAULTS.gbdt_min_samples_leaf)
    parser.add_argument("--q6-gbdt-subsample", type=float, default=Q6_E2E_TRAINING_DEFAULTS.gbdt_subsample)
    parser.add_argument(
        "--q6-gbdt-backend",
        default=Q6_E2E_TRAINING_DEFAULTS.gbdt_backend,
        choices=("xgboost", "sklearn"),
        help=(
            "Q6 GBDT backend. E2E defaults to sklearn because the promoted "
            "q6_gbdt_tight_n280_lr0.04_d2_l96_s0.7_t0.38 result was produced "
            "by the trainer's historical sklearn-first GBDT path."
        ),
    )
    parser.add_argument("--q6-decision-threshold", type=float, default=Q6_E2E_TRAINING_DEFAULTS.decision_threshold)
    parser.add_argument(
        "--q9-model-source",
        default="clean_q8_context_elasticnet",
        choices=("clean_q8_context_elasticnet",),
        help=(
            "Q9 model source. Uses the leak-safe Q8-context ElasticNet sweep path."
        ),
    )
    parser.add_argument("--q9-train-file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--q9-val-file-name", default="v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json")
    parser.add_argument(
        "--allow-val-features-during-training",
        action="store_true",
        help=(
            "Opt-in diagnostic mode. When enabled, exports val feature tables and "
            "passes them to Q3/Q4 trainers for report-only eval. Default is strict "
            "train-only fitting with no val feature export."
        ),
    )
    return parser


def run(command: Sequence[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_train_markdown(path: Path, manifest: dict[str, object]) -> None:
    model_paths = manifest.get("model_paths", {})
    frozen_policy_config = manifest.get("frozen_policy_config", {})
    if not isinstance(model_paths, dict):
        model_paths = {}
    if not isinstance(frozen_policy_config, dict):
        frozen_policy_config = {}
    q5 = frozen_policy_config.get("q5", {})
    q7 = frozen_policy_config.get("q7", {})
    if not isinstance(q5, dict):
        q5 = {}
    if not isinstance(q7, dict):
        q7 = {}

    lines = [
        f"# E2E Train Run `{manifest.get('run_name', '')}`",
        "",
        "## Object Motion Models",
        "",
        "| Task | QA Type | Model | Feature Set | Train Split | Artifact |",
        "| --- | ---: | --- | --- | --- | --- |",
        (
            f"| Q5 object motion | 15 | `{q5.get('model_family', '')}` | "
            f"`{q5.get('feature_set', '')}` | `train` | `{model_paths.get('q5_model_json', '')}` |"
        ),
        (
            f"| Q7 object motion | 17 | `{q7.get('model_family', '')}` | "
            f"`{q7.get('feature_set', '')}` | `train` | `{model_paths.get('q7_model_json', '')}` |"
        ),
        "",
        "## Q5/Q7 Evaluation Note",
        "",
        (
            "Q5 and Q7 are trained and archived as separate deployable models. "
            "The current V2V-GoT validation split has paired Q5/Q7 sample IDs with identical "
            "scored GT answer strings, while the questions differ in graph context. "
            "Equal official metrics for Q5 and Q7 therefore indicate duplicated validation "
            "targets, not shared e2e weights."
        ),
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        f"python3 scripts/e2e/run_e2e_train_pipeline.py --run-name {manifest.get('run_name', '')}",
        (
            "python3 scripts/e2e/run_e2e_validation_report.py "
            f"--manifest-json outputs/e2e_runs/{manifest.get('run_name', '')}/e2e_model_manifest.json"
        ),
        "```",
        "",
        "## Model Artifacts",
        "",
    ]
    for key, value in sorted(model_paths.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def summary_json_path(purpose: str, scenario_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / f"phase8_{purpose}"
        / "official_eval_reports"
        / f"{scenario_name}_official_export_manifest_official_qa_eval_summary.json"
    )


def summary_markdown_path(purpose: str, scenario_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / f"phase8_{purpose}"
        / "official_eval_reports"
        / f"{scenario_name}_official_export_manifest_official_qa_eval_summary.md"
    )


def export_manifest_path(purpose: str, scenario_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / f"phase8_{purpose}"
        / "official_exports"
        / f"{scenario_name}_official_export_manifest.json"
    )


def train_models(
    *,
    e2e_root: Path,
    v2vgot_root: str,
    workers: int,
    progress_every: int,
    allow_val_features_during_training: bool,
    q5_model_family: str,
    q5_l2_regularization: float,
    q5_max_match_distance: float,
    q5_max_abs_delta: float,
    q5_piecewise_min_rows: int,
    q5_tree_max_depth: int,
    q5_tree_min_leaf: int,
    q5_tree_min_gain: float,
    q5_feature_set: str,
    q7_model_family: str,
    q7_feature_set: str,
    q7_l2_regularization: float,
    q7_max_match_distance: float,
    q7_max_abs_delta: float,
    q7_piecewise_min_rows: int,
    q7_tree_max_depth: int,
    q7_tree_min_leaf: int,
    q7_tree_min_gain: float,
    q6_gbdt_n_estimators: int,
    q6_gbdt_learning_rate: float,
    q6_gbdt_max_depth: int,
    q6_gbdt_min_samples_leaf: int,
    q6_gbdt_subsample: float,
    q6_gbdt_backend: str,
    q6_decision_threshold: float,
    q9_model_source: str,
    q9_train_file_name: str,
    q9_val_file_name: str,
) -> dict[str, str]:
    features_dir = e2e_root / "features"
    models_dir = e2e_root / "models"
    reports_dir = e2e_root / "training_reports"
    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("Step 1/8: Q1-Q2 frozen policy snapshots")
    print("=" * 72)
    q1_policy_path = models_dir / "q1_notable_objects_policy.json"
    q2_policy_path = models_dir / "q2_occluding_objects_policy.json"
    write_json(
        q1_policy_path,
        {
            "task": "Q1 notable_objects",
            "policy_type": "heuristic",
            "note": "Frozen baseline policy; no learned model artifact.",
        },
    )
    write_json(
        q2_policy_path,
        {
            "task": "Q2 occluding_objects",
            "policy_type": "risk_adaptive",
            "note": "Frozen baseline policy; no learned model artifact.",
        },
    )
    print(f"saved_policy_snapshot: {q1_policy_path}")
    print(f"saved_policy_snapshot: {q2_policy_path}")

    print("\n" + "=" * 72)
    print("Step 2/8: Q3 feature export + training")
    print("=" * 72)
    q3_train_features = features_dir / "q3_invisible_train_features.jsonl"
    run(
        [
            sys.executable,
            "scripts/export_phase8_invisible_candidate_features.py",
            "--v2vgot-root",
            v2vgot_root,
            "--split",
            "train",
            "--baseline-mode",
            "cooperative",
            "--invisible-ranker",
            "legacy",
            "--shortlist-size",
            "64",
            "--invisible-max-distance-to-trajectory",
            "8.0",
            "--limit",
            "0",
            "--workers",
            str(workers),
            "--progress-every",
            str(progress_every),
            "--output-jsonl",
            str(q3_train_features),
        ]
    )
    q3_policy_dir = models_dir / "q3_policy_optimization"
    q3_policy_dir.mkdir(parents=True, exist_ok=True)
    q3_policy_run_name = "q3_invisible_policy_e2e"
    q3_opt_command = [
        sys.executable,
        "scripts/optimize_q3_invisible_candidate_policy.py",
        "--train-features-jsonl",
        str(q3_train_features),
        "--output-dir",
        str(q3_policy_dir),
        "--run-name",
        q3_policy_run_name,
        "--models",
        "logreg",
    ]
    if allow_val_features_during_training:
        q3_val_features = features_dir / "q3_invisible_val_features.jsonl"
        run(
            [
                sys.executable,
                "scripts/export_phase8_invisible_candidate_features.py",
                "--v2vgot-root",
                v2vgot_root,
                "--split",
                "val",
                "--baseline-mode",
                "cooperative",
                "--invisible-ranker",
                "legacy",
                "--shortlist-size",
                "64",
                "--invisible-max-distance-to-trajectory",
                "8.0",
                "--limit",
                "0",
                "--workers",
                str(workers),
                "--progress-every",
                str(progress_every),
                "--output-jsonl",
                str(q3_val_features),
            ]
        )
        q3_opt_command.extend(
            [
                "--eval-features-jsonl",
                str(q3_val_features),
            ]
        )
    run(q3_opt_command)
    q3_opt_report = read_json(q3_policy_dir / f"{q3_policy_run_name}_report.json")
    deployable_map = q3_opt_report.get("deployable_model_paths")
    q3_selected_model_path: Path | None = None
    q3_selected_result_key = ""
    q3_available_result_keys: list[str] = []
    q3_near_miss_keys: list[str] = []
    if isinstance(deployable_map, dict):
        q3_available_result_keys = sorted(str(key) for key in deployable_map.keys())
        for result_key, model_path in deployable_map.items():
            if not isinstance(result_key, str) or not isinstance(model_path, str):
                continue
            key_parts = result_key.split(":")
            if len(key_parts) < 3:
                continue
            model_name, policy_name = key_parts[0], key_parts[1]
            if model_name == "logreg" and policy_name == "max_recall_p0p5":
                q3_selected_model_path = Path(model_path).expanduser()
                q3_selected_result_key = result_key
                break
            if model_name == "logreg" and policy_name.startswith("max_recall_p0p5"):
                q3_near_miss_keys.append(result_key)
    if q3_selected_model_path is None:
        raise RuntimeError(
            "Unable to locate required Q3 deployable model for exact policy token "
            "'logreg:max_recall_p0p5'. "
            f"available_keys={q3_available_result_keys}, near_miss_keys={q3_near_miss_keys}"
        )
    q3_model = models_dir / "q3_invisible_logreg_acceptor_deployable.json"
    if not q3_selected_model_path.is_absolute():
        q3_selected_model_path = (REPO_ROOT / q3_selected_model_path).resolve()
    copy_if_exists(q3_selected_model_path, q3_model)
    q3_model_payload = read_json(q3_model)
    selected_policy = str(q3_model_payload.get("selected_policy", ""))
    if selected_policy != "max_recall_p0p5":
        raise RuntimeError(
            "Q3 selected deployable model does not match required policy family "
            f"'max_recall_p0p5'. Found selected_policy='{selected_policy}'."
        )
    q3_audit_path = models_dir / "q3_policy_selection_audit.json"
    write_json(
        q3_audit_path,
        {
            "required_result_key_policy_token": "logreg:max_recall_p0p5",
            "required_selected_policy": "max_recall_p0p5",
            "selected_result_key": q3_selected_result_key,
            "selected_model_path": str(q3_selected_model_path),
            "copied_model_path": str(q3_model),
            "selected_policy": selected_policy,
            "available_result_keys": q3_available_result_keys,
            "near_miss_keys": q3_near_miss_keys,
        },
    )
    if q3_near_miss_keys:
        print(f"ignored_q3_near_miss_keys: {sorted(q3_near_miss_keys)}")
    print(f"selected_q3_result_key: {q3_selected_result_key}")
    print(f"selected_q3_model: {q3_selected_model_path}")
    print(f"copied_q3_model: {q3_model}")
    print(f"selected_q3_policy: {selected_policy}")
    print(f"q3_policy_selection_audit: {q3_audit_path}")

    print("\n" + "=" * 72)
    print("Step 3/8: Q4 feature export + acceptor training + trajectory calibration")
    print("=" * 72)
    q4_train_features = features_dir / "q4_planning_train_features.jsonl"
    run(
        [
            sys.executable,
            "scripts/export_phase8_planning_candidate_features.py",
            "--v2vgot-root",
            v2vgot_root,
            "--split",
            "train",
            "--baseline-mode",
            "cooperative",
            "--planning-ranker",
            "relational_importance",
            "--limit",
            "0",
            "--progress-every",
            str(progress_every),
            "--output-jsonl",
            str(q4_train_features),
        ]
    )
    q4_final_model = models_dir / "q4_planning_rel_logreg_trajcal_deployable.json"
    q4_model_dir = models_dir / "q4_planning"
    q4_model_dir.mkdir(parents=True, exist_ok=True)
    q4_run_name = "q4_planning_rel_logreg_e2e"
    q4_train_command = [
        sys.executable,
        "scripts/train_q4_planning_acceptor.py",
        "--train-features-jsonl",
        str(q4_train_features),
        "--output-dir",
        str(q4_model_dir),
        "--run-name",
        q4_run_name,
        "--model-type",
        "logreg",
        "--regularization",
        "l2",
        "--l2",
        "0.001",
        "--min-precision",
        "0.55",
        "--max-results",
        "3",
        "--near-duplicate-distance",
        "1.0",
        "--log-every",
        "100",
    ]
    if allow_val_features_during_training:
        q4_val_features = features_dir / "q4_planning_val_features.jsonl"
        run(
            [
                sys.executable,
                "scripts/export_phase8_planning_candidate_features.py",
                "--v2vgot-root",
                v2vgot_root,
                "--split",
                "val",
                "--baseline-mode",
                "cooperative",
                "--planning-ranker",
                "relational_importance",
                "--limit",
                "0",
                "--progress-every",
                str(progress_every),
                "--output-jsonl",
                str(q4_val_features),
            ]
        )
        q4_train_command.extend(
            [
                "--eval-features-jsonl",
                str(q4_val_features),
            ]
        )
    run(q4_train_command)
    q4_report_json = q4_model_dir / f"{q4_run_name}_report.json"
    q4_report_payload = read_json(q4_report_json)
    q4_base_model = Path(str(q4_report_payload["deployable_model_path"]))
    run(
        [
            sys.executable,
            "scripts/configure_q4_trajectory_calibration.py",
            "--input-model-json",
            str(q4_base_model),
            "--output-model-json",
            str(q4_final_model),
        ]
    )

    print("\n" + "=" * 72)
    print("Step 4/8: Q5/Q7 object-motion model training")
    print("=" * 72)
    q5_model = models_dir / "q5_object_motion_model_deployable.json"
    q5_report = reports_dir / "q5_object_motion_model_report.json"
    run(
        [
            sys.executable,
            "scripts/train_q5_object_motion_predictor.py",
            "--v2vgot-root",
            v2vgot_root,
            "--split",
            "train",
            "--baseline-mode",
            "cooperative",
            "--model-family",
            q5_model_family,
            "--feature-set",
            q5_feature_set,
            "--l2-regularization",
            str(q5_l2_regularization),
            "--max-match-distance",
            str(q5_max_match_distance),
            "--max-abs-delta",
            str(q5_max_abs_delta),
            "--piecewise-min-rows",
            str(q5_piecewise_min_rows),
            "--tree-max-depth",
            str(q5_tree_max_depth),
            "--tree-min-leaf",
            str(q5_tree_min_leaf),
            "--tree-min-gain",
            str(q5_tree_min_gain),
            "--output-json",
            str(q5_model),
            "--output-report",
            str(q5_report),
        ]
    )
    q7_model = models_dir / "q7_object_motion_model_deployable.json"
    q7_report = reports_dir / "q7_object_motion_model_report.json"
    run(
        [
            sys.executable,
            "scripts/train_q7_object_motion_predictor.py",
            "--v2vgot-root",
            v2vgot_root,
            "--split",
            "train",
            "--baseline-mode",
            "cooperative",
            "--model-family",
            q7_model_family,
            "--feature-set",
            q7_feature_set,
            "--l2-regularization",
            str(q7_l2_regularization),
            "--max-match-distance",
            str(q7_max_match_distance),
            "--max-abs-delta",
            str(q7_max_abs_delta),
            "--piecewise-min-rows",
            str(q7_piecewise_min_rows),
            "--tree-max-depth",
            str(q7_tree_max_depth),
            "--tree-min-leaf",
            str(q7_tree_min_leaf),
            "--tree-min-gain",
            str(q7_tree_min_gain),
            "--output-json",
            str(q7_model),
            "--output-report",
            str(q7_report),
        ]
    )

    print("\n" + "=" * 72)
    print("Step 5/8: Q6 model training")
    print("=" * 72)
    q6_model = models_dir / "q6_agent_motion_gbdt_deployable.json"
    q6_report = reports_dir / "q6_agent_motion_gbdt_report.json"
    run(
        [
            sys.executable,
            "scripts/train_q6_agent_motion_notability.py",
            "--v2vgot-root",
            v2vgot_root,
            "--split",
            "train",
            "--model-family",
            "gbdt",
            "--gbdt-backend",
            q6_gbdt_backend,
            "--gbdt-n-estimators",
            str(q6_gbdt_n_estimators),
            "--gbdt-learning-rate",
            str(q6_gbdt_learning_rate),
            "--gbdt-max-depth",
            str(q6_gbdt_max_depth),
            "--gbdt-min-samples-leaf",
            str(q6_gbdt_min_samples_leaf),
            "--gbdt-subsample",
            str(q6_gbdt_subsample),
            "--decision-threshold",
            str(q6_decision_threshold),
            "--output-json",
            str(q6_model),
            "--output-report",
            str(q6_report),
        ]
    )

    print("\n" + "=" * 72)
    print("Step 6/8: Q8 model training")
    print("=" * 72)
    q8_model = models_dir / "q8_control_linear_classifier_deployable.json"
    q8_report = reports_dir / "q8_control_linear_classifier_report.json"
    run(
        [
            sys.executable,
            "scripts/train_q8_control_policy.py",
            "--v2vgot-root",
            v2vgot_root,
            "--split",
            "train",
            "--baseline-mode",
            "cooperative",
            "--feature-set",
            "extended_v1",
            "--speed-head-type",
            "ordinal",
            "--speed-class-weighting",
            "sqrt_inverse_freq",
            "--steering-class-weighting",
            "none",
            "--l2-regularization",
            "1e-4",
            "--speed-ordinal-threshold-policy",
            "risk3",
            "--speed-risk-split-low",
            "0.2",
            "--speed-risk-split-high",
            "0.5",
            "--output-json",
            str(q8_model),
            "--output-report",
            str(q8_report),
        ]
    )

    print("\n" + "=" * 72)
    print("Step 7/8: Q9 model training")
    print("=" * 72)
    if q9_model_source == "clean_q8_context_elasticnet":
        q9_run_name = "e2e_q9_clean_context_elasticnet"
        q9_sweep_root = e2e_root / "q9_clean_sweep"
        run(
            [
                sys.executable,
                "scripts/run_gmark_q9_model_sweep.py",
                "--run-name",
                q9_run_name,
                "--v2vgot-root",
                v2vgot_root,
                "--output-root",
                str(q9_sweep_root),
                "--train-file-name",
                q9_train_file_name,
                "--val-file-name",
                q9_val_file_name,
                "--models",
                "elasticnet",
                "--include-q8-pred-features",
                "--q8-feature-source",
                "question_context",
                "--progress-every",
                str(progress_every),
            ]
        )
        q9_model = q9_sweep_root / q9_run_name / f"{q9_run_name}_elasticnet_model.json"
        q9_report = reports_dir / "q9_clean_context_elasticnet_note.json"
        write_json(
            q9_report,
            {
                "q9_model_source": q9_model_source,
                "model_json": str(q9_model),
                "note": "Leak-safe Q9 sweep model using Q8 prompt-context features.",
            },
        )
    else:
        raise ValueError(f"Unsupported q9_model_source: {q9_model_source}")

    return {
        "q1_policy_json": str(q1_policy_path),
        "q2_policy_json": str(q2_policy_path),
        "q5_policy_json": "learned_router_object_motion_prediction",
        "q5_model_json": str(q5_model),
        "q7_model_json": str(q7_model),
        "q6_model_json": str(q6_model),
        "q3_model_json": str(q3_model),
        "q4_model_json": str(q4_final_model),
        "q8_model_json": str(q8_model),
        "q9_model_json": str(q9_model),
    }


def build_train_task_configs(model_paths: dict[str, str]) -> tuple[TaskRunConfig, ...]:
    q5_extra_args: tuple[str, ...] = ()
    q5_model_json = str(model_paths.get("q5_model_json", "")).strip()
    if q5_model_json:
        q5_extra_args = ("--object-motion-model-json", q5_model_json)
    q7_extra_args: tuple[str, ...] = ()
    q7_model_json = str(model_paths.get("q7_model_json", "")).strip()
    if q7_model_json:
        q7_extra_args = ("--object-motion-model-json", q7_model_json)
    q6_extra_args: tuple[str, ...] = ()
    q6_model_json = str(model_paths.get("q6_model_json", "")).strip()
    if q6_model_json:
        q6_extra_args = ("--agent-motion-model-json", q6_model_json)

    return (
        TaskRunConfig(
            name="q1_notable_objects",
            task_type="notable_objects",
            extra_args=("--notable-ranker", "heuristic"),
        ),
        TaskRunConfig(
            name="q2_occluding_objects",
            task_type="occluding_objects",
            extra_args=("--occluding-ranker", "risk_adaptive"),
        ),
        TaskRunConfig(
            name="q5_object_motion_prediction",
            task_type="object_motion_prediction",
            extra_args=("--qa-type-id", "15", *q5_extra_args),
        ),
        TaskRunConfig(
            name="q6_agent_motion_prediction",
            task_type="agent_motion_prediction",
            extra_args=q6_extra_args,
        ),
        TaskRunConfig(
            name="q7_object_motion_prediction",
            task_type="object_motion_prediction",
            extra_args=("--qa-type-id", "17", *q7_extra_args),
        ),
        TaskRunConfig(
            name="q3_invisible_objects",
            task_type="invisible_objects",
            extra_args=(
                "--invisible-ranker",
                "logreg_acceptor",
                "--invisible-acceptor-model-json",
                model_paths["q3_model_json"],
                "--invisible-max-results",
                "1",
                "--invisible-shortlist-size",
                "64",
                "--invisible-max-distance-to-trajectory",
                "8.0",
                "--invisible-min-risk",
                "0.58",
                "--invisible-min-relative-to-best",
                "0.75",
            ),
        ),
        TaskRunConfig(
            name="q4_planning_awareness",
            task_type="planning_awareness",
            extra_args=(
                "--planning-ranker",
                "relational_importance",
                "--planning-selection-policy",
                "trajectory_calibrated_acceptor",
                "--planning-selection-source",
                "orchestrator",
                "--planning-acceptor-model-json",
                model_paths["q4_model_json"],
            ),
        ),
        TaskRunConfig(
            name="q8_control_settings",
            task_type="control_settings",
            extra_args=(
                "--control-selection-policy",
                "linear_classifier",
                "--control-model-json",
                model_paths["q8_model_json"],
            ),
        ),
        TaskRunConfig(
            name="q9_future_trajectory",
            task_type="future_trajectory",
            extra_args=(
                "--future-trajectory-model-json",
                model_paths["q9_model_json"],
            ),
        ),
    )


def run_train_evals(
    *,
    run_name: str,
    v2vgot_root: str,
    workers: int,
    progress_every: int,
    e2e_root: Path,
    task_configs: tuple[TaskRunConfig, ...],
) -> list[dict[str, str]]:
    archived_dir = e2e_root / "train_eval"
    archived_dir.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, str]] = []
    print("\n" + "=" * 72)
    print("Step 8/8: Train-split official QA runs for Q1/Q2/Q3/Q4/Q5/Q6/Q7/Q8/Q9")
    print("=" * 72)
    for task in task_configs:
        scenario = f"e2e_{run_name}_train_{task.name}"
        command = [
            sys.executable,
            "scripts/run_qa_split_pipeline.py",
            "--purpose",
            "train_dev",
            "--split",
            "train",
            "--task-type",
            task.task_type,
            "--scenario-name",
            scenario,
            "--baseline-mode",
            "cooperative",
            "--v2vgot-root",
            v2vgot_root,
            "--workers",
            str(workers),
            "--progress-every",
            str(progress_every),
            *task.extra_args,
        ]
        run(command)
        src_json = summary_json_path("train_dev", scenario)
        src_md = summary_markdown_path("train_dev", scenario)
        src_export_manifest = export_manifest_path("train_dev", scenario)
        copy_if_exists(src_json, archived_dir / src_json.name)
        copy_if_exists(src_md, archived_dir / src_md.name)
        copy_if_exists(src_export_manifest, archived_dir / src_export_manifest.name)
        run_records.append(
            {
                "task_name": task.name,
                "task_type": task.task_type,
                "scenario_name": scenario,
                "summary_json": str(src_json),
                "summary_markdown": str(src_md),
                "official_export_manifest": str(src_export_manifest),
            }
        )
    return run_records


def main() -> int:
    args = build_parser().parse_args()
    e2e_root = REPO_ROOT / "outputs" / "e2e_runs" / args.run_name
    e2e_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("E2E Train Pipeline")
    print("=" * 72)
    print(f"run_name: {args.run_name}")
    print(f"e2e_root: {e2e_root}")
    print(f"v2vgot_root: {args.v2vgot_root}")
    print(f"workers: {args.workers}")
    print(f"allow_val_features_during_training: {args.allow_val_features_during_training}")
    print(f"q5_model_family: {args.q5_model_family}")
    print(f"q5_feature_set: {args.q5_feature_set}")
    print(f"q7_model_family: {args.q7_model_family}")
    print(f"q7_feature_set: {args.q7_feature_set}")
    print(f"q9_model_source: {args.q9_model_source}")
    print("q6_weights_source: trained_during_e2e_run")

    model_paths = train_models(
        e2e_root=e2e_root,
        v2vgot_root=args.v2vgot_root,
        workers=args.workers,
        progress_every=args.progress_every,
        allow_val_features_during_training=args.allow_val_features_during_training,
        q5_model_family=args.q5_model_family,
        q5_l2_regularization=args.q5_l2_regularization,
        q5_max_match_distance=args.q5_max_match_distance,
        q5_max_abs_delta=args.q5_max_abs_delta,
        q5_piecewise_min_rows=args.q5_piecewise_min_rows,
        q5_tree_max_depth=args.q5_tree_max_depth,
        q5_tree_min_leaf=args.q5_tree_min_leaf,
        q5_tree_min_gain=args.q5_tree_min_gain,
        q5_feature_set=args.q5_feature_set,
        q7_model_family=args.q7_model_family,
        q7_feature_set=args.q7_feature_set,
        q7_l2_regularization=args.q7_l2_regularization,
        q7_max_match_distance=args.q7_max_match_distance,
        q7_max_abs_delta=args.q7_max_abs_delta,
        q7_piecewise_min_rows=args.q7_piecewise_min_rows,
        q7_tree_max_depth=args.q7_tree_max_depth,
        q7_tree_min_leaf=args.q7_tree_min_leaf,
        q7_tree_min_gain=args.q7_tree_min_gain,
        q6_gbdt_n_estimators=args.q6_gbdt_n_estimators,
        q6_gbdt_learning_rate=args.q6_gbdt_learning_rate,
        q6_gbdt_max_depth=args.q6_gbdt_max_depth,
        q6_gbdt_min_samples_leaf=args.q6_gbdt_min_samples_leaf,
        q6_gbdt_subsample=args.q6_gbdt_subsample,
        q6_gbdt_backend=args.q6_gbdt_backend,
        q6_decision_threshold=args.q6_decision_threshold,
        q9_model_source=args.q9_model_source,
        q9_train_file_name=args.q9_train_file_name,
        q9_val_file_name=args.q9_val_file_name,
    )
    task_configs = build_train_task_configs(model_paths)
    train_runs = run_train_evals(
        run_name=args.run_name,
        v2vgot_root=args.v2vgot_root,
        workers=args.workers,
        progress_every=args.progress_every,
        e2e_root=e2e_root,
        task_configs=task_configs,
    )

    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "v2vgot_root": args.v2vgot_root,
        "workers": args.workers,
        "progress_every": args.progress_every,
        "allow_val_features_during_training": args.allow_val_features_during_training,
        "model_paths": model_paths,
        "frozen_policy_config": {
            "q1": {"notable_ranker": "heuristic"},
            "q2": {"occluding_ranker": "risk_adaptive"},
            "q5": {
                "task_type": "object_motion_prediction",
                "qa_type_id": 15,
                "policy_type": (
                    "learned_router_motion_projection"
                    if str(model_paths.get("q5_model_json", "")).strip()
                    else "deterministic_router_motion_projection"
                ),
                "object_motion_model_json": str(model_paths.get("q5_model_json", "")),
                "weights_source": "trained_during_e2e_run",
                "training_split": "train",
                "model_family": args.q5_model_family,
                "feature_set": args.q5_feature_set,
                "tree_max_depth": args.q5_tree_max_depth,
                "tree_min_leaf": args.q5_tree_min_leaf,
                "tree_min_gain": args.q5_tree_min_gain,
            },
            "q6": {
                "task_type": "agent_motion_prediction",
                "policy_type": "learned_gbdt_notability",
                "weights_source": "trained_during_e2e_run",
                "training_split": "train",
                "agent_motion_model_json": str(model_paths.get("q6_model_json", "")),
                "gbdt_backend": args.q6_gbdt_backend,
                "gbdt_n_estimators": args.q6_gbdt_n_estimators,
                "gbdt_learning_rate": args.q6_gbdt_learning_rate,
                "gbdt_max_depth": args.q6_gbdt_max_depth,
                "gbdt_min_samples_leaf": args.q6_gbdt_min_samples_leaf,
                "gbdt_subsample": args.q6_gbdt_subsample,
                "decision_threshold": args.q6_decision_threshold,
            },
            "q7": {
                "task_type": "object_motion_prediction",
                "qa_type_id": 17,
                "policy_type": "learned_router_motion_projection",
                "object_motion_model_json": str(model_paths.get("q7_model_json", "")),
                "weights_source": "trained_during_e2e_run",
                "training_split": "train",
                "model_family": args.q7_model_family,
                "feature_set": args.q7_feature_set,
                "tree_max_depth": args.q7_tree_max_depth,
                "tree_min_leaf": args.q7_tree_min_leaf,
                "tree_min_gain": args.q7_tree_min_gain,
                "notes": (
                    "Q7 trains a separate deployable model on qa_type_id=17 train rows. "
                    "Current V2V-GoT val labels for paired Q5/Q7 rows are identical, "
                    "so equal official metrics are expected when models produce the same outputs."
                ),
            },
            "q3": {
                "invisible_ranker": "logreg_acceptor",
                "invisible_max_results": 1,
                "invisible_shortlist_size": 64,
                "invisible_max_distance_to_trajectory": 8.0,
                "invisible_min_risk": 0.58,
                "invisible_min_relative_to_best": 0.75,
            },
            "q4": {
                "planning_ranker": "relational_importance",
                "planning_selection_policy": "trajectory_calibrated_acceptor",
                "planning_selection_source": "orchestrator",
            },
            "q8": {"control_selection_policy": "linear_classifier"},
            "q9": {
                "future_trajectory_model_family": "clean_q8_context_elasticnet",
                "q9_model_source": args.q9_model_source,
            },
        },
        "train_runs": train_runs,
    }
    manifest_path = e2e_root / "e2e_model_manifest.json"
    write_json(manifest_path, manifest)
    manifest_md_path = e2e_root / "e2e_model_manifest.md"
    write_train_markdown(manifest_md_path, manifest)

    print("\n" + "=" * 72)
    print("E2E train pipeline complete")
    print("=" * 72)
    print(f"manifest: {manifest_path}")
    print(f"manifest_markdown: {manifest_md_path}")
    print("model artifacts:")
    for key, value in model_paths.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
