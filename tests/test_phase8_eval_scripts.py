from __future__ import annotations

from kg_coop_drive.domain.benchmark import BenchmarkTaskType

from scripts.evaluate_v2vgotqa_phase5a import apply_sample_limit as apply_phase5a_sample_limit
from scripts.e2e.run_e2e_train_pipeline import (
    Q6_E2E_TRAINING_DEFAULTS,
    build_parser as build_e2e_train_parser,
)
from scripts.e2e.run_e2e_validation_report import (
    build_val_task_configs,
    extract_primary_metric,
)
from scripts.train_q5_object_motion_predictor import (
    Q5ObjectMotionTrainer,
    feature_names_for_qa_type,
    resolve_feature_set,
    target_waypoint_count_for_qa_type,
)
from scripts.train_q7_object_motion_predictor import Q7ObjectMotionTrainer
from scripts.run_phase5_closeout import apply_sample_limit as apply_closeout_sample_limit
from scripts.run_qa_split_pipeline import num_future_waypoints_for_official_eval
from scripts.run_v2vgot_official_qa_eval import patch_evaluator_text
from scripts.score_phase5_closeout import infer_task_types
from scripts.export_qa_predictions import export_task


def test_score_closeout_infers_run_task_types_missing_from_manifest() -> None:
    manifest = {"task_types": ["occluding_objects"]}
    runs = [
        {"task_type": "notable_objects"},
        {"task_type": "occluding_objects"},
        {"task_type": "planning_awareness"},
    ]

    assert infer_task_types(manifest, runs) == (
        BenchmarkTaskType.OCCLUDING_OBJECTS,
        BenchmarkTaskType.NOTABLE_OBJECTS,
        BenchmarkTaskType.PLANNING_AWARENESS,
    )


def test_limit_zero_means_full_sample_set() -> None:
    samples = ("a", "b", "c")

    assert apply_phase5a_sample_limit(samples, 0) == samples
    assert apply_closeout_sample_limit(samples, 0) == samples
    assert apply_phase5a_sample_limit(samples, 2) == ("a", "b")
    assert apply_closeout_sample_limit(samples, 2) == ("a", "b")


def test_official_qa_patch_guards_empty_action_accuracy_denominator() -> None:
    source = (
        "def evaluate():\n"
        "    action_accuracy = 1.0 * num_matched_gt_output_correct_action_count / (\n"
        "        num_matched_gt_output_correct_action_count\n"
        "        + num_matched_gt_output_incorrect_action_count\n"
        "    )\n"
    )

    patched, missing = patch_evaluator_text(source)

    assert "action_accuracy_denominator =" in patched
    assert "if action_accuracy_denominator == 0" in patched
    assert "action_accuracy = 1.0 * num_matched_gt_output_correct_action_count /" not in patched
    assert all("action_accuracy = 1.0" not in item for item in missing)


def test_official_qa_patch_guards_empty_binary_and_localization_metrics() -> None:
    source = (
        "def evaluate():\n"
        "    localization_precision = 1.0 * num_matched_gt_output[threshold_id] / num_outputs\n"
        "    localization_recall = 1.0 * num_matched_gt_output[threshold_id] / num_gts\n"
        "    localization_f1 = 2.0 * localization_precision * localization_recall / "
        "(localization_precision + localization_recall)\n"
        "    binary_precision = 1.0 * binary_tp / (binary_tp + binary_fp)\n"
        "    binary_recall = 1.0 * binary_tp / (binary_tp + binary_fn)\n"
        "    binary_f1 = 2.0 * binary_precision * binary_recall / "
        "(binary_precision + binary_recall)\n"
    )

    patched, missing = patch_evaluator_text(source)

    assert "localization_precision_denominator =" in patched
    assert "localization_recall_denominator =" in patched
    assert "localization_f1_denominator =" in patched
    assert "binary_precision_denominator =" in patched
    assert "binary_recall_denominator =" in patched
    assert "binary_f1_denominator =" in patched
    assert "if binary_precision_denominator == 0" in patched
    assert "binary_precision = 1.0 * binary_tp /" not in patched
    assert "localization_precision = 1.0 * num_matched_gt_output[threshold_id] /" not in patched
    assert all("denominator guard" not in item for item in missing)


def test_e2e_train_defaults_use_q6_gbdt_training_config() -> None:
    args = build_e2e_train_parser().parse_args([])

    assert args.q6_gbdt_backend == Q6_E2E_TRAINING_DEFAULTS.gbdt_backend == "sklearn"
    assert args.q6_gbdt_n_estimators == Q6_E2E_TRAINING_DEFAULTS.gbdt_n_estimators
    assert args.q6_gbdt_learning_rate == Q6_E2E_TRAINING_DEFAULTS.gbdt_learning_rate
    assert args.q6_gbdt_max_depth == Q6_E2E_TRAINING_DEFAULTS.gbdt_max_depth
    assert args.q6_gbdt_min_samples_leaf == Q6_E2E_TRAINING_DEFAULTS.gbdt_min_samples_leaf
    assert args.q6_gbdt_subsample == Q6_E2E_TRAINING_DEFAULTS.gbdt_subsample
    assert args.q6_decision_threshold == Q6_E2E_TRAINING_DEFAULTS.decision_threshold
    assert args.q5_model_family == "regression_tree"
    assert args.q5_feature_set == "path_relative"
    assert args.q7_model_family == "regression_tree"
    assert args.q7_feature_set == "path_relative"


def test_e2e_validation_runs_q6_with_manifest_model_and_metric() -> None:
    task_configs = build_val_task_configs(
        {
            "q3_model_json": "q3.json",
            "q4_model_json": "q4.json",
            "q5_model_json": "q5.json",
            "q7_model_json": "q7.json",
            "q6_model_json": "q6.json",
            "q8_model_json": "q8.json",
            "q9_model_json": "q9.json",
        }
    )

    q6_config = next(config for config in task_configs if config.name == "q6_agent_motion_prediction")
    assert q6_config.task_type == "agent_motion_prediction"
    assert q6_config.metric_label == "Binary Accuracy"
    assert q6_config.extra_args == ("--agent-motion-model-json", "q6.json")

    metric = extract_primary_metric(
        "q6_agent_motion_prediction",
        {"runs": [{"metrics": {"binary_classification_accuracy": 0.9045269878119558}}]},
    )
    assert metric == 0.9045269878119558


def test_e2e_validation_splits_q5_and_q7_by_qa_type_id() -> None:
    task_configs = build_val_task_configs(
        {
            "q3_model_json": "q3.json",
            "q4_model_json": "q4.json",
            "q5_model_json": "q5.json",
            "q7_model_json": "q7.json",
            "q6_model_json": "q6.json",
            "q8_model_json": "q8.json",
            "q9_model_json": "q9.json",
        }
    )

    q5_config = next(config for config in task_configs if config.name == "q5_object_motion_prediction")
    q7_config = next(config for config in task_configs if config.name == "q7_object_motion_prediction")

    assert q5_config.task_type == q7_config.task_type == "object_motion_prediction"
    assert q5_config.extra_args == ("--qa-type-id", "15", "--object-motion-model-json", "q5.json")
    assert q7_config.extra_args == ("--qa-type-id", "17", "--object-motion-model-json", "q7.json")

    metric = extract_primary_metric(
        "q7_object_motion_prediction",
        {"runs": [{"metrics": {"l2_error_avg_123_all": 6.5}}]},
    )
    assert metric == 6.5


def test_q5_q7_motion_trainers_have_separate_entrypoints() -> None:
    q5 = Q5ObjectMotionTrainer()
    q7 = Q7ObjectMotionTrainer()
    args = q7.build_parser().parse_args(
        [
            "--model-family",
            "gradient_boosting",
            "--output-json",
            "model.json",
        ]
    )

    assert q5.qa_type_id == 15
    assert q7.qa_type_id == 17
    assert args.model_family == "gradient_boosting"
    assert not hasattr(args, "qa_type_id")


def test_object_motion_eval_uses_one_future_waypoint_for_q5_and_q7() -> None:
    assert num_future_waypoints_for_official_eval("object_motion_prediction", [15]) == 1
    assert num_future_waypoints_for_official_eval("object_motion_prediction", [17]) == 1
    assert num_future_waypoints_for_official_eval("future_trajectory", []) == 6
    assert target_waypoint_count_for_qa_type(15) == 1
    assert target_waypoint_count_for_qa_type(17) == 1


def test_object_motion_feature_set_defaults_keep_q5_independent_from_q7() -> None:
    q5_auto_features = feature_names_for_qa_type(15)
    q5_path_features = feature_names_for_qa_type(15, "path_relative")
    q7_auto_features = feature_names_for_qa_type(17)

    assert resolve_feature_set(15, "auto") == "base"
    assert resolve_feature_set(17, "auto") == "path_relative"
    assert "asker_goal_x_from_object" not in q5_auto_features
    assert "asker_goal_x_from_object" in q5_path_features
    assert q7_auto_features == q5_path_features


def test_export_task_does_not_fallback_across_explicit_qa_type_ids(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        '{"sample_id":"0","qa_type_id":15,"answer_text":"x","object_ids":[]}\n',
        encoding="utf-8",
    )

    run = {
        "task_type": "object_motion_prediction",
        "qa_type_id": 15,
        "output_jsonl": str(prediction_path),
    }

    result = export_task(
        task_type=BenchmarkTaskType.OBJECT_MOTION_PREDICTION,
        run=run,
        samples_by_task_qa_and_id={},
        evaluator=object(),
        output_dir=tmp_path,
        baseline_mode="cooperative",
        scenario_name="explicit_q5_missing_exact_sample",
    )

    assert result["exported_count"] == 0
    assert result["missing_samples"] == 1
