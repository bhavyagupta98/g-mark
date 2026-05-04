from __future__ import annotations

from kg_coop_drive.domain.benchmark import BenchmarkTaskType

from scripts.evaluate_v2vgotqa_phase5a import apply_sample_limit as apply_phase5a_sample_limit
from scripts.run_phase5_closeout import apply_sample_limit as apply_closeout_sample_limit
from scripts.run_v2vgot_official_qa_eval import patch_evaluator_text
from scripts.score_phase5_closeout import infer_task_types


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
