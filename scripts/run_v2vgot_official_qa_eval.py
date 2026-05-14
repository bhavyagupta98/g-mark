#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)

TASK_LABELS = {
    11: "notable_objects",
    12: "occluding_objects",
    13: "invisible_objects",
    14: "planning_awareness",
    15: "object_motion_prediction",
    16: "agent_motion_prediction",
    17: "object_motion_prediction",
    18: "control_settings",
    19: "future_trajectory",
}

HEAVY_IMPORT_REPLACEMENTS = {
    "import cv2": "try:\n    import cv2\nexcept ModuleNotFoundError:\n    cv2 = None",
    "from AB3DMOT.AB3DMOT_libs.box import Box3D": (
        "try:\n"
        "    from AB3DMOT.AB3DMOT_libs.box import Box3D\n"
        "except ModuleNotFoundError:\n"
        "    Box3D = None"
    ),
    "from AB3DMOT_libs.dist_metrics import iou": (
        "try:\n"
        "    from AB3DMOT_libs.dist_metrics import iou\n"
        "except ModuleNotFoundError:\n"
        "    iou = None"
    ),
    "from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing": (
        "try:\n"
        "    from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing\n"
        "except ModuleNotFoundError:\n"
        "    def mkdir_if_missing(path):\n"
        "        os.makedirs(path, exist_ok=True)"
    ),
    "from V2V4Real.opencood.utils.transformation_utils import x1_to_x2": (
        "try:\n"
        "    from V2V4Real.opencood.utils.transformation_utils import x1_to_x2\n"
        "except ModuleNotFoundError:\n"
        "    x1_to_x2 = None"
    ),
    "from V2V4Real.opencood.utils.box_utils import boxes_to_corners_3d, corner_to_center, project_points_by_matrix_torch": (
        "try:\n"
        "    from V2V4Real.opencood.utils.box_utils import boxes_to_corners_3d, corner_to_center, project_points_by_matrix_torch\n"
        "except ModuleNotFoundError:\n"
        "    boxes_to_corners_3d = corner_to_center = None\n"
        "    def project_points_by_matrix_torch(points, transformation_matrix):\n"
        "        points_array = np.asarray(points)\n"
        "        transform_array = np.asarray(transformation_matrix)\n"
        "        ones = np.ones((points_array.shape[0], 1), dtype=points_array.dtype)\n"
        "        homogeneous = np.concatenate([points_array, ones], axis=1)\n"
        "        return (homogeneous @ transform_array.T)[:, :3]"
    ),
    "import matplotlib.pyplot as plt": (
        "try:\n"
        "    import matplotlib.pyplot as plt\n"
        "except ModuleNotFoundError:\n"
        "    plt = None"
    ),
}

UNSAFE_ACTION_ACCURACY_PATTERN = re.compile(
    r"(?P<indent>^[ \t]*)"
    r"action_accuracy\s*=\s*"
    r"1\.0\s*\*\s*num_matched_gt_output_correct_action_count\s*/\s*"
    r"\(\s*num_matched_gt_output_correct_action_count\s*\+\s*"
    r"num_matched_gt_output_incorrect_action_count\s*\)",
    re.MULTILINE,
)
UNSAFE_QA_DIVISION_PATTERNS = {
    "localization_precision": re.compile(
        r"(?P<indent>^[ \t]*)"
        r"localization_precision\s*=\s*"
        r"(?P<numerator>1\.0\s*\*\s*num_matched_gt_output\s*\[\s*threshold_id\s*\])\s*/\s*"
        r"(?P<denominator>num_outputs)\b",
        re.MULTILINE,
    ),
    "localization_recall": re.compile(
        r"(?P<indent>^[ \t]*)"
        r"localization_recall\s*=\s*"
        r"(?P<numerator>1\.0\s*\*\s*num_matched_gt_output\s*\[\s*threshold_id\s*\])\s*/\s*"
        r"(?P<denominator>num_gts)\b",
        re.MULTILINE,
    ),
    "localization_f1": re.compile(
        r"(?P<indent>^[ \t]*)"
        r"localization_f1\s*=\s*"
        r"(?P<numerator>2(?:\.0)?\s*\*\s*localization_precision\s*\*\s*localization_recall)\s*/\s*"
        r"(?P<denominator>\(\s*localization_precision\s*\+\s*localization_recall\s*\))",
        re.MULTILINE,
    ),
    "binary_precision": re.compile(
        r"(?P<indent>^[ \t]*)"
        r"binary_precision\s*=\s*"
        r"(?P<numerator>1\.0\s*\*\s*binary_tp)\s*/\s*"
        r"(?P<denominator>\(\s*binary_tp\s*\+\s*binary_fp\s*\))",
        re.MULTILINE,
    ),
    "binary_recall": re.compile(
        r"(?P<indent>^[ \t]*)"
        r"binary_recall\s*=\s*"
        r"(?P<numerator>1\.0\s*\*\s*binary_tp)\s*/\s*"
        r"(?P<denominator>\(\s*binary_tp\s*\+\s*binary_fn\s*\))",
        re.MULTILINE,
    ),
    "binary_f1": re.compile(
        r"(?P<indent>^[ \t]*)"
        r"binary_f1\s*=\s*"
        r"(?P<numerator>2(?:\.0)?\s*\*\s*binary_precision\s*\*\s*binary_recall)\s*/\s*"
        r"(?P<denominator>\(\s*binary_precision\s*\+\s*binary_recall\s*\))",
        re.MULTILINE,
    ),
}

METRIC_PATTERNS = {
    "binary_f1": re.compile(r"binary_f1:\s+([0-9.eE+-]+)"),
    "binary_precision": re.compile(r"binary_precision:\s+([0-9.eE+-]+)"),
    "binary_recall": re.compile(r"binary_recall:\s+([0-9.eE+-]+)"),
    "gt_parse_error_rate": re.compile(r"gt_parse_error_rate:\s+([0-9.eE+-]+)"),
    "output_parse_error_rate": re.compile(r"output_parse_error_rate:\s+([0-9.eE+-]+)"),
    "l2_error_avg_1s": re.compile(r"l2_error_avg_1s:\s+([0-9.eE+-]+)"),
    "l2_error_avg_2s": re.compile(r"l2_error_avg_2s:\s+([0-9.eE+-]+)"),
    "l2_error_avg_3s": re.compile(r"l2_error_avg_3s:\s+([0-9.eE+-]+)"),
    "l2_error_avg_all": re.compile(r"l2_error_avg_all:\s+([0-9.eE+-]+)"),
    "l2_error_avg_123_all": re.compile(r"l2_error_avg_123_all:\s+([0-9.eE+-]+)"),
    "l2_error_avg_03_all": re.compile(r"l2_error_avg_03_all:\s+([0-9.eE+-]+)"),
    "collision_rate_1s": re.compile(r"collision_rate_1s:\s+([0-9.eE+-]+)"),
    "collision_rate_2s": re.compile(r"collision_rate_2s:\s+([0-9.eE+-]+)"),
    "collision_rate_3s": re.compile(r"collision_rate_3s:\s+([0-9.eE+-]+)"),
    "collision_rate_avg_all": re.compile(r"collision_rate_avg_all:\s+([0-9.eE+-]+)"),
    "NC": re.compile(r"NC:\s+([0-9.eE+-]+)"),
    "TTC": re.compile(r"TTC:\s+([0-9.eE+-]+)"),
    "C": re.compile(r"C:\s+([0-9.eE+-]+)"),
    "PDMS": re.compile(r"PDMS:\s+([0-9.eE+-]+)"),
    "PDMS_sample_average": re.compile(r"PDMS_sample_average:\s+([0-9.eE+-]+)"),
    "speed_accuracy": re.compile(r"speed_accuracy:\s+([0-9.eE+-]+)"),
    "steering_accuracy": re.compile(r"steering_accuracy:\s+([0-9.eE+-]+)"),
    "action_accuracy": re.compile(r"action_accuracy:\s+([0-9.eE+-]+)"),
    "speed_edit_dist": re.compile(r"speed_edit_dist:\s+([0-9.eE+-]+)"),
    "steering_edit_dist": re.compile(r"steering_edit_dist:\s+([0-9.eE+-]+)"),
    "action_edit_dist": re.compile(r"action_edit_dist:\s+([0-9.eE+-]+)"),
    "binary_classification_accuracy": re.compile(r"binary classification accuracy:\s+([0-9.eE+-]+)"),
}
LOCALIZATION_PATTERN = re.compile(
    r"localization_(f1|precision|recall)\s*@?\s*([0-9.]+)?:\s+([0-9.eE+-]+)"
)
PLANNING_THRESHOLD_PATTERN = re.compile(r"threshold:\s+([0-9.eE+-]+)")
PLANNING_METRIC_PATTERN = re.compile(r"^(precision|recall|f1):\s+([0-9.eE+-]+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V2V-GoT/LLaVA simplified Q1-Q9 official evaluator on "
            "official-style export files, using a persistent QA-only evaluator copy."
        )
    )
    parser.add_argument("--export-manifest", required=True)
    parser.add_argument("--v2vgot-root", default="")
    parser.add_argument("--output-dir", default="outputs/phase8_official_exports/eval_reports")
    parser.add_argument(
        "--tools-dir",
        default="outputs/phase8_official_exports/tools",
        help="Mounted repo directory where the patched QA-only evaluator copy is stored.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-num-answer-objects", type=int, default=3)
    parser.add_argument("--num-future-waypoints", type=int, default=0)
    parser.add_argument(
        "--npy-save-path",
        default="",
        help=(
            "Optional V2V-GoT npy asset root passed through to the upstream evaluator. "
            "Q9/Q5 trajectory metrics need lidar poses and GT boxes from this path."
        ),
    )
    parser.add_argument(
        "--task-type",
        action="append",
        default=[],
        help="Optional task_type from the export manifest. Repeatable.",
    )
    return parser


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_v2vgot_root(raw_value: str) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def patch_evaluator_text(text: str) -> tuple[str, list[str]]:
    missing_replacements: list[str] = []
    for old, new in HEAVY_IMPORT_REPLACEMENTS.items():
        if old not in text:
            missing_replacements.append(old)
            continue
        text = text.replace(old, new)

    def replace_action_accuracy(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}action_accuracy_denominator = "
            "num_matched_gt_output_correct_action_count + "
            "num_matched_gt_output_incorrect_action_count\n"
            f"{indent}action_accuracy = 0.0 if action_accuracy_denominator == 0 "
            "else 1.0 * num_matched_gt_output_correct_action_count / "
            "action_accuracy_denominator"
        )

    text, replacement_count = UNSAFE_ACTION_ACCURACY_PATTERN.subn(replace_action_accuracy, text)
    if replacement_count == 0 and "num_matched_gt_output_correct_action_count" in text:
        missing_replacements.append("unsafe action_accuracy denominator guard")

    collision_dependency_line = (
        "    current_ego_lidar_pose = np.load(os.path.join(npy_save_path, 'ego', "
        "'%04d_lidar_pose.npy' % global_timestamp_index))"
    )
    collision_dependency_guard = (
        "    if any(helper is None for helper in (x1_to_x2, project_points_by_matrix_torch, corner_to_center, Box3D, iou)):\n"
        "      return False\n"
        "\n"
        f"{collision_dependency_line}"
    )
    if collision_dependency_line in text:
        text = text.replace(collision_dependency_line, collision_dependency_guard, 1)
    elif "def check_has_collision" in text:
        missing_replacements.append("Q9 collision dependency guard")

    for metric_name, pattern in UNSAFE_QA_DIVISION_PATTERNS.items():
        def replace_metric_division(match: re.Match[str], metric_name: str = metric_name) -> str:
            indent = match.group("indent")
            numerator = " ".join(match.group("numerator").split())
            denominator = " ".join(match.group("denominator").split())
            denominator_name = f"{metric_name}_denominator"
            return (
                f"{indent}{denominator_name} = {denominator}\n"
                f"{indent}{metric_name} = 0.0 if {denominator_name} == 0 "
                f"else {numerator} / {denominator_name}"
            )

        text, replacement_count = pattern.subn(replace_metric_division, text)
        if replacement_count == 0 and metric_name in text:
            missing_replacements.append(f"unsafe {metric_name} denominator guard")
    return text, missing_replacements


def create_qa_only_evaluator(v2vgot_root: Path, tools_dir: Path) -> Path:
    source_path = v2vgot_root / "LLaVA/scripts/eval_v2v4real_3d_grounding.py"
    if not source_path.exists():
        raise FileNotFoundError(f"missing upstream evaluator: {source_path}")

    text = source_path.read_text(encoding="utf-8")
    text, missing_replacements = patch_evaluator_text(text)

    header = (
        "#!/usr/bin/env python3\n"
        "# Auto-generated by kg_coop_drive/scripts/run_v2vgot_official_qa_eval.py.\n"
        "# This persistent copy guards import-time dependencies and empty-count QA metric divisions in the simplified Q1-Q9 path.\n\n"
    )
    if text.startswith("#!"):
        text = "\n".join(text.splitlines()[1:])

    tools_dir.mkdir(parents=True, exist_ok=True)
    evaluator_path = tools_dir / "eval_v2v4real_3d_grounding_qa_only.py"
    evaluator_path.write_text(header + text, encoding="utf-8")

    generated_text = evaluator_path.read_text(encoding="utf-8")
    unsafe_generated_metrics = [
        metric_name
        for metric_name, pattern in UNSAFE_QA_DIVISION_PATTERNS.items()
        if pattern.search(generated_text)
    ]
    if UNSAFE_ACTION_ACCURACY_PATTERN.search(generated_text) or unsafe_generated_metrics:
        raise RuntimeError(
            "Generated QA-only evaluator still contains unsafe metric divisions "
            f"{unsafe_generated_metrics}. "
            f"Inspect {evaluator_path} before rerunning the official evaluation."
        )

    if missing_replacements:
        print("warning: these upstream import patterns were not found:")
        for item in missing_replacements:
            print(f"  - {item}")
    return evaluator_path


def build_pythonpath(v2vgot_root: Path) -> str:
    additions = [
        v2vgot_root / "DMSTrack",
        v2vgot_root / "DMSTrack/AB3DMOT",
        v2vgot_root / "DMSTrack/AB3DMOT/Xinshuo_PyToolbox",
        v2vgot_root / "DMSTrack/V2V4Real",
        v2vgot_root / "LLaVA",
    ]
    existing = os.environ.get("PYTHONPATH", "")
    values = [str(path) for path in additions]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def parse_metrics(stdout: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(stdout)
        if match:
            metrics[name] = float(match.group(1))

    localization: dict[str, dict[str, float]] = {}
    for metric_name, raw_threshold, raw_value in LOCALIZATION_PATTERN.findall(stdout):
        threshold = str(float(raw_threshold or "0.5"))
        localization.setdefault(threshold, {})[metric_name] = float(raw_value)

    current_threshold = ""
    for line in stdout.splitlines():
        threshold_match = PLANNING_THRESHOLD_PATTERN.search(line)
        if threshold_match:
            current_threshold = str(float(threshold_match.group(1)))
            localization.setdefault(current_threshold, {})
            continue
        metric_match = PLANNING_METRIC_PATTERN.search(line.strip())
        if metric_match and current_threshold:
            metric_name, raw_value = metric_match.groups()
            localization.setdefault(current_threshold, {})[metric_name] = float(raw_value)

    if localization:
        metrics["localization"] = localization
    return metrics


def run_one(
    python_bin: str,
    evaluator_path: Path,
    v2vgot_root: Path,
    output_dir: Path,
    run: dict[str, object],
    max_num_answer_objects: int,
    num_future_waypoints: int,
    npy_save_path: str,
) -> dict[str, object]:
    task_type = str(run["task_type"])
    qa_type_id = int(run["qa_type_id"])
    answer_path = Path(str(run["output_jsonl"]))
    if not answer_path.is_absolute():
        answer_path = (REPO_ROOT / answer_path).resolve()

    command = [
        python_bin,
        str(evaluator_path),
        "--answers-file",
        str(answer_path),
        "--simplified",
        "--multiple-output",
        "--max-num-answer-objects",
        str(max_num_answer_objects),
        "--qa-type-id",
        str(qa_type_id),
        "--num-future-waypoints",
        str(num_future_waypoints),
    ]
    if npy_save_path:
        command.extend(["--npy-save-path", npy_save_path])

    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(v2vgot_root)
    completed = subprocess.run(
        command,
        cwd=str(v2vgot_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    log_path = output_dir / f"{task_type}_qa_type_{qa_type_id}_official_eval.log"
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )

    return {
        "task_type": task_type,
        "qa_type_id": qa_type_id,
        "answers_file": str(answer_path),
        "returncode": completed.returncode,
        "log_path": str(log_path),
        "metrics": parse_metrics(completed.stdout),
    }


def write_markdown(summary: dict[str, object], markdown_path: Path) -> None:
    runs = summary.get("runs", [])
    task_scope = ",".join(
        sorted(
            str(run.get("task_type"))
            for run in runs
            if isinstance(run, dict) and run.get("task_type") is not None
        )
    ) or "none"
    lines = [
        f"# Official QA Evaluation (`task_scope={task_scope}`)",
        "",
        f"- `export_manifest`: `{summary['export_manifest']}`",
        f"- `v2vgot_root`: `{summary['v2vgot_root']}`",
        f"- `evaluator_path`: `{summary['evaluator_path']}`",
        "",
        "| Task | QA Type | Return Code | F1 @ 0.5m | Precision @ 0.5m | Recall @ 0.5m | Binary F1 | Deferred Metric | Parse Error | Log |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for run in summary["runs"]:
        metrics = run.get("metrics", {})
        localization = metrics.get("localization", {}) if isinstance(metrics, dict) else {}
        threshold_metrics = localization.get("0.5", {}) if isinstance(localization, dict) else {}
        f1 = threshold_metrics.get("f1", "")
        precision = threshold_metrics.get("precision", "")
        recall = threshold_metrics.get("recall", "")
        binary_f1 = metrics.get("binary_f1", "") if isinstance(metrics, dict) else ""
        parse_error = metrics.get("output_parse_error_rate", "") if isinstance(metrics, dict) else ""
        deferred_metric = ""
        if isinstance(metrics, dict):
            if "l2_error_avg_all" in metrics:
                deferred_metric = f"l2_error_avg_all={metrics['l2_error_avg_all']}"
            elif "l2_error_avg_123_all" in metrics:
                deferred_metric = f"l2_error_avg_123_all={metrics['l2_error_avg_123_all']}"
            elif "action_accuracy" in metrics:
                deferred_metric = f"action_accuracy={metrics['action_accuracy']}"
            elif "binary_classification_accuracy" in metrics:
                deferred_metric = f"binary_classification_accuracy={metrics['binary_classification_accuracy']}"
        lines.append(
            "| "
            + f"`{run['task_type']}` | `{run['qa_type_id']}` | `{run['returncode']}` | "
            + f"`{f1}` | `{precision}` | `{recall}` | `{binary_f1}` | `{deferred_metric}` | `{parse_error}` | "
            + f"`{run['log_path']}` |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    export_manifest_path = resolve_repo_path(args.export_manifest)
    output_dir = resolve_repo_path(args.output_dir)
    tools_dir = resolve_repo_path(args.tools_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    v2vgot_root = resolve_v2vgot_root(args.v2vgot_root or str(manifest.get("repository_root", "")))
    evaluator_path = create_qa_only_evaluator(v2vgot_root, tools_dir)

    selected_tasks = set(args.task_type)
    runs = [
        run
        for run in manifest.get("runs", [])
        if isinstance(run, dict)
        and int(run.get("qa_type_id", -1)) in TASK_LABELS
        and (not selected_tasks or str(run.get("task_type")) in selected_tasks)
    ]

    results = [
        run_one(
            python_bin=args.python,
            evaluator_path=evaluator_path,
            v2vgot_root=v2vgot_root,
            output_dir=output_dir,
            run=run,
            max_num_answer_objects=args.max_num_answer_objects,
            num_future_waypoints=args.num_future_waypoints,
            npy_save_path=args.npy_save_path,
        )
        for run in runs
    ]

    summary = {
        "export_manifest": str(export_manifest_path),
        "v2vgot_root": str(v2vgot_root),
        "evaluator_path": str(evaluator_path),
        "runs": results,
    }
    json_path = output_dir / f"{export_manifest_path.stem}_official_qa_eval_summary.json"
    markdown_path = output_dir / f"{export_manifest_path.stem}_official_qa_eval_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, markdown_path)

    task_scope = ",".join(sorted(str(result["task_type"]) for result in results)) if results else "none"
    print("=" * 72)
    print("Official QA Evaluation")
    print("=" * 72)
    print(f"task_scope: {task_scope}")
    print(f"export_manifest: {export_manifest_path}")
    print(f"v2vgot_root: {v2vgot_root}")
    print(f"evaluator_path: {evaluator_path}")
    for result in results:
        print(
            f"[{result['task_type']}] qa_type_id={result['qa_type_id']} "
            f"returncode={result['returncode']} log={result['log_path']}"
        )
        metrics = result.get("metrics", {})
        if metrics:
            print(f"  metrics: {json.dumps(metrics, sort_keys=True)}")
        if result["returncode"] != 0:
            log_text = Path(str(result["log_path"])).read_text(encoding="utf-8")
            stderr = log_text.split("\nSTDERR:\n", 1)[-1].strip()
            if stderr:
                tail = "\n".join(stderr.splitlines()[-12:])
                print("  stderr_tail:")
                for line in tail.splitlines():
                    print(f"    {line}")
    print(f"saved_json: {json_path}")
    print(f"saved_markdown: {markdown_path}")

    return 0 if all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
