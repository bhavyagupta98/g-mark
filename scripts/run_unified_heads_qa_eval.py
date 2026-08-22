#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmark.features.leakage_checks import Q9_LEAKAGE_FIELDS  # noqa: E402
from gmark.models.unified_artifacts import (  # noqa: E402
    load_all_unified_artifacts,
    predict_with_motion_artifact,
    predict_with_object_retrieval_artifact,
    predict_with_scene_action_artifact,
)
from gmark.output.unified_prediction_formatters import (  # noqa: E402
    base_prediction_record,
    format_motion_answer,
    format_object_grounding_answer,
    format_q6_answer,
    format_q8_answer,
)
from kg_coop_drive.domain.benchmark import BenchmarkTaskType  # noqa: E402

LOGGER = logging.getLogger("run_unified_heads_qa_eval")

QA_TASK = {
    11: BenchmarkTaskType.NOTABLE_OBJECTS.value,
    12: BenchmarkTaskType.OCCLUDING_OBJECTS.value,
    13: BenchmarkTaskType.INVISIBLE_OBJECTS.value,
    14: BenchmarkTaskType.PLANNING_AWARENESS.value,
    15: BenchmarkTaskType.OBJECT_MOTION_PREDICTION.value,
    16: BenchmarkTaskType.AGENT_MOTION_PREDICTION.value,
    17: BenchmarkTaskType.OBJECT_MOTION_PREDICTION.value,
    18: BenchmarkTaskType.CONTROL_SETTINGS.value,
    19: BenchmarkTaskType.FUTURE_TRAJECTORY.value,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run unified_heads Stage 4 inference/eval.")
    p.add_argument("--config", default="configs/unified_heads/default.yaml")
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--feature-dir", default="")
    p.add_argument("--labeled-dir", default="")
    p.add_argument("--reuse-feature-rows", action="store_true")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=50000)
    p.add_argument("--log-every", type=int, default=100000)
    p.add_argument("--memory-log-every", type=int, default=500000)
    p.add_argument("--skip-official-eval", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def iter_jsonl(path: Path, max_rows: int = 0) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as h:
        n = 0
        for line in h:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)
            n += 1
            if max_rows > 0 and n >= max_rows:
                break


def get_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _obj_id(candidate_id: str) -> str:
    if "::" in candidate_id:
        return candidate_id.split("::", 1)[0]
    return candidate_id


_SAFE_OBJ_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_object_id(raw: str, fallback: str = "obj0") -> str:
    value = _SAFE_OBJ_RE.sub("_", str(raw or "").strip())
    value = value.strip("_")
    return value or fallback


def _to_six_points(target: list[float]) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in range(0, len(target), 2):
        if i + 1 >= len(target):
            break
        pts.append((float(target[i]), float(target[i + 1])))
    if not pts:
        pts = [(0.0, 0.0)]
    if len(pts) > 6:
        pts = pts[:6]
    if len(pts) < 6:
        pts = pts + [pts[-1]] * (6 - len(pts))
    return pts


def _strict_q5_output(object_id: str, start_x: float, start_y: float, target: list[float]) -> str:
    pts = _to_six_points(target)
    end_x, end_y = pts[-1]
    dx = end_x - float(start_x)
    dy = end_y - float(start_y)
    speed = (dx * dx + dy * dy) ** 0.5
    if speed < 0.1:
        motion = "staying at the same location"
    elif abs(dy) > abs(dx):
        motion = "turning right" if dy >= 0.0 else "turning left"
    elif dx >= 0.0:
        motion = "moving forward"
    else:
        motion = "turning right" if dy >= 0.0 else "turning left"
    rendered = ",".join(f"({x:.1f},{y:.1f})" for x, y in pts)
    return (
        f"There is a car at ({start_x:.1f},{start_y:.1f}) {motion}. "
        f"The predicted future trajectory is [{rendered}]."
    )


def run_object_retrieval(
    feature_path: Path,
    artifact: dict[str, Any],
    *,
    batch_size: int,
    max_rows: int,
    log_every: int,
    memory_log_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    thresholds = {int(k): float(v) for k, v in artifact.get("per_task_thresholds", {}).items()}
    max_results_raw = artifact.get("per_task_max_results", {}) if isinstance(artifact.get("per_task_max_results"), dict) else {}
    max_results = {int(k): (None if v is None else int(v)) for k, v in max_results_raw.items()}

    preds: list[dict[str, Any]] = []
    rows = 0
    grouped = 0
    selected_counts = Counter()
    empty_counts = Counter()
    curr_key: tuple[str, int] | None = None
    curr_rows: list[dict[str, Any]] = []

    def flush_group() -> None:
        nonlocal grouped
        if not curr_rows:
            return
        grouped += 1
        qa = int(curr_rows[0].get("qa_type_id", -1) or -1)
        sid = str(curr_rows[0].get("sample_id", ""))
        scores = predict_with_object_retrieval_artifact(curr_rows, artifact, config=None)
        thr = thresholds.get(qa, 0.5)
        k = max_results.get(qa)
        scored = []
        for row, score in zip(curr_rows, scores):
            cid = _obj_id(str(row.get("candidate_id", "")))
            if score >= thr and cid:
                scored.append((cid, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        if k is not None and k > 0:
            scored = scored[:k]
        obj_ids = [cid for cid, _ in scored]
        if not obj_ids:
            empty_counts[qa] += 1
        selected_counts[qa] += len(obj_ids)
        preds.append(
            base_prediction_record(
                sample_id=sid,
                qa_type_id=qa,
                task_type=QA_TASK.get(qa, "unknown"),
                answer_text=format_object_grounding_answer(qa, obj_ids),
                object_ids=obj_ids,
                extra={"scores": [s for _, s in scored], "threshold": thr},
            )
        )

    for row in iter_jsonl(feature_path, max_rows=max_rows):
        qa = int(row.get("qa_type_id", -1) or -1)
        if qa not in {11, 12, 13, 14}:
            continue
        key = (str(row.get("sample_id", "")), qa)
        if curr_key is None:
            curr_key = key
        if key != curr_key:
            flush_group()
            curr_rows = []
            curr_key = key
        curr_rows.append(row)
        rows += 1
        if rows % log_every == 0:
            LOGGER.info("object_retrieval rows=%d groups=%d preds=%d", rows, grouped, len(preds))
        if rows % memory_log_every == 0:
            rss = get_rss_mb()
            if rss is not None:
                LOGGER.info("object_retrieval rss_mb=%.1f", rss)
        if len(curr_rows) >= batch_size * 4:
            # safety valve for unexpected unsorted streams
            flush_group()
            curr_rows = []
            curr_key = None
    flush_group()

    summary = {
        "rows_processed": rows,
        "groups_processed": grouped,
        "selected_object_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "empty_prediction_counts": {str(k): int(v) for k, v in empty_counts.items()},
        "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "max_results": {str(k): v for k, v in max_results.items()},
    }
    return preds, summary


def run_motion(feature_path: Path, art_q57: dict[str, Any] | None, art_q9: dict[str, Any] | None, *, max_rows: int, log_every: int, memory_log_every: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], list[float]]]] = defaultdict(list)
    rows = 0
    failures = 0

    for row in iter_jsonl(feature_path, max_rows=max_rows):
        qa = int(row.get("qa_type_id", -1) or -1)
        if qa not in {15, 17, 19}:
            continue
        try:
            if qa == 19:
                if art_q9 is None:
                    continue
                pred = predict_with_motion_artifact([row], art_q9, config=None)[0]
            else:
                if art_q57 is None:
                    continue
                pred = predict_with_motion_artifact([row], art_q57, config=None)[0]
            grouped[(str(row.get("sample_id", "")), qa)].append((row, pred))
        except Exception:
            failures += 1
        rows += 1
        if rows % log_every == 0:
            LOGGER.info("motion rows=%d grouped_samples=%d", rows, len(grouped))
        if rows % memory_log_every == 0:
            rss = get_rss_mb()
            if rss is not None:
                LOGGER.info("motion rss_mb=%.1f", rss)

    preds: list[dict[str, Any]] = []
    per_qa = Counter()
    for (sid, qa), items in grouped.items():
        if not items:
            continue
        pred_vec = np.mean(np.asarray([x[1] for x in items], dtype=np.float32), axis=0).tolist()
        ref_row = max(items, key=lambda x: float(x[0].get("features", {}).get("confidence", 0.0)))[0]
        object_id = _safe_object_id(_obj_id(str(ref_row.get("candidate_id", "obj"))), fallback="obj0")
        start_x = float(ref_row.get("features", {}).get("x", 0.0))
        start_y = float(ref_row.get("features", {}).get("y", 0.0))
        text = format_motion_answer(qa, object_id, [float(v) for v in pred_vec], (start_x, start_y))
        obj_ids = [] if qa == 19 else [object_id]
        preds.append(
            base_prediction_record(
                sample_id=sid,
                qa_type_id=qa,
                task_type=QA_TASK.get(qa, "unknown"),
                answer_text=text,
                object_ids=obj_ids,
                extra={
                    "predicted_target": [float(v) for v in pred_vec],
                    "start_x": float(start_x),
                    "start_y": float(start_y),
                    "primary_object_id": object_id,
                },
            )
        )
        per_qa[qa] += 1

    summary = {
        "rows_processed": rows,
        "failed_rows": failures,
        "predictions_per_qa": {str(k): int(v) for k, v in per_qa.items()},
        "q9_leakage_check_status": bool(art_q9 is not None and art_q9.get("leakage_check_passed") is True),
    }
    return preds, summary


def patch_official_q5_q7_outputs(export_root: Path, scenario: str, raw_preds: list[dict[str, Any]]) -> dict[str, int]:
    by_key = {
        (str(r.get("sample_id", "")), int(r.get("qa_type_id", -1) or -1)): r
        for r in raw_preds
        if int(r.get("qa_type_id", -1) or -1) in {15, 17}
    }
    patched: dict[str, int] = {}
    for qa in (15, 17):
        path = export_root / f"object_motion_prediction_qa_type_{qa}_{scenario}_official.jsonl"
        if not path.exists():
            patched[str(qa)] = 0
            continue
        out_lines: list[str] = []
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kp = rec.get("kg_prediction", {}) if isinstance(rec.get("kg_prediction"), dict) else {}
            sample_id = str(kp.get("sample_id", rec.get("id", "")))
            raw = by_key.get((sample_id, qa))
            if raw is not None:
                target = raw.get("predicted_target", [])
                sx = float(raw.get("start_x", 0.0))
                sy = float(raw.get("start_y", 0.0))
                obj = _safe_object_id(str(raw.get("primary_object_id", "obj0")), fallback="obj0")
                rec["outputs"] = _strict_q5_output(obj, sx, sy, [float(v) for v in target] if isinstance(target, list) else [])
                count += 1
            out_lines.append(json.dumps(rec, ensure_ascii=True))
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        patched[str(qa)] = count
    return patched


def run_scene_action(feature_path: Path, art_q6: dict[str, Any] | None, art_q8s: dict[str, Any] | None, art_q8t: dict[str, Any] | None, *, max_rows: int, log_every: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = 0
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in iter_jsonl(feature_path, max_rows=max_rows):
        qa = int(row.get("qa_type_id", -1) or -1)
        if qa not in {16, 18}:
            continue
        grouped[(str(row.get("sample_id", "")), qa)] = row
        rows += 1
        if rows % log_every == 0:
            LOGGER.info("scene_action rows=%d grouped_samples=%d", rows, len(grouped))

    preds: list[dict[str, Any]] = []
    q6_dist = Counter()
    q8_speed_dist = Counter()
    q8_steer_dist = Counter()

    q8_speed_map = {int(v): str(v) for v in range(5)}
    q8_steer_map = {int(v): str(v) for v in range(5)}
    if art_q8s is not None and isinstance(art_q8s.get("label_map"), dict):
        q8_speed_map = {int(k): str(k) for k in art_q8s.get("label_map", {}).keys()}
    if art_q8t is not None and isinstance(art_q8t.get("label_map"), dict):
        q8_steer_map = {int(k): str(k) for k in art_q8t.get("label_map", {}).keys()}

    speed_labels = ["fast", "moderate", "slow", "very slow", "stop"]
    steer_labels = ["left", "slightly left", "straight", "slightly right", "right"]

    for (sid, qa), row in grouped.items():
        if qa == 16 and art_q6 is not None:
            out = predict_with_scene_action_artifact([row], art_q6, config=None)[0]
            threshold = float(art_q6.get("threshold", 0.5) or 0.5)
            text = format_q6_answer(float(out), threshold=threshold)
            label = 1 if float(out) >= threshold else 0
            q6_dist[label] += 1
            preds.append(
                base_prediction_record(
                    sample_id=sid,
                    qa_type_id=qa,
                    task_type=QA_TASK[qa],
                    answer_text=text,
                    object_ids=[],
                    extra={"score": float(out), "threshold": threshold},
                )
            )
        elif qa == 18 and art_q8s is not None and art_q8t is not None:
            s_pred = int(predict_with_scene_action_artifact([row], art_q8s, config=None)[0])
            t_pred = int(predict_with_scene_action_artifact([row], art_q8t, config=None)[0])
            s_label = speed_labels[s_pred] if 0 <= s_pred < len(speed_labels) else str(q8_speed_map.get(s_pred, s_pred))
            t_label = steer_labels[t_pred] if 0 <= t_pred < len(steer_labels) else str(q8_steer_map.get(t_pred, t_pred))
            q8_speed_dist[s_pred] += 1
            q8_steer_dist[t_pred] += 1
            preds.append(
                base_prediction_record(
                    sample_id=sid,
                    qa_type_id=qa,
                    task_type=QA_TASK[qa],
                    answer_text=format_q8_answer(s_label, t_label),
                    object_ids=[],
                    extra={
                        "speed_label_id": s_pred,
                        "steering_label_id": t_pred,
                        "speed_label": s_label,
                        "steering_label": t_label,
                    },
                )
            )

    summary = {
        "rows_processed": rows,
        "q6_distribution": {str(k): int(v) for k, v in q6_dist.items()},
        "q8_speed_distribution": {str(k): int(v) for k, v in q8_speed_dist.items()},
        "q8_steering_distribution": {str(k): int(v) for k, v in q8_steer_dist.items()},
        "q8_label_map_used": {"speed": q8_speed_map, "steering": q8_steer_map},
    }
    return preds, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_manifest_for_export(raw_path: Path, output_dir: Path, split: str, scenario: str) -> Path:
    task_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(raw_path):
        qa = int(row.get("qa_type_id", -1) or -1)
        task = str(row.get("task_type", "unknown"))
        task_rows[(task, qa)].append(row)

    runs = []
    for (task, qa), rows in sorted(task_rows.items(), key=lambda x: (x[0][0], x[0][1])):
        out = output_dir / f"{task}_qa_type_{qa}_{scenario}.jsonl"
        write_jsonl(out, rows)
        runs.append(
            {
                "task_type": task,
                "qa_type_id": qa,
                "output_jsonl": str(out),
                "baseline_mode": "cooperative",
                "supported_predictions": len(rows),
                "unsupported_predictions": 0,
                "total_samples": len(rows),
                "qa_type_ids": [qa],
            }
        )

    manifest = {
        "repository_root": str(REPO_ROOT),
        "split": split,
        "file_name": "v2v4real_3d_grounding_qa_dataset_v2vgot.json",
        "scenario_name": scenario,
        "task_types": sorted({k[0] for k in task_rows.keys()}),
        "runs": runs,
    }
    path = output_dir / f"{scenario}_manifest.json"
    atomic_json(path, manifest)
    return path


def compute_diagnostic_metrics(preds: list[dict[str, Any]], labeled_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"diagnostic_only": True}
    by_key = {(str(p.get("sample_id", "")), int(p.get("qa_type_id", -1) or -1)): p for p in preds}

    q6_true: list[int] = []
    q6_pred: list[int] = []
    q8_s_true: list[int] = []
    q8_s_pred: list[int] = []
    q8_t_true: list[int] = []
    q8_t_pred: list[int] = []
    reg_errs: dict[int, list[float]] = {15: [], 17: [], 19: []}

    scene_path = labeled_dir / "scene_action_labeled.jsonl"
    for row in iter_jsonl(scene_path):
        key = (str(row.get("sample_id", "")), int(row.get("qa_type_id", -1) or -1))
        pred = by_key.get(key)
        if pred is None:
            continue
        qa = key[1]
        label = row.get("label", {})
        if qa == 16 and isinstance(label, dict) and label.get("label") is not None:
            q6_true.append(int(label.get("label", 0)))
            q6_pred.append(1 if "not notable" not in str(pred.get("answer_text", "")).lower() else 0)
        if qa == 18 and isinstance(label, dict):
            s = label.get("speed_label_id", label.get("speed_class"))
            t = label.get("steering_label_id", label.get("steering_class"))
            ps = pred.get("speed_label_id")
            pt = pred.get("steering_label_id")
            if isinstance(s, int) and isinstance(ps, int):
                q8_s_true.append(s)
                q8_s_pred.append(ps)
            if isinstance(t, int) and isinstance(pt, int):
                q8_t_true.append(t)
                q8_t_pred.append(pt)

    motion_path = labeled_dir / "motion_regression_labeled.jsonl"
    for row in iter_jsonl(motion_path):
        key = (str(row.get("sample_id", "")), int(row.get("qa_type_id", -1) or -1))
        pred = by_key.get(key)
        if pred is None:
            continue
        qa = key[1]
        gt = row.get("label", {}).get("target", []) if isinstance(row.get("label"), dict) else []
        pr = pred.get("predicted_target", [])
        if isinstance(gt, list) and isinstance(pr, list) and gt and len(gt) == len(pr):
            diff = np.asarray(pr, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
            reg_errs[qa].append(float(np.mean(np.abs(diff))))

    if q6_true:
        q6_true_arr = np.asarray(q6_true, dtype=np.int32)
        q6_pred_arr = np.asarray(q6_pred, dtype=np.int32)
        result["q6"] = {"accuracy": float(np.mean(q6_true_arr == q6_pred_arr)), "n": int(len(q6_true_arr))}
    if q8_s_true:
        a = np.asarray(q8_s_true, dtype=np.int32) == np.asarray(q8_s_pred, dtype=np.int32)
        result["q8_speed"] = {"accuracy": float(np.mean(a)), "n": int(len(a))}
    if q8_t_true:
        a = np.asarray(q8_t_true, dtype=np.int32) == np.asarray(q8_t_pred, dtype=np.int32)
        result["q8_steering"] = {"accuracy": float(np.mean(a)), "n": int(len(a))}
    for qa in (15, 17, 19):
        if reg_errs[qa]:
            result[f"q{qa}_motion"] = {"mae": float(np.mean(reg_errs[qa])), "n": int(len(reg_errs[qa]))}
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    batch_size = max(1, int(args.batch_size or runtime_cfg.get("batch_size", 50000)))
    log_every = max(1, int(args.log_every or runtime_cfg.get("log_every", 100000)))
    memory_log_every = max(1, int(args.memory_log_every or runtime_cfg.get("memory_log_every", 500000)))

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if "outputs/unified_heads" not in str(out_dir):
        raise SystemExit("output_dir must be under outputs/unified_heads")

    feature_dir = Path(args.feature_dir).expanduser().resolve() if args.feature_dir else None
    if not args.reuse_feature_rows:
        raise SystemExit("Stage 4 currently requires --reuse-feature-rows for memory-safe inference")
    if feature_dir is None:
        raise SystemExit("--feature-dir is required with --reuse-feature-rows")

    LOGGER.info(
        "config=%s split=%s artifact_dir=%s output_dir=%s feature_dir=%s labeled_dir=%s workers=%d batch_size=%d execution_mode=sequential reuse_feature_rows=%s skip_export=%s skip_official_eval=%s",
        config_path,
        args.split,
        args.artifact_dir,
        out_dir,
        feature_dir,
        args.labeled_dir,
        args.num_workers,
        batch_size,
        args.reuse_feature_rows,
        args.skip_export,
        args.skip_official_eval,
    )

    artifacts, load_summary = load_all_unified_artifacts(args.artifact_dir)
    atomic_json(out_dir / "artifact_load_summary.json", load_summary)
    missing_artifacts = set(load_summary.get("missing_keys", []))

    preds_all: list[dict[str, Any]] = []
    family_runtime: dict[str, float] = {}
    family_summaries: dict[str, Any] = {}

    t0 = time.time()
    obj_preds, obj_summary = run_object_retrieval(
        feature_dir / "object_retrieval_rows.jsonl",
        artifacts.get("object_retrieval", {}),
        batch_size=batch_size,
        max_rows=args.max_rows,
        log_every=log_every,
        memory_log_every=memory_log_every,
    ) if "object_retrieval" in artifacts else ([], {"skipped": "missing artifact"})
    family_runtime["object_retrieval"] = time.time() - t0
    family_summaries["object_retrieval"] = obj_summary
    preds_all.extend(obj_preds)

    t0 = time.time()
    motion_preds, motion_summary = run_motion(
        feature_dir / "motion_regression_rows.jsonl",
        artifacts.get("motion_q57"),
        artifacts.get("motion_q9"),
        max_rows=args.max_rows,
        log_every=log_every,
        memory_log_every=memory_log_every,
    )
    family_runtime["motion_regression"] = time.time() - t0
    family_summaries["motion_regression"] = motion_summary
    preds_all.extend(motion_preds)

    t0 = time.time()
    scene_preds, scene_summary = run_scene_action(
        feature_dir / "scene_action_rows.jsonl",
        artifacts.get("q6"),
        artifacts.get("q8_speed"),
        artifacts.get("q8_steering"),
        max_rows=args.max_rows,
        log_every=log_every,
    )
    family_runtime["scene_action"] = time.time() - t0
    family_summaries["scene_action"] = scene_summary
    preds_all.extend(scene_preds)

    preds_all.sort(key=lambda x: (int(x.get("sample_id", 0)) if str(x.get("sample_id", "")).isdigit() else str(x.get("sample_id", "")), int(x.get("qa_type_id", -1) or -1)))
    raw_path = out_dir / "raw_predictions.jsonl"
    write_jsonl(raw_path, preds_all)

    by_qa = Counter(int(p.get("qa_type_id", -1) or -1) for p in preds_all)
    summary = {
        "split": args.split,
        "num_samples": len({str(p.get("sample_id", "")) for p in preds_all}),
        "num_predictions_per_qa_type": {str(k): int(v) for k, v in sorted(by_qa.items())},
        "num_rows_read_per_family": {
            "object_retrieval": int(family_summaries.get("object_retrieval", {}).get("rows_processed", 0)),
            "motion_regression": int(family_summaries.get("motion_regression", {}).get("rows_processed", 0)),
            "scene_action": int(family_summaries.get("scene_action", {}).get("rows_processed", 0)),
        },
        "missing_artifact_qa_types": sorted(list(missing_artifacts)),
        "failed_rows": int(family_summaries.get("motion_regression", {}).get("failed_rows", 0)),
        "output_paths": {
            "raw_predictions": str(raw_path),
            "artifact_load_summary": str(out_dir / "artifact_load_summary.json"),
        },
        "runtime_sec": round(time.time() - started, 3),
        "per_family_runtime_sec": {k: round(v, 3) for k, v in family_runtime.items()},
        "family_details": family_summaries,
        "execution_mode": "sequential",
    }

    if args.labeled_dir:
        labeled_dir = Path(args.labeled_dir).expanduser().resolve()
        if labeled_dir.exists():
            diag = compute_diagnostic_metrics(preds_all, labeled_dir)
            atomic_json(out_dir / "inference_metrics_diagnostic.json", diag)
            summary["output_paths"]["diagnostic_metrics"] = str(out_dir / "inference_metrics_diagnostic.json")

    export_ok = False
    official_ok = False
    export_error = ""
    official_error = ""
    official_warnings: list[str] = []

    if not args.skip_export:
        try:
            export_root = out_dir / "official_exports"
            export_root.mkdir(parents=True, exist_ok=True)
            manifest_path = build_manifest_for_export(raw_path, out_dir, args.split, f"unified_heads_{args.split}")
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "export_qa_predictions.py"),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(export_root),
                "--split",
                args.split,
            ]
            LOGGER.info("running_export cmd=%s", " ".join(cmd))
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
            export_manifest = export_root / f"unified_heads_{args.split}_official_export_manifest.json"
            q5q7_patch = patch_official_q5_q7_outputs(export_root, f"unified_heads_{args.split}", preds_all)
            LOGGER.info("patched_official_q5_q7_outputs=%s", q5q7_patch)
            summary["output_paths"]["official_export_manifest"] = str(export_manifest)
            export_ok = True

            if not args.skip_official_eval:
                eval_out = out_dir / "official_eval"
                eval_out.mkdir(parents=True, exist_ok=True)
                v2vgot_root = Path(os.environ.get("V2VGOT_ROOT", "/workspace/repos/V2V-GoT")).expanduser().resolve()
                npy_path = v2vgot_root / "DMSTrack" / "V2V4Real" / "official_models" / "no_fusion_keep_all" / "npy"
                ecmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "evaluate_official_qa.py"),
                    "--export-manifest",
                    str(export_manifest),
                    "--output-dir",
                    str(eval_out),
                    "--num-future-waypoints",
                    "6",
                ]
                if npy_path.exists():
                    ecmd.extend(["--npy-save-path", str(npy_path)])
                LOGGER.info("running_official_eval cmd=%s", " ".join(ecmd))
                subprocess.run(ecmd, cwd=str(REPO_ROOT), check=True)
                # pick latest summary json if present
                summaries = sorted(eval_out.glob("*_official_qa_eval_summary.json"))
                if summaries:
                    summary["output_paths"]["official_metrics"] = str(summaries[-1])
                official_ok = True
        except Exception as exc:  # noqa: BLE001
            if not export_ok:
                export_error = str(exc)
            else:
                official_error = str(exc)
                if "x1_to_x2" in official_error or "NoneType' object is not callable" in official_error:
                    official_warnings.append(
                        "Q9 official evaluator tool runtime appears broken in this environment "
                        "(x1_to_x2 unresolved). Raw predictions/export are valid; evaluate Q9 with a fixed evaluator runtime."
                    )
            if args.fail_fast:
                raise

    summary["official_export_succeeded"] = export_ok
    summary["official_eval_succeeded"] = official_ok
    if export_error:
        summary["official_export_error"] = export_error
    if official_error:
        summary["official_eval_error"] = official_error
    if official_warnings:
        summary["official_eval_warnings"] = official_warnings

    atomic_json(out_dir / "prediction_summary.json", summary)
    LOGGER.info("saved_summary=%s", out_dir / "prediction_summary.json")


if __name__ == "__main__":
    main()
