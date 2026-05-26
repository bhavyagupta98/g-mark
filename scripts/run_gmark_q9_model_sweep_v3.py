#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import run_gmark_q9_model_sweep as legacy
from kg_coop_drive.application.planning.control_settings_policy import (
    build_control_feature_vector,
    control_feature_names,
    rank_control_candidates,
    visibility_lookup,
)
from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator
from kg_coop_drive.domain.scene import VisibilityState

try:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
    from sklearn.neural_network import MLPRegressor  # type: ignore
    from sklearn.multioutput import MultiOutputRegressor  # type: ignore

    SKLEARN_EXTRA_AVAILABLE = True
except Exception:
    SKLEARN_EXTRA_AVAILABLE = False

try:
    from xgboost import XGBRegressor  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE1_Q9_FILE_NAME = legacy.TABLE1_Q9_FILE_NAME
DEFAULT_TRAIN_FILE_NAME = legacy.DEFAULT_TRAIN_FILE_NAME


@dataclass(frozen=True)
class SweepModelResult:
    model_name: str
    model_json: Path
    prediction_jsonl: Path
    prediction_manifest_json: Path
    train_rows: int
    val_rows: int
    train_l2_avg: float
    val_l2_avg: float
    val_l2_1s: float
    val_l2_2s: float
    val_l2_3s: float
    official_export_manifest_json: str | None
    official_summary_json: str | None


class Q8KGFeatureTimeoutError(TimeoutError):
    pass


def _raise_q8_kg_feature_timeout(signum: int, frame: object) -> None:
    raise Q8KGFeatureTimeoutError("Q8 KG feature row timed out")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Q9 sweep v3: keeps v2 behavior and adds optional fixed-width top-k static "
            "interaction features derived from cooperative scene at timestamp T."
        )
    )
    parser.add_argument("--run-name", default=f"gmark_q9_sweep_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--output-root", default="outputs/v2vgot_table1_reproduction/gmark_q9_sweep")
    parser.add_argument("--train-file-name", default=DEFAULT_TRAIN_FILE_NAME)
    parser.add_argument("--val-file-name", default=TABLE1_Q9_FILE_NAME)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--allow-train-val-overlap", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--models", nargs="+", default=("elasticnet",))
    parser.add_argument(
        "--feature-profile",
        choices=("full", "reduced_from_importance"),
        default="full",
        help=(
            "Feature-profile selector. full keeps all assembled features. "
            "reduced_from_importance keeps a compact subset from permutation diagnostics."
        ),
    )
    parser.add_argument("--include-q8-context-label-features", action="store_true")
    parser.add_argument(
        "--q8-context-feature-mode",
        choices=("label_onehot12", "model_probs14"),
        default="label_onehot12",
        help=(
            "label_onehot12 keeps the legacy 12-column Q8 block. "
            "model_probs14 uses Q8 JSONL probabilities and confidence margins."
        ),
    )
    parser.add_argument(
        "--q8-context-value-source",
        choices=("mapped_from_label", "float_jsonl"),
        default="mapped_from_label",
        help=(
            "How to populate the last 2 Q8 context columns. mapped_from_label uses "
            "fixed class-to-value mapping. float_jsonl loads per-sample Q8 float values."
        ),
    )
    parser.add_argument("--q8-float-train-jsonl", default="")
    parser.add_argument("--q8-float-val-jsonl", default="")
    parser.add_argument(
        "--q8-context-debug-samples",
        type=int,
        default=0,
        help="If >0, print this many Q8 context feature rows per split for inspection.",
    )
    parser.add_argument("--include-q8-kg-control-features", action="store_true")
    parser.add_argument("--q8-kg-feature-set", default="extended_v1", choices=("base", "extended_v1"))
    parser.add_argument(
        "--q8-kg-feature-subset",
        default="all",
        help=(
            "Comma-separated subset of q8_kg feature names (from control_feature_names "
            "for the selected feature set), or 'all'."
        ),
    )
    parser.add_argument("--q8-kg-feature-timeout-seconds", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-official-eval", action="store_true")
    parser.add_argument("--include-q8-topk-static-features", action="store_true")
    parser.add_argument("--q8-topk-k", type=int, default=3)
    parser.add_argument(
        "--q8-topk-debug-samples",
        type=int,
        default=0,
        help="If >0, print this many top-k feature rows per split for auditing.",
    )
    return parser


def resolve_output_root(raw_output_root: str) -> Path:
    output_root = Path(raw_output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def build_q8_kg_control_feature_lookup(
    *,
    samples: tuple[legacy.BenchmarkSample, ...],
    evaluator: V2VGoTQAPhase5AEvaluator,
    feature_set: str,
    timeout_seconds: int,
    progress_every: int,
    output_jsonl: Path,
) -> dict[str, list[float]]:
    lookup: dict[str, list[float]] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    default_features = [0.0] * len(control_feature_names(feature_set))
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples, start=1):
            started = time.monotonic()
            timed_out = False
            if timeout_seconds > 0:
                previous_handler = signal.signal(signal.SIGALRM, _raise_q8_kg_feature_timeout)
                signal.alarm(timeout_seconds)
            else:
                previous_handler = None
            try:
                prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode="cooperative")
                features = list(build_control_feature_vector(prepared_scene, feature_set=feature_set))
                status = "ok"
            except Q8KGFeatureTimeoutError:
                timed_out = True
                features = list(default_features)
                status = f"timeout_after_{timeout_seconds}s"
            finally:
                if timeout_seconds > 0:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_handler)
            lookup[sample.sample_id] = features
            elapsed = time.monotonic() - started
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "split_name": sample.split_name,
                        "file_name": sample.file_name,
                        "scenario_index": sample.raw_record.get("scenario_index"),
                        "global_timestamp_index": sample.raw_record.get("global_timestamp_index"),
                        "local_timestamp_index": sample.raw_record.get("local_timestamp_index"),
                        "asker_cav_id": sample.raw_record.get("asker_cav_id"),
                        "q8_kg_feature_set": feature_set,
                        "q8_kg_feature_elapsed_seconds": round(elapsed, 6),
                        "q8_kg_feature_status": status,
                        "q8_kg_feature_vector": features,
                    }
                )
                + "\n"
            )
            handle.flush()
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(samples)):
                print(
                    f"q8_kg_feature_progress: {sample.split_name} {idx}/{len(samples)} "
                    f"last_sample_id={sample.sample_id} elapsed={elapsed:.3f}s timed_out={timed_out}",
                    flush=True,
                )
    return lookup


def feature_names(
    *,
    include_q8_context_label_features: bool,
    include_q8_kg_control_features: bool,
    q8_kg_selected_names: tuple[str, ...],
) -> list[str]:
    names = legacy.feature_names(include_q8_pred_features=False)
    if include_q8_context_label_features:
        # Default names assume legacy onehot+2 mode; caller may override by profile mode.
        names.extend(
            [f"q8_speed_{label.replace(' ', '_')}" for label in legacy.SPEED_LABELS]
            + [f"q8_steer_{label.replace(' ', '_')}" for label in legacy.STEER_LABELS]
            + [
                "q8_pred_speed_control_value",
                "q8_pred_steering_control_value",
            ]
        )
    if include_q8_kg_control_features:
        names.extend([f"q8kg_{name}" for name in q8_kg_selected_names])
    return names


def q8_topk_static_feature_names(topk_k: int) -> list[str]:
    names: list[str] = []
    for idx in range(1, topk_k + 1):
        names.extend(
            [
                f"q8topk_{idx}_valid",
                f"q8topk_{idx}_risk",
                f"q8topk_{idx}_dist",
                f"q8topk_{idx}_long_gap",
                f"q8topk_{idx}_lat_gap",
                f"q8topk_{idx}_bearing_sin",
                f"q8topk_{idx}_bearing_cos",
                f"q8topk_{idx}_conflict",
                f"q8topk_{idx}_uncertainty",
                f"q8topk_{idx}_support_count",
                f"q8topk_{idx}_visible",
                f"q8topk_{idx}_uncertain",
                f"q8topk_{idx}_occluded",
            ]
        )
    names.extend(
        [
            "q8topk_num_objects_total",
            "q8topk_num_within_5m",
            "q8topk_num_within_10m",
            "q8topk_num_occluded_within_10m",
            "q8topk_num_uncertain_within_10m",
            "q8topk_min_dist_any",
            "q8topk_mean_dist_any",
        ]
    )
    return names


def build_q8_topk_static_feature_lookup(
    *,
    samples: tuple[legacy.BenchmarkSample, ...],
    evaluator: V2VGoTQAPhase5AEvaluator,
    topk_k: int,
    progress_every: int,
    output_jsonl: Path,
    debug_samples: int,
) -> dict[str, list[float]]:
    lookup: dict[str, list[float]] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    expected_names = q8_topk_static_feature_names(topk_k)
    row_width = len(expected_names)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        debug_left = debug_samples
        for idx, sample in enumerate(samples, start=1):
            prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode="cooperative")
            asker = next((agent for agent in prepared_scene.agents if agent.agent_id == prepared_scene.asker_agent_id), None)
            visibility_by_object = visibility_lookup(prepared_scene, prepared_scene.asker_agent_id)
            ranked = list(rank_control_candidates(prepared_scene))
            features: list[float] = []
            if asker is None:
                features = [0.0] * row_width
            else:
                yaw = float(asker.pose.yaw_radians)
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                asker_x = float(asker.pose.position.x)
                asker_y = float(asker.pose.position.y)
                all_dists: list[float] = []
                within_5 = 0
                within_10 = 0
                occluded_10 = 0
                uncertain_10 = 0
                for object_track in prepared_scene.object_tracks:
                    dx = float(object_track.position.x) - asker_x
                    dy = float(object_track.position.y) - asker_y
                    dist = math.hypot(dx, dy)
                    all_dists.append(dist)
                    if dist <= 5.0:
                        within_5 += 1
                    if dist <= 10.0:
                        within_10 += 1
                        state = visibility_by_object.get(object_track.object_id)
                        if state == VisibilityState.OCCLUDED:
                            occluded_10 += 1
                        elif state == VisibilityState.UNCERTAIN:
                            uncertain_10 += 1

                for slot in range(topk_k):
                    if slot < len(ranked):
                        object_track, risk = ranked[slot]
                        dx = float(object_track.position.x) - asker_x
                        dy = float(object_track.position.y) - asker_y
                        dist = math.hypot(dx, dy)
                        long_gap = cos_yaw * dx + sin_yaw * dy
                        lat_gap = -sin_yaw * dx + cos_yaw * dy
                        bearing = math.atan2(dy, dx) - yaw
                        state = visibility_by_object.get(object_track.object_id)
                        support_count = float(len(object_track.provenance.source_agent_ids))
                        features.extend(
                            [
                                1.0,
                                float(risk),
                                float(dist),
                                float(long_gap),
                                float(lat_gap),
                                float(math.sin(bearing)),
                                float(math.cos(bearing)),
                                float(object_track.conflict_score),
                                float(object_track.uncertainty_score),
                                support_count,
                                1.0 if state == VisibilityState.VISIBLE else 0.0,
                                1.0 if state == VisibilityState.UNCERTAIN else 0.0,
                                1.0 if state == VisibilityState.OCCLUDED else 0.0,
                            ]
                        )
                    else:
                        features.extend([0.0] * 13)

                min_dist = min(all_dists) if all_dists else 0.0
                mean_dist = float(sum(all_dists) / len(all_dists)) if all_dists else 0.0
                features.extend(
                    [
                        float(len(prepared_scene.object_tracks)),
                        float(within_5),
                        float(within_10),
                        float(occluded_10),
                        float(uncertain_10),
                        float(min_dist),
                        float(mean_dist),
                    ]
                )

            if len(features) != row_width:
                raise ValueError(
                    f"Top-k static feature width mismatch for sample {sample.sample_id}: "
                    f"got={len(features)} expected={row_width}"
                )
            if debug_left > 0:
                preview_pairs = list(zip(expected_names, features))[:50]
                preview_text = ", ".join(f"{name}={value:.6f}" for name, value in preview_pairs)
                print(
                    "[DEBUG] q8_topk_row: "
                    f"split={sample.split_name} sample_id={sample.sample_id} "
                    f"preview50=[{preview_text}]",
                    flush=True,
                )
                debug_left -= 1
            lookup[sample.sample_id] = features
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "split_name": sample.split_name,
                        "file_name": sample.file_name,
                        "scenario_index": sample.raw_record.get("scenario_index"),
                        "global_timestamp_index": sample.raw_record.get("global_timestamp_index"),
                        "local_timestamp_index": sample.raw_record.get("local_timestamp_index"),
                        "asker_cav_id": sample.raw_record.get("asker_cav_id"),
                        "q8_topk_k": topk_k,
                        "q8_topk_static_feature_vector": features,
                    }
                )
                + "\n"
            )
            handle.flush()
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(samples)):
                print(
                    f"q8_topk_feature_progress: {sample.split_name} {idx}/{len(samples)} "
                    f"last_sample_id={sample.sample_id}",
                    flush=True,
                )
    return lookup


def resolve_feature_profile_indices(
    all_feature_names: list[str],
    feature_profile: str,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if feature_profile == "full":
        names = tuple(all_feature_names)
        return tuple(range(len(names))), names
    if feature_profile != "reduced_from_importance":
        raise ValueError(f"Unsupported feature_profile: {feature_profile}")

    keep_set = {
        "current_x",
        "current_y",
        "asker_is_cav1",
        "q8_pred_speed_control_value",
        "q8_pred_steering_control_value",
        "q8_speed_fast",
        "q8_speed_moderate",
        "q8_speed_slow",
        "q8_speed_very_slow",
        "q8_speed_stop",
        "q8_steer_straight",
        "q8_steer_slightly_left",
        "q8_steer_slightly_right",
        "q8_steer_right",
    }
    selected_names = tuple(name for name in all_feature_names if name in keep_set)
    if not selected_names:
        raise ValueError(
            "Feature profile reduced_from_importance selected zero features. "
            "Ensure the run includes compatible context features."
        )
    selected_indices = tuple(idx for idx, name in enumerate(all_feature_names) if name in keep_set)
    return selected_indices, selected_names


def resolve_q8_kg_selected_indices(feature_set: str, raw_subset: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    full_names = tuple(control_feature_names(feature_set))
    raw = raw_subset.strip().lower()
    if raw in {"", "all"}:
        return tuple(range(len(full_names))), full_names
    requested = tuple(name.strip() for name in raw_subset.split(",") if name.strip())
    unknown = [name for name in requested if name not in full_names]
    if unknown:
        raise ValueError(
            "Unknown q8 kg feature names in --q8-kg-feature-subset: "
            f"{unknown}. Allowed names: {list(full_names)}"
        )
    return tuple(full_names.index(name) for name in requested), requested


def _parse_float_value(row: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        return parsed
    return None


def load_q8_float_lookup(path: str) -> dict[str, tuple[float, float]]:
    jsonl_path = Path(path).expanduser()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Q8 float JSONL not found: {jsonl_path}")
    lookup: dict[str, tuple[float, float]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            speed_value = _parse_float_value(
                row,
                (
                    "q8_pred_speed_control_value_float",
                    "q8_pred_speed_control_value",
                    "speed_control_value",
                ),
            )
            steering_value = _parse_float_value(
                row,
                (
                    "q8_pred_steering_control_value_float",
                    "q8_pred_steering_control_value",
                    "steering_control_value",
                ),
            )
            if speed_value is None or steering_value is None:
                continue
            lookup[sample_id] = (speed_value, steering_value)
    return lookup


def _parse_prob_list(row: dict[str, object], keys: tuple[str, ...]) -> list[float] | None:
    for key in keys:
        raw = row.get(key)
        if not isinstance(raw, list):
            continue
        try:
            values = [float(item) for item in raw]
        except (TypeError, ValueError):
            continue
        if len(values) != 5:
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        return values
    return None


def load_q8_rich_lookup(path: str) -> dict[str, dict[str, object]]:
    jsonl_path = Path(path).expanduser()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Q8 rich JSONL not found: {jsonl_path}")
    lookup: dict[str, dict[str, object]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            speed_probs = _parse_prob_list(row, ("q8_pred_speed_probs", "speed_probs"))
            steering_probs = _parse_prob_list(row, ("q8_pred_steering_probs", "steering_probs"))
            if speed_probs is None or steering_probs is None:
                continue
            lookup[sample_id] = {
                "speed_probs": speed_probs,
                "steering_probs": steering_probs,
            }
    return lookup


def build_q8_context_feature_lookup(
    *,
    samples: tuple[legacy.BenchmarkSample, ...],
    output_jsonl: Path,
    progress_every: int,
    context_feature_mode: str,
    value_source: str,
    float_lookup: dict[str, tuple[float, float]],
    rich_lookup: dict[str, dict[str, object]],
    debug_samples: int,
) -> dict[str, list[float]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lookup: dict[str, list[float]] = {}
    total = len(samples)
    debug_left = debug_samples
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples, start=1):
            started_at = time.monotonic()
            speed_label, steering_label = legacy.parse_q8_labels(sample.scene.raw_question)
            if context_feature_mode == "model_probs14" and sample.sample_id in rich_lookup:
                speed_probs = list(rich_lookup[sample.sample_id]["speed_probs"])  # type: ignore[index]
                steering_probs = list(rich_lookup[sample.sample_id]["steering_probs"])  # type: ignore[index]
                sorted_speed = sorted(speed_probs, reverse=True)
                sorted_steer = sorted(steering_probs, reverse=True)
                speed_top1 = sorted_speed[0] if sorted_speed else 0.0
                steer_top1 = sorted_steer[0] if sorted_steer else 0.0
                speed_margin = (sorted_speed[0] - sorted_speed[1]) if len(sorted_speed) > 1 else 0.0
                steer_margin = (sorted_steer[0] - sorted_steer[1]) if len(sorted_steer) > 1 else 0.0
                features = speed_probs + steering_probs + [speed_top1, steer_top1, speed_margin, steer_margin]
                speed_value = speed_top1
                steering_value = steer_top1
                value_status = "from_float_jsonl_probs"
            else:
                speed_one_hot = [1.0 if label == speed_label else 0.0 for label in legacy.SPEED_LABELS]
                steer_one_hot = [1.0 if label == steering_label else 0.0 for label in legacy.STEER_LABELS]
                if value_source == "float_jsonl" and sample.sample_id in float_lookup:
                    speed_value, steering_value = float_lookup[sample.sample_id]
                    value_status = "from_float_jsonl"
                else:
                    speed_value = legacy.Q8_SPEED_CONTROL_VALUES.get(speed_label, 0.0)
                    steering_value = legacy.Q8_STEERING_CONTROL_VALUES.get(steering_label, 0.0)
                    value_status = "mapped_from_label"
                features = speed_one_hot + steer_one_hot + [float(speed_value), float(steering_value)]
            lookup[sample.sample_id] = features
            elapsed = time.monotonic() - started_at
            row = {
                "sample_id": sample.sample_id,
                "split_name": sample.split_name,
                "file_name": sample.file_name,
                "scenario_index": sample.raw_record.get("scenario_index"),
                "global_timestamp_index": sample.raw_record.get("global_timestamp_index"),
                "local_timestamp_index": sample.raw_record.get("local_timestamp_index"),
                "asker_cav_id": sample.raw_record.get("asker_cav_id"),
                "q8_pred_speed_label": speed_label,
                "q8_pred_steering_label": steering_label,
                "q8_context_value_source": value_status,
                "q8_pred_speed_control_value": float(speed_value),
                "q8_pred_steering_control_value": float(steering_value),
                "q8_feature_vector": features,
                "q8_feature_elapsed_seconds": round(elapsed, 6),
                "q8_context_feature_mode": context_feature_mode,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if debug_left > 0:
                print(
                    "[DEBUG] q8_context_row: "
                    f"split={sample.split_name} sample_id={sample.sample_id} "
                    f"speed={speed_label} steer={steering_label} "
                    f"values=({speed_value:.6f},{steering_value:.6f}) source={value_status}",
                    flush=True,
                )
                debug_left -= 1
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
                print(
                    f"q8_context_feature_progress: {sample.split_name} {idx}/{total} "
                    f"last_sample_id={sample.sample_id}",
                    flush=True,
                )
    return lookup


def build_feature_row(
    sample: legacy.BenchmarkSample,
    *,
    include_q8_context_label_features: bool,
    include_q8_kg_control_features: bool,
    q8_context_lookup: dict[str, list[float]],
    q8_kg_lookup: dict[str, list[float]],
    q8_kg_feature_set: str,
    q8_kg_selected_indices: tuple[int, ...],
    include_q8_topk_static_features: bool,
    q8_topk_static_lookup: dict[str, list[float]],
    profile_selected_indices: tuple[int, ...],
) -> list[float]:
    all_features = legacy.build_feature_row(
        sample,
        include_q8_pred_features=False,
        q8_lookup={},
    )
    if include_q8_context_label_features:
        all_features.extend(q8_context_lookup.get(sample.sample_id, [0.0] * legacy.q8_feature_width()))
    if include_q8_kg_control_features:
        full = q8_kg_lookup.get(sample.sample_id, [0.0] * len(control_feature_names(q8_kg_feature_set)))
        all_features.extend([full[index] for index in q8_kg_selected_indices])
    if include_q8_topk_static_features:
        all_features.extend(q8_topk_static_lookup.get(sample.sample_id, []))
    return [all_features[index] for index in profile_selected_indices]


def build_xy(
    samples: tuple[legacy.BenchmarkSample, ...],
    *,
    include_q8_context_label_features: bool,
    include_q8_kg_control_features: bool,
    q8_context_lookup: dict[str, list[float]],
    q8_kg_lookup: dict[str, list[float]],
    q8_kg_feature_set: str,
    q8_kg_selected_indices: tuple[int, ...],
    include_q8_topk_static_features: bool,
    q8_topk_static_lookup: dict[str, list[float]],
    profile_selected_indices: tuple[int, ...],
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    usable = 0
    for idx, sample in enumerate(samples, start=1):
        gt_waypoints = legacy.parse_waypoints(legacy.raw_answer(sample), limit=6)
        if len(gt_waypoints) != 6:
            continue
        x_rows.append(
            build_feature_row(
                sample,
                include_q8_context_label_features=include_q8_context_label_features,
                include_q8_kg_control_features=include_q8_kg_control_features,
                q8_context_lookup=q8_context_lookup,
                q8_kg_lookup=q8_kg_lookup,
                q8_kg_feature_set=q8_kg_feature_set,
                q8_kg_selected_indices=q8_kg_selected_indices,
                include_q8_topk_static_features=include_q8_topk_static_features,
                q8_topk_static_lookup=q8_topk_static_lookup,
                profile_selected_indices=profile_selected_indices,
            )
        )
        y_rows.append([value for point in gt_waypoints for value in point])
        usable += 1
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(samples)):
            print(f"feature_progress: {idx}/{len(samples)} usable={usable}", flush=True)
    if not x_rows:
        raise ValueError("No usable Q9 rows after feature extraction.")
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), usable


def train_predict_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    learning_rate: float = 0.05,
    max_iter: int = 400,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not SKLEARN_EXTRA_AVAILABLE:
        raise RuntimeError("scikit-learn HistGradientBoosting is not available; hgb cannot run.")
    print(
        "[hgb] start fit: "
        f"train_shape={x_train.shape} target_shape={y_train.shape} val_shape={x_val.shape} "
        f"lr={learning_rate} max_iter={max_iter} max_leaf_nodes={max_leaf_nodes} "
        f"min_samples_leaf={min_samples_leaf} l2={l2_regularization}",
        flush=True,
    )
    base = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=42,
    )
    reg = MultiOutputRegressor(base)
    print("[hgb] fitting MultiOutputRegressor...", flush=True)
    reg.fit(x_train, y_train)
    print("[hgb] fit complete; predicting train/val...", flush=True)
    y_train_pred = reg.predict(x_train)
    y_val_pred = reg.predict(x_val)
    model = {
        "family": "hgb",
        "learning_rate": float(learning_rate),
        "max_iter": int(max_iter),
        "max_leaf_nodes": int(max_leaf_nodes),
        "min_samples_leaf": int(min_samples_leaf),
        "l2_regularization": float(l2_regularization),
    }
    print("[hgb] prediction complete", flush=True)
    return y_val_pred, y_train_pred, model


def train_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, int] = (128, 64),
    alpha: float = 1e-4,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not SKLEARN_EXTRA_AVAILABLE:
        raise RuntimeError("scikit-learn MLPRegressor is not available; mlp cannot run.")
    print(
        "[mlp] start fit: "
        f"train_shape={x_train.shape} target_shape={y_train.shape} val_shape={x_val.shape} "
        f"hidden={hidden_layer_sizes} alpha={alpha} max_iter={max_iter}",
        flush=True,
    )
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    mean[0] = 0.0
    scale[0] = 1.0
    x_train_norm = (x_train - mean) / scale
    x_val_norm = (x_val - mean) / scale
    base = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate="adaptive",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        max_iter=max_iter,
        random_state=42,
    )
    reg = MultiOutputRegressor(base)
    print("[mlp] fitting MultiOutputRegressor...", flush=True)
    reg.fit(x_train_norm, y_train)
    print("[mlp] fit complete; predicting train/val...", flush=True)
    y_train_pred = reg.predict(x_train_norm)
    y_val_pred = reg.predict(x_val_norm)
    model = {
        "family": "mlp",
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "alpha": float(alpha),
        "max_iter": int(max_iter),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }
    print("[mlp] prediction complete", flush=True)
    return y_val_pred, y_train_pred, model


def train_predict_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    n_estimators: int = 600,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not SKLEARN_EXTRA_AVAILABLE:
        raise RuntimeError("scikit-learn is not available; xgb cannot run with MultiOutputRegressor.")
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("xgboost is not installed; xgb cannot run.")
    print(
        "[xgb] start fit: "
        f"train_shape={x_train.shape} target_shape={y_train.shape} val_shape={x_val.shape} "
        f"n_estimators={n_estimators} max_depth={max_depth} lr={learning_rate}",
        flush=True,
    )
    base = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    reg = MultiOutputRegressor(base)
    print("[xgb] fitting MultiOutputRegressor...", flush=True)
    reg.fit(x_train, y_train)
    print("[xgb] fit complete; predicting train/val...", flush=True)
    y_train_pred = reg.predict(x_train)
    y_val_pred = reg.predict(x_val)
    model = {
        "family": "xgb",
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_lambda": float(reg_lambda),
    }
    print("[xgb] prediction complete", flush=True)
    return y_val_pred, y_train_pred, model


def main() -> int:
    args = build_parser().parse_args()
    if not args.include_q8_context_label_features and not args.include_q8_kg_control_features:
        raise ValueError("Enable at least one Q8 feature family: context labels and/or KG control features.")

    output_root = resolve_output_root(args.output_root)
    run_root = output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    adapter = legacy.V2VGoTQABenchmarkAdapter(str(Path(args.v2vgot_root).expanduser().resolve()))
    train_samples = legacy.load_q9_samples(
        adapter=adapter,
        split_name="train",
        file_name=args.train_file_name,
        limit=args.limit_train,
    )
    val_samples = legacy.load_q9_samples(
        adapter=adapter,
        split_name="val",
        file_name=args.val_file_name,
        limit=args.limit_val,
    )
    if not train_samples:
        raise ValueError("No Q9 train samples loaded.")
    if not val_samples:
        raise ValueError("No Q9 val samples loaded.")
    if not args.allow_train_val_overlap:
        val_keys = {legacy.qa_overlap_key(sample) for sample in val_samples}
        train_samples = tuple(sample for sample in train_samples if legacy.qa_overlap_key(sample) not in val_keys)
        if not train_samples:
            raise ValueError("All train samples were removed by train/val overlap filtering.")

    q8_context_train_lookup: dict[str, list[float]] = {}
    q8_context_val_lookup: dict[str, list[float]] = {}
    q8_kg_train_lookup: dict[str, list[float]] = {}
    q8_kg_val_lookup: dict[str, list[float]] = {}
    q8_feature_source: dict[str, object] = {"mode": "v2_explicit_q8_features"}
    q8_topk_train_lookup: dict[str, list[float]] = {}
    q8_topk_val_lookup: dict[str, list[float]] = {}
    q8_kg_selected_indices, q8_kg_selected_names = resolve_q8_kg_selected_indices(
        args.q8_kg_feature_set, args.q8_kg_feature_subset
    )

    if args.include_q8_context_label_features:
        q8_ctx_dir = run_root / "q8_question_context_features"
        q8_ctx_train_jsonl = q8_ctx_dir / f"{args.run_name}_q8_context_features_train.jsonl"
        q8_ctx_val_jsonl = q8_ctx_dir / f"{args.run_name}_q8_context_features_val.jsonl"
        train_float_lookup: dict[str, tuple[float, float]] = {}
        val_float_lookup: dict[str, tuple[float, float]] = {}
        train_rich_lookup: dict[str, dict[str, object]] = {}
        val_rich_lookup: dict[str, dict[str, object]] = {}
        if args.q8_context_value_source == "float_jsonl":
            if not args.q8_float_train_jsonl or not args.q8_float_val_jsonl:
                raise ValueError(
                    "--q8-context-value-source=float_jsonl requires both "
                    "--q8-float-train-jsonl and --q8-float-val-jsonl."
                )
            train_float_lookup = load_q8_float_lookup(args.q8_float_train_jsonl)
            val_float_lookup = load_q8_float_lookup(args.q8_float_val_jsonl)
            train_rich_lookup = load_q8_rich_lookup(args.q8_float_train_jsonl)
            val_rich_lookup = load_q8_rich_lookup(args.q8_float_val_jsonl)
            print(
                "[INFO] Loaded Q8 float lookups: "
                f"train={len(train_float_lookup)} val={len(val_float_lookup)}; "
                f"rich_train={len(train_rich_lookup)} rich_val={len(val_rich_lookup)}",
                flush=True,
            )
        print(
            "[INFO] Building Q8 context-label features from Q9 question context: "
            f"value_source={args.q8_context_value_source} mode={args.q8_context_feature_mode}",
            flush=True,
        )
        q8_context_train_lookup = build_q8_context_feature_lookup(
            samples=train_samples,
            output_jsonl=q8_ctx_train_jsonl,
            progress_every=args.progress_every,
            context_feature_mode=args.q8_context_feature_mode,
            value_source=args.q8_context_value_source,
            float_lookup=train_float_lookup,
            rich_lookup=train_rich_lookup,
            debug_samples=args.q8_context_debug_samples,
        )
        q8_context_val_lookup = build_q8_context_feature_lookup(
            samples=val_samples,
            output_jsonl=q8_ctx_val_jsonl,
            progress_every=args.progress_every,
            context_feature_mode=args.q8_context_feature_mode,
            value_source=args.q8_context_value_source,
            float_lookup=val_float_lookup,
            rich_lookup=val_rich_lookup,
            debug_samples=args.q8_context_debug_samples,
        )
        q8_feature_source["context_label_features"] = {
            "source": "q9_question_context",
            "context_feature_mode": args.q8_context_feature_mode,
            "value_source": args.q8_context_value_source,
            "q8_float_train_jsonl": args.q8_float_train_jsonl if args.q8_context_value_source == "float_jsonl" else "",
            "q8_float_val_jsonl": args.q8_float_val_jsonl if args.q8_context_value_source == "float_jsonl" else "",
            "train_jsonl": str(q8_ctx_train_jsonl),
            "val_jsonl": str(q8_ctx_val_jsonl),
        }

    if args.include_q8_kg_control_features:
        evaluator = V2VGoTQAPhase5AEvaluator(str(Path(args.v2vgot_root).expanduser().resolve()))
        q8_kg_dir = run_root / "q8_kg_control_features"
        q8_kg_train_jsonl = q8_kg_dir / f"{args.run_name}_q8_kg_control_features_train.jsonl"
        q8_kg_val_jsonl = q8_kg_dir / f"{args.run_name}_q8_kg_control_features_val.jsonl"
        print(
            "[INFO] Building Q8 KG control features from cooperative scene graph: "
            f"feature_set={args.q8_kg_feature_set}",
            flush=True,
        )
        q8_kg_train_lookup = build_q8_kg_control_feature_lookup(
            samples=train_samples,
            evaluator=evaluator,
            feature_set=args.q8_kg_feature_set,
            timeout_seconds=args.q8_kg_feature_timeout_seconds,
            progress_every=args.progress_every,
            output_jsonl=q8_kg_train_jsonl,
        )
        q8_kg_val_lookup = build_q8_kg_control_feature_lookup(
            samples=val_samples,
            evaluator=evaluator,
            feature_set=args.q8_kg_feature_set,
            timeout_seconds=args.q8_kg_feature_timeout_seconds,
            progress_every=args.progress_every,
            output_jsonl=q8_kg_val_jsonl,
        )
        q8_feature_source["kg_control_features"] = {
            "source": "cooperative_scene_graph",
            "feature_set": args.q8_kg_feature_set,
            "feature_names": list(control_feature_names(args.q8_kg_feature_set)),
            "selected_feature_names": list(q8_kg_selected_names),
            "train_jsonl": str(q8_kg_train_jsonl),
            "val_jsonl": str(q8_kg_val_jsonl),
            "timeout_seconds": args.q8_kg_feature_timeout_seconds,
            "leakage_note": "No benchmark metadata fields are used for Q8 KG features.",
        }

    if args.include_q8_topk_static_features:
        evaluator = V2VGoTQAPhase5AEvaluator(str(Path(args.v2vgot_root).expanduser().resolve()))
        q8_topk_dir = run_root / "q8_topk_static_features"
        q8_topk_train_jsonl = q8_topk_dir / f"{args.run_name}_q8_topk_static_features_train.jsonl"
        q8_topk_val_jsonl = q8_topk_dir / f"{args.run_name}_q8_topk_static_features_val.jsonl"
        print(
            f"[INFO] Building Q8 top-k static features from cooperative scene graph: k={args.q8_topk_k}",
            flush=True,
        )
        q8_topk_train_lookup = build_q8_topk_static_feature_lookup(
            samples=train_samples,
            evaluator=evaluator,
            topk_k=args.q8_topk_k,
            progress_every=args.progress_every,
            output_jsonl=q8_topk_train_jsonl,
            debug_samples=args.q8_topk_debug_samples,
        )
        q8_topk_val_lookup = build_q8_topk_static_feature_lookup(
            samples=val_samples,
            evaluator=evaluator,
            topk_k=args.q8_topk_k,
            progress_every=args.progress_every,
            output_jsonl=q8_topk_val_jsonl,
            debug_samples=args.q8_topk_debug_samples,
        )
        q8_feature_source["topk_static_features"] = {
            "source": "cooperative_scene_graph_current_timestamp",
            "k": int(args.q8_topk_k),
            "feature_names": q8_topk_static_feature_names(args.q8_topk_k),
            "train_jsonl": str(q8_topk_train_jsonl),
            "val_jsonl": str(q8_topk_val_jsonl),
            "leakage_note": "Uses only prepared scene state at current timestamp T.",
        }

    if args.include_q8_context_label_features and args.q8_context_feature_mode == "model_probs14":
        q8_context_names = (
            [f"q8_speed_prob_{label.replace(' ', '_')}" for label in legacy.SPEED_LABELS]
            + [f"q8_steer_prob_{label.replace(' ', '_')}" for label in legacy.STEER_LABELS]
            + ["q8_speed_top1_prob", "q8_steer_top1_prob", "q8_speed_margin", "q8_steer_margin"]
        )
        all_feature_names = legacy.feature_names(include_q8_pred_features=False) + q8_context_names
        if args.include_q8_kg_control_features:
            all_feature_names.extend([f"q8kg_{name}" for name in q8_kg_selected_names])
    else:
        all_feature_names = feature_names(
            include_q8_context_label_features=args.include_q8_context_label_features,
            include_q8_kg_control_features=args.include_q8_kg_control_features,
            q8_kg_selected_names=q8_kg_selected_names,
        )
        if args.include_q8_topk_static_features:
            all_feature_names.extend(q8_topk_static_feature_names(args.q8_topk_k))
    profile_selected_indices, profile_selected_names = resolve_feature_profile_indices(
        all_feature_names, args.feature_profile
    )
    print(
        f"[INFO] feature profile: {args.feature_profile} "
        f"selected={len(profile_selected_names)}/{len(all_feature_names)}",
        flush=True,
    )

    x_train, y_train, usable_train = build_xy(
        train_samples,
        include_q8_context_label_features=args.include_q8_context_label_features,
        include_q8_kg_control_features=args.include_q8_kg_control_features,
        q8_context_lookup=q8_context_train_lookup,
        q8_kg_lookup=q8_kg_train_lookup,
        q8_kg_feature_set=args.q8_kg_feature_set,
        q8_kg_selected_indices=q8_kg_selected_indices,
        include_q8_topk_static_features=args.include_q8_topk_static_features,
        q8_topk_static_lookup=q8_topk_train_lookup,
        profile_selected_indices=profile_selected_indices,
        progress_every=args.progress_every,
    )
    x_val, y_val, usable_val = build_xy(
        val_samples,
        include_q8_context_label_features=args.include_q8_context_label_features,
        include_q8_kg_control_features=args.include_q8_kg_control_features,
        q8_context_lookup=q8_context_val_lookup,
        q8_kg_lookup=q8_kg_val_lookup,
        q8_kg_feature_set=args.q8_kg_feature_set,
        q8_kg_selected_indices=q8_kg_selected_indices,
        include_q8_topk_static_features=args.include_q8_topk_static_features,
        q8_topk_static_lookup=q8_topk_val_lookup,
        profile_selected_indices=profile_selected_indices,
        progress_every=args.progress_every,
    )
    print(
        "[INFO] feature matrices ready: "
        f"x_train={x_train.shape} y_train={y_train.shape} usable_train={usable_train}; "
        f"x_val={x_val.shape} y_val={y_val.shape} usable_val={usable_val}",
        flush=True,
    )

    model_results: list[SweepModelResult] = []
    for model_name in args.models:
        model_name = str(model_name).strip().lower()
        print(f"\n[INFO] Training model: {model_name}", flush=True)
        if model_name == "ridge":
            y_val_pred, model_payload = legacy.train_predict_ridge(x_train, y_train, x_val, alpha=1.0)
            y_train_pred = legacy.predict_with_saved_model(model_payload, x_train)
        elif model_name == "elasticnet":
            y_val_pred, model_payload = legacy.train_predict_elasticnet(
                x_train, y_train, x_val, alpha=0.001, l1_ratio=0.1
            )
            y_train_pred = legacy.predict_with_saved_model(model_payload, x_train)
        elif model_name == "rf":
            y_val_pred, model_payload = legacy.train_predict_rf(x_train, y_train, x_val, n_estimators=200, max_depth=12)
            y_train_pred = legacy.train_predict_rf(x_train, y_train, x_train, n_estimators=200, max_depth=12)[0]
        elif model_name == "hgb":
            y_val_pred, y_train_pred, model_payload = train_predict_hgb(x_train, y_train, x_val)
        elif model_name == "mlp":
            y_val_pred, y_train_pred, model_payload = train_predict_mlp(x_train, y_train, x_val)
        elif model_name == "xgb":
            y_val_pred, y_train_pred, model_payload = train_predict_xgb(x_train, y_train, x_val)
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Use ridge elasticnet rf hgb mlp xgb.")

        train_metrics = legacy.l2_metrics(y_train, y_train_pred)
        val_metrics = legacy.l2_metrics(y_val, y_val_pred)
        print(
            f"[INFO] model={model_name} metrics: "
            f"train_l2_avg={train_metrics['l2_error_avg_all']:.6f} "
            f"val_l2_avg={val_metrics['l2_error_avg_all']:.6f}",
            flush=True,
        )

        scenario_name = f"{args.run_name}_{model_name}"
        model_json = run_root / f"{scenario_name}_model.json"
        pred_jsonl = run_root / f"{scenario_name}_predictions.jsonl"
        pred_manifest = run_root / f"{scenario_name}_manifest.json"

        model_record = {
            "run_name": args.run_name,
            "scenario_name": scenario_name,
            "model_name": model_name,
            "feature_names": list(profile_selected_names),
            "feature_profile": args.feature_profile,
            "feature_names_all_before_profile": list(all_feature_names),
            "include_q8_context_label_features": bool(args.include_q8_context_label_features),
            "include_q8_kg_control_features": bool(args.include_q8_kg_control_features),
            "include_q8_topk_static_features": bool(args.include_q8_topk_static_features),
            "q8_topk_k": int(args.q8_topk_k),
            "q8_kg_feature_set": args.q8_kg_feature_set,
            "q8_kg_feature_subset": args.q8_kg_feature_subset,
            "train_rows": len(train_samples),
            "val_rows": len(val_samples),
            "usable_train_rows": usable_train,
            "usable_val_rows": usable_val,
            "train_metrics_local": train_metrics,
            "val_metrics_local": val_metrics,
            "model_payload": model_payload,
            "leakage_policy": {
                "excluded_fields": [
                    "dist",
                    "angle",
                    "suggested_speed_idx",
                    "suggested_steering_idx",
                    "future_trajectory_str_in_ego",
                    "future_trajectory_str_in_self",
                ],
                "q8_feature_source": q8_feature_source,
            },
        }
        model_json.write_text(json.dumps(model_record, indent=2), encoding="utf-8")

        legacy.write_predictions_jsonl(
            path=pred_jsonl,
            val_samples=val_samples,
            y_val_pred=y_val_pred,
            scenario_name=scenario_name,
        )
        legacy.write_prediction_manifest(
            path=pred_manifest,
            v2vgot_root=str(Path(args.v2vgot_root).expanduser().resolve()),
            val_file_name=args.val_file_name,
            scenario_name=scenario_name,
            output_jsonl=pred_jsonl,
            total_samples=len(val_samples),
        )

        export_manifest_json: str | None = None
        official_summary_json: str | None = None
        if args.run_official_eval:
            export_manifest_json, official_summary_json = legacy.run_official_export_eval(
                python_bin=args.python,
                v2vgot_root=str(Path(args.v2vgot_root).expanduser().resolve()),
                prediction_manifest=pred_manifest,
                output_root=run_root,
                scenario_name=scenario_name,
                val_file_name=args.val_file_name,
            )

        model_results.append(
            SweepModelResult(
                model_name=model_name,
                model_json=model_json,
                prediction_jsonl=pred_jsonl,
                prediction_manifest_json=pred_manifest,
                train_rows=len(train_samples),
                val_rows=len(val_samples),
                train_l2_avg=float(train_metrics["l2_error_avg_all"]),
                val_l2_avg=float(val_metrics["l2_error_avg_all"]),
                val_l2_1s=float(val_metrics["l2_error_avg_1s"]),
                val_l2_2s=float(val_metrics["l2_error_avg_2s"]),
                val_l2_3s=float(val_metrics["l2_error_avg_3s"]),
                official_export_manifest_json=export_manifest_json,
                official_summary_json=official_summary_json,
            )
        )

    consolidated = {
        "run_name": args.run_name,
        "v2vgot_root": str(Path(args.v2vgot_root).expanduser().resolve()),
        "train_file_name": args.train_file_name,
        "val_file_name": args.val_file_name,
        "include_q8_context_label_features": bool(args.include_q8_context_label_features),
        "include_q8_kg_control_features": bool(args.include_q8_kg_control_features),
        "feature_profile": args.feature_profile,
        "feature_names": list(profile_selected_names),
        "q8_kg_feature_set": args.q8_kg_feature_set,
        "q8_kg_feature_subset": args.q8_kg_feature_subset,
        "q8_feature_source": q8_feature_source,
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
        "models": [
            {
                "model_name": item.model_name,
                "model_json": str(item.model_json),
                "prediction_jsonl": str(item.prediction_jsonl),
                "prediction_manifest_json": str(item.prediction_manifest_json),
                "train_rows": item.train_rows,
                "val_rows": item.val_rows,
                "train_l2_avg": item.train_l2_avg,
                "val_l2_1s": item.val_l2_1s,
                "val_l2_2s": item.val_l2_2s,
                "val_l2_3s": item.val_l2_3s,
                "val_l2_avg": item.val_l2_avg,
                "official_export_manifest_json": item.official_export_manifest_json,
                "official_summary_json": item.official_summary_json,
            }
            for item in model_results
        ],
    }
    consolidated_path = run_root / f"{args.run_name}_consolidated_manifest.json"
    consolidated_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")

    print("\n============================================================")
    print("Q9 Sweep v2 Complete")
    print("============================================================")
    print(f"run_root: {run_root}")
    print(f"consolidated_manifest: {consolidated_path}")
    print("\nModel summary:")
    for item in model_results:
        print(
            f"  - {item.model_name}: val_l2_avg={item.val_l2_avg:.4f} "
            f"(1s={item.val_l2_1s:.4f}, 2s={item.val_l2_2s:.4f}, 3s={item.val_l2_3s:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
