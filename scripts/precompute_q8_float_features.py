#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import run_gmark_q9_model_sweep as legacy
from kg_coop_drive.application.planning.control_settings_policy import (
    SPEED_CLASSES,
    STEERING_CLASSES,
    build_control_feature_vector,
    control_feature_names,
    select_top_control_objects,
)
from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator


SPEED_VALUE_MAP = {
    "fast": 1.0,
    "moderate": 0.65,
    "slow": 0.35,
    "very slow": 0.15,
    "stop": 0.0,
}
STEER_VALUE_MAP = {
    "left": -1.0,
    "slightly left": -0.5,
    "straight": 0.0,
    "slightly right": 0.5,
    "right": 1.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute split-correct Q8 float features for Q9 samples using a deployable "
            "Q8 linear classifier model."
        )
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--train-file-name", default=legacy.DEFAULT_TRAIN_FILE_NAME)
    parser.add_argument("--val-file-name", default=legacy.TABLE1_Q9_FILE_NAME)
    parser.add_argument("--q8-model-json", required=True)
    parser.add_argument("--output-train-jsonl", required=True)
    parser.add_argument("--output-val-jsonl", required=True)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--allow-train-val-overlap", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--debug-samples", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser


def load_q8_model(path: str) -> dict[str, object]:
    model_path = Path(path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Q8 model file not found: {model_path}")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Q8 model JSON: {model_path}")
    return payload


def _linear_scores(features: tuple[float, ...], weights: list[object]) -> list[float]:
    scores: list[float] = []
    for row in weights:
        if not isinstance(row, list) or len(row) != len(features):
            raise ValueError("Weight shape does not match feature length.")
        scores.append(sum(float(weight) * value for weight, value in zip(row, features)))
    return scores


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(-50.0, min(50.0, v)) for v in values]
    max_v = max(clipped)
    exps = [math.exp(v - max_v) for v in clipped]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in exps]


def _sigmoid(value: float) -> float:
    clipped = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _expected_value(probs: list[float], classes: tuple[str, ...], value_map: dict[str, float]) -> float:
    return sum(prob * value_map[str(label)] for prob, label in zip(probs, classes))


def compute_q8_float_features(
    *,
    prepared_scene: object,
    model: dict[str, object],
) -> dict[str, object]:
    feature_set = str(model.get("feature_set", "base"))
    raw_features = build_control_feature_vector(prepared_scene, feature_set=feature_set)
    expected_feature_names = control_feature_names(feature_set)
    model_feature_names = model.get("feature_names")
    if isinstance(model_feature_names, list) and tuple(model_feature_names) != expected_feature_names:
        raise ValueError("Model feature_names do not match control feature set.")

    feature_mean = model.get("feature_mean")
    feature_std = model.get("feature_std")
    if not (isinstance(feature_mean, list) and isinstance(feature_std, list)):
        raise ValueError("Missing feature_mean/feature_std in model.")
    if len(raw_features) != len(feature_mean) or len(raw_features) != len(feature_std):
        raise ValueError("Model normalization shape mismatch.")

    normalized = tuple(
        (value - float(mean)) / max(float(std), 1e-6)
        for value, mean, std in zip(raw_features, feature_mean, feature_std)
    )
    model_input = (1.0,) + normalized

    steering_weights = model.get("steering_weights")
    if not isinstance(steering_weights, list):
        raise ValueError("Missing steering_weights in model.")
    steering_scores = _linear_scores(model_input, steering_weights)
    steering_probs = _softmax(steering_scores)
    steer_idx = max(range(len(steering_scores)), key=steering_scores.__getitem__)
    steer_label = STEERING_CLASSES[steer_idx]

    speed_head_type = str(model.get("speed_head_type", "multiclass"))
    if speed_head_type == "ordinal":
        speed_ordinal_weights = model.get("speed_ordinal_weights")
        if not isinstance(speed_ordinal_weights, list):
            raise ValueError("Missing speed_ordinal_weights for ordinal speed head.")
        ordinal_scores = _linear_scores(model_input, speed_ordinal_weights)
        ordinal_probs = [_sigmoid(score) for score in ordinal_scores]
        # Convert monotonic ordinal probs P(y > k) into class probs.
        thresholds = [max(0.0, min(1.0, p)) for p in ordinal_probs]
        class_probs = [0.0] * len(SPEED_CLASSES)
        class_probs[0] = max(0.0, 1.0 - thresholds[0]) if thresholds else 1.0
        for idx in range(1, len(SPEED_CLASSES) - 1):
            class_probs[idx] = max(0.0, thresholds[idx - 1] - thresholds[idx])
        if len(SPEED_CLASSES) > 1:
            class_probs[-1] = max(0.0, thresholds[-1]) if thresholds else 0.0
        total = sum(class_probs)
        speed_probs = [p / total for p in class_probs] if total > 0 else [1.0 / len(SPEED_CLASSES)] * len(SPEED_CLASSES)
    else:
        speed_weights = model.get("speed_weights")
        if not isinstance(speed_weights, list):
            raise ValueError("Missing speed_weights for multiclass speed head.")
        speed_scores = _linear_scores(model_input, speed_weights)
        speed_probs = _softmax(speed_scores)
    speed_idx = max(range(len(speed_probs)), key=speed_probs.__getitem__)
    speed_label = SPEED_CLASSES[speed_idx]

    speed_value_float = _expected_value(speed_probs, SPEED_CLASSES, SPEED_VALUE_MAP)
    steering_value_float = _expected_value(steering_probs, STEERING_CLASSES, STEER_VALUE_MAP)
    top_objects = select_top_control_objects(prepared_scene)
    return {
        "q8_pred_speed_label": speed_label,
        "q8_pred_steering_label": steer_label,
        "q8_pred_speed_probs": [float(v) for v in speed_probs],
        "q8_pred_steering_probs": [float(v) for v in steering_probs],
        "q8_pred_speed_control_value_float": float(speed_value_float),
        "q8_pred_steering_control_value_float": float(steering_value_float),
        "q8_pred_object_ids": [item.object_id for item in top_objects[:2]],
    }


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                sample_id = str(row.get("sample_id", "")).strip()
                if sample_id:
                    ids.add(sample_id)
    return ids


def run_split(
    *,
    split_name: str,
    samples: tuple[legacy.BenchmarkSample, ...],
    evaluator: V2VGoTQAPhase5AEvaluator,
    model: dict[str, object],
    output_jsonl: Path,
    progress_every: int,
    debug_samples: int,
    resume: bool,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(output_jsonl) if resume else set()
    mode = "a" if resume else "w"
    debug_left = debug_samples
    with output_jsonl.open(mode, encoding="utf-8") as handle:
        processed_count = 0
        for idx, sample in enumerate(samples, start=1):
            if sample.sample_id in processed_ids:
                continue
            started = time.monotonic()
            prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode="cooperative")
            result = compute_q8_float_features(prepared_scene=prepared_scene, model=model)
            elapsed = time.monotonic() - started
            row = {
                "sample_id": sample.sample_id,
                "split_name": sample.split_name,
                "file_name": sample.file_name,
                "scenario_index": sample.raw_record.get("scenario_index"),
                "global_timestamp_index": sample.raw_record.get("global_timestamp_index"),
                "local_timestamp_index": sample.raw_record.get("local_timestamp_index"),
                "asker_cav_id": sample.raw_record.get("asker_cav_id"),
                "q8_feature_elapsed_seconds": round(elapsed, 6),
                **result,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            processed_count += 1
            if debug_left > 0:
                print(
                    "[DEBUG] q8_float_row: "
                    f"split={split_name} sample_id={sample.sample_id} "
                    f"speed_label={row['q8_pred_speed_label']} steer_label={row['q8_pred_steering_label']} "
                    f"speed_float={row['q8_pred_speed_control_value_float']:.6f} "
                    f"steer_float={row['q8_pred_steering_control_value_float']:.6f}",
                    flush=True,
                )
                debug_left -= 1
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(samples)):
                print(
                    f"q8_float_progress: {split_name} {idx}/{len(samples)} "
                    f"written_this_run={processed_count}",
                    flush=True,
                )
    print(f"[INFO] split={split_name} output={output_jsonl}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    q8_model = load_q8_model(args.q8_model_json)
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

    evaluator = V2VGoTQAPhase5AEvaluator(str(Path(args.v2vgot_root).expanduser().resolve()))
    run_split(
        split_name="train",
        samples=train_samples,
        evaluator=evaluator,
        model=q8_model,
        output_jsonl=Path(args.output_train_jsonl).expanduser().resolve(),
        progress_every=args.progress_every,
        debug_samples=args.debug_samples,
        resume=args.resume,
    )
    run_split(
        split_name="val",
        samples=val_samples,
        evaluator=evaluator,
        model=q8_model,
        output_jsonl=Path(args.output_val_jsonl).expanduser().resolve(),
        progress_every=args.progress_every,
        debug_samples=args.debug_samples,
        resume=args.resume,
    )
    print("[INFO] Q8 float precompute complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
