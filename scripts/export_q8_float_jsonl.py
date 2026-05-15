#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export explicit Q8 float-value JSONLs keyed by sample_id for Q9 feature joins. "
            "Writes q8_pred_speed_control_value_float and q8_pred_steering_control_value_float."
        )
    )
    parser.add_argument("--input-train-jsonl", required=True)
    parser.add_argument("--input-val-jsonl", required=True)
    parser.add_argument("--output-train-jsonl", required=True)
    parser.add_argument("--output-val-jsonl", required=True)
    parser.add_argument(
        "--strict-float-keys",
        action="store_true",
        help=(
            "Require explicit float keys in input rows. If disabled, falls back to "
            "q8_pred_*_control_value fields when explicit *_float keys are absent."
        ),
    )
    return parser


def _parse_float_value(row: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def export_file(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    strict_float_keys: bool,
) -> tuple[int, int]:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_jsonl}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                skipped += 1
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                skipped += 1
                continue

            if strict_float_keys:
                speed = _parse_float_value(row, ("q8_pred_speed_control_value_float",))
                steer = _parse_float_value(row, ("q8_pred_steering_control_value_float",))
                value_source = "explicit_float_keys"
            else:
                speed = _parse_float_value(
                    row,
                    ("q8_pred_speed_control_value_float", "q8_pred_speed_control_value", "speed_control_value"),
                )
                steer = _parse_float_value(
                    row,
                    ("q8_pred_steering_control_value_float", "q8_pred_steering_control_value", "steering_control_value"),
                )
                value_source = "float_or_fallback"

            # Legacy fallback: some older q8 feature dumps only store the full
            # feature vector; last two entries are speed/steering numeric values.
            if speed is None or steer is None:
                vector = row.get("q8_feature_vector")
                if isinstance(vector, list) and len(vector) >= 2:
                    try:
                        tail_speed = float(vector[-2])
                        tail_steer = float(vector[-1])
                        speed = tail_speed if speed is None else speed
                        steer = tail_steer if steer is None else steer
                        if value_source == "float_or_fallback":
                            value_source = "q8_feature_vector_tail"
                    except (TypeError, ValueError):
                        pass

            if speed is None or steer is None:
                skipped += 1
                continue

            out = {
                "sample_id": sample_id,
                "split_name": row.get("split_name", ""),
                "file_name": row.get("file_name", ""),
                "scenario_index": row.get("scenario_index"),
                "global_timestamp_index": row.get("global_timestamp_index"),
                "local_timestamp_index": row.get("local_timestamp_index"),
                "asker_cav_id": row.get("asker_cav_id", ""),
                "q8_pred_speed_control_value_float": float(speed),
                "q8_pred_steering_control_value_float": float(steer),
                "q8_float_value_source": value_source,
            }
            dst.write(json.dumps(out) + "\n")
            written += 1
    return written, skipped


def main() -> int:
    args = build_parser().parse_args()
    input_train = Path(args.input_train_jsonl).expanduser().resolve()
    input_val = Path(args.input_val_jsonl).expanduser().resolve()
    output_train = Path(args.output_train_jsonl).expanduser().resolve()
    output_val = Path(args.output_val_jsonl).expanduser().resolve()

    train_written, train_skipped = export_file(
        input_jsonl=input_train,
        output_jsonl=output_train,
        strict_float_keys=bool(args.strict_float_keys),
    )
    val_written, val_skipped = export_file(
        input_jsonl=input_val,
        output_jsonl=output_val,
        strict_float_keys=bool(args.strict_float_keys),
    )

    print("[INFO] Q8 float export complete")
    print(f"[INFO] train_input={input_train}")
    print(f"[INFO] train_output={output_train}")
    print(f"[INFO] train_written={train_written} train_skipped={train_skipped}")
    print(f"[INFO] val_input={input_val}")
    print(f"[INFO] val_output={output_val}")
    print(f"[INFO] val_written={val_written} val_skipped={val_skipped}")
    if args.strict_float_keys:
        print("[INFO] mode=strict_float_keys")
    else:
        print("[INFO] mode=float_or_fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
