#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a Q4 planning acceptor JSON and add trajectory-calibration policy knobs."
    )
    parser.add_argument("--input-model-json", required=True)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--far-distance-to-trajectory", type=float, default=10.0)
    parser.add_argument("--far-abs-y", type=float, default=5.0)
    parser.add_argument("--far-moderate-max-probability", type=float, default=0.65)
    parser.add_argument("--rescue-min-probability", type=float, default=0.50)
    parser.add_argument("--rescue-max-rank", type=int, default=6)
    parser.add_argument("--rescue-max-distance-to-trajectory", type=float, default=4.0)
    return parser


def resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = build_parser().parse_args()
    input_path = resolve(args.input_model_json)
    output_path = resolve(args.output_model_json)
    model = json.loads(input_path.read_text(encoding="utf-8"))
    model["trajectory_calibration"] = {
        "far_distance_to_trajectory": args.far_distance_to_trajectory,
        "far_abs_y": args.far_abs_y,
        "far_moderate_max_probability": args.far_moderate_max_probability,
        "rescue_min_probability": args.rescue_min_probability,
        "rescue_max_rank": args.rescue_max_rank,
        "rescue_max_distance_to_trajectory": args.rescue_max_distance_to_trajectory,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"saved_model: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
