#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper for clean Q9 sweep: generate Q8 predictions from a trained Q8 model, "
            "then run Q9 sweep with optional Q8-derived features."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--output-root", default="outputs/v2vgot_table1_reproduction/gmark_q9_sweep")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--q8-model-json", default="")
    parser.add_argument(
        "--e2e-manifest-json",
        default="",
        help="Optional E2E manifest; if set and --q8-model-json is empty, uses model_paths.q8_model_json.",
    )
    parser.add_argument("--val-file-name", default="v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json")
    parser.add_argument(
        "--q8-file-name",
        default="v2v4real_3d_grounding_qa_dataset_v2vgot.json",
        help=(
            "QA JSON used to generate Q8 predictions. Keep this as the general "
            "val QA file; the released nq9 file is Q9-only and has no Q8 rows."
        ),
    )
    parser.add_argument("--models", nargs="+", default=("ridge", "elasticnet", "rf"))
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--allow-train-val-overlap", action="store_true")
    parser.add_argument("--run-official-eval", action="store_true")
    parser.add_argument(
        "--skip-q8-feature-branch",
        action="store_true",
        help="If set, runs only no-Q8-feature sweep branch.",
    )
    return parser


def run(command: Sequence[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(list(command), cwd=str(REPO_ROOT), check=True)


def resolve_q8_model_json(args: argparse.Namespace) -> str:
    if args.q8_model_json:
        return str(Path(args.q8_model_json).expanduser().resolve())
    if not args.e2e_manifest_json:
        raise ValueError("Provide either --q8-model-json or --e2e-manifest-json.")
    manifest_path = Path(args.e2e_manifest_json).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_paths = payload.get("model_paths", {})
    if not isinstance(model_paths, dict):
        raise ValueError(f"Invalid model_paths in {manifest_path}")
    q8_model_json = str(model_paths.get("q8_model_json", "")).strip()
    if not q8_model_json:
        raise ValueError(f"q8_model_json missing in {manifest_path}")
    return str(Path(q8_model_json).expanduser().resolve())


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    run_root = output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    base_sweep = [
        args.python,
        "scripts/run_gmark_q9_model_sweep.py",
        "--v2vgot-root",
        args.v2vgot_root,
        "--run-name",
        f"{args.run_name}_noq8feat",
        "--output-root",
        str(output_root),
        "--val-file-name",
        args.val_file_name,
        "--progress-every",
        str(args.progress_every),
        "--limit-train",
        str(args.limit_train),
        "--limit-val",
        str(args.limit_val),
        "--models",
        *args.models,
    ]
    if args.allow_train_val_overlap:
        base_sweep.append("--allow-train-val-overlap")
    if args.run_official_eval:
        base_sweep.append("--run-official-eval")
    run(base_sweep)

    if args.skip_q8_feature_branch:
        print("[INFO] Completed no-Q8-feature branch only.", flush=True)
        return 0

    q8_model_json = resolve_q8_model_json(args)
    q8_scenario = f"{args.run_name}_q8_pred_for_q9"
    q8_output_root = run_root / "q8_predictions"
    q8_output_root.mkdir(parents=True, exist_ok=True)
    q8_prediction_jsonl = q8_output_root / f"{q8_scenario}.jsonl"

    run(
        [
            args.python,
            "scripts/run_qa_split_pipeline.py",
            "--purpose",
            "val_report",
            "--split",
            "val",
            "--task-type",
            "control_settings",
            "--scenario-name",
            q8_scenario,
            "--file-name",
            args.q8_file_name,
            "--baseline-mode",
            "cooperative",
            "--control-selection-policy",
            "linear_classifier",
            "--control-model-json",
            q8_model_json,
            "--output-root",
            str(q8_output_root),
            "--v2vgot-root",
            args.v2vgot_root,
            "--progress-every",
            str(args.progress_every),
            "--skip-official-eval",
        ]
    )
    if not q8_prediction_jsonl.exists():
        raise FileNotFoundError(f"Expected Q8 predictions not found: {q8_prediction_jsonl}")

    q8_sweep = [
        args.python,
        "scripts/run_gmark_q9_model_sweep.py",
        "--v2vgot-root",
        args.v2vgot_root,
        "--run-name",
        f"{args.run_name}_withq8feat",
        "--output-root",
        str(output_root),
        "--val-file-name",
        args.val_file_name,
        "--progress-every",
        str(args.progress_every),
        "--limit-train",
        str(args.limit_train),
        "--limit-val",
        str(args.limit_val),
        "--models",
        *args.models,
        "--include-q8-pred-features",
        "--q8-predictions-jsonl",
        str(q8_prediction_jsonl),
    ]
    if args.allow_train_val_overlap:
        q8_sweep.append("--allow-train-val-overlap")
    if args.run_official_eval:
        q8_sweep.append("--run-official-eval")
    run(q8_sweep)

    wrapper_manifest = {
        "run_name": args.run_name,
        "output_root": str(output_root),
        "q8_model_json": q8_model_json,
        "q8_prediction_jsonl": str(q8_prediction_jsonl),
        "branches": {
            "no_q8_features_run_name": f"{args.run_name}_noq8feat",
            "with_q8_features_run_name": f"{args.run_name}_withq8feat",
        },
        "models": list(args.models),
    }
    wrapper_manifest_path = run_root / f"{args.run_name}_wrapper_manifest.json"
    wrapper_manifest_path.write_text(json.dumps(wrapper_manifest, indent=2), encoding="utf-8")
    print(f"[INFO] wrapper_manifest: {wrapper_manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
