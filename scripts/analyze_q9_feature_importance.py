#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_gmark_q9_model_sweep as legacy
import run_gmark_q9_model_sweep_v2 as sweep_v2


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Permutation feature-importance analysis for Q9 sweep v2 model artifacts. "
            "Supports grouped and per-feature modes."
        )
    )
    parser.add_argument("--model-json", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--val-file-name", default=legacy.TABLE1_Q9_FILE_NAME)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--mode", choices=("grouped", "per_feature"), default="grouped")
    parser.add_argument("--num-shuffles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--output-dir", required=True)
    return parser


def resolve_model_json(args: argparse.Namespace) -> Path:
    if args.model_json:
        path = Path(args.model_json).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"model-json not found: {path}")
        return path
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        if not manifest.exists():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        if manifest.name.endswith("_manifest.json"):
            candidate = manifest.with_name(manifest.name.replace("_manifest.json", "_model.json"))
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not infer model json from manifest. Pass --model-json explicitly."
        )
    raise ValueError("Provide either --model-json or --manifest.")


def build_feature_matrix_for_model(
    *,
    model_record: dict[str, object],
    v2vgot_root: str,
    val_file_name: str,
    limit_val: int,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    include_q8_context = bool(model_record.get("include_q8_context_label_features", False))
    include_q8_kg = bool(model_record.get("include_q8_kg_control_features", False))
    q8_kg_feature_set = str(model_record.get("q8_kg_feature_set", "extended_v1"))
    q8_kg_feature_subset = str(model_record.get("q8_kg_feature_subset", "all"))
    feature_names = list(model_record.get("feature_names", []))

    adapter = legacy.V2VGoTQABenchmarkAdapter(str(Path(v2vgot_root).expanduser().resolve()))
    val_samples = legacy.load_q9_samples(
        adapter=adapter,
        split_name="val",
        file_name=val_file_name,
        limit=limit_val,
    )
    if not val_samples:
        raise ValueError("No Q9 val samples loaded.")

    q8_context_lookup: dict[str, list[float]] = {}
    if include_q8_context:
        context_jsonl = REPO_ROOT / "outputs" / "tmp" / "q9_featimp_q8_context_val.jsonl"
        context_jsonl.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] building val Q8 context features from Q9 prompt context", flush=True)
        q8_context_lookup = legacy.build_q8_prediction_lookup_from_question_context(
            samples=val_samples,
            output_jsonl=context_jsonl,
            progress_every=progress_every,
        )

    q8_kg_lookup: dict[str, list[float]] = {}
    q8_kg_selected_indices, _q8_kg_selected_names = sweep_v2.resolve_q8_kg_selected_indices(
        q8_kg_feature_set, q8_kg_feature_subset
    )
    if include_q8_kg:
        evaluator = sweep_v2.V2VGoTQAPhase5AEvaluator(str(Path(v2vgot_root).expanduser().resolve()))
        kg_jsonl = REPO_ROOT / "outputs" / "tmp" / "q9_featimp_q8_kg_val.jsonl"
        kg_jsonl.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] building val Q8 KG control features", flush=True)
        q8_kg_lookup = sweep_v2.build_q8_kg_control_feature_lookup(
            samples=val_samples,
            evaluator=evaluator,
            feature_set=q8_kg_feature_set,
            timeout_seconds=0,
            progress_every=progress_every,
            output_jsonl=kg_jsonl,
        )

    x_val, y_val, usable = sweep_v2.build_xy(
        val_samples,
        include_q8_context_label_features=include_q8_context,
        include_q8_kg_control_features=include_q8_kg,
        q8_context_lookup=q8_context_lookup,
        q8_kg_lookup=q8_kg_lookup,
        q8_kg_feature_set=q8_kg_feature_set,
        q8_kg_selected_indices=q8_kg_selected_indices,
        progress_every=progress_every,
    )
    print(f"[INFO] usable val rows for analysis: {usable}", flush=True)
    if feature_names and len(feature_names) != x_val.shape[1]:
        raise ValueError(
            f"Feature-name mismatch: model has {len(feature_names)} names but built matrix has {x_val.shape[1]} columns."
        )
    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(x_val.shape[1])]
    return x_val, y_val, feature_names


def predict_from_payload(model_payload: dict[str, object], x: np.ndarray) -> np.ndarray:
    family = str(model_payload.get("family", ""))
    if family in {"ridge", "elasticnet"}:
        return legacy.predict_with_saved_model(model_payload, x)
    raise ValueError(
        f"Permutation analysis currently supports ridge/elasticnet payloads only; got family={family}."
    )


def build_groups(mode: str, feature_names: list[str]) -> list[tuple[str, list[int]]]:
    if mode == "per_feature":
        return [(name, [idx]) for idx, name in enumerate(feature_names)]

    grouped: dict[str, list[int]] = {
        "base_geometry": [],
        "q8_context_speed_onehot": [],
        "q8_context_steer_onehot": [],
        "q8_context_numeric": [],
        "q8_kg": [],
        "other": [],
    }
    for idx, name in enumerate(feature_names):
        if name.startswith("q8_speed_"):
            grouped["q8_context_speed_onehot"].append(idx)
        elif name.startswith("q8_steer_"):
            grouped["q8_context_steer_onehot"].append(idx)
        elif name in {"q8_pred_speed_control_value", "q8_pred_steering_control_value"}:
            grouped["q8_context_numeric"].append(idx)
        elif name.startswith("q8kg_"):
            grouped["q8_kg"].append(idx)
        elif name.startswith("traj_") or name in {"bias", "current_x", "current_y", "asker_is_cav1"}:
            grouped["base_geometry"].append(idx)
        else:
            grouped["other"].append(idx)
    return [(group_name, indices) for group_name, indices in grouped.items() if indices]


def permutation_importance(
    *,
    model_payload: dict[str, object],
    x_val: np.ndarray,
    y_val: np.ndarray,
    groups: list[tuple[str, list[int]]],
    num_shuffles: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    baseline_pred = predict_from_payload(model_payload, x_val)
    baseline_metrics = legacy.l2_metrics(y_val, baseline_pred)
    rng = np.random.default_rng(seed)
    rows = x_val.shape[0]
    results: list[dict[str, object]] = []
    for group_name, indices in groups:
        deltas_1s: list[float] = []
        deltas_2s: list[float] = []
        deltas_3s: list[float] = []
        deltas_all: list[float] = []
        for _ in range(num_shuffles):
            perm = rng.permutation(rows)
            x_perm = x_val.copy()
            x_perm[:, indices] = x_val[perm][:, indices]
            pred = predict_from_payload(model_payload, x_perm)
            metrics = legacy.l2_metrics(y_val, pred)
            deltas_1s.append(metrics["l2_error_avg_1s"] - baseline_metrics["l2_error_avg_1s"])
            deltas_2s.append(metrics["l2_error_avg_2s"] - baseline_metrics["l2_error_avg_2s"])
            deltas_3s.append(metrics["l2_error_avg_3s"] - baseline_metrics["l2_error_avg_3s"])
            deltas_all.append(metrics["l2_error_avg_all"] - baseline_metrics["l2_error_avg_all"])
        results.append(
            {
                "name": group_name,
                "indices": indices,
                "num_features": len(indices),
                "delta_l2_error_avg_1s_mean": float(np.mean(deltas_1s)),
                "delta_l2_error_avg_2s_mean": float(np.mean(deltas_2s)),
                "delta_l2_error_avg_3s_mean": float(np.mean(deltas_3s)),
                "delta_l2_error_avg_all_mean": float(np.mean(deltas_all)),
                "delta_l2_error_avg_all_std": float(np.std(deltas_all)),
            }
        )
    results.sort(key=lambda item: float(item["delta_l2_error_avg_all_mean"]), reverse=True)
    return baseline_metrics, results


def write_reports(
    *,
    output_dir: Path,
    model_json: Path,
    mode: str,
    baseline_metrics: dict[str, float],
    results: list[dict[str, object]],
    feature_names: list[str],
    num_shuffles: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_json": str(model_json),
        "mode": mode,
        "num_shuffles": num_shuffles,
        "seed": seed,
        "baseline_metrics": baseline_metrics,
        "feature_count": len(feature_names),
        "results": results,
    }
    json_path = output_dir / "q9_feature_importance.json"
    md_path = output_dir / "q9_feature_importance.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Q9 Feature Importance",
        "",
        f"- model_json: `{model_json}`",
        f"- mode: `{mode}`",
        f"- num_shuffles: `{num_shuffles}`",
        f"- seed: `{seed}`",
        "",
        "## Baseline Metrics",
        "",
        f"- l2_error_avg_1s: `{baseline_metrics['l2_error_avg_1s']:.6f}`",
        f"- l2_error_avg_2s: `{baseline_metrics['l2_error_avg_2s']:.6f}`",
        f"- l2_error_avg_3s: `{baseline_metrics['l2_error_avg_3s']:.6f}`",
        f"- l2_error_avg_all: `{baseline_metrics['l2_error_avg_all']:.6f}`",
        "",
        "## Ranked Permutation Deltas",
        "",
        "| name | n_features | delta_all_mean | delta_all_std | delta_1s_mean | delta_2s_mean | delta_3s_mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| "
            f"{item['name']} | {item['num_features']} | "
            f"{float(item['delta_l2_error_avg_all_mean']):.6f} | {float(item['delta_l2_error_avg_all_std']):.6f} | "
            f"{float(item['delta_l2_error_avg_1s_mean']):.6f} | {float(item['delta_l2_error_avg_2s_mean']):.6f} | "
            f"{float(item['delta_l2_error_avg_3s_mean']):.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] saved_json: {json_path}", flush=True)
    print(f"[INFO] saved_markdown: {md_path}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    model_json = resolve_model_json(args)
    model_record = json.loads(model_json.read_text(encoding="utf-8"))
    if not isinstance(model_record, dict):
        raise ValueError(f"Invalid model json payload: {model_json}")
    model_payload = model_record.get("model_payload")
    if not isinstance(model_payload, dict):
        raise ValueError(f"Missing model_payload in {model_json}")

    x_val, y_val, feature_names = build_feature_matrix_for_model(
        model_record=model_record,
        v2vgot_root=args.v2vgot_root,
        val_file_name=args.val_file_name,
        limit_val=args.limit_val,
        progress_every=args.progress_every,
    )
    groups = build_groups(args.mode, feature_names)
    print(f"[INFO] analysis groups/features: {len(groups)}", flush=True)
    baseline_metrics, results = permutation_importance(
        model_payload=model_payload,
        x_val=x_val,
        y_val=y_val,
        groups=groups,
        num_shuffles=args.num_shuffles,
        seed=args.seed,
    )
    write_reports(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        model_json=model_json,
        mode=args.mode,
        baseline_metrics=baseline_metrics,
        results=results,
        feature_names=feature_names,
        num_shuffles=args.num_shuffles,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
