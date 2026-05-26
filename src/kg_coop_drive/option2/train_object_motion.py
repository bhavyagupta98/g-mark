from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kg_coop_drive.option2.dataset import ObjectMotionDatasetBuilder
from kg_coop_drive.option2.models import build_regression_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Option2 isolated object-motion trainer (Q5/Q7 family). "
            "Trains on train split and validates on val split with strict split guards."
        )
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--train-split", default="train", choices=("train", "val"))
    parser.add_argument("--val-split", default="val", choices=("train", "val"))
    parser.add_argument("--qa-type-ids", default="15,17", help="Comma-separated QA type ids, default merges Q5/Q7.")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--graph-ablation-mode", default="full")
    parser.add_argument("--max-match-distance", type=float, default=2.0)
    parser.add_argument("--include-kg-features", action="store_true")
    parser.add_argument("--backend", default="auto", choices=("auto", "lightgbm", "sklearn_gbdt"))
    parser.add_argument("--n-estimators", type=int, default=280)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=96)
    parser.add_argument("--subsample", type=float, default=0.7)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--fail-on-sample-overlap",
        action="store_true",
        help=(
            "Optional strict mode. If set, fail when train/val sample_ids overlap. "
            "Default behavior logs overlap but does not fail."
        ),
    )
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    return parser


def _as_arrays(rows) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.asarray([list(row.feature_values) for row in rows], dtype=float)
    y = np.asarray([[row.target_dx, row.target_dy] for row in rows], dtype=float)
    sample_ids = [row.sample_id for row in rows]
    return x, y, sample_ids


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    endpoint_l2 = np.linalg.norm(err, axis=1)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    return {
        "mae": mae,
        "rmse": rmse,
        "endpoint_l2_avg": float(np.mean(endpoint_l2)),
        "endpoint_l2_p90": float(np.quantile(endpoint_l2, 0.9)),
    }


def main() -> None:
    args = build_parser().parse_args()
    qa_type_ids = tuple(int(item.strip()) for item in args.qa_type_ids.split(",") if item.strip())
    if not qa_type_ids:
        raise SystemExit("No qa_type_ids resolved from --qa-type-ids")

    print("=" * 72)
    print("Option2 Object Motion Training")
    print("=" * 72)
    print(f"train_split: {args.train_split}")
    print(f"val_split: {args.val_split}")
    print(f"qa_type_ids: {qa_type_ids}")
    print(f"include_kg_features: {args.include_kg_features}")

    builder = ObjectMotionDatasetBuilder(
        v2vgot_root=args.v2vgot_root,
        file_name=args.file_name,
        baseline_mode=args.baseline_mode,
        graph_ablation_mode=args.graph_ablation_mode,
        max_match_distance=args.max_match_distance,
        include_kg_features=args.include_kg_features,
    )

    print("[1/4] Building train dataset...")
    train_rows, train_summary = builder.build(
        split_name=args.train_split,
        qa_type_ids=qa_type_ids,
        limit=int(args.train_limit),
        progress_every=int(args.progress_every),
    )
    print("[2/4] Building val dataset...")
    val_rows, val_summary = builder.build(
        split_name=args.val_split,
        qa_type_ids=qa_type_ids,
        limit=int(args.val_limit),
        progress_every=int(args.progress_every),
    )
    if not train_rows:
        raise SystemExit("No train rows were built. Check split/data/matching threshold.")
    if not val_rows:
        raise SystemExit("No val rows were built. Check split/data/matching threshold.")

    x_train, y_train, train_sample_ids = _as_arrays(train_rows)
    x_val, y_val, val_sample_ids = _as_arrays(val_rows)

    overlap = sorted(set(train_sample_ids).intersection(val_sample_ids))
    overlap_count = len(overlap)
    overlap_preview = overlap[:10]
    if overlap and args.fail_on_sample_overlap:
        raise SystemExit(
            "Strict overlap guard triggered: overlapping sample_id across train/val: "
            f"count={overlap_count} preview={overlap_preview}"
        )
    if overlap:
        print(
            "split_overlap_notice: overlapping sample_id across train/val "
            f"count={overlap_count} preview={overlap_preview}"
        )

    print("[3/4] Training backend...")
    backend = build_regression_backend(
        backend=args.backend,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        subsample=float(args.subsample),
        num_leaves=int(args.num_leaves),
    )
    backend.fit(x_train, y_train)

    print("[4/4] Evaluating and writing artifacts...")
    train_pred = backend.predict(x_train)
    val_pred = backend.predict(x_val)

    report = {
        "module": "option2_object_motion",
        "task_scope": "object_motion_prediction_q5_q7",
        "qa_type_ids": list(qa_type_ids),
        "feature_set": "flat_plus_kg" if args.include_kg_features else "flat_only",
        "feature_names": list(builder.feature_names),
        "train_summary": train_summary.__dict__,
        "val_summary": val_summary.__dict__,
        "train_metrics": _metric_block(y_train, train_pred),
        "val_metrics": _metric_block(y_val, val_pred),
        "backend": backend.export_metadata(),
        "guards": {
            "train_val_sample_overlap_count": overlap_count,
            "train_val_sample_overlap_preview": overlap_preview,
            "fail_on_sample_overlap": bool(args.fail_on_sample_overlap),
            "train_split_name": args.train_split,
            "val_split_name": args.val_split,
            "feature_source": "scene_only_no_result_metadata",
        },
    }

    model_payload = {
        "model_type": "option2_object_motion_regression_backend_v1",
        "task_scope": "object_motion_prediction_q5_q7",
        "qa_type_ids": list(qa_type_ids),
        "feature_set": "flat_plus_kg" if args.include_kg_features else "flat_only",
        "feature_names": list(builder.feature_names),
        "backend": backend.export_metadata(),
        "note": (
            "This artifact captures training config and metrics. "
            "For deployment wiring, add backend-specific serialization in next step."
        ),
    }

    output_model = Path(args.output_model_json).expanduser().resolve()
    output_report = Path(args.output_report_json).expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"saved_model: {output_model}")
    print(f"saved_report: {output_report}")


if __name__ == "__main__":
    main()
