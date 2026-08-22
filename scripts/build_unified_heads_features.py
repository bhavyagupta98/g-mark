#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmark.features.task_family_views import (  # noqa: E402
    build_motion_regression_view,
    build_object_retrieval_view,
    build_scene_action_view,
)
from gmark.features.unified_feature_bank import build_unified_feature_bank  # noqa: E402
from kg_coop_drive.application.qa.v2vgotqa_evaluator import GraphAblationMode, V2VGoTQAPhase5AEvaluator  # noqa: E402
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

LOGGER = logging.getLogger("build_unified_heads_features")


class _NoopRouter:
    """Router placeholder for feature-only flows that never call answer()."""

    def answer(self, sample):  # pragma: no cover
        raise RuntimeError(
            "Unified-heads feature build should not call router.answer()."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified-heads feature rows from KG-prepared V2V-GoT QA samples.")
    parser.add_argument("--config", default="configs/unified_heads/default.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--baseline-mode", default="cooperative", choices=("cooperative", "ego_only"))
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required for --config YAML files. Install pyyaml.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _family_for_qa_type(qa_type_id: int | None, config: dict[str, Any]) -> str | None:
    if qa_type_id is None:
        return None
    families = config.get("families", {}) if isinstance(config.get("families"), dict) else {}
    for name in ("object_retrieval", "motion_regression", "scene_action"):
        family_cfg = families.get(name, {}) if isinstance(families.get(name), dict) else {}
        tasks = family_cfg.get("tasks", []) if isinstance(family_cfg.get("tasks"), list) else []
        if int(qa_type_id) in [int(v) for v in tasks]:
            return name
    return None


def _validate_rows(rows: list[dict[str, Any]], *, leakage_fields: set[str]) -> tuple[int, int]:
    invalid = 0
    empty = 0
    for row in rows:
        if "sample_id" not in row or "qa_type_id" not in row:
            invalid += 1
        features = row.get("model_input", {})
        metadata = row.get("metadata", {})
        if not isinstance(features, dict) or not isinstance(metadata, dict):
            invalid += 1
            continue
        if not features:
            empty += 1
        row_qa_type_id = row.get("qa_type_id")
        for key, value in features.items():
            if key in leakage_fields and int(row_qa_type_id or -1) == 19:
                invalid += 1
            if not isinstance(value, (int, float)):
                invalid += 1
    return invalid, empty


def _process_sample(
    sample,
    evaluator: V2VGoTQAPhase5AEvaluator,
    config: dict[str, Any],
    baseline_mode: str,
) -> tuple[str | None, list[dict[str, Any]], dict[str, int]]:
    prepared_scene = evaluator.prepare_sample(sample=sample, baseline_mode=baseline_mode)
    bank = build_unified_feature_bank(sample=sample, kg=prepared_scene, config=config)
    family = _family_for_qa_type(sample.qa_type_id, config)
    if family == "object_retrieval":
        rows = build_object_retrieval_view(sample, prepared_scene, bank, sample.qa_type_id, config)
    elif family == "motion_regression":
        rows = build_motion_regression_view(sample, prepared_scene, bank, sample.qa_type_id, config)
    elif family == "scene_action":
        rows = build_scene_action_view(sample, prepared_scene, bank, sample.qa_type_id, config)
    else:
        rows = []
    return family, rows, bank.missing_counts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    run_name = str(config.get("run_name", "unified_heads_v1"))

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else REPO_ROOT / "outputs" / "unified_heads" / run_name / "features" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    evaluator = V2VGoTQAPhase5AEvaluator(
        args.v2vgot_root,
        router=_NoopRouter(),
        graph_ablation=GraphAblationMode.FULL,
    )
    samples = list(adapter.load_samples(split_name=args.split, file_name=args.file_name))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    num_workers = max(1, int(args.num_workers or runtime.get("num_workers", 8)))
    log_every = max(1, int(args.log_every or runtime.get("log_every", 100)))

    LOGGER.info("split=%s samples=%d workers=%d", args.split, len(samples), num_workers)
    if num_workers > 1:
        LOGGER.warning("Using ThreadPoolExecutor for KG safety (pickling-sensitive objects).")

    family_rows: dict[str, list[dict[str, Any]]] = {
        "object_retrieval": [],
        "motion_regression": [],
        "scene_action": [],
    }
    missing_counter: Counter[str] = Counter()
    failure_rows: list[dict[str, Any]] = []
    processed = 0

    if num_workers == 1:
        for sample in samples:
            processed += 1
            try:
                family, rows, missing_counts = _process_sample(sample, evaluator, config, args.baseline_mode)
                missing_counter.update(missing_counts)
                if family is not None:
                    family_rows[family].extend(rows)
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "sample_id": sample.sample_id,
                    "qa_type_id": sample.qa_type_id,
                    "error": str(exc),
                }
                failure_rows.append(failure)
                LOGGER.exception("feature_build_failed sample_id=%s qa_type_id=%s", sample.sample_id, sample.qa_type_id)
                if args.fail_fast:
                    raise
            if processed == 1 or processed % log_every == 0 or processed == len(samples):
                LOGGER.info("progress=%d/%d sample_id=%s qa_type_id=%s", processed, len(samples), sample.sample_id, sample.qa_type_id)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_sample, sample, evaluator, config, args.baseline_mode): sample
                for sample in samples
            }
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    family, rows, missing_counts = future.result()
                    if family is not None:
                        family_rows[family].extend(rows)
                    missing_counter.update(missing_counts)
                except Exception as exc:  # noqa: BLE001
                    failure = {
                        "sample_id": sample.sample_id,
                        "qa_type_id": sample.qa_type_id,
                        "error": str(exc),
                    }
                    failure_rows.append(failure)
                    LOGGER.exception("feature_build_failed sample_id=%s qa_type_id=%s", sample.sample_id, sample.qa_type_id)
                    if args.fail_fast:
                        raise
                processed += 1
                if processed == 1 or processed % log_every == 0 or processed == len(samples):
                    LOGGER.info("progress=%d/%d sample_id=%s qa_type_id=%s", processed, len(samples), sample.sample_id, sample.qa_type_id)

    leakage_fields = set(str(v) for v in (config.get("features", {}) or {}).get("exclude_leakage_fields", []))
    invalid_total = 0
    empty_total = 0
    for family, rows in family_rows.items():
        family_invalid, family_empty = _validate_rows(rows, leakage_fields=leakage_fields)
        invalid_total += family_invalid
        empty_total += family_empty
        LOGGER.info("family=%s rows=%d invalid=%d empty=%d", family, len(rows), family_invalid, family_empty)

    out_paths = {
        "object_retrieval": output_dir / "object_retrieval_rows.jsonl",
        "motion_regression": output_dir / "motion_regression_rows.jsonl",
        "scene_action": output_dir / "scene_action_rows.jsonl",
    }
    for family, path in out_paths.items():
        with path.open("w", encoding="utf-8") as handle:
            for row in family_rows[family]:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "run_name": run_name,
        "split": args.split,
        "num_samples_loaded": len(samples),
        "num_workers": num_workers,
        "rows_per_family": {family: len(rows) for family, rows in family_rows.items()},
        "failed_samples": len(failure_rows),
        "failures": failure_rows,
        "invalid_rows": invalid_total,
        "empty_rows": empty_total,
        "top_missing_features": missing_counter.most_common(30),
        "outputs": {k: str(v) for k, v in out_paths.items()},
    }
    summary_path = output_dir / "feature_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("summary=%s", summary_path)


if __name__ == "__main__":
    main()
