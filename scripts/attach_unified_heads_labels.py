#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmark.features.leakage_checks import assert_no_leakage_features  # noqa: E402
from gmark.training.unified_label_builders import (  # noqa: E402
    build_motion_regression_label,
    build_object_retrieval_label,
    build_scene_action_label,
)
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter  # noqa: E402

LOGGER = logging.getLogger("attach_unified_heads_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach labels to unified-heads feature rows.")
    parser.add_argument("--config", default="configs/unified_heads/default.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--debug-labels", action="store_true")
    parser.add_argument("--debug-samples", type=int, default=5)
    parser.add_argument("--debug-qa-types", default="11,12,13,14,16,18")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required for --config YAML files. Install pyyaml.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                yield json.loads(raw)


def _key(sample_id: str, qa_type_id: int) -> str:
    return f"{sample_id}::{qa_type_id}"


def _answer_text(sample: Any) -> str:
    conversations = sample.raw_record.get("conversations", [])
    if isinstance(conversations, list) and len(conversations) > 1 and isinstance(conversations[1], dict):
        value = conversations[1].get("value", "")
        return value if isinstance(value, str) else ""
    return ""


def _label_row(
    row: dict[str, Any],
    sample_lookup: dict[str, Any],
    fail_fast: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    sample_id = str(row.get("sample_id", ""))
    qa_type_id = int(row.get("qa_type_id", -1) or -1)
    join_key = _key(sample_id, qa_type_id)
    sample = sample_lookup.get(join_key)

    debug_info = {
        "sample_id": sample_id,
        "qa_type_id": qa_type_id,
        "candidate_id": str(row.get("candidate_id", "")),
        "joined": sample is not None,
    }

    if sample is None:
        return None, {"sample_id": sample_id, "qa_type_id": qa_type_id, "error": "sample_not_found"}, debug_info

    try:
        feature_names = row.get("feature_names", [])
        if isinstance(feature_names, list):
            assert_no_leakage_features((str(v) for v in feature_names), qa_type_id=qa_type_id, strict=True)

        family = str(row.get("family", ""))
        if family == "object_retrieval":
            result = build_object_retrieval_label(sample, row)
        elif family == "motion_regression":
            result = build_motion_regression_label(sample, row)
        elif family == "scene_action":
            result = build_scene_action_label(sample, row)
        else:
            raise ValueError(f"Unknown family: {family}")

        out = dict(row)
        out["label"] = result.label_payload
        if family == "object_retrieval":
            label_value = None
            if isinstance(result.label_payload, dict):
                label_value = result.label_payload.get("label")
            out["has_label"] = label_value is not None
        else:
            out["has_label"] = not result.missing_label

        if qa_type_id in {11, 12, 13, 14, 16, 18}:
            debug_info.update(
                {
                    "raw_answer": _answer_text(sample)[:500],
                    "label": result.label_payload,
                    "has_label": out["has_label"],
                }
            )
            features = row.get("features", {})
            if isinstance(features, dict):
                x = features.get("x")
                y = features.get("y")
                if x is not None and y is not None:
                    debug_info["candidate_xy"] = [x, y]

        return out, None, debug_info
    except Exception as exc:  # noqa: BLE001
        if fail_fast:
            raise
        return None, {"sample_id": sample_id, "qa_type_id": qa_type_id, "error": str(exc)}, debug_info


def _update_stats(stats: dict[str, Any], labeled: dict[str, Any]) -> None:
    qa = int(labeled.get("qa_type_id", -1) or -1)
    label = labeled.get("label", {})
    has_label = bool(labeled.get("has_label", False))
    stats["total_rows"] += 1

    if qa in {11, 12, 13, 14, 16}:
        bucket = stats["qa_binary"][str(qa)]
        bucket["total_rows"] += 1
        if not has_label:
            bucket["missing_label"] += 1
            return
        lv = int(label.get("label", 0)) if isinstance(label, dict) else 0
        if lv == 1:
            bucket["positive"] += 1
            bucket["samples_with_at_least_one_positive"].add(str(labeled.get("sample_id", "")))
        else:
            bucket["negative"] += 1

        if qa in {11, 12, 13, 14} and isinstance(label, dict):
            if int(label.get("parsed_reference_count", 0)) > 0:
                bucket["parsed_reference_samples"].add(str(labeled.get("sample_id", "")))
            else:
                bucket["unparsed_reference_samples"].add(str(labeled.get("sample_id", "")))
            dist = label.get("min_match_distance")
            if isinstance(dist, (float, int)):
                bucket["distance_values"].append(float(dist))
        return

    if qa == 18:
        bucket = stats["q8"]
        bucket["total_rows"] += 1
        if not has_label or not isinstance(label, dict):
            bucket["missing_label"] += 1
            return
        speed_id = label.get("speed_label_id", label.get("speed_class"))
        steering_id = label.get("steering_label_id", label.get("steering_class"))
        if isinstance(speed_id, int):
            bucket["speed_label_counts"][str(speed_id)] += 1
        else:
            bucket["missing_speed_label"] += 1
        if isinstance(steering_id, int):
            bucket["steering_label_counts"][str(steering_id)] += 1
        else:
            bucket["missing_steering_label"] += 1
        return

    if qa in {15, 17, 19}:
        bucket = stats["qa_regression"][str(qa)]
        bucket["total_rows"] += 1
        if not has_label or not isinstance(label, dict):
            bucket["missing_target"] += 1
            return
        target = label.get("target")
        if isinstance(target, list) and len(target) > 0:
            bucket["valid_target"] += 1
            bucket["target_dim"] = int(label.get("target_dim", len(target)))
        else:
            bucket["missing_target"] += 1


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    qa_binary: dict[str, Any] = {}
    for qa, bucket in stats["qa_binary"].items():
        dvals = bucket.pop("distance_values")
        parsed = bucket.pop("parsed_reference_samples")
        unparsed = bucket.pop("unparsed_reference_samples")
        samples_pos = bucket.pop("samples_with_at_least_one_positive")
        out = dict(bucket)
        out["parsed_reference_samples"] = len(parsed)
        out["unparsed_reference_samples"] = len(unparsed)
        out["samples_with_at_least_one_positive"] = len(samples_pos)
        if dvals:
            out["min_match_distance_stats"] = {
                "count": len(dvals),
                "min": min(dvals),
                "p50": sorted(dvals)[len(dvals) // 2],
                "max": max(dvals),
            }
        qa_binary[qa] = out

    return {
        "total_rows": stats["total_rows"],
        "qa_binary": qa_binary,
        "q8": {
            "total_rows": stats["q8"]["total_rows"],
            "speed_label_counts": dict(stats["q8"]["speed_label_counts"]),
            "steering_label_counts": dict(stats["q8"]["steering_label_counts"]),
            "missing_speed_label": stats["q8"]["missing_speed_label"],
            "missing_steering_label": stats["q8"]["missing_steering_label"],
            "missing_label": stats["q8"]["missing_label"],
        },
        "qa_regression": dict(stats["qa_regression"]),
    }


def _process_family(
    *,
    input_path: Path,
    output_path: Path,
    sample_lookup: dict[str, Any],
    num_workers: int,
    fail_fast: bool,
    log_every: int,
    debug_enabled: bool,
    debug_samples: int,
    debug_qa_types: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int, int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    q9_leakage_checked = 0
    rows_in = 0
    rows_out = 0
    rows_join_missing = 0
    processed = 0
    debug_rows: list[dict[str, Any]] = []

    stats = {
        "total_rows": 0,
        "qa_binary": defaultdict(
            lambda: {
                "positive": 0,
                "negative": 0,
                "missing_label": 0,
                "total_rows": 0,
                "parsed_reference_samples": set(),
                "unparsed_reference_samples": set(),
                "samples_with_at_least_one_positive": set(),
                "distance_values": [],
            }
        ),
        "q8": {
            "total_rows": 0,
            "speed_label_counts": Counter(),
            "steering_label_counts": Counter(),
            "missing_speed_label": 0,
            "missing_steering_label": 0,
            "missing_label": 0,
        },
        "qa_regression": defaultdict(lambda: {"total_rows": 0, "valid_target": 0, "missing_target": 0, "target_dim": 0}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("reading_family_file=%s", input_path)

    def _handle_result(labeled: dict[str, Any] | None, failure: dict[str, Any] | None, debug_info: dict[str, Any], out_handle: Any) -> None:
        nonlocal rows_out, q9_leakage_checked, rows_join_missing, processed
        if labeled is not None:
            out_handle.write(json.dumps(labeled, ensure_ascii=True) + "\n")
            rows_out += 1
            qa = int(labeled.get("qa_type_id", -1) or -1)
            if qa == 19:
                q9_leakage_checked += 1
            _update_stats(stats, labeled)
            if debug_enabled and qa in debug_qa_types and len([d for d in debug_rows if d.get("qa_type_id") == qa]) < debug_samples:
                debug_rows.append(debug_info)
        if failure is not None:
            failures.append(failure)
            if failure.get("error") == "sample_not_found":
                rows_join_missing += 1
        processed += 1
        if processed == 1 or processed % log_every == 0:
            LOGGER.info("label_progress=%d rows_in=%d rows_out=%d join_missing=%d", processed, rows_in, rows_out, rows_join_missing)

    if num_workers == 1:
        with output_path.open("w", encoding="utf-8") as out_handle:
            for row in _iter_jsonl(input_path):
                rows_in += 1
                labeled, failure, debug_info = _label_row(row, sample_lookup, fail_fast)
                _handle_result(labeled, failure, debug_info, out_handle)
    else:
        inflight_limit = max(32, num_workers * 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with output_path.open("w", encoding="utf-8") as out_handle:
                pending = set()
                for row in _iter_jsonl(input_path):
                    rows_in += 1
                    pending.add(executor.submit(_label_row, row, sample_lookup, fail_fast))
                    if len(pending) >= inflight_limit:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            labeled, failure, debug_info = future.result()
                            _handle_result(labeled, failure, debug_info, out_handle)
                if pending:
                    done, _ = wait(pending)
                    for future in done:
                        labeled, failure, debug_info = future.result()
                        _handle_result(labeled, failure, debug_info, out_handle)

    return _finalize_stats(stats), failures, rows_in, q9_leakage_checked, rows_join_missing, debug_rows


def _parse_debug_qa_types(raw: str) -> set[int]:
    out: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(Path(args.config).expanduser().resolve())
    run_name = str(config.get("run_name", "unified_heads_v1"))
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    num_workers = max(1, int(args.num_workers or runtime.get("num_workers", 8)))
    log_every = max(1, int(args.log_every or runtime.get("log_every", 100)))
    debug_qa_types = _parse_debug_qa_types(args.debug_qa_types)

    feature_dir = Path(args.feature_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("split=%s feature_dir=%s output_dir=%s workers=%d", args.split, feature_dir, output_dir, num_workers)

    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    samples = list(adapter.load_samples(split_name=args.split, file_name=args.file_name))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    sample_lookup: dict[str, Any] = {}
    duplicate_keys = 0
    for sample in samples:
        key = _key(str(sample.sample_id), int(sample.qa_type_id or -1))
        if key in sample_lookup:
            duplicate_keys += 1
        sample_lookup[key] = sample

    LOGGER.info(
        "loaded_qa_records=%d lookup_keys=%d duplicate_keys=%d",
        len(samples),
        len(sample_lookup),
        duplicate_keys,
    )

    input_paths = {
        "object_retrieval": feature_dir / "object_retrieval_rows.jsonl",
        "motion_regression": feature_dir / "motion_regression_rows.jsonl",
        "scene_action": feature_dir / "scene_action_rows.jsonl",
    }
    output_paths = {
        "object_retrieval": output_dir / "object_retrieval_labeled.jsonl",
        "motion_regression": output_dir / "motion_regression_labeled.jsonl",
        "scene_action": output_dir / "scene_action_labeled.jsonl",
    }

    all_failures: list[dict[str, Any]] = []
    q9_leakage_checked = 0
    rows_read_per_family: dict[str, int] = {}
    rows_written_per_family: dict[str, int] = {}
    join_missing_rows = 0
    family_summaries: dict[str, Any] = {}
    debug_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for family in ("object_retrieval", "motion_regression", "scene_action"):
        per_family_summary, failures, rows_in, q9_rows, joins_miss, debug_rows = _process_family(
            input_path=input_paths[family],
            output_path=output_paths[family],
            sample_lookup=sample_lookup,
            num_workers=num_workers,
            fail_fast=args.fail_fast,
            log_every=log_every,
            debug_enabled=bool(args.debug_labels),
            debug_samples=max(1, int(args.debug_samples)),
            debug_qa_types=debug_qa_types,
        )
        q9_leakage_checked += q9_rows
        join_missing_rows += joins_miss
        rows_read_per_family[family] = rows_in
        rows_written_per_family[family] = (
            sum(1 for _ in output_paths[family].open("r", encoding="utf-8"))
            if output_paths[family].exists()
            else 0
        )
        family_summaries[family] = per_family_summary
        all_failures.extend(failures)
        LOGGER.info(
            "family=%s rows_in=%d rows_out=%d failures=%d join_missing=%d",
            family,
            rows_in,
            rows_written_per_family[family],
            len(failures),
            joins_miss,
        )
        for row in debug_rows:
            qa_key = str(row.get("qa_type_id", -1))
            if len(debug_examples[qa_key]) < max(1, int(args.debug_samples)):
                debug_examples[qa_key].append(row)

    summary = {
        "run_name": run_name,
        "split": args.split,
        "num_samples_loaded": len(samples),
        "num_workers": num_workers,
        "rows_read_per_family": rows_read_per_family,
        "rows_written_per_family": rows_written_per_family,
        "family_label_stats": family_summaries,
        "join_missing_rows": join_missing_rows,
        "failed_rows": len(all_failures),
        "failures": all_failures[:500],
        "q9_leakage_checks_passed_rows": q9_leakage_checked,
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }

    if args.debug_labels:
        summary["debug_examples"] = {k: v for k, v in sorted(debug_examples.items(), key=lambda x: int(x[0]))}
        for qa, examples in summary["debug_examples"].items():
            LOGGER.info("debug_qa_type=%s examples=%s", qa, json.dumps(examples[: args.debug_samples], ensure_ascii=True)[:3000])

    summary_path = output_dir / "label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("label_summary=%s", summary_path)


if __name__ == "__main__":
    main()
