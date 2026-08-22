#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmark.models.unified_trainers import (  # noqa: E402
    train_elasticnet_motion_heads,
    train_gbdt_scene_action_heads,
    train_shared_object_retrieval_logreg,
)

LOGGER = logging.getLogger("train_unified_heads_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified_heads Stage 3 family models.")
    parser.add_argument("--config", default="configs/unified_heads/default.yaml")
    parser.add_argument("--labeled-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", choices=("object_retrieval", "motion_regression", "scene_action"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required for --config YAML files. Install pyyaml.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.flush()
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    num_workers = max(1, int(args.num_workers or runtime.get("num_workers", 8)))
    log_every = max(1, int(args.log_every or runtime.get("log_every", 100)))

    labeled_dir = Path(args.labeled_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    families = [args.family] if args.family else ["object_retrieval", "motion_regression", "scene_action"]
    # Run families sequentially by default to avoid parallel peak-memory spikes
    # when loading very large labeled JSONL files (notably object_retrieval).
    execution_mode = "sequential"

    LOGGER.info(
        "config=%s labeled_dir=%s output_dir=%s families=%s workers=%d execution_mode=%s log_every=%d seed=42",
        config_path,
        labeled_dir,
        output_dir,
        families,
        num_workers,
        execution_mode,
        log_every,
    )

    model_cfg = config.get("models", {}) if isinstance(config.get("models"), dict) else {}
    tasks: list[tuple[str, Any]] = []

    if "object_retrieval" in families:
        tasks.append((
            "object_retrieval",
            lambda: train_shared_object_retrieval_logreg(
                rows_path=labeled_dir / "object_retrieval_labeled.jsonl",
                model_cfg=model_cfg.get("object_retrieval", {}),
                run_cfg=config,
                output_path=output_dir / "object_retrieval" / "object_retrieval_logreg_shared_q1_q4.json",
                max_rows=max(0, int(args.max_rows)),
                overwrite=bool(args.overwrite),
                log_every=log_every,
            ),
        ))

    if "motion_regression" in families:
        tasks.append((
            "motion_regression",
            lambda: train_elasticnet_motion_heads(
                rows_path=labeled_dir / "motion_regression_labeled.jsonl",
                model_cfg=model_cfg.get("motion_regression", {}),
                run_cfg=config,
                output_dir=output_dir / "motion_regression",
                max_rows=max(0, int(args.max_rows)),
                overwrite=bool(args.overwrite),
                log_every=log_every,
            ),
        ))

    if "scene_action" in families:
        tasks.append((
            "scene_action",
            lambda: train_gbdt_scene_action_heads(
                rows_path=labeled_dir / "scene_action_labeled.jsonl",
                model_cfg=model_cfg.get("scene_action", {}),
                run_cfg=config,
                output_dir=output_dir / "scene_action",
                max_rows=max(0, int(args.max_rows)),
                overwrite=bool(args.overwrite),
                log_every=log_every,
            ),
        ))

    started = time.time()
    summaries: dict[str, Any] = {"artifacts": [], "metrics": {}, "failures": []}
    per_family_runtime: dict[str, float] = {}

    def _run_family(name: str, fn: Any) -> tuple[str, Any, float]:
        t0 = time.time()
        out = fn()
        return name, out, time.time() - t0

    for name, fn in tasks:
        try:
            fam_name, out, dt = _run_family(name, fn)
            per_family_runtime[fam_name] = dt
            _consume_family_result(fam_name, out, summaries)
        except Exception as exc:  # noqa: BLE001
            if str(exc).startswith("SKIP_ARTIFACT_EXISTS:"):
                LOGGER.warning("%s", str(exc))
                continue
            LOGGER.exception("family training failed: %s", name)
            summaries["failures"].append({"family": name, "error": str(exc)})
            if args.fail_fast:
                raise

    report = {
        "run_name": str(config.get("run_name", "unified_heads_v1")),
        "labeled_dir": str(labeled_dir),
        "output_dir": str(output_dir),
        "families": families,
        "num_workers": num_workers,
        "execution_mode": execution_mode,
        "max_rows": int(args.max_rows),
        "artifacts": summaries["artifacts"],
        "metrics": summaries["metrics"],
        "failures": summaries["failures"],
        "runtime_sec": round(time.time() - started, 3),
        "per_family_runtime_sec": {k: round(v, 3) for k, v in per_family_runtime.items()},
        "assumptions": [
            "Stage-3 trains only from labeled train rows.",
            "Val split is not used for training/threshold tuning.",
            "Object retrieval trains one shared model with task one-hot features.",
        ],
    }
    summary_path = output_dir / "train_unified_heads_summary.json"
    _atomic_write_json(summary_path, report)

    LOGGER.info("artifacts_produced=%d failures=%d summary=%s", len(summaries["artifacts"]), len(summaries["failures"]), summary_path)
    LOGGER.info("total_runtime_sec=%.3f per_family_runtime=%s", time.time() - started, {k: round(v, 2) for k, v in per_family_runtime.items()})


def _consume_family_result(name: str, out: Any, summaries: dict[str, Any]) -> None:
    if out is None:
        LOGGER.warning("family=%s produced no artifact", name)
        return

    if isinstance(out, list):
        for item in out:
            summaries["artifacts"].append(item.artifact_relpath)
            summaries["metrics"][item.head_name] = item.metrics
            size = Path(item.artifact_relpath).stat().st_size if Path(item.artifact_relpath).exists() else -1
            LOGGER.info("artifact_written family=%s head=%s path=%s size_bytes=%d", name, item.head_name, item.artifact_relpath, size)
        return

    summaries["artifacts"].append(out.artifact_relpath)
    summaries["metrics"][out.head_name] = out.metrics
    size = Path(out.artifact_relpath).stat().st_size if Path(out.artifact_relpath).exists() else -1
    LOGGER.info("artifact_written family=%s head=%s path=%s size_bytes=%d", name, out.head_name, out.artifact_relpath, size)


if __name__ == "__main__":
    main()
