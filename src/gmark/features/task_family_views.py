from __future__ import annotations

from typing import Any

from kg_coop_drive.domain.benchmark import BenchmarkSample
from kg_coop_drive.domain.scene import CooperativeScene

from .unified_feature_bank import UnifiedFeatureBank


def _split_feature_metadata(row: dict[str, Any], *, exclude: set[str]) -> tuple[dict[str, float], dict[str, Any]]:
    features: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    for key, value in row.items():
        if key in exclude:
            metadata[key] = value
            continue
        if isinstance(value, (int, float)):
            if key not in exclude:
                features[key] = float(value)
            continue
        metadata[key] = value
    return features, metadata


def build_object_retrieval_view(
    sample: BenchmarkSample,
    kg: CooperativeScene,
    feature_bank: UnifiedFeatureBank,
    qa_type_id: int | None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    del sample, kg, config
    rows: list[dict[str, Any]] = []
    base_meta = {"sample_id", "qa_type_id", "task_type", "asker_id", "candidate_id", "track_id"}
    for row in feature_bank.object_rows:
        features, metadata = _split_feature_metadata(row, exclude=base_meta)
        rows.append(
            {
                "sample_id": feature_bank.sample_id,
                "qa_type_id": qa_type_id,
                "family": "object_retrieval",
                "candidate_id": row.get("candidate_id", ""),
                "feature_names": list(features.keys()),
                "feature_values": list(features.values()),
                "features": features,
                "metadata": metadata,
                "model_input": features,
            }
        )
    for row in feature_bank.object_pair_rows:
        pair_meta = {"sample_id", "qa_type_id", "task_type", "asker_id", "blocker_candidate_id", "hidden_candidate_id"}
        features, metadata = _split_feature_metadata(row, exclude=pair_meta)
        rows.append(
            {
                "sample_id": feature_bank.sample_id,
                "qa_type_id": qa_type_id,
                "family": "object_retrieval",
                "candidate_id": f"{row.get('blocker_candidate_id', '')}::{row.get('hidden_candidate_id', '')}",
                "feature_names": list(features.keys()),
                "feature_values": list(features.values()),
                "features": features,
                "metadata": metadata,
                "model_input": features,
            }
        )
    return rows


def build_motion_regression_view(
    sample: BenchmarkSample,
    kg: CooperativeScene,
    feature_bank: UnifiedFeatureBank,
    qa_type_id: int | None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    del sample, kg
    features_cfg = (config or {}).get("features", {}) if isinstance((config or {}).get("features"), dict) else {}
    excluded = set(str(v) for v in features_cfg.get("exclude_leakage_fields", [])) if isinstance(features_cfg.get("exclude_leakage_fields", []), list) else set()
    meta_keys = {"sample_id", "qa_type_id", "task_type", "asker_id", "candidate_id", "track_id"}
    rows: list[dict[str, Any]] = []
    for row in feature_bank.motion_rows:
        features, metadata = _split_feature_metadata(row, exclude=meta_keys)
        for key in list(features.keys()):
            if key in excluded:
                features.pop(key, None)
        rows.append(
            {
                "sample_id": feature_bank.sample_id,
                "qa_type_id": qa_type_id,
                "family": "motion_regression",
                "candidate_id": row.get("candidate_id", ""),
                "feature_names": list(features.keys()),
                "feature_values": list(features.values()),
                "features": features,
                "metadata": metadata,
                "model_input": features,
            }
        )
    return rows


def build_scene_action_view(
    sample: BenchmarkSample,
    kg: CooperativeScene,
    feature_bank: UnifiedFeatureBank,
    qa_type_id: int | None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    del sample, kg, config
    meta_keys = {"sample_id", "qa_type_id", "task_type", "asker_id"}
    features, metadata = _split_feature_metadata(feature_bank.scene_row, exclude=meta_keys)
    return [
        {
            "sample_id": feature_bank.sample_id,
            "qa_type_id": qa_type_id,
            "family": "scene_action",
            "candidate_id": "scene",
            "feature_names": list(features.keys()),
            "feature_values": list(features.values()),
            "features": features,
            "metadata": metadata,
            "model_input": features,
        }
    ]
