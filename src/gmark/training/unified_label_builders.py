from __future__ import annotations

import re
from dataclasses import dataclass
from math import dist
from typing import Any

from gmark.features.leakage_checks import assert_no_leakage_features
from kg_coop_drive.application.planning.control_settings_policy import (
    SPEED_CLASSES,
    STEERING_CLASSES,
    parse_speed_steering_idx,
)
from kg_coop_drive.domain.benchmark import BenchmarkSample

COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
Q6_NOTABLE_RE = re.compile(r"\bnotable\b", re.IGNORECASE)
Q6_NOT_NOTABLE_RE = re.compile(r"\bnot notable\b|\bnon[- ]notable\b", re.IGNORECASE)
Q6_YES_RE = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
Q6_NO_RE = re.compile(r"\b(no|false)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LabelBuildResult:
    label_payload: dict[str, Any]
    missing_label: bool = False


def _answer_text(sample: BenchmarkSample) -> str:
    conversations = sample.raw_record.get("conversations", [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        return ""
    second = conversations[1]
    if not isinstance(second, dict):
        return ""
    value = second.get("value", "")
    return value if isinstance(value, str) else ""


def _parse_coords(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORD_RE.findall(text)]


def _sample_track_lookup(sample: BenchmarkSample) -> dict[str, tuple[float, float]]:
    return {
        str(track.object_id): (float(track.position.x), float(track.position.y))
        for track in sample.scene.object_tracks
    }


def _candidate_center(sample: BenchmarkSample, row: dict[str, Any]) -> tuple[float, float] | None:
    features = row.get("features", {})
    if isinstance(features, dict):
        try:
            return float(features["x"]), float(features["y"])
        except (KeyError, TypeError, ValueError):
            pass
    candidate_id = str(row.get("candidate_id", "")).strip()
    if not candidate_id or candidate_id == "scene":
        return None
    candidate_id = candidate_id.split("::", 1)[0]
    lookup = _sample_track_lookup(sample)
    return lookup.get(candidate_id)


def build_object_retrieval_label(
    sample: BenchmarkSample,
    row: dict[str, Any],
    *,
    localization_threshold_m: float = 0.5,
) -> LabelBuildResult:
    answer_text = _answer_text(sample)
    gt_coords = _parse_coords(answer_text)
    center = _candidate_center(sample, row)
    if center is None:
        return LabelBuildResult(
            {
                "label_type": "binary",
                "label": 0,
                "matched_gt_count": 0,
                "min_match_distance": None,
                "match_threshold": float(localization_threshold_m),
                "parsed_reference_count": len(gt_coords),
            },
            missing_label=True,
        )

    if not gt_coords:
        return LabelBuildResult(
            {
                "label_type": "binary",
                "label": 0,
                "matched_gt_count": 0,
                "min_match_distance": None,
                "match_threshold": float(localization_threshold_m),
                "parsed_reference_count": 0,
            },
            missing_label=True,
        )
    distances = [dist(center, (gx, gy)) for gx, gy in gt_coords]
    min_distance = min(distances) if distances else None
    matched = sum(1 for d in distances if d <= localization_threshold_m)
    return LabelBuildResult(
        {
            "label_type": "binary",
            "label": 1 if matched > 0 else 0,
            "matched_gt_count": matched,
            "min_match_distance": float(min_distance) if min_distance is not None else None,
            "match_threshold": float(localization_threshold_m),
            "parsed_reference_count": len(gt_coords),
        },
    )


def _flatten_waypoints(answer_text: str, waypoint_count: int = 6) -> list[float] | None:
    points = _parse_coords(answer_text)[:waypoint_count]
    if not points:
        return None
    if len(points) < waypoint_count:
        points = points + [points[-1]] * (waypoint_count - len(points))
    return [float(v) for xy in points for v in xy]


def build_motion_regression_label(sample: BenchmarkSample, row: dict[str, Any]) -> LabelBuildResult:
    qa_type_id = int(sample.qa_type_id or -1)
    feature_names = row.get("feature_names", [])
    if isinstance(feature_names, list):
        assert_no_leakage_features((str(x) for x in feature_names), qa_type_id=qa_type_id, strict=True)

    target = _flatten_waypoints(_answer_text(sample), waypoint_count=6)
    if target is None:
        return LabelBuildResult(
            {"target": [], "target_dim": 0, "label_type": "regression"},
            missing_label=True,
        )
    return LabelBuildResult(
        {
            "target": target,
            "target_dim": len(target),
            "waypoint_count": 6,
            "label_type": "regression",
        }
    )


def build_scene_action_label(sample: BenchmarkSample, _row: dict[str, Any]) -> LabelBuildResult:
    qa_type_id = int(sample.qa_type_id or -1)
    answer = _answer_text(sample)
    if qa_type_id == 16:
        lower = answer.strip().lower()
        if Q6_NOT_NOTABLE_RE.search(lower):
            label = 0
        elif Q6_NOTABLE_RE.search(lower):
            label = 1
        elif Q6_NO_RE.search(lower):
            label = 0
        elif Q6_YES_RE.search(lower):
            label = 1
        else:
            return LabelBuildResult({"label": 0, "label_type": "binary", "raw_answer": answer}, missing_label=True)
        return LabelBuildResult(
            {
                "label_type": "binary",
                "label": int(label),
                "raw_answer": answer,
                "parsed_from": "reference_answer",
            }
        )
    if qa_type_id == 18:
        speed_idx, steering_idx = parse_speed_steering_idx(answer)
        return LabelBuildResult(
            {
                "label_type": "control",
                "speed_label": SPEED_CLASSES[int(speed_idx)],
                "steering_label": STEERING_CLASSES[int(steering_idx)],
                "speed_label_id": int(speed_idx),
                "steering_label_id": int(steering_idx),
                "speed_class": int(speed_idx),
                "steering_class": int(steering_idx),
                "raw_answer": answer,
                "parsed_from": "reference_answer",
            }
        )
    return LabelBuildResult({"label_type": "unknown"}, missing_label=True)
