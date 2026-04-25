from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kg_coop_drive.domain.scene import CooperativeScene


class BenchmarkTaskType(str, Enum):
    """Stable task categories for benchmark-level routing and evaluation."""

    VISIBLE_OBJECTS = "visible_objects"
    OCCLUDING_OBJECTS = "occluding_objects"
    INVISIBLE_OBJECTS = "invisible_objects"
    NOTABLE_OBJECTS = "notable_objects"
    PLANNING_AWARENESS = "planning_awareness"
    OBJECT_MOTION_PREDICTION = "object_motion_prediction"
    AGENT_MOTION_PREDICTION = "agent_motion_prediction"
    CONTROL_SETTINGS = "control_settings"
    FUTURE_TRAJECTORY = "future_trajectory"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BenchmarkSample:
    """One benchmark sample enriched with a canonical scene and task metadata."""

    sample_id: str
    dataset_name: str
    split_name: str
    file_name: str
    task_type: BenchmarkTaskType
    scene: CooperativeScene
    raw_record: dict[str, object]
    qa_type_id: int | None = None


@dataclass(frozen=True)
class BenchmarkPrediction:
    """One serialized benchmark prediction for evaluation and debugging."""

    sample_id: str
    dataset_name: str
    split_name: str
    task_type: BenchmarkTaskType
    qa_type_id: int | None
    supported: bool
    answer_text: str
    object_ids: tuple[str, ...]
    baseline_mode: str


@dataclass(frozen=True)
class BenchmarkEvaluationSummary:
    """Aggregated counters for one evaluation run."""

    dataset_name: str
    split_name: str
    baseline_mode: str
    total_samples: int
    evaluated_samples: int
    supported_predictions: int
    unsupported_predictions: int
