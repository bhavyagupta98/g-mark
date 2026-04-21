from __future__ import annotations

from dataclasses import dataclass, field

from kg_coop_drive.domain.scene import ObjectTrack, ObservationEvidence, VisibilityFact


@dataclass(frozen=True)
class ProcessedFrameSceneData:
    """Scene content extracted from processed V2V-GoT timestamped assets."""

    timestamp_index: int
    observations: tuple[ObservationEvidence, ...]
    object_tracks: tuple[ObjectTrack, ...]
    visibility_facts: tuple[VisibilityFact, ...]
    source_paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProcessedSceneAvailability:
    """Availability summary for processed scene assets near a timestamp."""

    npy_root: str
    timestamp_index: int
    has_gt_boxes: bool
    has_gt_ids: bool
    has_visibility_for_ego: bool
    has_visibility_for_cav1: bool
    has_pred_for_ego: bool
    has_pred_for_cav1: bool
