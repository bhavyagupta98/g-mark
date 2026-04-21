#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.candidate_track_creator import CandidateTrackCreator
from kg_coop_drive.application.candidate_track_resolver import CandidateTrackResolver
from kg_coop_drive.application.metrics_reporter import TemporalMetricsReporter
from kg_coop_drive.application.observation_associator import ObservationAssociator
from kg_coop_drive.application.processed_scene_service import ProcessedSceneEnricher
from kg_coop_drive.application.relation_builder import RelationBuilder
from kg_coop_drive.application.temporal_track_manager import TemporalTrackManager
from kg_coop_drive.application.track_quality_assessor import TrackQualityAssessor
from kg_coop_drive.application.track_lifecycle_manager import TrackLifecycleManager
from kg_coop_drive.application.track_merger import TrackMerger
from kg_coop_drive.application.track_support_enricher import TrackSupportEnricher
from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader
from kg_coop_drive.infrastructure.v2vgot_scene_adapter import V2VGoTSceneAdapter


DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def build_scene_for_timestamp(repository_root: Path, timestamp_index: int):
    adapter = V2VGoTSceneAdapter(str(repository_root))
    base_scene = adapter.load_first_scene()
    scene = replace(
        base_scene,
        local_timestamp_index=timestamp_index,
        global_timestamp_index=timestamp_index,
    )
    loader = V2VGoTProcessedAssetLoader(str(repository_root))
    availability = loader.inspect_availability(timestamp_index=timestamp_index, split_name="val")
    processed_data = loader.load_frame_scene_data(timestamp_index=timestamp_index, split_name="val")

    if processed_data is None:
        return scene, availability

    scene = ProcessedSceneEnricher().enrich(scene, processed_data)
    association_report = ObservationAssociator().associate(scene, max_distance=3.0)
    scene = TrackSupportEnricher().enrich(scene, association_report)
    scene = CandidateTrackCreator().promote(scene, association_report)
    scene, _candidate_report = CandidateTrackResolver().resolve(scene, min_candidate_confidence=0.25)
    scene, _merge_report = TrackMerger().merge(scene, max_distance=1.0)
    scene = RelationBuilder().build(scene)
    return scene, availability


def main() -> None:
    repository_root = resolve_v2vgot_root()
    temporal_metrics_reporter = TemporalMetricsReporter()
    track_lifecycle_manager = TrackLifecycleManager()
    track_quality_assessor = TrackQualityAssessor()
    frame0, availability0 = build_scene_for_timestamp(repository_root, 0)
    frame1, availability1 = build_scene_for_timestamp(repository_root, 1)

    print_section("Temporal Availability")
    print(f"Frame 0 root: {availability0.npy_root}")
    print(f"Frame 0 has_gt_boxes: {availability0.has_gt_boxes}")
    print(f"Frame 1 root: {availability1.npy_root}")
    print(f"Frame 1 has_gt_boxes: {availability1.has_gt_boxes}")

    if not availability0.has_gt_boxes or not availability1.has_gt_boxes:
        print("Both frame 0 and frame 1 processed assets are required for the temporal demo.")
        return

    updated_scene, report = TemporalTrackManager().update(
        frame0,
        frame1,
        max_distance=2.0,
        max_missed_frames=1,
    )
    updated_scene = track_lifecycle_manager.update(
        updated_scene,
        promotion_age_frames=2,
        max_supported_miss_count=1,
    )
    updated_scene = track_quality_assessor.assess(updated_scene)
    temporal_metrics = temporal_metrics_reporter.compute(updated_scene, report)

    print_section("Temporal Summary")
    print(
        f"Updated frame 1 contains {len(updated_scene.object_tracks)} tracks after carrying forward identities from frame 0."
    )

    print_section("Temporal Decisions")
    print(f"Persisted track ids: {list(report.persisted_track_ids)}")
    print(f"New track ids: {list(report.new_track_ids)}")
    print(f"Retained stale track ids: {list(report.retained_stale_track_ids)}")
    print(f"Pruned stale track ids: {list(report.pruned_stale_track_ids)}")

    print_section("Temporal Metrics")
    print(f"total_tracks: {temporal_metrics.total_tracks}")
    print(f"persisted_tracks: {temporal_metrics.persisted_tracks}")
    print(f"new_tracks: {temporal_metrics.new_tracks}")
    print(f"retained_stale_tracks: {temporal_metrics.retained_stale_tracks}")
    print(f"pruned_stale_tracks: {temporal_metrics.pruned_stale_tracks}")
    print(f"persistence_rate: {temporal_metrics.persistence_rate:.2f}")
    print(f"average_track_age: {temporal_metrics.average_track_age:.2f}")
    print(f"average_miss_count: {temporal_metrics.average_miss_count:.2f}")

    print_section("Updated Tracks")
    for object_track in updated_scene.object_tracks:
        print(
            f"- object_id={object_track.object_id}, status={object_track.status.value}, "
            f"age_frames={object_track.age_frames}, miss_count={object_track.miss_count}, "
            f"uncertainty={object_track.uncertainty_score:.2f}, "
            f"conflict={object_track.conflict_score:.2f}, "
            f"position=({object_track.position.x:.2f}, {object_track.position.y:.2f})"
        )


if __name__ == "__main__":
    main()
