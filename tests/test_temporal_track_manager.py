import pytest

from kg_coop_drive.application.temporal_track_manager import TemporalTrackManager
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    TrackStatus,
    Trajectory,
)


def test_temporal_track_manager_persists_close_track_identity_and_reports_new_and_stale() -> None:
    previous_scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="track-persist",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-persist_0",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                age_frames=2,
                observations=(
                    ObservationEvidence(
                        observation_id="obs_prev",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(10.1, 0.1),
                        confidence=0.9,
                        timestamp_index=0,
                    ),
                ),
            ),
            ObjectTrack(
                object_id="track-stale",
                object_type="car",
                position=Point2D(40.0, 0.0),
                confidence=0.8,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-stale_0",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CONFIRMED,
            ),
        ),
    )

    current_scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=previous_scene.agents,
        future_trajectory=previous_scene.future_trajectory,
        object_tracks=(
            ObjectTrack(
                object_id="current-temp",
                object_type="car",
                position=Point2D(10.8, 0.2),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs_curr",),
                    latest_timestamp_index=1,
                ),
                status=TrackStatus.SUPPORTED,
                observations=(
                    ObservationEvidence(
                        observation_id="obs_curr",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(10.8, 0.2),
                        confidence=0.95,
                        timestamp_index=1,
                    ),
                ),
            ),
            ObjectTrack(
                object_id="new-track",
                object_type="car",
                position=Point2D(70.0, 0.0),
                confidence=0.4,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs_new",),
                    latest_timestamp_index=1,
                ),
                status=TrackStatus.CANDIDATE,
            ),
        ),
    )

    updated_scene, report = TemporalTrackManager().update(
        previous_scene,
        current_scene,
        max_distance=2.0,
        max_missed_frames=1,
    )

    assert tuple(track.object_id for track in updated_scene.object_tracks) == (
        "track-persist",
        "new-track",
        "track-stale",
    )
    persisted_track = updated_scene.object_tracks[0]
    assert persisted_track.age_frames == 3
    assert persisted_track.velocity is not None
    assert persisted_track.velocity.x == pytest.approx(0.8)
    assert persisted_track.velocity.y == pytest.approx(0.2)
    assert len(persisted_track.observations) == 2
    retained_stale_track = updated_scene.object_tracks[2]
    assert retained_stale_track.object_id == "track-stale"
    assert retained_stale_track.miss_count == 1
    assert report.persisted_track_ids == ("track-persist",)
    assert report.new_track_ids == ("new-track",)
    assert report.retained_stale_track_ids == ("track-stale",)
    assert report.pruned_stale_track_ids == ()


def test_temporal_track_manager_prunes_stale_track_after_miss_budget_is_exceeded() -> None:
    previous_scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="track-stale",
                object_type="car",
                position=Point2D(40.0, 0.0),
                confidence=0.8,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-stale_1",),
                    latest_timestamp_index=1,
                ),
                status=TrackStatus.CONFIRMED,
                miss_count=1,
            ),
        ),
    )
    current_scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=2,
        global_timestamp_index=2,
        asker_agent_id="CAV_EGO",
        agents=previous_scene.agents,
        future_trajectory=previous_scene.future_trajectory,
        object_tracks=(),
    )

    updated_scene, report = TemporalTrackManager().update(
        previous_scene,
        current_scene,
        max_distance=2.0,
        max_missed_frames=1,
    )

    assert updated_scene.object_tracks == ()
    assert report.persisted_track_ids == ()
    assert report.new_track_ids == ()
    assert report.retained_stale_track_ids == ()
    assert report.pruned_stale_track_ids == ("track-stale",)
