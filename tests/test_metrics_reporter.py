from kg_coop_drive.application.metrics_reporter import (
    SceneMetricsReporter,
    TemporalMetricsReporter,
)
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    CrossAgentAssociationReport,
    ObservationAssociationReport,
    ObservationEvidence,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    TemporalTrackUpdateReport,
    TrackStatus,
    Trajectory,
)


def test_scene_metrics_reporter_computes_expected_counts() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(
            ObservationEvidence(
                observation_id="obs-1",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=0.8,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id="obs-2",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(20.0, 0.0),
                confidence=0.3,
                timestamp_index=0,
            ),
        ),
        object_tracks=(
            ObjectTrack(
                object_id="track-1",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_EGO"),
                    observation_ids=("gt_track-1_0", "obs-1"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
            ObjectTrack(
                object_id="track-2",
                object_type="car",
                position=Point2D(20.0, 0.0),
                confidence=0.3,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-2",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
            ),
        ),
    )
    association_report = ObservationAssociationReport(
        matches=(),
        unmatched_track_ids=(),
        unmatched_observation_ids=("obs-2",),
    )
    cross_agent_report = CrossAgentAssociationReport(
        matches=(),
        participating_agents=("CAV_EGO",),
    )

    metrics = SceneMetricsReporter().compute(scene, association_report, cross_agent_report)

    assert metrics.total_tracks == 2
    assert metrics.supported_tracks == 1
    assert metrics.candidate_tracks == 1
    assert metrics.total_observations == 2
    assert metrics.unmatched_observations == 1
    assert metrics.support_coverage == 0.5
    assert metrics.cross_agent_match_count == 0


def test_temporal_metrics_reporter_computes_expected_rates() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=(AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="track-1",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-1_0",),
                    latest_timestamp_index=1,
                ),
                status=TrackStatus.SUPPORTED,
                age_frames=2,
            ),
            ObjectTrack(
                object_id="track-2",
                object_type="car",
                position=Point2D(20.0, 0.0),
                confidence=0.3,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-2",),
                    latest_timestamp_index=1,
                ),
                status=TrackStatus.CANDIDATE,
                age_frames=1,
                miss_count=1,
            ),
        ),
    )
    report = TemporalTrackUpdateReport(
        persisted_track_ids=("track-1",),
        new_track_ids=(),
        retained_stale_track_ids=("track-2",),
        pruned_stale_track_ids=(),
    )

    metrics = TemporalMetricsReporter().compute(scene, report)

    assert metrics.total_tracks == 2
    assert metrics.persisted_tracks == 1
    assert metrics.retained_stale_tracks == 1
    assert metrics.persistence_rate == 0.5
    assert metrics.average_track_age == 1.5
    assert metrics.average_miss_count == 0.5
