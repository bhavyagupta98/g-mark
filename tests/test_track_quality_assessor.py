from kg_coop_drive.application.track_quality_assessor import TrackQualityAssessor
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


def test_track_quality_assessor_computes_support_confidence_and_conflict() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="track-1",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_EGO"),
                    observation_ids=("gt-1", "obs-1"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                observations=(
                    ObservationEvidence(
                        observation_id="obs-1",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(10.2, 0.1),
                        confidence=0.8,
                        timestamp_index=0,
                    ),
                ),
            ),
        ),
    )

    updated_scene = TrackQualityAssessor().assess(scene)
    track = updated_scene.object_tracks[0]

    assert track.last_support_confidence == 0.8
    assert track.conflict_score > 0.0
    assert 0.0 <= track.uncertainty_score <= 1.0
