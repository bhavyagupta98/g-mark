from kg_coop_drive.application.tracking.observation_associator import ObservationAssociator
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    Trajectory,
)


def test_observation_associator_matches_nearest_detection_and_leaves_rest_unmatched() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(
            ObservationEvidence(
                observation_id="obs-close",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(10.2, 0.3),
                confidence=0.8,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id="obs-far",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(25.0, 0.0),
                confidence=0.7,
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
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-1_0",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
    )

    report = ObservationAssociator().associate(scene, max_distance=3.0)

    assert len(report.matches) == 1
    assert report.matches[0].track_id == "track-1"
    assert report.matches[0].observation_id == "obs-close"
    assert report.unmatched_track_ids == ()
    assert report.unmatched_observation_ids == ("obs-far",)


def test_observation_associator_reports_unmatched_tracks_when_nothing_is_close() -> None:
    scene = CooperativeScene(
        scene_id="scene-2",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(
            ObservationEvidence(
                observation_id="obs-far",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(30.0, 0.0),
                confidence=0.6,
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
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-1_0",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
    )

    report = ObservationAssociator().associate(scene, max_distance=3.0)

    assert report.matches == ()
    assert report.unmatched_track_ids == ("track-1",)
    assert report.unmatched_observation_ids == ("obs-far",)
