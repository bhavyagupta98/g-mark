from kg_coop_drive.application.tracking.track_lifecycle_manager import TrackLifecycleManager
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


def test_track_lifecycle_manager_promotes_stable_candidate() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=(AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="candidate-1",
                object_type="car",
                position=Point2D(20.0, 0.0),
                confidence=0.4,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-1",),
                    latest_timestamp_index=1,
                ),
                status=TrackStatus.CANDIDATE,
                age_frames=2,
                miss_count=0,
                observations=(
                    ObservationEvidence(
                        observation_id="obs-1",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(20.0, 0.0),
                        confidence=0.4,
                        timestamp_index=1,
                    ),
                ),
            ),
        ),
    )

    updated_scene = TrackLifecycleManager().update(scene, promotion_age_frames=2)

    assert updated_scene.object_tracks[0].status == TrackStatus.SUPPORTED


def test_track_lifecycle_manager_downgrades_repeatedly_missed_supported_track() -> None:
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
                position=Point2D(20.0, 0.0),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_EGO"),
                    observation_ids=("gt-1", "obs-1"),
                    latest_timestamp_index=1,
                ),
                status=TrackStatus.SUPPORTED,
                miss_count=2,
            ),
        ),
    )

    updated_scene = TrackLifecycleManager().update(scene, max_supported_miss_count=1)

    assert updated_scene.object_tracks[0].status == TrackStatus.CANDIDATE
