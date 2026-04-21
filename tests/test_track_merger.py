from kg_coop_drive.application.track_merger import TrackMerger
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


def test_track_merger_merges_close_candidate_into_supported_track() -> None:
    scene = CooperativeScene(
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
                object_id="track-1",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_EGO"),
                    observation_ids=("gt_track-1_0", "obs_support"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                observations=(
                    ObservationEvidence(
                        observation_id="obs_support",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(10.1, 0.1),
                        confidence=0.8,
                        timestamp_index=0,
                    ),
                ),
            ),
            ObjectTrack(
                object_id="candidate-near",
                object_type="car",
                position=Point2D(10.6, 0.3),
                confidence=0.4,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs_candidate",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
                observations=(
                    ObservationEvidence(
                        observation_id="obs_candidate",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(10.6, 0.3),
                        confidence=0.4,
                        timestamp_index=0,
                    ),
                ),
            ),
            ObjectTrack(
                object_id="candidate-far",
                object_type="car",
                position=Point2D(40.0, 0.0),
                confidence=0.4,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs_far",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
                observations=(
                    ObservationEvidence(
                        observation_id="obs_far",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(40.0, 0.0),
                        confidence=0.4,
                        timestamp_index=0,
                    ),
                ),
            ),
        ),
    )

    merged_scene, report = TrackMerger().merge(scene, max_distance=1.0)

    assert len(report.merges) == 1
    assert report.merges[0].source_track_id == "candidate-near"
    assert report.merges[0].target_track_id == "track-1"
    assert report.remaining_candidate_ids == ("candidate-far",)
    assert tuple(track.object_id for track in merged_scene.object_tracks) == (
        "track-1",
        "candidate-far",
    )
    merged_track = merged_scene.object_tracks[0]
    assert len(merged_track.observations) == 2
    assert merged_track.provenance.source_agent_ids == ("GT", "CAV_EGO")
