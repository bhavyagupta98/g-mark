from kg_coop_drive.application.tracking.candidate_track_resolver import CandidateTrackResolver
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


def test_candidate_track_resolver_prunes_only_weak_candidates() -> None:
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
                object_id="confirmed-1",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_confirmed-1_0",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CONFIRMED,
            ),
            ObjectTrack(
                object_id="candidate-keep",
                object_type="car",
                position=Point2D(20.0, 0.0),
                confidence=0.30,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs_keep",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
                observations=(
                    ObservationEvidence(
                        observation_id="obs_keep",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(20.0, 0.0),
                        confidence=0.30,
                        timestamp_index=0,
                    ),
                ),
            ),
            ObjectTrack(
                object_id="candidate-prune",
                object_type="car",
                position=Point2D(30.0, 0.0),
                confidence=0.20,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs_prune",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
                observations=(
                    ObservationEvidence(
                        observation_id="obs_prune",
                        source_agent_id="CAV_EGO",
                        object_type="car",
                        position=Point2D(30.0, 0.0),
                        confidence=0.20,
                        timestamp_index=0,
                    ),
                ),
            ),
        ),
    )

    resolved_scene, report = CandidateTrackResolver().resolve(
        scene,
        min_candidate_confidence=0.25,
    )

    assert tuple(track.object_id for track in resolved_scene.object_tracks) == (
        "confirmed-1",
        "candidate-keep",
    )
    assert report.kept_candidate_ids == ("candidate-keep",)
    assert report.pruned_candidate_ids == ("candidate-prune",)
