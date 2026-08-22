from kg_coop_drive.application.tracking.candidate_track_creator import CandidateTrackCreator
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


def test_candidate_track_creator_promotes_unmatched_observations_without_mutating_existing_tracks() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=7,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(
            ObservationEvidence(
                observation_id="obs-match",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(10.1, 0.1),
                confidence=0.9,
                timestamp_index=7,
            ),
            ObservationEvidence(
                observation_id="obs-unmatched",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(30.0, 0.0),
                confidence=0.4,
                timestamp_index=7,
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
                    observation_ids=("gt_track-1_7",),
                    latest_timestamp_index=7,
                ),
            ),
        ),
    )

    association_report = ObservationAssociator().associate(scene, max_distance=3.0)
    enriched = CandidateTrackCreator().promote(scene, association_report)

    assert len(enriched.object_tracks) == 2
    assert enriched.object_tracks[0].object_id == "track-1"
    candidate_track = enriched.object_tracks[1]
    assert candidate_track.object_id == "pred_candidate_7_0"
    assert candidate_track.position.x == 30.0
    assert candidate_track.confidence == 0.4
    assert candidate_track.provenance.source_agent_ids == ("CAV_EGO",)
    assert candidate_track.provenance.observation_ids == ("obs-unmatched",)
    assert len(candidate_track.observations) == 1
