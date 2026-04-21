from kg_coop_drive.application.visibility_reasoner import VisibilityReasoner
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
    VisibilityFact,
    VisibilityState,
)


def test_visibility_reasoner_preserves_existing_gt_facts() -> None:
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
                position=Point2D(8.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-1_0",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CONFIRMED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="track-1",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )

    updated_scene, report = VisibilityReasoner().infer(scene, uncertain_distance=30.0)

    assert updated_scene.visibility_facts == scene.visibility_facts
    assert report.preserved_fact_count == 1
    assert not report.inferred_visible_pairs
    assert not report.inferred_uncertain_pairs


def test_visibility_reasoner_infers_visible_and_uncertain_pairs_conservatively() -> None:
    observations = (
        ObservationEvidence(
            observation_id="obs-1",
            source_agent_id="CAV_EGO",
            object_type="car",
            position=Point2D(12.0, 0.0),
            confidence=0.8,
            timestamp_index=0,
        ),
    )
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(4.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=observations,
        object_tracks=(
            ObjectTrack(
                object_id="track-1",
                object_type="car",
                position=Point2D(12.0, 0.0),
                confidence=0.8,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-1",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                observations=observations,
            ),
        ),
    )

    updated_scene, report = VisibilityReasoner().infer(scene, uncertain_distance=30.0)

    fact_map = {
        (fact.agent_id, fact.object_id): fact.state for fact in updated_scene.visibility_facts
    }

    assert fact_map[("CAV_EGO", "track-1")] == VisibilityState.VISIBLE
    assert fact_map[("CAV_1", "track-1")] == VisibilityState.UNCERTAIN
    assert report.inferred_visible_pairs == ("CAV_EGO:track-1",)
    assert report.inferred_uncertain_pairs == ("CAV_1:track-1",)


def test_visibility_reasoner_keeps_weak_candidates_uncertain() -> None:
    observations = (
        ObservationEvidence(
            observation_id="obs-2",
            source_agent_id="CAV_EGO",
            object_type="car",
            position=Point2D(15.0, 0.0),
            confidence=0.33,
            timestamp_index=0,
        ),
    )
    scene = CooperativeScene(
        scene_id="scene-2",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=observations,
        object_tracks=(
            ObjectTrack(
                object_id="candidate-1",
                object_type="car",
                position=Point2D(15.0, 0.0),
                confidence=0.33,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-2",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
                observations=observations,
            ),
        ),
    )

    updated_scene, report = VisibilityReasoner().infer(
        scene,
        uncertain_distance=30.0,
        min_candidate_visible_confidence=0.5,
    )

    fact_map = {
        (fact.agent_id, fact.object_id): fact.state for fact in updated_scene.visibility_facts
    }

    assert fact_map[("CAV_EGO", "candidate-1")] == VisibilityState.UNCERTAIN
    assert not report.inferred_visible_pairs
    assert report.inferred_uncertain_pairs == ("CAV_EGO:candidate-1",)
