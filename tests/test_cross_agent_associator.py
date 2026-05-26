from kg_coop_drive.application.tracking.cross_agent_associator import CrossAgentAssociator
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObservationEvidence,
    Point2D,
    Pose2D,
    Trajectory,
)


def test_cross_agent_associator_matches_close_observations_from_different_agents() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(1.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(
            ObservationEvidence(
                observation_id="ego-obs-1",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=0.8,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id="cav1-obs-1",
                source_agent_id="CAV_1",
                object_type="car",
                position=Point2D(10.5, 0.2),
                confidence=0.7,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id="ego-obs-2",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(40.0, 0.0),
                confidence=0.5,
                timestamp_index=0,
            ),
        ),
    )

    report = CrossAgentAssociator().associate(scene, max_distance=3.0)

    assert report.participating_agents == ("CAV_1", "CAV_EGO")
    assert len(report.matches) == 1
    match = report.matches[0]
    assert {match.left_observation_id, match.right_observation_id} == {
        "ego-obs-1",
        "cav1-obs-1",
    }


def test_cross_agent_associator_reports_no_matches_when_only_one_agent_is_present() -> None:
    scene = CooperativeScene(
        scene_id="scene-2",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(
            ObservationEvidence(
                observation_id="ego-obs-1",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=0.8,
                timestamp_index=0,
            ),
        ),
    )

    report = CrossAgentAssociator().associate(scene, max_distance=3.0)

    assert report.participating_agents == ("CAV_EGO",)
    assert report.matches == ()
