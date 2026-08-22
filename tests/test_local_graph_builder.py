from kg_coop_drive.application.scene_graph.local_graph_builder import LocalGraphBuilder
from kg_coop_drive.domain.processed_scene import ProcessedFrameSceneData
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    Trajectory,
    VisibilityFact,
    VisibilityState,
)


def test_local_graph_builder_filters_to_one_agent_and_builds_local_facts() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(2.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
    )
    processed_data = ProcessedFrameSceneData(
        timestamp_index=0,
        observations=(
            ObservationEvidence(
                observation_id="ego-obs-1",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(10.2, 0.2),
                confidence=0.8,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id="cav1-obs-1",
                source_agent_id="CAV_1",
                object_type="car",
                position=Point2D(11.0, 0.1),
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
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="track-1",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_1",
                object_id="track-1",
                state=VisibilityState.UNCERTAIN,
            ),
        ),
        source_paths=(),
    )

    local_scene = LocalGraphBuilder().build(scene, processed_data, agent_id="CAV_EGO")

    assert local_scene.asker_agent_id == "CAV_EGO"
    assert len(local_scene.agents) == 1
    assert local_scene.agents[0].agent_id == "CAV_EGO"
    assert len(local_scene.object_tracks) == 1
    assert len(local_scene.object_tracks[0].observations) == 1
    assert local_scene.object_tracks[0].observations[0].source_agent_id == "CAV_EGO"
    assert all(fact.agent_id == "CAV_EGO" for fact in local_scene.visibility_facts)
    assert any(relation.object_id == "CAV_EGO" for relation in local_scene.relations)
