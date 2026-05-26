from kg_coop_drive.application.scene_graph.local_graph_builder import LocalGraphBuilder
from kg_coop_drive.application.scene_graph.query_engine import SceneQueryEngine
from kg_coop_drive.domain.processed_scene import ProcessedFrameSceneData
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    RelationType,
    Trajectory,
    VisibilityFact,
    VisibilityState,
)


def test_local_query_path_returns_visible_local_object() -> None:
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
        ),
        source_paths=(),
    )

    local_scene = LocalGraphBuilder().build(scene, processed_data, agent_id="CAV_EGO")
    engine = SceneQueryEngine()

    selected = engine.select_objects(local_scene)
    visible = engine.filter_by_visibility(
        selected,
        agent_id="CAV_EGO",
        visibility=VisibilityState.VISIBLE,
    )
    front_or_behind = engine.filter_by_relation(
        selected,
        relation_type=RelationType.FRONT_OF,
        reference_id="CAV_EGO",
    )

    assert selected.count() == 1
    assert visible.count() == 1
    assert visible.objects[0].object_id == "track-1"
    assert front_or_behind.exists()
