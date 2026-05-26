from kg_coop_drive.application.scene_graph.relation_builder import RelationBuilder
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    RelationType,
    Trajectory,
)


def test_relation_builder_derives_front_left_and_near_trajectory() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(3.0, -1.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="car-1",
                object_type="car",
                position=Point2D(11.0, 1.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_car-1_0",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
    )

    enriched = RelationBuilder().build(scene, near_ego_distance=12.0)
    relation_types = {relation.relation_type for relation in enriched.relations}

    assert RelationType.FRONT_OF in relation_types
    assert RelationType.LEFT_OF in relation_types
    assert RelationType.NEAR in relation_types
    assert RelationType.NEAR_TRAJECTORY in relation_types
    assert RelationType.NEAR_FIRST_WAYPOINT in relation_types
    assert RelationType.PATH_RELEVANT in relation_types
    assert RelationType.LOW_CONFLICT in relation_types


def test_relation_builder_derives_behind_and_right_of() -> None:
    scene = CooperativeScene(
        scene_id="scene-2",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="car-2",
                object_type="car",
                position=Point2D(-5.0, -2.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_car-2_0",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
    )

    enriched = RelationBuilder().build(scene)
    relation_types = {relation.relation_type for relation in enriched.relations}

    assert RelationType.BEHIND in relation_types
    assert RelationType.RIGHT_OF in relation_types
    assert RelationType.NEAR in relation_types


def test_relation_builder_marks_cooperative_support_when_multiple_sources_exist() -> None:
    scene = CooperativeScene(
        scene_id="scene-3",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="car-3",
                object_type="car",
                position=Point2D(9.0, 0.5),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO", "CAV_1"),
                    observation_ids=("obs-1", "obs-2"),
                    latest_timestamp_index=0,
                ),
            ),
        ),
    )

    enriched = RelationBuilder().build(scene)
    relation_types = {relation.relation_type for relation in enriched.relations}

    assert RelationType.COOPERATIVELY_SUPPORTED in relation_types
