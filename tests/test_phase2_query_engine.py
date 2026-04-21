from kg_coop_drive.application.query_engine import SceneQueryEngine
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    RelationFact,
    RelationType,
    Trajectory,
    VisibilityFact,
    VisibilityState,
)


def build_toy_scene() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=10,
        global_timestamp_index=100,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="car-1",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-1",),
                    latest_timestamp_index=10,
                ),
            ),
            ObjectTrack(
                object_id="ped-1",
                object_type="pedestrian",
                position=Point2D(50.0, 8.0),
                confidence=0.8,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-2",),
                    latest_timestamp_index=10,
                ),
            ),
        ),
        relations=(
            RelationFact(
                subject_id="car-1",
                relation_type=RelationType.FRONT_OF,
                object_id="CAV_EGO",
                confidence=0.95,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="car-1",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="ped-1",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def test_query_engine_filters_visible_objects_near_trajectory() -> None:
    scene = build_toy_scene()
    engine = SceneQueryEngine()

    result = engine.select_objects(scene)
    result = engine.filter_by_visibility(result, agent_id="CAV_EGO")
    result = engine.filter_near_trajectory(result, max_distance=3.0)

    assert result.count() == 1
    assert result.objects[0].object_id == "car-1"


def test_query_engine_filters_by_relation_and_type() -> None:
    scene = build_toy_scene()
    engine = SceneQueryEngine()

    result = engine.select_objects(scene)
    result = engine.filter_by_relation(
        result,
        relation_type=RelationType.FRONT_OF,
        reference_id="CAV_EGO",
    )
    result = engine.filter_by_type(result, object_type="car")

    assert result.exists()
    assert result.objects[0].object_id == "car-1"


def test_query_engine_filters_by_new_relation_types() -> None:
    scene = build_toy_scene()
    scene = CooperativeScene(
        scene_id=scene.scene_id,
        local_timestamp_index=scene.local_timestamp_index,
        global_timestamp_index=scene.global_timestamp_index,
        asker_agent_id=scene.asker_agent_id,
        agents=scene.agents,
        future_trajectory=scene.future_trajectory,
        object_tracks=scene.object_tracks,
        relations=scene.relations
        + (
            RelationFact(
                subject_id="ped-1",
                relation_type=RelationType.RIGHT_OF,
                object_id="CAV_EGO",
                confidence=0.9,
            ),
            RelationFact(
                subject_id="ped-1",
                relation_type=RelationType.BEHIND,
                object_id="CAV_EGO",
                confidence=0.9,
            ),
        ),
        visibility_facts=scene.visibility_facts,
    )
    engine = SceneQueryEngine()

    right_of = engine.filter_by_relation(
        engine.select_objects(scene),
        relation_type=RelationType.RIGHT_OF,
        reference_id="CAV_EGO",
    )
    behind = engine.filter_by_relation(
        engine.select_objects(scene),
        relation_type=RelationType.BEHIND,
        reference_id="CAV_EGO",
    )

    assert right_of.exists()
    assert right_of.objects[0].object_id == "ped-1"
    assert behind.exists()
    assert behind.objects[0].object_id == "ped-1"


def test_query_engine_supports_source_agent_count_and_exists_tools() -> None:
    scene = build_toy_scene()
    engine = SceneQueryEngine()

    result = engine.select_objects(scene)
    result = engine.filter_by_source_agent(result, source_agent_id="CAV_EGO")

    assert engine.exists(result) is True
    assert engine.count(result) == 1
    assert result.objects[0].object_id == "car-1"


def test_query_engine_get_attribute_compare_and_trace_provenance() -> None:
    scene = build_toy_scene()
    engine = SceneQueryEngine()

    result = engine.select_objects(scene)
    attributes = engine.get_attribute(result, attribute_name="confidence")
    comparisons = engine.compare(result, attribute_name="confidence")
    provenance = engine.trace_provenance(result)

    assert attributes[0].object_id == "car-1"
    assert attributes[0].attribute_name == "confidence"
    assert attributes[0].value == 0.9

    assert len(comparisons) == 1
    assert comparisons[0].attribute_name == "confidence"
    assert comparisons[0].relation == "greater_than"
    assert comparisons[0].left_object_id == "car-1"
    assert comparisons[0].right_object_id == "ped-1"

    assert len(provenance) == 2
    assert provenance[0].object_id == "car-1"
    assert provenance[0].source_agent_ids == ("CAV_EGO",)
    assert provenance[0].observation_ids == ("obs-1",)


def test_query_engine_handles_empty_results_safely() -> None:
    scene = build_toy_scene()
    engine = SceneQueryEngine()

    result = engine.select_objects(scene)
    result = engine.filter_by_type(result, object_type="truck")

    assert engine.exists(result) is False
    assert engine.count(result) == 0
    assert engine.get_attribute(result, attribute_name="confidence") == ()
    assert engine.compare(result, attribute_name="confidence") == ()
    assert engine.trace_provenance(result) == ()


def test_query_engine_returns_none_for_unknown_attributes_without_failing() -> None:
    scene = build_toy_scene()
    engine = SceneQueryEngine()

    result = engine.select_objects(scene)
    attributes = engine.get_attribute(result, attribute_name="unknown_field")
    comparisons = engine.compare(result, attribute_name="unknown_field")

    assert len(attributes) == 2
    assert all(item.value is None for item in attributes)
    assert comparisons == ()


def test_query_engine_marks_non_numeric_compare_requests_as_not_comparable() -> None:
    scene = build_toy_scene()
    engine = SceneQueryEngine()

    result = engine.select_objects(scene)
    comparisons = engine.compare(result, attribute_name="object_type")

    assert len(comparisons) == 1
    assert comparisons[0].attribute_name == "object_type"
    assert comparisons[0].relation == "not_comparable"
