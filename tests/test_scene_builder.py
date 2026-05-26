from kg_coop_drive.application.scene_graph.scene_builder import QueryInterpreter, SceneBuilder
from kg_coop_drive.application.scene_graph.query_engine import SceneQueryEngine
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    Point2D,
    Pose2D,
    Trajectory,
    VisibilityState,
)


def build_scene_seed() -> CooperativeScene:
    return CooperativeScene(
        scene_id="demo-scene",
        local_timestamp_index=3,
        global_timestamp_index=30,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(4.0, 1.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(1.0, 0.0), Point2D(2.0, 0.0))),
        raw_question="What is visible near my planned future trajectory?",
        raw_answer="There is no notable object visible to you.",
    )


def test_scene_builder_reports_missing_graph_contents_for_seed_scene() -> None:
    report = SceneBuilder().build(build_scene_seed())

    assert "metadata-rich seed" in report.interpretation.assumptions[0]
    assert report.scene.raw_question.startswith("What is visible")
    assert len(report.build_steps) >= 4


def test_query_interpreter_describes_empty_selection_clearly() -> None:
    scene = build_scene_seed()
    result = SceneQueryEngine().select_objects(scene)
    explanation = QueryInterpreter().explain_visibility_filter(
        agent_id="CAV_EGO",
        visibility=VisibilityState.VISIBLE,
        result=result,
    )

    assert explanation.title == "Visibility Filter"
    assert "No objects remain" in explanation.outcome
