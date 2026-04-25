from __future__ import annotations

from kg_coop_drive.application.v2vgotqa_router import (
    AgentMotionPredictionHandler,
    ControlSettingsHandler,
    FutureTrajectoryHandler,
    NotableObjectsHandler,
    NotableObjectLLMRankItem,
    NotableObjectLLMRankedItem,
    ObjectMotionPredictionHandler,
    OccludingObjectLLMRankItem,
    OccludingObjectLLMRankedItem,
    OccludingObjectsHandler,
    V2VGoTQARouter,
)
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    Trajectory,
    Vector2D,
    VisibilityFact,
    VisibilityState,
)


def _scene() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
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
                object_id="visible-car",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-1",),
                    latest_timestamp_index=0,
                ),
                velocity=Vector2D(2.0, 0.0),
            ),
            ObjectTrack(
                object_id="occluded-car",
                object_type="car",
                position=Point2D(12.0, -0.5),
                confidence=0.8,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-2",),
                    latest_timestamp_index=0,
                ),
                velocity=Vector2D(0.0, -1.0),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="visible-car",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="occluded-car",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def _sample(task_type: BenchmarkTaskType) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"sample-{task_type.value}",
        dataset_name="V2V-GoT-QA",
        split_name="val",
        file_name="demo.json",
        task_type=task_type,
        scene=_scene(),
        raw_record={},
    )


def _scene_with_competing_objects() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-2",
        local_timestamp_index=0,
        global_timestamp_index=0,
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
                object_id="good-visible",
                object_type="car",
                position=Point2D(11.0, 0.3),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO", "CAV_1"),
                    observation_ids=("obs-1", "obs-2"),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="far-visible",
                object_type="car",
                position=Point2D(11.0, 5.5),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-3",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="blocker-visible",
                object_type="car",
                position=Point2D(9.0, 0.0),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-4",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="hidden-target",
                object_type="car",
                position=Point2D(16.0, 0.1),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-5",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="good-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="far-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="blocker-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="hidden-target",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def _sample_with_scene(task_type: BenchmarkTaskType, scene: CooperativeScene) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"sample-{task_type.value}-custom",
        dataset_name="V2V-GoT-QA",
        split_name="val",
        file_name="demo.json",
        task_type=task_type,
        scene=scene,
        raw_record={},
    )


class _FakeOccludingLLMClient:
    def rerank_occluding_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[OccludingObjectLLMRankItem, ...],
    ) -> tuple[OccludingObjectLLMRankedItem, ...]:
        del asker_agent_id, raw_question
        return tuple(
            OccludingObjectLLMRankedItem(
                object_id=candidate.object_id,
                score=1.0 if candidate.object_id == "blocker-visible" else 0.1,
            )
            for candidate in candidates
        )


class _FakeNotableLLMClient:
    def rerank_notable_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[NotableObjectLLMRankItem, ...],
    ) -> tuple[NotableObjectLLMRankedItem, ...]:
        del asker_agent_id, raw_question
        return tuple(
            NotableObjectLLMRankedItem(
                object_id=candidate.object_id,
                score=1.0 if candidate.object_id == "good-visible" else 0.1,
            )
            for candidate in candidates
        )


def test_v2vgotqa_router_answers_notable_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.NOTABLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car",)
    assert "Notable visible objects" in answer.answer_text


def test_v2vgotqa_router_prioritizes_top_visible_relevant_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.NOTABLE_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ("good-visible", "blocker-visible")


def test_v2vgotqa_router_supports_energy_ranker_for_notable_objects() -> None:
    router = V2VGoTQARouter(
        handlers=(NotableObjectsHandler(ranker="energy"),)
    )

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.NOTABLE_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids[0] == "good-visible"


def test_v2vgotqa_router_supports_llm_ranker_for_notable_objects() -> None:
    router = V2VGoTQARouter(
        handlers=(NotableObjectsHandler(ranker="llm", llm_client=_FakeNotableLLMClient()),)
    )

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.NOTABLE_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids[0] == "good-visible"


def test_v2vgotqa_router_answers_occluding_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.OCCLUDING_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car",)
    assert "occluding" in answer.answer_text.lower()


def test_v2vgotqa_router_prefers_visible_blockers_for_occluding_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.OCCLUDING_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids[0] == "blocker-visible"
    assert "blocker-visible" in answer.object_ids


def test_v2vgotqa_router_uses_llm_rerank_for_occluding_objects() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(llm_client=_FakeOccludingLLMClient()),)
    )

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.OCCLUDING_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ("blocker-visible", "good-visible")


def test_v2vgotqa_router_answers_invisible_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.INVISIBLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car",)
    assert "invisible" in answer.answer_text.lower()


def test_v2vgotqa_router_answers_object_motion_prediction() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.OBJECT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car", "occluded-car")
    assert answer.answer_text == (
        "Predicted object motion: visible-car=moving forward from (11.0, 0.5) to (13.0, 0.5); "
        "occluded-car=moving left from (12.0, -0.5) to (12.0, -1.5)."
    )


def test_v2vgotqa_router_answers_agent_motion_prediction() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.AGENT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert answer.answer_text == (
        "Predicted agent motion: CAV_1=hold position near (5.0, 0.0)."
    )


def test_v2vgotqa_router_answers_future_trajectory() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.FUTURE_TRAJECTORY))

    assert answer.supported is True
    assert answer.object_ids == ()
    assert answer.answer_text == (
        "Suggested future trajectory: [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]."
    )


def test_v2vgotqa_router_answers_control_settings() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.CONTROL_SETTINGS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car", "visible-car")
    assert answer.answer_text == (
        "Suggested control settings: speed=reduce speed sharply; steering=steer left; key objects: occluded-car, visible-car."
    )


def test_future_trajectory_handler_renders_points_directly() -> None:
    handler = FutureTrajectoryHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.FUTURE_TRAJECTORY))

    assert answer.supported is True
    assert answer.object_ids == ()
    assert answer.answer_text.endswith("[(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)].")


def test_control_settings_handler_renders_speed_and_steering() -> None:
    handler = ControlSettingsHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.CONTROL_SETTINGS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car", "visible-car")
    assert "speed=reduce speed sharply" in answer.answer_text
    assert "steering=steer left" in answer.answer_text


def test_object_motion_prediction_handler_projects_object_velocity() -> None:
    handler = ObjectMotionPredictionHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.OBJECT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car", "occluded-car")
    assert "visible-car=moving forward" in answer.answer_text
    assert "to (13.0, 0.5)" in answer.answer_text


def test_agent_motion_prediction_handler_renders_other_agent_positions() -> None:
    handler = AgentMotionPredictionHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.AGENT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert "CAV_1=hold position near (5.0, 0.0)" in answer.answer_text


def test_agent_motion_prediction_handler_projects_agent_velocity() -> None:
    handler = AgentMotionPredictionHandler()
    scene = CooperativeScene(
        scene_id="scene-agent-motion",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(
                agent_id="CAV_1",
                pose=Pose2D(position=Point2D(5.0, 1.0)),
                velocity=Vector2D(0.0, 2.0),
            ),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.AGENT_MOTION_PREDICTION, scene)
    )

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert answer.answer_text == (
        "Predicted agent motion: CAV_1=move right from (5.0, 1.0) to (5.0, 3.0)."
    )


def test_agent_motion_prediction_handler_uses_planned_trajectory_when_velocity_is_static() -> None:
    handler = AgentMotionPredictionHandler()
    scene = CooperativeScene(
        scene_id="scene-agent-trajectory",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(
                agent_id="CAV_1",
                pose=Pose2D(position=Point2D(5.0, 1.0)),
                velocity=Vector2D(0.0, 0.0),
                planned_trajectory=Trajectory(points=(Point2D(8.0, 0.2), Point2D(20.0, 0.5))),
            ),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.AGENT_MOTION_PREDICTION, scene)
    )

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert answer.answer_text == (
        "Predicted agent motion: CAV_1=move forward from (5.0, 1.0) to (25.0, 1.5)."
    )


def test_v2vgotqa_router_answers_planning_awareness() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.PLANNING_AWARENESS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car", "visible-car")
    assert "aware" in answer.answer_text.lower()


def test_v2vgotqa_router_marks_unsupported_tasks_explicitly() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.UNKNOWN))

    assert answer.supported is False
    assert answer.object_ids == ()
    assert "Unsupported Phase 5 task" in answer.answer_text
