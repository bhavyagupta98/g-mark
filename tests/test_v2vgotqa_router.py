from kg_coop_drive.application.v2vgotqa_router import (
    NotableObjectsHandler,
    NotableObjectLLMRankItem,
    NotableObjectLLMRankedItem,
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


def test_v2vgotqa_router_answers_planning_awareness() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.PLANNING_AWARENESS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car", "visible-car")
    assert "aware" in answer.answer_text.lower()


def test_v2vgotqa_router_marks_unsupported_tasks_explicitly() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.FUTURE_TRAJECTORY))

    assert answer.supported is False
    assert answer.object_ids == ()
    assert "Unsupported Phase 5 task" in answer.answer_text
