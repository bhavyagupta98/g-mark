from __future__ import annotations

from kg_coop_drive.application.qa.planning_awareness import (
    DiverseTopKDecisionPolicy,
    EnergyBasedDecisionPolicy,
    EnergyBasedPlanningAwarenessScorer,
    HeuristicPlanningAwarenessScorer,
    LLMRerankDecisionPolicy,
    LLMPlanningAwarenessScorer,
    PlanningAwarenessBatchLLMClient,
    PlanningAwarenessLLMContext,
    PlanningAwarenessLLMRankItem,
    PlanningAwarenessLLMRankedItem,
    PlanningAwarenessOrchestrator,
    PlanningAwarenessRanker,
    PlanningAwarenessSelectionPolicy,
    RelationalImportancePlanningAwarenessScorer,
    RiskAwarePlanningAwarenessScorer,
    TopScoreDecisionPolicy,
    build_planning_awareness_decision_policy,
    build_planning_awareness_orchestrator,
    build_planning_awareness_scorer,
)
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    TrackStatus,
    Trajectory,
    VisibilityFact,
    VisibilityState,
)


def _scene() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_1",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="supported-occluded",
                object_type="car",
                position=Point2D(21.0, 0.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_1"),
                    observation_ids=("gt-1", "obs-1"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                uncertainty_score=0.0,
                conflict_score=0.0,
            ),
            ObjectTrack(
                object_id="visible-candidate",
                object_type="car",
                position=Point2D(29.0, 0.1),
                confidence=0.55,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-2",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
                uncertainty_score=0.55,
                conflict_score=0.0,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_1",
                object_id="supported-occluded",
                state=VisibilityState.OCCLUDED,
            ),
            VisibilityFact(
                agent_id="CAV_1",
                object_id="visible-candidate",
                state=VisibilityState.VISIBLE,
            ),
        ),
    )


def test_planning_awareness_orchestrator_prefers_supported_occluded_track_over_weaker_visible_candidate() -> None:
    orchestrator = PlanningAwarenessOrchestrator()

    decision = orchestrator.select(_scene())

    assert decision.selected_candidates
    assert decision.selected_candidates[0].object_track.object_id == "supported-occluded"


def test_planning_awareness_orchestrator_can_filter_by_threshold() -> None:
    scorer = HeuristicPlanningAwarenessScorer()
    orchestrator = PlanningAwarenessOrchestrator(
        scorer=scorer,
        decision_policy=TopScoreDecisionPolicy(min_score=0.95, max_results=3),
    )

    decision = orchestrator.select(_scene())

    assert tuple(candidate.object_track.object_id for candidate in decision.selected_candidates) == (
        "supported-occluded",
    )


class _StubLLMClient:
    def score_candidate(self, context: PlanningAwarenessLLMContext) -> tuple[float, tuple[str, ...]]:
        if context.object_id == "visible-candidate":
            return 0.9, ("llm_prefers_visible_candidate",)
        return 0.2, ("llm_downranks_other_objects",)


class _StubBatchLLMClient(PlanningAwarenessBatchLLMClient):
    def rerank_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[PlanningAwarenessLLMRankItem, ...],
    ) -> tuple[PlanningAwarenessLLMRankedItem, ...]:
        ranked: list[PlanningAwarenessLLMRankedItem] = []
        for candidate in candidates:
            if candidate.object_id == "visible-candidate":
                ranked.append(
                    PlanningAwarenessLLMRankedItem(
                        object_id=candidate.object_id,
                        score=0.95,
                        rationale=("llm_prefers_visible_candidate",),
                    )
                )
            else:
                ranked.append(
                    PlanningAwarenessLLMRankedItem(
                        object_id=candidate.object_id,
                        score=0.15,
                        rationale=("llm_downranks_other_objects",),
                    )
                )
        return tuple(ranked)


def test_planning_awareness_orchestrator_supports_llm_ranker() -> None:
    orchestrator = PlanningAwarenessOrchestrator(
        scorer=LLMPlanningAwarenessScorer(client=_StubLLMClient(), min_score=0.3),
        decision_policy=TopScoreDecisionPolicy(min_score=0.3, max_results=3),
    )

    decision = orchestrator.select(_scene())

    assert decision.selected_candidates
    assert decision.selected_candidates[0].object_track.object_id == "visible-candidate"
    assert "llm_prefers_visible_candidate" in decision.selected_candidates[0].rationale


def test_llm_rerank_policy_can_reorder_risk_shortlist() -> None:
    orchestrator = PlanningAwarenessOrchestrator(
        scorer=RiskAwarePlanningAwarenessScorer(min_score=0.1),
        decision_policy=LLMRerankDecisionPolicy(
            client=_StubBatchLLMClient(),
            min_score=0.1,
            max_results=2,
            shortlist_size=2,
            blend_weight=0.9,
        ),
    )

    decision = orchestrator.select(_scene())

    assert decision.selected_candidates
    assert decision.selected_candidates[0].object_track.object_id == "visible-candidate"
    assert "llm_score=0.95" in decision.selected_candidates[0].rationale


def test_all_non_llm_rankers_can_build_and_score_candidates() -> None:
    for ranker in (
        PlanningAwarenessRanker.HEURISTIC,
        PlanningAwarenessRanker.RELATIONAL_IMPORTANCE,
        PlanningAwarenessRanker.RISK_AWARE,
        PlanningAwarenessRanker.ENERGY_BASED,
    ):
        orchestrator = build_planning_awareness_orchestrator(ranker)
        decision = orchestrator.select(_scene())
        assert decision.considered_candidates


def test_relational_importance_prefers_supported_object() -> None:
    orchestrator = PlanningAwarenessOrchestrator(
        scorer=RelationalImportancePlanningAwarenessScorer(),
        decision_policy=TopScoreDecisionPolicy(min_score=0.2, max_results=3),
    )

    decision = orchestrator.select(_scene())

    assert decision.selected_candidates
    assert decision.selected_candidates[0].object_track.object_id == "supported-occluded"


def test_risk_aware_prefers_occluded_supported_object() -> None:
    orchestrator = PlanningAwarenessOrchestrator(
        scorer=RiskAwarePlanningAwarenessScorer(),
        decision_policy=TopScoreDecisionPolicy(min_score=0.2, max_results=3),
    )

    decision = orchestrator.select(_scene())

    assert decision.selected_candidates
    assert decision.selected_candidates[0].object_track.object_id == "supported-occluded"


def test_energy_based_policy_penalizes_redundant_candidates() -> None:
    scene = CooperativeScene(
        scene_id="scene-2",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_1",
        agents=_scene().agents,
        future_trajectory=_scene().future_trajectory,
        object_tracks=(
            ObjectTrack(
                object_id="gt-supported",
                object_type="car",
                position=Point2D(20.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_1"),
                    observation_ids=("gt-1", "obs-1"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                uncertainty_score=0.0,
                conflict_score=0.0,
            ),
            ObjectTrack(
                object_id="dup-a",
                object_type="car",
                position=Point2D(20.5, 0.1),
                confidence=0.92,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_1"),
                    observation_ids=("gt-2", "obs-2"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                uncertainty_score=0.0,
                conflict_score=0.0,
            ),
            ObjectTrack(
                object_id="far-bus",
                object_type="bus",
                position=Point2D(28.0, 0.0),
                confidence=0.85,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_1"),
                    observation_ids=("gt-3", "obs-3"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                uncertainty_score=0.0,
                conflict_score=0.0,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_1", object_id="gt-supported", state=VisibilityState.OCCLUDED),
            VisibilityFact(agent_id="CAV_1", object_id="dup-a", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_1", object_id="far-bus", state=VisibilityState.VISIBLE),
        ),
    )
    scorer = EnergyBasedPlanningAwarenessScorer(min_score=0.1)
    orchestrator = PlanningAwarenessOrchestrator(
        scorer=scorer,
        decision_policy=EnergyBasedDecisionPolicy(min_score=0.1, max_results=2),
    )

    decision = orchestrator.select(scene)
    selected_ids = tuple(candidate.object_track.object_id for candidate in decision.selected_candidates)

    assert "gt-supported" in selected_ids
    assert len(selected_ids) == 2
    assert not {"gt-supported", "dup-a"} <= set(selected_ids)


def test_build_planning_awareness_decision_policy_rejects_llm_without_client() -> None:
    scorer = build_planning_awareness_scorer(PlanningAwarenessRanker.LLM)

    try:
        build_planning_awareness_decision_policy(
            PlanningAwarenessRanker.LLM,
            scorer=scorer,
        )
    except ValueError as exc:
        assert "llm_client" in str(exc)
    else:
        raise AssertionError("Expected ValueError when building llm decision policy without client.")


def test_diverse_top2_policy_prefers_one_occluded_and_one_visible_candidate() -> None:
    orchestrator = PlanningAwarenessOrchestrator(
        scorer=RiskAwarePlanningAwarenessScorer(min_score=0.2),
        decision_policy=DiverseTopKDecisionPolicy(min_score=0.2, max_results=2),
    )

    decision = orchestrator.select(_scene())

    selected_ids = tuple(candidate.object_track.object_id for candidate in decision.selected_candidates)
    assert selected_ids == ("supported-occluded", "visible-candidate")


def test_top2_policy_limits_results_to_two() -> None:
    scene = CooperativeScene(
        scene_id="scene-3",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_1",
        agents=_scene().agents,
        future_trajectory=_scene().future_trajectory,
        object_tracks=(
            ObjectTrack(
                object_id="occluded-best",
                object_type="car",
                position=Point2D(21.0, 0.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_1"),
                    observation_ids=("gt-1", "obs-1"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                uncertainty_score=0.0,
                conflict_score=0.0,
            ),
            ObjectTrack(
                object_id="visible-best",
                object_type="car",
                position=Point2D(22.0, 0.1),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_1"),
                    observation_ids=("gt-2", "obs-2"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                uncertainty_score=0.0,
                conflict_score=0.0,
            ),
            ObjectTrack(
                object_id="visible-third",
                object_type="car",
                position=Point2D(23.0, 0.0),
                confidence=0.90,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_1"),
                    observation_ids=("gt-3", "obs-3"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                uncertainty_score=0.0,
                conflict_score=0.0,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_1", object_id="occluded-best", state=VisibilityState.OCCLUDED),
            VisibilityFact(agent_id="CAV_1", object_id="visible-best", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_1", object_id="visible-third", state=VisibilityState.VISIBLE),
        ),
    )
    orchestrator = build_planning_awareness_orchestrator(
        PlanningAwarenessRanker.RISK_AWARE,
        selection_policy=PlanningAwarenessSelectionPolicy.TOP2,
    )

    decision = orchestrator.select(scene)

    assert len(decision.selected_candidates) == 2
