from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, dist, tanh
from typing import Protocol

from kg_coop_drive.application.planning_awareness import (
    PlanningAwarenessOrchestrator,
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.application.future_trajectory_planner import (
    ControlConditionedFutureTrajectoryPlanner,
)
from kg_coop_drive.application.control_settings_policy import (
    ControlSettingsDecision,
    decide_control_settings,
)
from kg_coop_drive.application.qa_selection_policies import (
    InvisibleSelectionPolicy,
    OccludingSelectionPolicy,
)
from kg_coop_drive.application.query_engine import SceneQueryEngine
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType
from kg_coop_drive.domain.scene import (
    ObjectTrack,
    QueryResult,
    RelationType,
    TrackStatus,
    VisibilityState,
)


@dataclass(frozen=True)
class BenchmarkAnswer:
    """Normalized benchmark answer returned by one task handler."""

    sample_id: str
    task_type: BenchmarkTaskType
    answer_text: str
    object_ids: tuple[str, ...]
    supported: bool


class BenchmarkTaskHandler(Protocol):
    """Strategy interface for task-specific benchmark answer generation."""

    task_type: BenchmarkTaskType

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        """Return a normalized benchmark answer for one sample."""


@dataclass(frozen=True)
class NotableObjectLLMRankItem:
    """Structured visible notable-object candidate for local LLM reranking."""

    object_id: str
    object_type: str
    distance_to_asker: float
    distance_to_trajectory: float
    distance_to_first_waypoint: float
    base_score: float
    status: str
    support_count: int
    conflict_score: float
    uncertainty_score: float


@dataclass(frozen=True)
class NotableObjectLLMRankedItem:
    """LLM rerank result for one visible notable-object candidate."""

    object_id: str
    score: float


class NotableObjectsBatchLLMClient(Protocol):
    """LLM interface for reranking visible notable-object candidates."""

    def rerank_notable_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[NotableObjectLLMRankItem, ...],
    ) -> tuple[NotableObjectLLMRankedItem, ...]:
        """Return ranked visible notable-object candidates."""


@dataclass(frozen=True)
class OccludingObjectLLMRankItem:
    """Structured blocker candidate for local LLM reranking."""

    object_id: str
    object_type: str
    distance_to_asker: float
    distance_to_trajectory: float
    base_score: float
    status: str
    support_count: int
    aligned_hidden_object_ids: tuple[str, ...]
    aligned_hidden_distances_to_trajectory: tuple[float, ...]
    best_alignment_radians: float


@dataclass(frozen=True)
class OccludingObjectLLMRankedItem:
    """LLM rerank result for one blocker candidate."""

    object_id: str
    score: float


class OccludingObjectsBatchLLMClient(Protocol):
    """LLM interface for reranking visible blocker candidates."""

    def rerank_occluding_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[OccludingObjectLLMRankItem, ...],
    ) -> tuple[OccludingObjectLLMRankedItem, ...]:
        """Return ranked blocker candidates for an occluding-object question."""


@dataclass(frozen=True)
class _RoleScoredObject:
    """Shared geometric ranking record for QA object selection."""

    object_track: object
    role: str
    score: float
    distance_to_trajectory: float
    distance_to_asker: float
    support_count: int
    visibility_state: VisibilityState | None
    aligned_hidden_object_ids: tuple[str, ...] = ()
    aligned_hidden_distances_to_trajectory: tuple[float, ...] = ()
    best_alignment_radians: float = 0.0


class V2VGoTQARouter:
    """Routes V2V-GoT-QA task categories to deterministic graph strategies."""

    def __init__(self, handlers: tuple[BenchmarkTaskHandler, ...] | None = None) -> None:
        default_handlers = (
            NotableObjectsHandler(),
            OccludingObjectsHandler(),
            InvisibleObjectsHandler(),
            ObjectMotionPredictionHandler(),
            AgentMotionPredictionHandler(),
            FutureTrajectoryHandler(),
            ControlSettingsHandler(),
            PlanningAwarenessHandler(),
        )
        resolved_handlers = {handler.task_type: handler for handler in default_handlers}
        if handlers:
            resolved_handlers.update({handler.task_type: handler for handler in handlers})
        self._handlers = resolved_handlers

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        """Answer one benchmark sample or return an explicit unsupported result."""

        handler = self._handlers.get(sample.task_type)
        if handler is None:
            return BenchmarkAnswer(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                answer_text=f"Unsupported Phase 5 task: {sample.task_type.value}.",
                object_ids=(),
                supported=False,
            )
        return handler.answer(sample)

    def supported_task_types(self) -> tuple[BenchmarkTaskType, ...]:
        """Return the currently registered Phase 5A task set."""

        return tuple(sorted(self._handlers.keys(), key=lambda item: item.value))


class _BaseQueryHandler:
    """Shared helpers for benchmark task handlers."""

    _visible_notable_max_distance = 7.0

    def __init__(self) -> None:
        self._engine = SceneQueryEngine()

    def _select_all(self, sample: BenchmarkSample) -> QueryResult:
        return self._engine.select_objects(sample.scene)

    def _near_trajectory(self, result: QueryResult, max_distance: float = 3.0) -> QueryResult:
        return self._engine.filter_near_trajectory(result, max_distance=max_distance)

    @staticmethod
    def _distance_to_trajectory(scene, object_track) -> float:
        if not scene.future_trajectory.points:
            return float("inf")
        return min(
            dist(
                (object_track.position.x, object_track.position.y),
                (point.x, point.y),
            )
            for point in scene.future_trajectory.points
        )

    @staticmethod
    def _distance_to_asker(scene, object_track) -> float:
        asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
        if asker is None:
            return float("inf")
        return dist(
            (object_track.position.x, object_track.position.y),
            (asker.pose.position.x, asker.pose.position.y),
        )

    @staticmethod
    def _distance_to_first_waypoint(scene, object_track) -> float:
        if not scene.future_trajectory.points:
            return float("inf")
        first_point = scene.future_trajectory.points[0]
        return dist(
            (object_track.position.x, object_track.position.y),
            (first_point.x, first_point.y),
        )

    @staticmethod
    def _bearing_from_asker(scene, object_track) -> float:
        asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
        if asker is None:
            return 0.0
        dx = object_track.position.x - asker.pose.position.x
        dy = object_track.position.y - asker.pose.position.y
        return atan2(dy, dx)

    @staticmethod
    def _visibility_lookup(scene, agent_id: str) -> dict[str, VisibilityState]:
        return {
            fact.object_id: fact.state
            for fact in scene.visibility_facts
            if fact.agent_id == agent_id
        }

    @staticmethod
    def _status_rank(object_track) -> int:
        order = {"confirmed": 0, "supported": 1, "candidate": 2}
        return order.get(object_track.status.value, 3)

    @staticmethod
    def _support_count(object_track) -> int:
        return len(object_track.provenance.source_agent_ids)

    @staticmethod
    def _wrapped_angle_difference(left: float, right: float) -> float:
        difference = abs(left - right)
        return min(difference, 6.283185307179586 - difference)

    @staticmethod
    def _relation_confidence(scene, object_id: str, relation_type: RelationType, reference_id: str) -> float:
        best_confidence = 0.0
        for relation in scene.relations:
            if (
                relation.subject_id == object_id
                and relation.relation_type == relation_type
                and relation.object_id == reference_id
            ):
                best_confidence = max(best_confidence, relation.confidence)
        return best_confidence

    def _blocker_role_scores(
        self,
        sample: BenchmarkSample,
        max_results: int,
    ) -> tuple[_RoleScoredObject, ...]:
        scene = sample.scene
        visibility_by_object = self._visibility_lookup(scene, scene.asker_agent_id)
        visible_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) == VisibilityState.VISIBLE
        )
        hidden_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) in (
                VisibilityState.OCCLUDED,
                VisibilityState.UNCERTAIN,
            )
        )

        role_scores: list[_RoleScoredObject] = []
        for visible_object in visible_objects:
            visible_distance = self._distance_to_asker(scene, visible_object)
            visible_bearing = self._bearing_from_asker(scene, visible_object)
            blocker_distance_to_trajectory = self._distance_to_trajectory(scene, visible_object)
            pair_evidence: list[tuple[float, str, float, float]] = []

            for hidden_object in hidden_objects:
                hidden_distance = self._distance_to_asker(scene, hidden_object)
                hidden_bearing = self._bearing_from_asker(scene, hidden_object)
                angle_difference = self._wrapped_angle_difference(visible_bearing, hidden_bearing)
                if angle_difference > 0.65:
                    continue

                hidden_distance_to_trajectory = self._distance_to_trajectory(scene, hidden_object)
                pair_distance = dist(
                    (visible_object.position.x, visible_object.position.y),
                    (hidden_object.position.x, hidden_object.position.y),
                )

                alignment_term = max(0.0, 1.0 - angle_difference / 0.65)
                depth_margin = hidden_distance - visible_distance
                depth_term = max(0.0, min(1.5, (depth_margin + 1.0) / 8.0))
                proximity_term = max(0.0, 1.0 - pair_distance / 20.0)
                hidden_relevance_term = 1.0 / (1.0 + hidden_distance_to_trajectory)

                pair_score = (
                    2.4 * alignment_term
                    + 1.8 * depth_term
                    + 1.2 * proximity_term
                    + 1.6 * hidden_relevance_term
                )
                if pair_score <= 0.0:
                    continue
                pair_evidence.append(
                    (
                        pair_score,
                        hidden_object.object_id,
                        hidden_distance_to_trajectory,
                        angle_difference,
                    )
                )

            if not pair_evidence:
                continue

            pair_evidence.sort(key=lambda item: (-item[0], item[2], item[3], item[1]))
            top_pair_score = pair_evidence[0][0]
            best_alignment_radians = pair_evidence[0][3]
            aggregated_pair_score = sum(item[0] for item in pair_evidence[:2])
            score = (
                0.65 * aggregated_pair_score
                + 0.35 * top_pair_score
                + 0.35 * self._support_count(visible_object)
                + 0.2 * visible_object.confidence
                - 0.18 * blocker_distance_to_trajectory
                - 0.12 * visible_distance
                - 0.28 * self._status_rank(visible_object)
                - 0.35 * visible_object.conflict_score
            )
            role_scores.append(
                _RoleScoredObject(
                    object_track=visible_object,
                    role="blocker",
                    score=score,
                    distance_to_trajectory=blocker_distance_to_trajectory,
                    distance_to_asker=visible_distance,
                    support_count=self._support_count(visible_object),
                    visibility_state=visibility_by_object.get(visible_object.object_id),
                    aligned_hidden_object_ids=tuple(item[1] for item in pair_evidence),
                    aligned_hidden_distances_to_trajectory=tuple(item[2] for item in pair_evidence),
                    best_alignment_radians=best_alignment_radians,
                )
            )

        return tuple(
            sorted(
                role_scores,
                key=lambda item: (
                    -item.score,
                    item.distance_to_trajectory,
                    item.distance_to_asker,
                    -item.support_count,
                    item.object_track.object_id,
                ),
            )[:max_results]
        )

    def _rank_by_role(
        self,
        sample: BenchmarkSample,
        role: str,
        max_results: int,
    ) -> QueryResult:
        scene = sample.scene
        visibility_by_object = self._visibility_lookup(scene, scene.asker_agent_id)
        role_scores: list[_RoleScoredObject] = []

        visible_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) == VisibilityState.VISIBLE
        )
        occluded_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) == VisibilityState.OCCLUDED
        )

        if role == "visible_relevant":
            for object_track in visible_objects:
                distance_to_trajectory = self._distance_to_trajectory(scene, object_track)
                if distance_to_trajectory > self._visible_notable_max_distance:
                    continue
                distance_to_asker = self._distance_to_asker(scene, object_track)
                distance_to_first_waypoint = self._distance_to_first_waypoint(scene, object_track)
                score = (
                    2.4 / (1.0 + distance_to_trajectory)
                    + 1.2 / (1.0 + distance_to_first_waypoint)
                    + 0.4 * self._support_count(object_track)
                    + 0.2 * object_track.confidence
                    - 0.2 * self._status_rank(object_track)
                    - 0.45 * object_track.conflict_score
                    - 0.35 * object_track.uncertainty_score
                    - 0.06 * distance_to_asker
                )
                role_scores.append(
                    _RoleScoredObject(
                        object_track=object_track,
                        role=role,
                        score=score,
                        distance_to_trajectory=distance_to_trajectory,
                        distance_to_asker=distance_to_asker,
                        support_count=self._support_count(object_track),
                        visibility_state=visibility_by_object.get(object_track.object_id),
                    )
                )

        elif role == "blocker":
            role_scores.extend(self._blocker_role_scores(sample, max_results=max_results))

        elif role == "hidden_relevant":
            for object_track in occluded_objects:
                distance_to_trajectory = self._distance_to_trajectory(scene, object_track)
                if distance_to_trajectory > 5.0:
                    continue
                distance_to_asker = self._distance_to_asker(scene, object_track)
                score = (
                    2.5 / (1.0 + distance_to_trajectory)
                    + 0.7 / (1.0 + distance_to_asker)
                    + 0.35 * self._support_count(object_track)
                    + 0.25 * object_track.confidence
                    - 0.25 * self._status_rank(object_track)
                    - 0.45 * object_track.conflict_score
                    - 0.3 * object_track.uncertainty_score
                )
                role_scores.append(
                    _RoleScoredObject(
                        object_track=object_track,
                        role=role,
                        score=score,
                        distance_to_trajectory=distance_to_trajectory,
                        distance_to_asker=distance_to_asker,
                        support_count=self._support_count(object_track),
                        visibility_state=visibility_by_object.get(object_track.object_id),
                    )
                )

        ordered = tuple(
            item.object_track
            for item in sorted(
                role_scores,
                key=lambda item: (
                    -item.score,
                    item.distance_to_trajectory,
                    item.distance_to_asker,
                    -item.support_count,
                    item.object_track.object_id,
                ),
            )[:max_results]
        )
        return QueryResult(scene=scene, objects=ordered)

    def _top_notable_visible(self, sample: BenchmarkSample, max_results: int = 2) -> QueryResult:
        return self._rank_by_role(sample, role="visible_relevant", max_results=max_results)

    def _top_hidden_relevant_risk_adaptive(
        self,
        sample: BenchmarkSample,
        policy: InvisibleSelectionPolicy,
    ) -> QueryResult:
        shortlist_size = max(policy.shortlist_size, policy.max_results)
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=shortlist_size,
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=5.0,
        )
        if not ranked_scores:
            return QueryResult(scene=sample.scene, objects=())

        risks = self._invisible_relative_risks(ranked_scores, policy)
        best_risk = max(risks, default=0.0)
        selected: list[_RoleScoredObject] = []
        for index, candidate in enumerate(ranked_scores):
            risk = risks[index]
            relative_to_best = risk / best_risk if best_risk > 0.0 else 0.0
            if candidate.distance_to_asker < policy.min_distance_to_asker:
                continue
            if candidate.distance_to_trajectory > policy.max_distance_to_trajectory:
                continue
            if candidate.distance_to_asker > policy.max_distance_to_asker:
                continue
            if risk < policy.min_risk:
                continue
            if relative_to_best < policy.min_relative_to_best:
                continue
            selected.append(candidate)
            if len(selected) >= policy.max_results:
                break

        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in selected),
        )

    @classmethod
    def _invisible_relative_risks(
        cls,
        ranked_scores: tuple[_RoleScoredObject, ...],
        policy: InvisibleSelectionPolicy,
    ) -> tuple[float, ...]:
        model_scores = tuple(item.score for item in ranked_scores)
        model_components = cls._normalize_high_values(model_scores)
        risks: list[float] = []

        for index, item in enumerate(ranked_scores):
            trajectory_component = max(
                0.0,
                1.0 - item.distance_to_trajectory / max(policy.max_distance_to_trajectory, 1e-6),
            )
            asker_component = max(
                0.0,
                1.0 - item.distance_to_asker / max(policy.max_distance_to_asker, 1e-6),
            )
            provenance_component = min(1.0, item.support_count / 2.0)
            confidence_component = max(0.0, min(1.0, float(item.object_track.confidence)))
            risk = (
                policy.trajectory_weight * trajectory_component
                + policy.asker_weight * asker_component
                + policy.provenance_weight * provenance_component
                + policy.confidence_weight * confidence_component
                + policy.model_score_weight * model_components[index]
                - policy.conflict_penalty * item.object_track.conflict_score
                - policy.uncertainty_penalty * item.object_track.uncertainty_score
            )
            if item.object_track.status == TrackStatus.CANDIDATE:
                risk -= policy.candidate_penalty
            risks.append(max(0.0, min(1.0, risk)))

        return tuple(risks)

    def _top_notable_visible_energy(self, sample: BenchmarkSample, max_results: int = 2) -> QueryResult:
        scene = sample.scene
        visibility_by_object = self._visibility_lookup(scene, scene.asker_agent_id)
        visible_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) == VisibilityState.VISIBLE
        )
        hidden_or_uncertain_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) in (
                VisibilityState.OCCLUDED,
                VisibilityState.UNCERTAIN,
            )
        )

        scored_objects: list[tuple[float, object]] = []
        for object_track in visible_objects:
            distance_to_trajectory = self._distance_to_trajectory(scene, object_track)
            if distance_to_trajectory > 5.0:
                continue

            distance_to_asker = self._distance_to_asker(scene, object_track)
            distance_to_first_waypoint = self._distance_to_first_waypoint(scene, object_track)
            path_relevant_confidence = self._relation_confidence(
                scene,
                object_track.object_id,
                RelationType.PATH_RELEVANT,
                scene.asker_agent_id,
            )
            cooperative_support_confidence = self._relation_confidence(
                scene,
                object_track.object_id,
                RelationType.COOPERATIVELY_SUPPORTED,
                scene.asker_agent_id,
            )

            neighbor_interaction_strength = 0.0
            hidden_neighbor_strength = 0.0
            for other_track in scene.object_tracks:
                if other_track.object_id == object_track.object_id:
                    continue
                pair_distance = dist(
                    (object_track.position.x, object_track.position.y),
                    (other_track.position.x, other_track.position.y),
                )
                if pair_distance > 12.0:
                    continue
                interaction_term = max(0.0, 1.0 - pair_distance / 12.0)
                other_path_term = 1.0 / (1.0 + self._distance_to_trajectory(scene, other_track))
                if visibility_by_object.get(other_track.object_id) == VisibilityState.VISIBLE:
                    neighbor_interaction_strength += 0.4 * interaction_term + 0.2 * other_path_term
                else:
                    hidden_neighbor_strength += 0.7 * interaction_term + 0.4 * other_path_term

            energy = (
                1.4 * distance_to_trajectory
                + 1.0 * distance_to_first_waypoint
                + 0.04 * distance_to_asker
                + 2.0 * object_track.conflict_score
                + 1.2 * object_track.uncertainty_score
                + 0.8 * self._status_rank(object_track)
                - 0.7 * self._support_count(object_track)
                - 1.4 * path_relevant_confidence
                - 0.8 * cooperative_support_confidence
                - 0.7 * min(2.0, neighbor_interaction_strength)
                - 1.1 * min(2.0, hidden_neighbor_strength)
                - 0.25 * object_track.confidence
            )
            scored_objects.append((energy, object_track))

        ordered = tuple(
            object_track
            for _energy, object_track in sorted(
                scored_objects,
                key=lambda item: (
                    item[0],
                    self._distance_to_trajectory(scene, item[1]),
                    self._distance_to_asker(scene, item[1]),
                    item[1].object_id,
                ),
            )[:max_results]
        )
        return QueryResult(scene=scene, objects=ordered)

    def _ranked_role_scores(
        self,
        sample: BenchmarkSample,
        role: str,
        max_results: int,
        min_distance_to_asker: float = 0.0,
        max_distance_to_trajectory: float = 5.0,
    ) -> tuple[_RoleScoredObject, ...]:
        if role == "blocker":
            return self._blocker_role_scores(sample, max_results=max_results)
        if role == "hidden_relevant":
            return self._hidden_relevant_role_scores(
                sample,
                max_results=max_results,
                min_distance_to_asker=min_distance_to_asker,
                max_distance_to_trajectory=max_distance_to_trajectory,
            )
        return ()

    def _hidden_relevant_role_scores(
        self,
        sample: BenchmarkSample,
        max_results: int,
        min_distance_to_asker: float = 0.0,
        max_distance_to_trajectory: float = 5.0,
    ) -> tuple[_RoleScoredObject, ...]:
        scene = sample.scene
        visibility_by_object = self._visibility_lookup(scene, scene.asker_agent_id)
        occluded_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) == VisibilityState.OCCLUDED
        )

        role_scores: list[_RoleScoredObject] = []
        for object_track in occluded_objects:
            distance_to_trajectory = self._distance_to_trajectory(scene, object_track)
            if distance_to_trajectory > max_distance_to_trajectory:
                continue
            distance_to_asker = self._distance_to_asker(scene, object_track)
            if distance_to_asker < min_distance_to_asker:
                continue
            score = (
                2.5 / (1.0 + distance_to_trajectory)
                + 0.7 / (1.0 + distance_to_asker)
                + 0.35 * self._support_count(object_track)
                + 0.25 * object_track.confidence
                - 0.25 * self._status_rank(object_track)
                - 0.45 * object_track.conflict_score
                - 0.3 * object_track.uncertainty_score
            )
            role_scores.append(
                _RoleScoredObject(
                    object_track=object_track,
                    role="hidden_relevant",
                    score=score,
                    distance_to_trajectory=distance_to_trajectory,
                    distance_to_asker=distance_to_asker,
                    support_count=self._support_count(object_track),
                    visibility_state=visibility_by_object.get(object_track.object_id),
                )
            )

        return tuple(
            sorted(
                role_scores,
                key=lambda item: (
                    -item.score,
                    item.distance_to_trajectory,
                    item.distance_to_asker,
                    -item.support_count,
                    item.object_track.object_id,
                ),
            )[:max_results]
        )

    def _notable_role_scores(
        self,
        sample: BenchmarkSample,
        max_results: int,
    ) -> tuple[_RoleScoredObject, ...]:
        scene = sample.scene
        visibility_by_object = self._visibility_lookup(scene, scene.asker_agent_id)
        visible_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) == VisibilityState.VISIBLE
        )
        role_scores: list[_RoleScoredObject] = []
        for object_track in visible_objects:
            distance_to_trajectory = self._distance_to_trajectory(scene, object_track)
            if distance_to_trajectory > self._visible_notable_max_distance:
                continue
            distance_to_asker = self._distance_to_asker(scene, object_track)
            distance_to_first_waypoint = self._distance_to_first_waypoint(scene, object_track)
            path_relevant_confidence = self._relation_confidence(
                scene,
                object_track.object_id,
                RelationType.PATH_RELEVANT,
                scene.asker_agent_id,
            )
            first_waypoint_confidence = self._relation_confidence(
                scene,
                object_track.object_id,
                RelationType.NEAR_FIRST_WAYPOINT,
                scene.asker_agent_id,
            )
            cooperative_support_confidence = self._relation_confidence(
                scene,
                object_track.object_id,
                RelationType.COOPERATIVELY_SUPPORTED,
                scene.asker_agent_id,
            )
            low_conflict_confidence = self._relation_confidence(
                scene,
                object_track.object_id,
                RelationType.LOW_CONFLICT,
                scene.asker_agent_id,
            )
            score = (
                1.6 / (1.0 + distance_to_trajectory)
                + 0.8 / (1.0 + distance_to_first_waypoint)
                + 1.2 * path_relevant_confidence
                + 0.9 * first_waypoint_confidence
                + 0.6 * cooperative_support_confidence
                + 0.5 * low_conflict_confidence
                + 0.25 * self._support_count(object_track)
                + 0.2 * object_track.confidence
                - 0.2 * self._status_rank(object_track)
                - 0.45 * object_track.conflict_score
                - 0.35 * object_track.uncertainty_score
                - 0.06 * distance_to_asker
            )
            role_scores.append(
                _RoleScoredObject(
                    object_track=object_track,
                    role="visible_relevant",
                    score=score,
                    distance_to_trajectory=distance_to_trajectory,
                    distance_to_asker=distance_to_asker,
                    support_count=self._support_count(object_track),
                    visibility_state=visibility_by_object.get(object_track.object_id),
                )
            )

        return tuple(
            sorted(
                role_scores,
                key=lambda item: (
                    -item.score,
                    item.distance_to_trajectory,
                    item.distance_to_asker,
                    -item.support_count,
                    item.object_track.object_id,
                ),
            )[:max_results]
        )

    def _top_occluding_objects(self, sample: BenchmarkSample, max_results: int = 3) -> QueryResult:
        ranked_scores = self._ranked_role_scores(sample, role="blocker", max_results=max_results)
        if ranked_scores:
            ranked_scores = self._select_occluding_score_set(ranked_scores, max_results=max_results)
            return QueryResult(
                scene=sample.scene,
                objects=tuple(item.object_track for item in ranked_scores),
            )

        fallback = self._rank_by_role(sample, role="visible_relevant", max_results=min(2, max_results))
        return QueryResult(scene=sample.scene, objects=fallback.objects)

    def _top_occluding_objects_open_top3(self, sample: BenchmarkSample, max_results: int = 3) -> QueryResult:
        ranked_scores = self._ranked_role_scores(sample, role="blocker", max_results=max_results)
        if ranked_scores:
            return QueryResult(
                scene=sample.scene,
                objects=tuple(item.object_track for item in ranked_scores[:max_results]),
            )

        fallback = self._rank_by_role(sample, role="visible_relevant", max_results=max_results)
        return QueryResult(scene=sample.scene, objects=fallback.objects)

    def _top_occluding_objects_far_supported_top3(
        self,
        sample: BenchmarkSample,
        max_results: int = 3,
    ) -> QueryResult:
        ranked_scores = self._ranked_role_scores(sample, role="blocker", max_results=max_results)
        if ranked_scores:
            ranked_scores = self._select_occluding_far_supported_score_set(
                ranked_scores,
                max_results=max_results,
            )
            return QueryResult(
                scene=sample.scene,
                objects=tuple(item.object_track for item in ranked_scores),
            )

        fallback = self._rank_by_role(sample, role="visible_relevant", max_results=min(2, max_results))
        return QueryResult(scene=sample.scene, objects=fallback.objects)

    def _top_occluding_objects_hybrid_top3(
        self,
        sample: BenchmarkSample,
        max_results: int = 3,
    ) -> QueryResult:
        ranked_scores = self._ranked_role_scores(sample, role="blocker", max_results=max_results)
        if ranked_scores:
            ranked_scores = self._select_occluding_hybrid_score_set(
                ranked_scores,
                max_results=max_results,
            )
            return QueryResult(
                scene=sample.scene,
                objects=tuple(item.object_track for item in ranked_scores),
            )

        fallback = self._rank_by_role(sample, role="visible_relevant", max_results=min(2, max_results))
        return QueryResult(scene=sample.scene, objects=fallback.objects)

    def _top_occluding_objects_risk_adaptive(
        self,
        sample: BenchmarkSample,
        policy: OccludingSelectionPolicy,
    ) -> QueryResult:
        shortlist_size = max(policy.max_results, 4)
        ranked_scores = self._ranked_role_scores(sample, role="blocker", max_results=shortlist_size)
        if ranked_scores:
            ranked_scores = self._select_occluding_risk_adaptive_score_set(
                ranked_scores,
                policy=policy,
            )
            ranked_scores = self._backfill_occluding_scores_with_visible_risk(
                sample=sample,
                selected_scores=ranked_scores,
                policy=policy,
            )
            return QueryResult(
                scene=sample.scene,
                objects=tuple(item.object_track for item in ranked_scores),
            )

        fallback_scores = self._visible_occluding_fallback_scores(
            sample=sample,
            excluded_object_ids=(),
            max_results=min(policy.min_results_with_visible_fallback, policy.max_results),
        )
        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in fallback_scores),
        )

    def _backfill_occluding_scores_with_visible_risk(
        self,
        sample: BenchmarkSample,
        selected_scores: tuple[_RoleScoredObject, ...],
        policy: OccludingSelectionPolicy,
    ) -> tuple[_RoleScoredObject, ...]:
        if not policy.enable_visible_fallback:
            return selected_scores[: policy.max_results]
        if len(selected_scores) >= policy.min_results_with_visible_fallback:
            return selected_scores[: policy.max_results]

        excluded_object_ids = tuple(item.object_track.object_id for item in selected_scores)
        fallback_scores = self._visible_occluding_fallback_scores(
            sample=sample,
            excluded_object_ids=excluded_object_ids,
            max_results=policy.min_results_with_visible_fallback - len(selected_scores),
        )
        return tuple((*selected_scores, *fallback_scores))[: policy.max_results]

    def _visible_occluding_fallback_scores(
        self,
        sample: BenchmarkSample,
        excluded_object_ids: tuple[str, ...],
        max_results: int,
    ) -> tuple[_RoleScoredObject, ...]:
        if max_results <= 0:
            return ()

        scene = sample.scene
        excluded_ids = set(excluded_object_ids)
        visibility_by_object = self._visibility_lookup(scene, scene.asker_agent_id)
        visible_objects = tuple(
            object_track
            for object_track in scene.object_tracks
            if visibility_by_object.get(object_track.object_id) == VisibilityState.VISIBLE
            and object_track.object_id not in excluded_ids
        )
        if not visible_objects:
            return ()

        trajectory_distances = tuple(
            self._distance_to_trajectory(scene, object_track)
            for object_track in visible_objects
        )
        asker_distances = tuple(
            self._distance_to_asker(scene, object_track)
            for object_track in visible_objects
        )
        supports = tuple(float(self._support_count(object_track)) for object_track in visible_objects)
        confidences = tuple(float(object_track.confidence) for object_track in visible_objects)
        trajectory_components = self._normalize_low_values(trajectory_distances)
        asker_components = self._normalize_low_values(asker_distances)
        support_components = self._normalize_high_values(supports)
        confidence_components = self._normalize_high_values(confidences)

        fallback_scores: list[_RoleScoredObject] = []
        for index, object_track in enumerate(visible_objects):
            score = (
                0.45 * trajectory_components[index]
                + 0.2 * asker_components[index]
                + 0.2 * support_components[index]
                + 0.15 * confidence_components[index]
                - 0.08 * self._status_rank(object_track)
                - 0.2 * object_track.conflict_score
                - 0.15 * object_track.uncertainty_score
            )
            fallback_scores.append(
                _RoleScoredObject(
                    object_track=object_track,
                    role="visible_occluding_fallback",
                    score=score,
                    distance_to_trajectory=trajectory_distances[index],
                    distance_to_asker=asker_distances[index],
                    support_count=self._support_count(object_track),
                    visibility_state=visibility_by_object.get(object_track.object_id),
                )
            )

        return tuple(
            sorted(
                fallback_scores,
                key=lambda item: (
                    -item.score,
                    item.distance_to_trajectory,
                    item.distance_to_asker,
                    -item.support_count,
                    item.object_track.object_id,
                ),
            )[:max_results]
        )

    @staticmethod
    def _select_occluding_score_set(
        ranked_scores: tuple[_RoleScoredObject, ...],
        max_results: int,
    ) -> tuple[_RoleScoredObject, ...]:
        """Keep top blockers while admitting a strong third candidate for Phase 8 recall."""

        if len(ranked_scores) <= 2 or max_results <= 2:
            return ranked_scores[:max_results]

        selected = list(ranked_scores[:2])
        third = ranked_scores[2]
        second = ranked_scores[1]
        best_hidden_distance = (
            min(third.aligned_hidden_distances_to_trajectory)
            if third.aligned_hidden_distances_to_trajectory
            else float("inf")
        )
        third_is_path_relevant = third.distance_to_trajectory <= 7.0 or best_hidden_distance <= 5.0
        third_is_well_aligned = third.best_alignment_radians <= 0.35
        third_is_score_competitive = third.score >= second.score - 0.75
        if third_is_path_relevant and third_is_well_aligned and third_is_score_competitive:
            selected.append(third)

        return tuple(selected[:max_results])

    @staticmethod
    def _select_occluding_far_supported_score_set(
        ranked_scores: tuple[_RoleScoredObject, ...],
        max_results: int,
    ) -> tuple[_RoleScoredObject, ...]:
        """Admit far, grounded third blockers while filtering mid-distance extras."""

        if len(ranked_scores) <= 2 or max_results <= 2:
            return ranked_scores[:max_results]

        selected = list(ranked_scores[:2])
        third = ranked_scores[2]
        third_is_grounded = third.object_track.status != TrackStatus.CANDIDATE
        third_is_far_context = third.distance_to_asker >= 70.0
        if third_is_grounded and third_is_far_context:
            selected.append(third)

        return tuple(selected[:max_results])

    @staticmethod
    def _select_occluding_hybrid_score_set(
        ranked_scores: tuple[_RoleScoredObject, ...],
        max_results: int,
    ) -> tuple[_RoleScoredObject, ...]:
        """Admit third blockers that match far-track or near-mid blocker evidence."""

        if len(ranked_scores) <= 2 or max_results <= 2:
            return ranked_scores[:max_results]

        selected = list(ranked_scores[:2])
        third = ranked_scores[2]
        third_is_grounded = third.object_track.status != TrackStatus.CANDIDATE
        third_is_far_supported_context = third_is_grounded and third.distance_to_asker >= 70.0
        third_is_near_mid_context = (
            third.distance_to_asker <= 45.0
            and third.distance_to_trajectory <= 16.0
            and third.best_alignment_radians <= 0.38
        )
        if third_is_far_supported_context or third_is_near_mid_context:
            selected.append(third)

        return tuple(selected[:max_results])

    @classmethod
    def _select_occluding_risk_adaptive_score_set(
        cls,
        ranked_scores: tuple[_RoleScoredObject, ...],
        policy: OccludingSelectionPolicy,
    ) -> tuple[_RoleScoredObject, ...]:
        """Select blockers by relative occlusion risk instead of fixed scene distances."""

        if len(ranked_scores) <= 2 or policy.max_results <= 2:
            return ranked_scores[: policy.max_results]

        risks = cls._occluding_relative_risks(ranked_scores, policy)
        selected = list(ranked_scores[:2])
        second_risk = risks[1]
        total_risk = sum(risks)
        selected_risk = sum(risks[:2])
        for index, candidate in enumerate(ranked_scores[2:], start=2):
            candidate_risk = risks[index]
            relative_to_second = (
                candidate_risk / second_risk
                if second_risk > 0.0
                else 0.0
            )
            selected_coverage = (
                selected_risk / total_risk
                if total_risk > 0.0
                else 1.0
            )
            if (
                (
                    candidate_risk >= policy.third_candidate_min_risk * policy.caution_multiplier
                    and relative_to_second >= policy.third_candidate_min_relative_to_second
                )
                or (
                    selected_coverage < policy.top_two_risk_coverage_target
                    and candidate_risk > 0.0
                )
            ):
                selected.append(candidate)
                selected_risk += candidate_risk
            if len(selected) >= policy.max_results:
                break

        return tuple(selected[: policy.max_results])

    @classmethod
    def _occluding_relative_risks(
        cls,
        ranked_scores: tuple[_RoleScoredObject, ...],
        policy: OccludingSelectionPolicy,
    ) -> tuple[float, ...]:
        model_scores = tuple(item.score for item in ranked_scores)
        trajectory_distances = tuple(item.distance_to_trajectory for item in ranked_scores)
        alignments = tuple(item.best_alignment_radians for item in ranked_scores)
        supports = tuple(float(item.support_count) for item in ranked_scores)
        hidden_distances = tuple(
            min(item.aligned_hidden_distances_to_trajectory)
            if item.aligned_hidden_distances_to_trajectory
            else max(trajectory_distances, default=1.0)
            for item in ranked_scores
        )

        model_components = cls._normalize_high_values(model_scores)
        trajectory_components = cls._normalize_low_values(trajectory_distances)
        alignment_components = cls._normalize_low_values(alignments)
        support_components = cls._normalize_high_values(supports)
        hidden_components = cls._normalize_low_values(hidden_distances)

        risks: list[float] = []
        for index, item in enumerate(ranked_scores):
            risk = (
                policy.geometric_weight * trajectory_components[index]
                + policy.alignment_weight * alignment_components[index]
                + policy.hidden_relevance_weight * hidden_components[index]
                + policy.provenance_weight * support_components[index]
                + policy.model_score_weight * model_components[index]
            )
            if item.object_track.status == TrackStatus.CANDIDATE:
                risk -= policy.candidate_penalty
            risks.append(max(0.0, min(1.0, risk)))
        return tuple(risks)

    @staticmethod
    def _normalize_high_values(values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            return ()
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return tuple(1.0 for _value in values)
        return tuple((value - minimum) / (maximum - minimum) for value in values)

    @staticmethod
    def _normalize_low_values(values: tuple[float, ...]) -> tuple[float, ...]:
        high_values = _BaseQueryHandler._normalize_high_values(values)
        return tuple(1.0 - value for value in high_values)

    def _top_occluding_objects_with_llm(
        self,
        sample: BenchmarkSample,
        llm_client: OccludingObjectsBatchLLMClient,
        shortlist_size: int = 4,
        max_results: int = 2,
    ) -> QueryResult:
        ranked_scores = self._ranked_role_scores(sample, role="blocker", max_results=shortlist_size)
        if not ranked_scores:
            return self._top_occluding_objects(sample, max_results=max_results)

        rank_items = tuple(
            OccludingObjectLLMRankItem(
                object_id=item.object_track.object_id,
                object_type=item.object_track.object_type,
                distance_to_asker=item.distance_to_asker,
                distance_to_trajectory=item.distance_to_trajectory,
                base_score=item.score,
                status=item.object_track.status.value,
                support_count=item.support_count,
                aligned_hidden_object_ids=item.aligned_hidden_object_ids,
                aligned_hidden_distances_to_trajectory=item.aligned_hidden_distances_to_trajectory,
                best_alignment_radians=item.best_alignment_radians,
            )
            for item in ranked_scores
        )
        reranked = llm_client.rerank_occluding_candidates(
            asker_agent_id=sample.scene.asker_agent_id,
            raw_question=sample.scene.raw_question,
            candidates=rank_items,
        )
        reranked_by_id = {item.object_id: item.score for item in reranked}
        blended = tuple(
            sorted(
                ranked_scores,
                key=lambda item: (
                    -(
                        0.65 * item.score
                        + 0.35 * reranked_by_id.get(item.object_track.object_id, 0.0)
                    ),
                    item.distance_to_trajectory,
                    item.distance_to_asker,
                    item.object_track.object_id,
                ),
            )[:max_results]
        )
        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in blended),
        )

    def _top_notable_visible_with_llm(
        self,
        sample: BenchmarkSample,
        llm_client: NotableObjectsBatchLLMClient,
        shortlist_size: int = 4,
        max_results: int = 2,
    ) -> QueryResult:
        ranked_scores = self._notable_role_scores(sample, max_results=shortlist_size)
        if not ranked_scores:
            return self._top_notable_visible(sample, max_results=max_results)

        rank_items = tuple(
            NotableObjectLLMRankItem(
                object_id=item.object_track.object_id,
                object_type=item.object_track.object_type,
                distance_to_asker=item.distance_to_asker,
                distance_to_trajectory=item.distance_to_trajectory,
                distance_to_first_waypoint=self._distance_to_first_waypoint(sample.scene, item.object_track),
                base_score=item.score,
                status=item.object_track.status.value,
                support_count=item.support_count,
                conflict_score=item.object_track.conflict_score,
                uncertainty_score=item.object_track.uncertainty_score,
            )
            for item in ranked_scores
        )
        reranked = llm_client.rerank_notable_candidates(
            asker_agent_id=sample.scene.asker_agent_id,
            raw_question=sample.scene.raw_question,
            candidates=rank_items,
        )
        reranked_by_id = {item.object_id: item.score for item in reranked}
        blended = tuple(
            sorted(
                ranked_scores,
                key=lambda item: (
                    -(
                        0.65 * item.score
                        + 0.35 * reranked_by_id.get(item.object_track.object_id, 0.0)
                    ),
                    item.distance_to_trajectory,
                    item.distance_to_asker,
                    item.object_track.object_id,
                ),
            )[:max_results]
        )
        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in blended),
        )

    def _top_hidden_relevant(
        self,
        sample: BenchmarkSample,
        max_results: int = 2,
        selection_policy: InvisibleSelectionPolicy | None = None,
    ) -> QueryResult:
        policy = selection_policy or InvisibleSelectionPolicy(
            min_distance_to_asker=0.0,
            max_distance_to_trajectory=5.0,
        )
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=max_results,
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=policy.max_distance_to_trajectory,
        )
        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in ranked_scores),
        )

    def _top_hidden_relevant_road_region(
        self,
        sample: BenchmarkSample,
        selection_policy: InvisibleSelectionPolicy,
    ) -> QueryResult:
        policy = selection_policy
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=max(policy.shortlist_size, policy.max_results),
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=policy.max_distance_to_trajectory,
        )
        rescored: list[tuple[float, _RoleScoredObject]] = []
        for item in ranked_scores:
            position = item.object_track.position
            abs_y = abs(position.y)
            road_region_score = item.score
            if policy.lateral_relevance_min_abs_y <= abs_y <= policy.lateral_relevance_max_abs_y:
                road_region_score += policy.lateral_relevance_bonus
            if (
                abs_y < policy.far_centerline_abs_y
                and item.distance_to_asker >= policy.far_centerline_min_distance_to_asker
            ):
                road_region_score -= policy.far_centerline_penalty
            if road_region_score < policy.road_region_min_score:
                continue
            rescored.append((road_region_score, item))

        selected = tuple(
            item
            for _, item in sorted(
                rescored,
                key=lambda value: (
                    -value[0],
                    value[1].distance_to_trajectory,
                    value[1].distance_to_asker,
                    -value[1].support_count,
                    value[1].object_track.object_id,
                ),
            )[: policy.max_results]
        )
        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in selected),
        )

    def _top_hidden_relevant_temporal_guard(
        self,
        sample: BenchmarkSample,
        selection_policy: InvisibleSelectionPolicy,
    ) -> QueryResult:
        policy = selection_policy
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=max(policy.shortlist_size, policy.max_results),
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=policy.max_distance_to_trajectory,
        )
        asker = next(
            (agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id),
            None,
        )
        selected: list[_RoleScoredObject] = []
        for item in ranked_scores:
            if asker is not None:
                relative_x = item.object_track.position.x - asker.pose.position.x
                relative_y = item.object_track.position.y - asker.pose.position.y
                is_far_behind_centerline = (
                    relative_x <= policy.far_behind_max_relative_x
                    and abs(relative_y) < policy.far_behind_centerline_abs_y
                    and item.distance_to_asker >= policy.far_behind_min_distance_to_asker
                )
                if is_far_behind_centerline:
                    continue
            selected.append(item)
            if len(selected) >= policy.max_results:
                break
        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in selected),
        )

    def _top_hidden_relevant_backtrack_guard(
        self,
        sample: BenchmarkSample,
        selection_policy: InvisibleSelectionPolicy,
    ) -> QueryResult:
        policy = selection_policy
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=max(policy.shortlist_size, policy.max_results),
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=policy.max_distance_to_trajectory,
        )
        asker = next(
            (agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id),
            None,
        )
        selected: list[_RoleScoredObject] = []
        for item in ranked_scores:
            if asker is not None:
                relative_x = item.object_track.position.x - asker.pose.position.x
                relative_y = item.object_track.position.y - asker.pose.position.y
                is_backtrack_centerline_clutter = (
                    relative_x <= policy.backtrack_max_relative_x
                    and abs(relative_y) < policy.backtrack_centerline_abs_y
                    and item.distance_to_trajectory < policy.backtrack_max_distance_to_trajectory
                )
                if is_backtrack_centerline_clutter:
                    continue
            selected.append(item)
            if len(selected) >= policy.max_results:
                break
        return QueryResult(
            scene=sample.scene,
            objects=tuple(item.object_track for item in selected),
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = 2.718281828459045 ** (-value)
            return 1.0 / (1.0 + z)
        z = 2.718281828459045 ** value
        return z / (1.0 + z)

    def _logreg_feature_values(
        self,
        sample: BenchmarkSample,
        item: _RoleScoredObject,
        rank: int,
        feature_names: list[str],
    ) -> list[float]:
        track = item.object_track
        asker = next(
            (agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id),
            None,
        )
        relative_x = track.position.x - asker.pose.position.x if asker is not None else track.position.x
        relative_y = track.position.y - asker.pose.position.y if asker is not None else track.position.y
        raw_values = {
            "rank": float(rank),
            "role_score": float(item.score),
            "relative_x": float(relative_x),
            "relative_y": float(relative_y),
            "abs_relative_x": abs(float(relative_x)),
            "abs_relative_y": abs(float(relative_y)),
            "distance_to_asker": float(item.distance_to_asker),
            "distance_to_trajectory": float(item.distance_to_trajectory),
            "support_count": float(item.support_count),
            "confidence": float(track.confidence),
            "conflict_score": float(track.conflict_score),
            "uncertainty_score": float(track.uncertainty_score),
            "age_frames": float(track.age_frames),
            "miss_count": float(track.miss_count),
            "status=confirmed": 1.0 if track.status.value == "confirmed" else 0.0,
            "status=supported": 1.0 if track.status.value == "supported" else 0.0,
            "status=candidate": 1.0 if track.status.value == "candidate" else 0.0,
        }
        return [raw_values.get(name, 0.0) for name in feature_names]

    def _top_hidden_relevant_logreg_acceptor(
        self,
        sample: BenchmarkSample,
        selection_policy: InvisibleSelectionPolicy,
        acceptor_model: dict[str, object],
    ) -> QueryResult:
        policy = selection_policy
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=max(policy.shortlist_size, policy.max_results),
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=policy.max_distance_to_trajectory,
        )
        selected = self._accepted_logreg_hidden_items(sample, ranked_scores, policy, acceptor_model)
        return QueryResult(scene=sample.scene, objects=tuple(item.object_track for item in selected))

    def _top_hidden_relevant_mlp_acceptor(
        self,
        sample: BenchmarkSample,
        selection_policy: InvisibleSelectionPolicy,
        acceptor_model: dict[str, object],
    ) -> QueryResult:
        policy = selection_policy
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=max(policy.shortlist_size, policy.max_results),
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=policy.max_distance_to_trajectory,
        )
        selected = self._accepted_mlp_hidden_items(sample, ranked_scores, policy, acceptor_model)
        return QueryResult(scene=sample.scene, objects=tuple(item.object_track for item in selected))

    def _accepted_logreg_hidden_items(
        self,
        sample: BenchmarkSample,
        ranked_scores: tuple[_RoleScoredObject, ...],
        policy: InvisibleSelectionPolicy,
        acceptor_model: dict[str, object],
    ) -> tuple[_RoleScoredObject, ...]:
        feature_names = [str(item) for item in acceptor_model.get("feature_names", [])]
        normalization = acceptor_model.get("normalization", {})
        if not isinstance(normalization, dict):
            normalization = {}
        means = [float(item) for item in normalization.get("mean", [])]
        stds = [float(item) if float(item) != 0.0 else 1.0 for item in normalization.get("std", [])]
        weights = [float(item) for item in acceptor_model.get("weights", [])]
        bias = float(acceptor_model.get("bias", 0.0))
        threshold = float(acceptor_model.get("threshold", 0.5))
        if not feature_names or len(feature_names) != len(means) or len(feature_names) != len(stds) or len(feature_names) != len(weights):
            return ()
        scored: list[tuple[float, _RoleScoredObject]] = []
        for rank, item in enumerate(ranked_scores, start=1):
            raw_features = self._logreg_feature_values(sample, item, rank, feature_names)
            normalized = [
                (value - means[index]) / stds[index]
                for index, value in enumerate(raw_features)
            ]
            logit = bias + sum(weight * value for weight, value in zip(weights, normalized))
            probability = self._sigmoid(logit)
            if probability >= threshold:
                scored.append((probability, item))

        return tuple(
            item
            for _, item in sorted(
                scored,
                key=lambda value: (
                    -value[0],
                    value[1].distance_to_trajectory,
                    value[1].distance_to_asker,
                    -value[1].support_count,
                    value[1].object_track.object_id,
                ),
            )[: policy.max_results]
        )

    def _accepted_mlp_hidden_items(
        self,
        sample: BenchmarkSample,
        ranked_scores: tuple[_RoleScoredObject, ...],
        policy: InvisibleSelectionPolicy,
        acceptor_model: dict[str, object],
    ) -> tuple[_RoleScoredObject, ...]:
        feature_names = [str(item) for item in acceptor_model.get("feature_names", [])]
        normalization = acceptor_model.get("normalization", {})
        if not isinstance(normalization, dict):
            normalization = {}
        means = [float(item) for item in normalization.get("mean", [])]
        stds = [float(item) if float(item) != 0.0 else 1.0 for item in normalization.get("std", [])]
        w1 = [
            [float(value) for value in row]
            for row in acceptor_model.get("w1", [])
            if isinstance(row, list)
        ]
        b1 = [float(value) for value in acceptor_model.get("b1", [])]
        w2 = [float(value) for value in acceptor_model.get("w2", [])]
        b2 = float(acceptor_model.get("b2", 0.0))
        threshold = float(acceptor_model.get("threshold", 0.5))
        if (
            not feature_names
            or len(feature_names) != len(means)
            or len(feature_names) != len(stds)
            or not w1
            or len(w1) != len(b1)
            or len(w1) != len(w2)
            or any(len(row) != len(feature_names) for row in w1)
        ):
            return ()

        scored: list[tuple[float, _RoleScoredObject]] = []
        for rank, item in enumerate(ranked_scores, start=1):
            raw_features = self._logreg_feature_values(sample, item, rank, feature_names)
            normalized = [
                (value - means[index]) / stds[index]
                for index, value in enumerate(raw_features)
            ]
            hidden_values = [
                tanh(bias + sum(weight * value for weight, value in zip(row, normalized)))
                for row, bias in zip(w1, b1)
            ]
            logit = b2 + sum(weight * value for weight, value in zip(w2, hidden_values))
            probability = self._sigmoid(logit)
            if probability >= threshold:
                scored.append((probability, item))

        return tuple(
            item
            for _, item in sorted(
                scored,
                key=lambda value: (
                    -value[0],
                    value[1].distance_to_trajectory,
                    value[1].distance_to_asker,
                    -value[1].support_count,
                    value[1].object_track.object_id,
                ),
            )[: policy.max_results]
        )

    def _top_hidden_relevant_logreg_lateral_rescue(
        self,
        sample: BenchmarkSample,
        selection_policy: InvisibleSelectionPolicy,
        acceptor_model: dict[str, object],
    ) -> QueryResult:
        policy = selection_policy
        ranked_scores = self._ranked_role_scores(
            sample,
            role="hidden_relevant",
            max_results=max(policy.shortlist_size, policy.max_results),
            min_distance_to_asker=policy.min_distance_to_asker,
            max_distance_to_trajectory=policy.max_distance_to_trajectory,
        )
        accepted = self._accepted_logreg_hidden_items(sample, ranked_scores, policy, acceptor_model)
        if accepted:
            return QueryResult(scene=sample.scene, objects=tuple(item.object_track for item in accepted))

        asker = next(
            (agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id),
            None,
        )
        selected: list[_RoleScoredObject] = []
        for item in ranked_scores:
            track = item.object_track
            if track.status == TrackStatus.CANDIDATE:
                continue
            if item.support_count < policy.rescue_min_support_count:
                continue
            if item.distance_to_trajectory < policy.rescue_min_distance_to_trajectory:
                continue
            if asker is not None:
                relative_x = track.position.x - asker.pose.position.x
                relative_y = track.position.y - asker.pose.position.y
                if relative_x <= policy.rescue_min_relative_x:
                    continue
                if abs(relative_y) < policy.rescue_min_abs_y:
                    continue
            selected.append(item)
            if len(selected) >= policy.max_results:
                break

        return QueryResult(scene=sample.scene, objects=tuple(item.object_track for item in selected))

    def _render_objects(
        self,
        sample: BenchmarkSample,
        task_type: BenchmarkTaskType,
        result: QueryResult,
        empty_message: str,
        prefix: str = "Relevant objects",
    ) -> BenchmarkAnswer:
        object_ids = tuple(object_track.object_id for object_track in result.objects)
        if not object_ids:
            return BenchmarkAnswer(
                sample_id=sample.sample_id,
                task_type=task_type,
                answer_text=empty_message,
                object_ids=(),
                supported=True,
            )
        rendered_objects = ", ".join(object_ids)
        return BenchmarkAnswer(
            sample_id=sample.sample_id,
            task_type=task_type,
            answer_text=f"{prefix}: {rendered_objects}.",
            object_ids=object_ids,
            supported=True,
        )

    @staticmethod
    def _prefer_grounded_objects(result: QueryResult) -> QueryResult:
        grounded_objects = tuple(
            object_track
            for object_track in result.objects
            if object_track.status != TrackStatus.CANDIDATE
        )
        if grounded_objects:
            return QueryResult(scene=result.scene, objects=grounded_objects)
        return result


class NotableObjectsHandler(_BaseQueryHandler):
    """Handles qa_type_id 11 notable-object questions."""

    task_type = BenchmarkTaskType.NOTABLE_OBJECTS

    def __init__(
        self,
        ranker: str = "heuristic",
        llm_client: NotableObjectsBatchLLMClient | None = None,
    ) -> None:
        super().__init__()
        self._ranker = ranker
        self._llm_client = llm_client

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        result = (
            self._top_notable_visible_with_llm(sample, self._llm_client)
            if self._ranker == "llm" and self._llm_client is not None
            else
            self._top_notable_visible_energy(sample)
            if self._ranker == "energy"
            else self._top_notable_visible(sample)
        )
        result = self._prefer_grounded_objects(result)
        return self._render_objects(
            sample,
            task_type=self.task_type,
            result=result,
            empty_message="There is no notable object visible to you near your planned future trajectory.",
            prefix="Notable visible objects",
        )


class OccludingObjectsHandler(_BaseQueryHandler):
    """Handles qa_type_id 12 occluding-object questions."""

    task_type = BenchmarkTaskType.OCCLUDING_OBJECTS

    def __init__(
        self,
        ranker: str = "risk_adaptive",
        llm_client: OccludingObjectsBatchLLMClient | None = None,
        selection_policy: OccludingSelectionPolicy | None = None,
    ) -> None:
        super().__init__()
        self._ranker = ranker
        self._llm_client = llm_client
        self._selection_policy = selection_policy or OccludingSelectionPolicy()

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        result = (
            self._top_occluding_objects_with_llm(sample, self._llm_client)
            if self._llm_client is not None
            else
            self._top_occluding_objects_risk_adaptive(sample, self._selection_policy)
            if self._ranker == "risk_adaptive"
            else
            self._top_occluding_objects_hybrid_top3(sample)
            if self._ranker == "top3_hybrid"
            else
            self._top_occluding_objects_far_supported_top3(sample)
            if self._ranker == "top3_far_supported"
            else
            self._top_occluding_objects_open_top3(sample)
            if self._ranker == "top3_open"
            else self._top_occluding_objects(sample)
        )
        return self._render_objects(
            sample,
            task_type=self.task_type,
            result=result,
            empty_message="There is no object currently marked as obstructing your view.",
            prefix="Potentially occluding objects",
        )


class InvisibleObjectsHandler(_BaseQueryHandler):
    """Handles qa_type_id 13 invisible-object questions."""

    task_type = BenchmarkTaskType.INVISIBLE_OBJECTS

    def __init__(
        self,
        ranker: str = "legacy",
        selection_policy: InvisibleSelectionPolicy | None = None,
        acceptor_model: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if ranker not in {
            "legacy",
            "risk_adaptive",
            "road_region",
            "road_region_strict",
            "temporal_guard",
            "backtrack_guard",
            "logreg_acceptor",
            "mlp_acceptor",
            "logreg_legacy_fallback",
            "logreg_lateral_rescue",
        }:
            raise ValueError(f"Unsupported invisible-object ranker: {ranker}")
        self._ranker = ranker
        self._selection_policy = selection_policy or InvisibleSelectionPolicy()
        self._acceptor_model = acceptor_model or {}

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        result = (
            self._top_hidden_relevant_risk_adaptive(sample, self._selection_policy)
            if self._ranker == "risk_adaptive"
            else self._top_hidden_relevant_road_region(sample, self._selection_policy)
            if self._ranker == "road_region"
            else self._top_hidden_relevant_road_region(
                sample,
                replace(
                    self._selection_policy,
                    lateral_relevance_bonus=max(self._selection_policy.lateral_relevance_bonus, 0.45),
                    far_centerline_abs_y=max(self._selection_policy.far_centerline_abs_y, 1.5),
                    far_centerline_penalty=max(self._selection_policy.far_centerline_penalty, 1.25),
                    road_region_min_score=max(self._selection_policy.road_region_min_score, 0.5),
                ),
            )
            if self._ranker == "road_region_strict"
            else self._top_hidden_relevant_temporal_guard(sample, self._selection_policy)
            if self._ranker == "temporal_guard"
            else self._top_hidden_relevant_backtrack_guard(sample, self._selection_policy)
            if self._ranker == "backtrack_guard"
            else self._top_hidden_relevant_logreg_acceptor(sample, self._selection_policy, self._acceptor_model)
            if self._ranker == "logreg_acceptor"
            else self._top_hidden_relevant_mlp_acceptor(sample, self._selection_policy, self._acceptor_model)
            if self._ranker == "mlp_acceptor"
            else self._top_hidden_relevant_logreg_lateral_rescue(sample, self._selection_policy, self._acceptor_model)
            if self._ranker == "logreg_lateral_rescue"
            else (
                logreg_result
                if (logreg_result := self._top_hidden_relevant_logreg_acceptor(sample, self._selection_policy, self._acceptor_model)).objects
                else self._top_hidden_relevant(sample, selection_policy=self._selection_policy)
            )
            if self._ranker == "logreg_legacy_fallback"
            else self._top_hidden_relevant(sample, selection_policy=self._selection_policy)
        )
        return self._render_objects(
            sample,
            task_type=self.task_type,
            result=result,
            empty_message="There is no notable object invisible to you near your planned future trajectory.",
            prefix="Notable invisible objects",
        )


@dataclass(frozen=True)
class MotionPrediction:
    """Projected one-step motion for a scene entity."""

    entity_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    motion_label: str


class ObjectMotionPredictionHandler(_BaseQueryHandler):
    """Handles qa_type_id 15/17 object-motion-prediction questions."""

    task_type = BenchmarkTaskType.OBJECT_MOTION_PREDICTION
    _prediction_horizon_seconds = 1.0

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        predictions = self._predict(sample)
        if not predictions:
            return BenchmarkAnswer(
                sample_id=sample.sample_id,
                task_type=self.task_type,
                answer_text="There are no object tracks available for motion prediction.",
                object_ids=(),
                supported=True,
            )

        rendered_predictions = "; ".join(
            (
                f"{prediction.entity_id}={prediction.motion_label} "
                f"from ({prediction.start_x:.1f}, {prediction.start_y:.1f}) "
                f"to ({prediction.end_x:.1f}, {prediction.end_y:.1f})"
            )
            for prediction in predictions
        )
        return BenchmarkAnswer(
            sample_id=sample.sample_id,
            task_type=self.task_type,
            answer_text=f"Predicted object motion: {rendered_predictions}.",
            object_ids=tuple(prediction.entity_id for prediction in predictions),
            supported=True,
        )

    def _predict(self, sample: BenchmarkSample) -> tuple[MotionPrediction, ...]:
        scene = sample.scene
        visibility_by_object = self._visibility_lookup(scene, scene.asker_agent_id)
        ranked_objects = sorted(
            scene.object_tracks,
            key=lambda object_track: (
                self._distance_to_trajectory(scene, object_track),
                self._status_rank(object_track),
                -object_track.confidence,
                object_track.object_id,
            ),
        )
        relevant_objects = tuple(
            object_track
            for object_track in ranked_objects
            if self._distance_to_trajectory(scene, object_track) <= 8.0
            or visibility_by_object.get(object_track.object_id) in {
                VisibilityState.OCCLUDED,
                VisibilityState.UNCERTAIN,
            }
        )[:3]
        if not relevant_objects:
            relevant_objects = tuple(ranked_objects[:3])

        return tuple(self._predict_object_motion(object_track) for object_track in relevant_objects)

    def _predict_object_motion(self, object_track) -> MotionPrediction:
        velocity = object_track.velocity
        start_x = object_track.position.x
        start_y = object_track.position.y
        if velocity is None:
            return MotionPrediction(
                entity_id=object_track.object_id,
                start_x=start_x,
                start_y=start_y,
                end_x=start_x,
                end_y=start_y,
                motion_label="stationary",
            )

        end_x = start_x + velocity.x * self._prediction_horizon_seconds
        end_y = start_y + velocity.y * self._prediction_horizon_seconds
        speed = (velocity.x**2 + velocity.y**2) ** 0.5
        if speed < 0.1:
            motion_label = "stationary"
        elif abs(velocity.x) >= abs(velocity.y):
            motion_label = "moving forward" if velocity.x >= 0.0 else "moving backward"
        else:
            motion_label = "moving right" if velocity.y >= 0.0 else "moving left"
        return MotionPrediction(
            entity_id=object_track.object_id,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            motion_label=motion_label,
        )


class AgentMotionPredictionHandler(_BaseQueryHandler):
    """Handles qa_type_id 16 agent-motion-prediction questions."""

    task_type = BenchmarkTaskType.AGENT_MOTION_PREDICTION

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        scene = sample.scene
        other_agents = tuple(
            agent for agent in scene.agents if agent.agent_id != scene.asker_agent_id
        )
        if other_agents:
            rendered_predictions = "; ".join(
                self._render_agent_motion(agent)
                for agent in other_agents
            )
            return BenchmarkAnswer(
                sample_id=sample.sample_id,
                task_type=self.task_type,
                answer_text=f"Predicted agent motion: {rendered_predictions}.",
                object_ids=tuple(agent.agent_id for agent in other_agents),
                supported=True,
            )

        points = scene.future_trajectory.points
        if not points:
            return BenchmarkAnswer(
                sample_id=sample.sample_id,
                task_type=self.task_type,
                answer_text="There is no agent trajectory context available for motion prediction.",
                object_ids=(),
                supported=True,
            )

        motion_label = self._trajectory_motion_label(scene)
        final_point = points[-1]
        return BenchmarkAnswer(
            sample_id=sample.sample_id,
            task_type=self.task_type,
            answer_text=(
                "Predicted agent motion: "
                f"{scene.asker_agent_id}={motion_label} toward "
                f"({final_point.x:.1f}, {final_point.y:.1f})."
            ),
            object_ids=(scene.asker_agent_id,),
            supported=True,
        )

    @staticmethod
    def _render_agent_motion(agent) -> str:
        if agent.velocity is not None:
            speed = (agent.velocity.x**2 + agent.velocity.y**2) ** 0.5
            if speed >= 0.1:
                if abs(agent.velocity.x) >= abs(agent.velocity.y):
                    motion_label = "move forward" if agent.velocity.x >= 0.0 else "move backward"
                else:
                    motion_label = "move right" if agent.velocity.y >= 0.0 else "move left"

                end_x = agent.pose.position.x + agent.velocity.x
                end_y = agent.pose.position.y + agent.velocity.y
                return (
                    f"{agent.agent_id}={motion_label} from "
                    f"({agent.pose.position.x:.1f}, {agent.pose.position.y:.1f}) "
                    f"to ({end_x:.1f}, {end_y:.1f})"
                )

        if agent.planned_trajectory is None or not agent.planned_trajectory.points:
            return (
                f"{agent.agent_id}=hold position near "
                f"({agent.pose.position.x:.1f}, {agent.pose.position.y:.1f})"
            )

        final_point = agent.planned_trajectory.points[-1]
        dx = final_point.x
        dy = final_point.y
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            motion_label = "hold position"
        elif abs(dx) >= abs(dy):
            motion_label = "move forward" if dx >= 0.0 else "move backward"
        else:
            motion_label = "move right" if dy >= 0.0 else "move left"
        end_x = agent.pose.position.x + final_point.x
        end_y = agent.pose.position.y + final_point.y
        return (
            f"{agent.agent_id}={motion_label} from "
            f"({agent.pose.position.x:.1f}, {agent.pose.position.y:.1f}) "
            f"to ({end_x:.1f}, {end_y:.1f})"
        )

    @staticmethod
    def _trajectory_motion_label(scene) -> str:
        asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
        points = scene.future_trajectory.points
        if asker is None or not points:
            return "continue along planned trajectory"

        dx = points[-1].x - asker.pose.position.x
        dy = points[-1].y - asker.pose.position.y
        if abs(dx) >= abs(dy):
            return "move forward" if dx >= 0.0 else "move backward"
        return "move right" if dy >= 0.0 else "move left"


class FutureTrajectoryHandler(_BaseQueryHandler):
    """Handles qa_type_id 19 future-trajectory questions."""

    task_type = BenchmarkTaskType.FUTURE_TRAJECTORY

    def __init__(
        self,
        planner: ControlConditionedFutureTrajectoryPlanner | None = None,
    ) -> None:
        super().__init__()
        self._planner = planner or ControlConditionedFutureTrajectoryPlanner()

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        plan = self._planner.plan(sample)
        points = plan.points
        if not points:
            return BenchmarkAnswer(
                sample_id=sample.sample_id,
                task_type=self.task_type,
                answer_text="There is no future trajectory prediction available.",
                object_ids=(),
                supported=True,
            )

        rendered_points = ", ".join(
            f"({point.x:.1f}, {point.y:.1f})" for point in points
        )
        return BenchmarkAnswer(
            sample_id=sample.sample_id,
            task_type=self.task_type,
            answer_text=f"The suggested future trajectory is [{rendered_points}].",
            object_ids=(),
            supported=True,
        )

class ControlSettingsHandler(_BaseQueryHandler):
    """Handles qa_type_id 18 control-settings questions."""

    task_type = BenchmarkTaskType.CONTROL_SETTINGS

    def __init__(
        self,
        selection_policy: str = "rule",
        model: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if selection_policy not in {"rule", "linear_classifier"}:
            raise ValueError(f"Unsupported control selection policy: {selection_policy}")
        self._selection_policy = selection_policy
        self._model = model or {}

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        decision = self._decide(sample)
        speed_label = decision.speed_instruction
        steering_label = decision.steering_instruction
        answer_text = (
            f"The suggested speed setting is: {speed_label}. "
            f"The suggested steering setting is: {steering_label}."
        )
        if decision.object_ids:
            rendered_objects = ", ".join(decision.object_ids)
            answer_text += f" Key objects: {rendered_objects}."
        return BenchmarkAnswer(
            sample_id=sample.sample_id,
            task_type=self.task_type,
            answer_text=answer_text,
            object_ids=decision.object_ids,
            supported=True,
        )

    def _decide(self, sample: BenchmarkSample) -> ControlSettingsDecision:
        return decide_control_settings(
            scene=sample.scene,
            selection_policy=self._selection_policy,
            model=self._model,
        )


class PlanningAwarenessHandler(_BaseQueryHandler):
    """Handles qa_type_id 14 planning-awareness questions."""

    task_type = BenchmarkTaskType.PLANNING_AWARENESS

    def __init__(
        self,
        orchestrator: PlanningAwarenessOrchestrator | None = None,
        selection_source: str = "composition",
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator or build_planning_awareness_orchestrator()
        if selection_source not in {"composition", "orchestrator"}:
            raise ValueError(f"Unsupported planning-awareness selection source: {selection_source}")
        self._selection_source = selection_source

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        planning_result = self._planning_awareness_objects(sample)
        if planning_result.objects:
            ordered_objects = tuple(
                sorted(
                    planning_result.objects,
                    key=lambda object_track: (
                        0
                        if self._visibility_lookup(sample.scene, sample.scene.asker_agent_id).get(
                            object_track.object_id
                        )
                        == VisibilityState.OCCLUDED
                        else 1,
                        self._distance_to_trajectory(sample.scene, object_track),
                        object_track.object_id,
                    ),
                )
            )
            object_ids = tuple(object_track.object_id for object_track in ordered_objects)
            rendered_objects = ", ".join(object_ids)
            return BenchmarkAnswer(
                sample_id=sample.sample_id,
                task_type=self.task_type,
                answer_text=f"Objects to be aware of: {rendered_objects}.",
                object_ids=object_ids,
                supported=True,
            )

        return BenchmarkAnswer(
            sample_id=sample.sample_id,
            task_type=self.task_type,
            answer_text="There is no notable object.",
            object_ids=(),
            supported=True,
        )

    def _planning_awareness_objects(self, sample: BenchmarkSample) -> QueryResult:
        if self._selection_source == "orchestrator":
            decision = self._orchestrator.select(sample.scene)
            return QueryResult(
                scene=sample.scene,
                objects=tuple(candidate.object_track for candidate in decision.selected_candidates),
            )

        hidden_result = self._prefer_grounded_objects(
            self._top_hidden_relevant(sample, max_results=1)
        )
        visible_result = self._prefer_grounded_objects(
            self._top_notable_visible(sample, max_results=1)
        )
        merged_objects = self._merge_unique_objects(hidden_result.objects, visible_result.objects)
        return QueryResult(scene=sample.scene, objects=merged_objects)

    @staticmethod
    def _merge_unique_objects(*object_groups: tuple[ObjectTrack, ...]) -> tuple[ObjectTrack, ...]:
        ordered: list[ObjectTrack] = []
        seen_object_ids: set[str] = set()
        for object_group in object_groups:
            for object_track in object_group:
                if object_track.object_id in seen_object_ids:
                    continue
                ordered.append(object_track)
                seen_object_ids.add(object_track.object_id)
        return tuple(ordered)
