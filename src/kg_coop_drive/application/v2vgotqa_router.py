from __future__ import annotations

from dataclasses import dataclass
from math import atan2, dist
from typing import Protocol

from kg_coop_drive.application.planning_awareness import (
    PlanningAwarenessOrchestrator,
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.application.query_engine import SceneQueryEngine
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType
from kg_coop_drive.domain.scene import QueryResult, RelationType, VisibilityState


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
                if distance_to_trajectory > 4.0:
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
    ) -> tuple[_RoleScoredObject, ...]:
        if role == "blocker":
            return self._blocker_role_scores(sample, max_results=max_results)
        return ()

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
            if distance_to_trajectory > 4.0:
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

    def _top_occluding_objects(self, sample: BenchmarkSample, max_results: int = 2) -> QueryResult:
        ranked_scores = self._ranked_role_scores(sample, role="blocker", max_results=max_results)
        if ranked_scores:
            return QueryResult(
                scene=sample.scene,
                objects=tuple(item.object_track for item in ranked_scores),
            )

        fallback = self._rank_by_role(sample, role="visible_relevant", max_results=max_results)
        return QueryResult(scene=sample.scene, objects=fallback.objects)

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

    def _top_hidden_relevant(self, sample: BenchmarkSample, max_results: int = 2) -> QueryResult:
        return self._rank_by_role(sample, role="hidden_relevant", max_results=max_results)

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
        llm_client: OccludingObjectsBatchLLMClient | None = None,
    ) -> None:
        super().__init__()
        self._llm_client = llm_client

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        result = (
            self._top_occluding_objects_with_llm(sample, self._llm_client)
            if self._llm_client is not None
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

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        result = self._top_hidden_relevant(sample)
        return self._render_objects(
            sample,
            task_type=self.task_type,
            result=result,
            empty_message="There is no notable object invisible to you near your planned future trajectory.",
            prefix="Notable invisible objects",
        )


class PlanningAwarenessHandler(_BaseQueryHandler):
    """Handles qa_type_id 14 planning-awareness questions."""

    task_type = BenchmarkTaskType.PLANNING_AWARENESS

    def __init__(
        self,
        orchestrator: PlanningAwarenessOrchestrator | None = None,
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator or build_planning_awareness_orchestrator()

    def answer(self, sample: BenchmarkSample) -> BenchmarkAnswer:
        decision = self._orchestrator.select(sample.scene)
        if decision.selected_candidates:
            ordered_candidates = tuple(
                sorted(
                    decision.selected_candidates,
                    key=lambda candidate: (
                        0 if candidate.visibility_state == VisibilityState.OCCLUDED else 1,
                        candidate.distance_to_trajectory,
                        candidate.object_track.object_id,
                    ),
                )
            )
            object_ids = tuple(
                candidate.object_track.object_id for candidate in ordered_candidates
            )
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
