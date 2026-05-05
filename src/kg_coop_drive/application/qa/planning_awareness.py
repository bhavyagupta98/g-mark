from __future__ import annotations

"""Planning-awareness ranking strategies for Phase 5A.

This module deliberately keeps the graph-building pipeline separate from the
decision/ranking layer so we can compare multiple planning-awareness rankers
over the same candidate set.

Implemented variants and their paper grounding:

1. ``heuristic``
   A transparent baseline using confidence, support, provenance, visibility,
   uncertainty, conflict, and trajectory proximity.

2. ``relational_importance``
   Inspired by Li et al., "Important Object Identification with
   Semi-Supervised Learning for Autonomous Driving" (ICRA 2022 /
   arXiv:2203.02634). Their paper frames object importance explicitly as an
   important/unimportant decision with relational reasoning over scene objects.
   We adapt that idea to our graph setting by scoring each object as an
   explicit "important for current planning query" candidate using graph-native
   relational features such as trajectory proximity, visibility, asker support,
   and cooperative provenance.
   Source: https://arxiv.org/abs/2203.02634

3. ``risk_aware``
   Inspired by Nyberg et al., "Risk-aware Motion Planning for Autonomous
   Vehicles with Safety Specifications" (IV 2021). That work models risk as a
   combination of violation likelihood and severity, and emphasizes that risk
   should increase with probability, severity, and uncertainty. We adapt that
   idea at the object-ranking level by estimating an object-centric risk signal
   from occlusion/uncertainty/conflict likelihood and trajectory-proximity
   severity.
   Source: https://doi.org/10.1109/IV48863.2021.9575928
   Abstract mirror: https://research.tudelft.nl/en/publications/risk-aware-motion-planning-for-autonomous-vehicles-with-safety-sp

4. ``energy_based``
   Inspired by Tian et al., "KLDrive: Fine-Grained 3D Scene Reasoning for
   Autonomous Driving based on Knowledge Graph" (arXiv:2603.21029). KLDrive
   introduces an energy-based scene fact construction module for consolidating
   multi-source evidence. The paper is aimed at scene-fact construction rather
   than our planning-awareness task specifically, so this implementation is an
   adaptation: we use evidence-derived unary energies plus pairwise redundancy
   penalties to select a more coherent set of awareness objects.
   Source: https://arxiv.org/abs/2603.21029
"""

from dataclasses import dataclass, replace
from enum import Enum
from math import exp, isfinite, dist, tanh
from typing import Any, Protocol

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    ObjectTrack,
    TrackStatus,
    VisibilityState,
)


class PlanningAwarenessRanker(str, Enum):
    """Supported planning-awareness ranking strategies."""

    HEURISTIC = "heuristic"
    RELATIONAL_IMPORTANCE = "relational_importance"
    RISK_AWARE = "risk_aware"
    ENERGY_BASED = "energy_based"
    LLM = "llm"


class PlanningAwarenessSelectionPolicy(str, Enum):
    """Supported final-selection policies for planning-awareness answers."""

    DEFAULT = "default"
    TOP2 = "top2"
    DIVERSE_TOP2 = "diverse_top2"
    COUNT_ADAPTIVE = "count_adaptive"
    LOGREG_ACCEPTOR = "logreg_acceptor"
    MLP_ACCEPTOR = "mlp_acceptor"
    COUNT_GATED_ACCEPTOR = "count_gated_acceptor"
    SOFT_COUNT_GATED_ACCEPTOR = "soft_count_gated_acceptor"
    TRAJECTORY_CALIBRATED_ACCEPTOR = "trajectory_calibrated_acceptor"


@dataclass(frozen=True)
class PlanningAwarenessFeatures:
    """Derived per-object features shared across rankers."""

    distance_to_trajectory: float
    distance_ratio: float
    is_visible: bool
    is_occluded: bool
    is_gt_backed: bool
    is_supported_like: bool
    is_candidate: bool
    is_cooperative: bool
    is_asker_observed: bool
    support_count: int
    confidence: float
    uncertainty_score: float
    conflict_score: float


@dataclass(frozen=True)
class PlanningAwarenessCandidate:
    """One candidate object considered by the planning-awareness orchestrator."""

    object_track: ObjectTrack
    visibility_state: VisibilityState
    distance_to_trajectory: float
    score: float
    rationale: tuple[str, ...]


class PlanningAwarenessScorer(Protocol):
    """Scores scene objects for planning-awareness selection."""

    min_score: float

    def score(
        self,
        scene: CooperativeScene,
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        distance_to_trajectory: float,
    ) -> PlanningAwarenessCandidate:
        """Return a scored planning-awareness candidate."""


@dataclass(frozen=True)
class PlanningAwarenessLLMContext:
    """Compact structured context passed to an LLM ranker implementation."""

    asker_agent_id: str
    raw_question: str
    object_id: str
    object_type: str
    visibility_state: str
    distance_to_trajectory: float
    confidence: float
    status: str
    uncertainty_score: float
    conflict_score: float
    provenance_agents: tuple[str, ...]


class PlanningAwarenessLLMClient(Protocol):
    """External LLM client used only for candidate-level relevance scoring."""

    def score_candidate(self, context: PlanningAwarenessLLMContext) -> tuple[float, tuple[str, ...]]:
        """Return a relevance score in [0, 1] plus a short rationale."""


@dataclass(frozen=True)
class PlanningAwarenessLLMRankItem:
    """One candidate sent to a local LLM reranker.

    We keep this compact and structured so a local LLM server only sees a
    short, graph-produced shortlist rather than the whole scene.
    """

    object_id: str
    object_type: str
    visibility_state: str
    distance_to_trajectory: float
    base_score: float
    confidence: float
    status: str
    uncertainty_score: float
    conflict_score: float
    provenance_agents: tuple[str, ...]


@dataclass(frozen=True)
class PlanningAwarenessLLMRankedItem:
    """One scored/reranked candidate returned by a local LLM."""

    object_id: str
    score: float
    rationale: tuple[str, ...]


class PlanningAwarenessBatchLLMClient(Protocol):
    """Scene-level reranker over a graph-produced shortlist."""

    def rerank_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[PlanningAwarenessLLMRankItem, ...],
    ) -> tuple[PlanningAwarenessLLMRankedItem, ...]:
        """Return ranked/scored shortlist items for final selection."""


@dataclass(frozen=True)
class PlanningAwarenessDecision:
    """Final selection result of the planning-awareness orchestrator."""

    selected_candidates: tuple[PlanningAwarenessCandidate, ...]
    considered_candidates: tuple[PlanningAwarenessCandidate, ...]


PLANNING_LOGREG_FEATURE_NAMES: tuple[str, ...] = (
    "rank",
    "candidate_count",
    "relational_score",
    "score_gap_prev",
    "score_ratio_prev",
    "distance_to_trajectory",
    "distance_to_first_waypoint",
    "distance_to_asker",
    "visibility_visible",
    "visibility_occluded",
    "confidence",
    "source_agent_count",
    "observation_count",
    "uncertainty_score",
    "conflict_score",
    "is_cooperative",
    "is_asker_observed",
    "is_candidate",
    "is_supported_like",
    "nearest_higher_rank_distance",
)

PLANNING_COUNT_GATE_FEATURE_NAMES: tuple[str, ...] = (
    "candidate_count",
    "eligible_count",
    "high_prob_count_0p55",
    "high_prob_count_0p60",
    "high_prob_count_0p65",
    "p1",
    "p2",
    "p3",
    "p4",
    "p_gap_1_2",
    "p_gap_2_3",
    "p_gap_3_4",
    "p_mean_top3",
    "score1",
    "score2",
    "score3",
    "score_gap_1_2",
    "score_gap_2_3",
    "distance_to_trajectory1",
    "distance_to_trajectory2",
    "distance_to_trajectory3",
    "abs_y1",
    "abs_y2",
    "abs_y3",
    "behind_count_top3",
    "visible_count_top3",
    "occluded_count_top3",
    "cooperative_count_top3",
)


def _safe_sigmoid(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _agent_distance_to_object(scene: CooperativeScene, object_track: ObjectTrack) -> float:
    asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
    if asker is None:
        return 0.0
    return dist(
        (object_track.position.x, object_track.position.y),
        (asker.pose.position.x, asker.pose.position.y),
    )


def _distance_to_first_waypoint(scene: CooperativeScene, object_track: ObjectTrack) -> float:
    if not scene.future_trajectory.points:
        return 0.0
    first = scene.future_trajectory.points[0]
    return dist(
        (object_track.position.x, object_track.position.y),
        (first.x, first.y),
    )


def planning_logreg_feature_values(
    scene: CooperativeScene,
    candidate: PlanningAwarenessCandidate,
    *,
    rank: int,
    ordered_candidates: tuple[PlanningAwarenessCandidate, ...],
    feature_names: tuple[str, ...] = PLANNING_LOGREG_FEATURE_NAMES,
) -> dict[str, float]:
    """Return generic graph-derived Q4 acceptor features for one candidate."""

    previous_score = ordered_candidates[rank - 2].score if rank > 1 else candidate.score
    score_gap_prev = previous_score - candidate.score if rank > 1 else 0.0
    score_ratio_prev = candidate.score / previous_score if rank > 1 and previous_score > 0.0 else 1.0
    candidate_position = candidate.object_track.position
    nearest_higher_rank_distance = 30.0
    for higher in ordered_candidates[: max(0, rank - 1)]:
        higher_position = higher.object_track.position
        nearest_higher_rank_distance = min(
            nearest_higher_rank_distance,
            dist(
                (candidate_position.x, candidate_position.y),
                (higher_position.x, higher_position.y),
            ),
        )

    track = candidate.object_track
    values = {
        "rank": float(rank),
        "candidate_count": float(len(ordered_candidates)),
        "relational_score": float(candidate.score),
        "score_gap_prev": float(score_gap_prev),
        "score_ratio_prev": float(score_ratio_prev),
        "distance_to_trajectory": float(candidate.distance_to_trajectory),
        "distance_to_first_waypoint": float(_distance_to_first_waypoint(scene, track)),
        "distance_to_asker": float(_agent_distance_to_object(scene, track)),
        "visibility_visible": 1.0 if candidate.visibility_state == VisibilityState.VISIBLE else 0.0,
        "visibility_occluded": 1.0 if candidate.visibility_state == VisibilityState.OCCLUDED else 0.0,
        "confidence": float(track.confidence),
        "source_agent_count": float(len(track.provenance.source_agent_ids)),
        "observation_count": float(len(track.observations)),
        "uncertainty_score": float(track.uncertainty_score),
        "conflict_score": float(track.conflict_score),
        "is_cooperative": 1.0 if len(track.provenance.source_agent_ids) >= 2 else 0.0,
        "is_asker_observed": 1.0 if scene.asker_agent_id in track.provenance.source_agent_ids else 0.0,
        "is_candidate": 1.0 if track.status == TrackStatus.CANDIDATE else 0.0,
        "is_supported_like": 1.0 if track.status in (TrackStatus.SUPPORTED, TrackStatus.CONFIRMED) else 0.0,
        "nearest_higher_rank_distance": float(nearest_higher_rank_distance),
    }
    return {
        name: values.get(name, 0.0) if isfinite(values.get(name, 0.0)) else 0.0
        for name in feature_names
    }


def planning_count_gate_feature_values(
    candidate_probabilities: tuple[tuple[PlanningAwarenessCandidate, float], ...],
    *,
    candidate_count: int,
    threshold: float,
    feature_names: tuple[str, ...] = PLANNING_COUNT_GATE_FEATURE_NAMES,
) -> dict[str, float]:
    """Return scene-level Q4 count-gate features from candidate probabilities."""

    ordered = tuple(
        sorted(
            candidate_probabilities,
            key=lambda item: (
                -item[1],
                -item[0].score,
                item[0].distance_to_trajectory,
                item[0].object_track.object_id,
            ),
        )
    )

    def probability(index: int) -> float:
        return float(ordered[index][1]) if len(ordered) > index else 0.0

    def score(index: int) -> float:
        return float(ordered[index][0].score) if len(ordered) > index else 0.0

    def trajectory_distance(index: int) -> float:
        return float(ordered[index][0].distance_to_trajectory) if len(ordered) > index else 30.0

    def abs_y(index: int) -> float:
        return abs(float(ordered[index][0].object_track.position.y)) if len(ordered) > index else 30.0

    top3 = ordered[:3]
    values = {
        "candidate_count": float(candidate_count),
        "eligible_count": float(sum(1 for _candidate, prob in ordered if prob >= threshold)),
        "high_prob_count_0p55": float(sum(1 for _candidate, prob in ordered if prob >= 0.55)),
        "high_prob_count_0p60": float(sum(1 for _candidate, prob in ordered if prob >= 0.60)),
        "high_prob_count_0p65": float(sum(1 for _candidate, prob in ordered if prob >= 0.65)),
        "p1": probability(0),
        "p2": probability(1),
        "p3": probability(2),
        "p4": probability(3),
        "p_gap_1_2": probability(0) - probability(1),
        "p_gap_2_3": probability(1) - probability(2),
        "p_gap_3_4": probability(2) - probability(3),
        "p_mean_top3": sum(probability(index) for index in range(3)) / 3.0,
        "score1": score(0),
        "score2": score(1),
        "score3": score(2),
        "score_gap_1_2": score(0) - score(1),
        "score_gap_2_3": score(1) - score(2),
        "distance_to_trajectory1": trajectory_distance(0),
        "distance_to_trajectory2": trajectory_distance(1),
        "distance_to_trajectory3": trajectory_distance(2),
        "abs_y1": abs_y(0),
        "abs_y2": abs_y(1),
        "abs_y3": abs_y(2),
        "behind_count_top3": float(sum(1 for candidate, _prob in top3 if candidate.object_track.position.x < 0.0)),
        "visible_count_top3": float(sum(1 for candidate, _prob in top3 if candidate.visibility_state == VisibilityState.VISIBLE)),
        "occluded_count_top3": float(sum(1 for candidate, _prob in top3 if candidate.visibility_state == VisibilityState.OCCLUDED)),
        "cooperative_count_top3": float(
            sum(1 for candidate, _prob in top3 if len(candidate.object_track.provenance.source_agent_ids) >= 2)
        ),
    }
    return {
        name: values.get(name, 0.0) if isfinite(values.get(name, 0.0)) else 0.0
        for name in feature_names
    }


class _BasePlanningAwarenessScorer:
    """Small utility base class for deterministic planning-awareness scorers."""

    def __init__(self, min_score: float = 0.45) -> None:
        self.min_score = min_score

    @staticmethod
    def _features(
        scene: CooperativeScene,
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        distance_to_trajectory: float,
        max_distance: float = 30.0,
    ) -> PlanningAwarenessFeatures:
        return PlanningAwarenessFeatures(
            distance_to_trajectory=distance_to_trajectory,
            distance_ratio=min(distance_to_trajectory / max_distance, 1.0),
            is_visible=visibility_state == VisibilityState.VISIBLE,
            is_occluded=visibility_state == VisibilityState.OCCLUDED,
            is_gt_backed="GT" in object_track.provenance.source_agent_ids,
            is_supported_like=object_track.status in (TrackStatus.SUPPORTED, TrackStatus.CONFIRMED),
            is_candidate=object_track.status == TrackStatus.CANDIDATE,
            is_cooperative=len(object_track.provenance.source_agent_ids) >= 2,
            is_asker_observed=scene.asker_agent_id in object_track.provenance.source_agent_ids,
            support_count=len(object_track.observations),
            confidence=object_track.confidence,
            uncertainty_score=object_track.uncertainty_score,
            conflict_score=object_track.conflict_score,
        )

    @staticmethod
    def _build_candidate(
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        features: PlanningAwarenessFeatures,
        score: float,
        rationale: list[str],
    ) -> PlanningAwarenessCandidate:
        return PlanningAwarenessCandidate(
            object_track=object_track,
            visibility_state=visibility_state,
            distance_to_trajectory=features.distance_to_trajectory,
            score=score,
            rationale=tuple(rationale),
        )


class HeuristicPlanningAwarenessScorer(_BasePlanningAwarenessScorer):
    """Transparent baseline scorer for planning-awareness ranking."""

    def __init__(
        self,
        min_score: float = 0.45,
        candidate_penalty: float = 0.20,
        uncertainty_penalty_weight: float = 0.25,
        conflict_penalty_weight: float = 0.15,
        distance_penalty_weight: float = 0.10,
        cooperative_bonus: float = 0.08,
        gt_bonus: float = 0.10,
        support_bonus: float = 0.12,
        occluded_bonus: float = 0.08,
    ) -> None:
        super().__init__(min_score=min_score)
        self.candidate_penalty = candidate_penalty
        self.uncertainty_penalty_weight = uncertainty_penalty_weight
        self.conflict_penalty_weight = conflict_penalty_weight
        self.distance_penalty_weight = distance_penalty_weight
        self.cooperative_bonus = cooperative_bonus
        self.gt_bonus = gt_bonus
        self.support_bonus = support_bonus
        self.occluded_bonus = occluded_bonus

    def score(
        self,
        scene: CooperativeScene,
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        distance_to_trajectory: float,
    ) -> PlanningAwarenessCandidate:
        features = self._features(scene, object_track, visibility_state, distance_to_trajectory)
        rationale: list[str] = []
        score = features.confidence
        rationale.append(f"base_confidence={features.confidence:.2f}")

        if features.is_occluded:
            score += self.occluded_bonus
            rationale.append(f"occluded_bonus={self.occluded_bonus:.2f}")

        if features.is_gt_backed:
            score += self.gt_bonus
            rationale.append(f"gt_bonus={self.gt_bonus:.2f}")

        if features.is_supported_like:
            score += self.support_bonus
            rationale.append(f"support_bonus={self.support_bonus:.2f}")

        if features.is_cooperative:
            score += self.cooperative_bonus
            rationale.append(f"cooperative_bonus={self.cooperative_bonus:.2f}")

        if features.is_candidate:
            score -= self.candidate_penalty
            rationale.append(f"candidate_penalty={self.candidate_penalty:.2f}")

        uncertainty_penalty = features.uncertainty_score * self.uncertainty_penalty_weight
        if uncertainty_penalty:
            score -= uncertainty_penalty
            rationale.append(f"uncertainty_penalty={uncertainty_penalty:.2f}")

        conflict_penalty = features.conflict_score * self.conflict_penalty_weight
        if conflict_penalty:
            score -= conflict_penalty
            rationale.append(f"conflict_penalty={conflict_penalty:.2f}")

        distance_penalty = features.distance_ratio * self.distance_penalty_weight
        if distance_penalty:
            score -= distance_penalty
            rationale.append(f"distance_penalty={distance_penalty:.2f}")

        return self._build_candidate(object_track, visibility_state, features, score, rationale)


class RelationalImportancePlanningAwarenessScorer(_BasePlanningAwarenessScorer):
    """Explicit importance scorer inspired by Li et al. (ICRA 2022).

    Paper-grounded adaptation:
    Li et al. model importance explicitly instead of relying on indirect
    attention through downstream tasks. We mirror that by producing a direct
    object-importance score from graph-native relational cues rather than only
    confidence. Because our current pipeline does not yet include a learned
    relational network, this implementation is a deterministic approximation of
    that explicit-importance decision boundary.
    """

    def __init__(
        self,
        min_score: float = 0.52,
        trajectory_bonus: float = 0.35,
        occluded_bonus: float = 0.18,
        asker_bonus: float = 0.12,
        support_bonus: float = 0.15,
        cooperative_bonus: float = 0.08,
        candidate_penalty: float = 0.18,
        uncertainty_penalty_weight: float = 0.22,
        conflict_penalty_weight: float = 0.16,
    ) -> None:
        super().__init__(min_score=min_score)
        self.trajectory_bonus = trajectory_bonus
        self.occluded_bonus = occluded_bonus
        self.asker_bonus = asker_bonus
        self.support_bonus = support_bonus
        self.cooperative_bonus = cooperative_bonus
        self.candidate_penalty = candidate_penalty
        self.uncertainty_penalty_weight = uncertainty_penalty_weight
        self.conflict_penalty_weight = conflict_penalty_weight

    def score(
        self,
        scene: CooperativeScene,
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        distance_to_trajectory: float,
    ) -> PlanningAwarenessCandidate:
        features = self._features(scene, object_track, visibility_state, distance_to_trajectory)
        rationale: list[str] = []
        score = 0.25 + 0.45 * features.confidence
        rationale.append(f"classification_prior={0.25 + 0.45 * features.confidence:.2f}")

        trajectory_bonus = (1.0 - features.distance_ratio) * self.trajectory_bonus
        score += trajectory_bonus
        rationale.append(f"trajectory_bonus={trajectory_bonus:.2f}")

        if features.is_occluded:
            score += self.occluded_bonus
            rationale.append(f"occluded_bonus={self.occluded_bonus:.2f}")

        if features.is_asker_observed:
            score += self.asker_bonus
            rationale.append(f"asker_bonus={self.asker_bonus:.2f}")

        if features.is_supported_like:
            score += self.support_bonus
            rationale.append(f"support_bonus={self.support_bonus:.2f}")

        if features.is_cooperative:
            score += self.cooperative_bonus
            rationale.append(f"cooperative_bonus={self.cooperative_bonus:.2f}")

        if features.is_candidate:
            score -= self.candidate_penalty
            rationale.append(f"candidate_penalty={self.candidate_penalty:.2f}")

        uncertainty_penalty = features.uncertainty_score * self.uncertainty_penalty_weight
        if uncertainty_penalty:
            score -= uncertainty_penalty
            rationale.append(f"uncertainty_penalty={uncertainty_penalty:.2f}")

        conflict_penalty = features.conflict_score * self.conflict_penalty_weight
        if conflict_penalty:
            score -= conflict_penalty
            rationale.append(f"conflict_penalty={conflict_penalty:.2f}")

        return self._build_candidate(object_track, visibility_state, features, score, rationale)


class RiskAwarePlanningAwarenessScorer(_BasePlanningAwarenessScorer):
    """Object-centric risk scorer inspired by Nyberg et al. (IV 2021).

    The cited work models risk using both violation likelihood and severity and
    argues risk should rise with probability, severity, and uncertainty. We
    adapt that at object level:

    - likelihood proxy: occlusion, uncertainty, conflict
    - severity proxy: closeness to planned trajectory, GT/support confidence
    - progress/evidence term: base confidence and support
    """

    def __init__(
        self,
        min_score: float = 0.50,
        evidence_weight: float = 0.55,
        risk_weight: float = 0.85,
        occlusion_probability_bonus: float = 0.20,
        support_bonus: float = 0.10,
        gt_bonus: float = 0.08,
        candidate_penalty: float = 0.16,
    ) -> None:
        super().__init__(min_score=min_score)
        self.evidence_weight = evidence_weight
        self.risk_weight = risk_weight
        self.occlusion_probability_bonus = occlusion_probability_bonus
        self.support_bonus = support_bonus
        self.gt_bonus = gt_bonus
        self.candidate_penalty = candidate_penalty

    def score(
        self,
        scene: CooperativeScene,
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        distance_to_trajectory: float,
    ) -> PlanningAwarenessCandidate:
        features = self._features(scene, object_track, visibility_state, distance_to_trajectory)
        rationale: list[str] = []

        evidence = self.evidence_weight * features.confidence
        rationale.append(f"evidence_term={evidence:.2f}")

        violation_probability = min(
            1.0,
            0.12
            + (self.occlusion_probability_bonus if features.is_occluded else 0.0)
            + 0.45 * features.uncertainty_score
            + 0.30 * features.conflict_score,
        )
        rationale.append(f"violation_probability={violation_probability:.2f}")

        severity = min(
            1.0,
            (1.0 - features.distance_ratio)
            + (0.12 if features.is_gt_backed else 0.0)
            + (0.10 if features.is_supported_like else 0.0),
        )
        rationale.append(f"violation_severity={severity:.2f}")

        risk_term = self.risk_weight * violation_probability * severity
        score = evidence + risk_term
        rationale.append(f"risk_term={risk_term:.2f}")

        if features.is_supported_like:
            score += self.support_bonus
            rationale.append(f"support_bonus={self.support_bonus:.2f}")

        if features.is_gt_backed:
            score += self.gt_bonus
            rationale.append(f"gt_bonus={self.gt_bonus:.2f}")

        if features.is_candidate:
            score -= self.candidate_penalty
            rationale.append(f"candidate_penalty={self.candidate_penalty:.2f}")

        return self._build_candidate(object_track, visibility_state, features, score, rationale)


class EnergyBasedPlanningAwarenessScorer(_BasePlanningAwarenessScorer):
    """Unary evidence scorer adapted from KLDrive's energy-based framing.

    KLDrive's published description focuses on energy-based scene-fact
    construction. We adapt that idea by building a negative-energy unary score
    from evidence consistency:

    - lower energy for GT-backed / supported / cooperative evidence
    - higher energy for candidate-only, uncertain, conflicting, or far-away
      objects

    The set-level coherence step is handled by ``EnergyBasedDecisionPolicy``.
    """

    def __init__(
        self,
        min_score: float = 0.20,
        gt_energy_bonus: float = 0.20,
        support_energy_bonus: float = 0.18,
        cooperative_energy_bonus: float = 0.10,
        occluded_energy_bonus: float = 0.10,
        candidate_energy_penalty: float = 0.22,
        uncertainty_energy_weight: float = 0.30,
        conflict_energy_weight: float = 0.22,
        distance_energy_weight: float = 0.25,
    ) -> None:
        super().__init__(min_score=min_score)
        self.gt_energy_bonus = gt_energy_bonus
        self.support_energy_bonus = support_energy_bonus
        self.cooperative_energy_bonus = cooperative_energy_bonus
        self.occluded_energy_bonus = occluded_energy_bonus
        self.candidate_energy_penalty = candidate_energy_penalty
        self.uncertainty_energy_weight = uncertainty_energy_weight
        self.conflict_energy_weight = conflict_energy_weight
        self.distance_energy_weight = distance_energy_weight

    def score(
        self,
        scene: CooperativeScene,
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        distance_to_trajectory: float,
    ) -> PlanningAwarenessCandidate:
        features = self._features(scene, object_track, visibility_state, distance_to_trajectory)
        rationale: list[str] = []

        energy = 1.0 - features.confidence
        rationale.append(f"base_energy={energy:.2f}")

        if features.is_gt_backed:
            energy -= self.gt_energy_bonus
            rationale.append(f"gt_energy_bonus={self.gt_energy_bonus:.2f}")
        if features.is_supported_like:
            energy -= self.support_energy_bonus
            rationale.append(f"support_energy_bonus={self.support_energy_bonus:.2f}")
        if features.is_cooperative:
            energy -= self.cooperative_energy_bonus
            rationale.append(f"cooperative_energy_bonus={self.cooperative_energy_bonus:.2f}")
        if features.is_occluded:
            energy -= self.occluded_energy_bonus
            rationale.append(f"occluded_energy_bonus={self.occluded_energy_bonus:.2f}")
        if features.is_candidate:
            energy += self.candidate_energy_penalty
            rationale.append(f"candidate_energy_penalty={self.candidate_energy_penalty:.2f}")

        uncertainty_energy = features.uncertainty_score * self.uncertainty_energy_weight
        conflict_energy = features.conflict_score * self.conflict_energy_weight
        distance_energy = features.distance_ratio * self.distance_energy_weight
        energy += uncertainty_energy + conflict_energy + distance_energy
        if uncertainty_energy:
            rationale.append(f"uncertainty_energy={uncertainty_energy:.2f}")
        if conflict_energy:
            rationale.append(f"conflict_energy={conflict_energy:.2f}")
        if distance_energy:
            rationale.append(f"distance_energy={distance_energy:.2f}")

        score = 1.0 - energy
        rationale.append(f"score_from_negative_energy={score:.2f}")
        return self._build_candidate(object_track, visibility_state, features, score, rationale)


class LLMPlanningAwarenessScorer(_BasePlanningAwarenessScorer):
    """LLM-backed scorer for relevance/ranking over graph-selected candidates."""

    def __init__(self, client: PlanningAwarenessLLMClient, min_score: float = 0.45) -> None:
        super().__init__(min_score=min_score)
        self._client = client

    def score(
        self,
        scene: CooperativeScene,
        object_track: ObjectTrack,
        visibility_state: VisibilityState,
        distance_to_trajectory: float,
    ) -> PlanningAwarenessCandidate:
        context = PlanningAwarenessLLMContext(
            asker_agent_id=scene.asker_agent_id,
            raw_question=scene.raw_question,
            object_id=object_track.object_id,
            object_type=object_track.object_type,
            visibility_state=visibility_state.value,
            distance_to_trajectory=distance_to_trajectory,
            confidence=object_track.confidence,
            status=object_track.status.value,
            uncertainty_score=object_track.uncertainty_score,
            conflict_score=object_track.conflict_score,
            provenance_agents=object_track.provenance.source_agent_ids,
        )
        score, rationale = self._client.score_candidate(context)
        clamped_score = max(0.0, min(1.0, score))
        return PlanningAwarenessCandidate(
            object_track=object_track,
            visibility_state=visibility_state,
            distance_to_trajectory=distance_to_trajectory,
            score=clamped_score,
            rationale=rationale,
        )


class PlanningAwarenessDecisionPolicy(Protocol):
    """Selects the final awareness set from scored candidates."""

    def select(
        self,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        """Return selected and considered candidates."""


class SceneAwarePlanningAwarenessDecisionPolicy(Protocol):
    """Decision policy variant that may use the raw benchmark question."""

    def select_with_scene(
        self,
        scene: CooperativeScene,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        """Return selected and considered candidates with scene context."""


class TopScoreDecisionPolicy:
    """Selects all candidates above threshold, ordered by descending score."""

    def __init__(self, min_score: float = 0.45, max_results: int = 3) -> None:
        self.min_score = min_score
        self.max_results = max_results

    def select(
        self,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        selected = tuple(
            candidate
            for candidate in ordered
            if candidate.score >= self.min_score
        )[: self.max_results]
        return PlanningAwarenessDecision(
            selected_candidates=selected,
            considered_candidates=ordered,
        )


class DiverseTopKDecisionPolicy:
    """Select up to K candidates while preferring visibility diversity.

    For planning-awareness questions, benchmark answers often naturally contain:
    - one occluded/invisible object to be aware of
    - one visible object to be aware of

    This policy therefore prefers taking the strongest occluded candidate and
    the strongest visible candidate first, then only fills remaining slots from
    the global order if needed.
    """

    def __init__(self, min_score: float = 0.45, max_results: int = 2) -> None:
        self.min_score = min_score
        self.max_results = max_results

    def select(
        self,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        eligible = tuple(
            candidate
            for candidate in ordered
            if candidate.score >= self.min_score
        )
        selected: list[PlanningAwarenessCandidate] = []

        best_occluded = next(
            (candidate for candidate in eligible if candidate.visibility_state == VisibilityState.OCCLUDED),
            None,
        )
        best_visible = next(
            (candidate for candidate in eligible if candidate.visibility_state == VisibilityState.VISIBLE),
            None,
        )

        for candidate in (best_occluded, best_visible):
            if candidate is None:
                continue
            if any(existing.object_track.object_id == candidate.object_track.object_id for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) >= self.max_results:
                break

        if len(selected) < self.max_results:
            for candidate in eligible:
                if any(existing.object_track.object_id == candidate.object_track.object_id for existing in selected):
                    continue
                selected.append(candidate)
                if len(selected) >= self.max_results:
                    break

        return PlanningAwarenessDecision(
            selected_candidates=tuple(selected),
            considered_candidates=ordered,
        )


class CountAdaptiveDecisionPolicy:
    """Select up to three objects while making extra objects pass stricter gates.

    Q4 train diagnostics show that a fixed top-2 policy under-recovers
    three-object references, while unconstrained top-3 over-predicts. This
    policy keeps the top two eligible objects but admits a third only when it is
    close in score to the second selected object and clears a higher absolute
    score. It also suppresses near-duplicate coordinates so strict localization
    errors do not turn adjacent predictions into extra false positives.
    """

    def __init__(
        self,
        min_score: float = 0.45,
        max_results: int = 3,
        third_extra_min_score: float = 0.10,
        third_min_relative_to_second: float = 0.92,
        near_duplicate_distance: float = 2.0,
    ) -> None:
        self.min_score = min_score
        self.max_results = max_results
        self.third_extra_min_score = third_extra_min_score
        self.third_min_relative_to_second = third_min_relative_to_second
        self.near_duplicate_distance = near_duplicate_distance

    def select(
        self,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        selected: list[PlanningAwarenessCandidate] = []
        for candidate in ordered:
            if candidate.score < self.min_score:
                continue
            if self._is_near_duplicate(candidate, selected):
                continue
            if len(selected) < 2:
                selected.append(candidate)
            elif len(selected) < self.max_results and self._third_candidate_allowed(candidate, selected):
                selected.append(candidate)
            if len(selected) >= self.max_results:
                break
        return PlanningAwarenessDecision(
            selected_candidates=tuple(selected),
            considered_candidates=ordered,
        )

    def _third_candidate_allowed(
        self,
        candidate: PlanningAwarenessCandidate,
        selected: list[PlanningAwarenessCandidate],
    ) -> bool:
        if len(selected) < 2:
            return True
        second_score = selected[1].score
        if candidate.score < self.min_score + self.third_extra_min_score:
            return False
        if second_score <= 0.0:
            return False
        return candidate.score / second_score >= self.third_min_relative_to_second

    def _is_near_duplicate(
        self,
        candidate: PlanningAwarenessCandidate,
        selected: list[PlanningAwarenessCandidate],
    ) -> bool:
        candidate_position = candidate.object_track.position
        for existing in selected:
            existing_position = existing.object_track.position
            if (
                dist(
                    (candidate_position.x, candidate_position.y),
                    (existing_position.x, existing_position.y),
                )
                <= self.near_duplicate_distance
            ):
                return True
        return False


class LogRegAcceptorDecisionPolicy:
    """Apply a train-frozen logistic acceptor to planning-awareness candidates."""

    def __init__(
        self,
        acceptor_model: dict[str, Any],
        *,
        min_probability: float | None = None,
        max_results: int = 3,
        near_duplicate_distance: float | None = None,
    ) -> None:
        self.acceptor_model = acceptor_model
        self.feature_names = tuple(
            str(item)
            for item in acceptor_model.get("feature_names", PLANNING_LOGREG_FEATURE_NAMES)
        )
        self.normalization = acceptor_model.get("normalization", {})
        self.weights = [float(item) for item in acceptor_model.get("weights", [])]
        self.bias = float(acceptor_model.get("bias", 0.0))
        self.threshold = (
            float(min_probability)
            if min_probability is not None
            else float(acceptor_model.get("threshold", 0.5))
        )
        self.max_results = max_results
        self.near_duplicate_distance = (
            float(near_duplicate_distance)
            if near_duplicate_distance is not None
            else float(acceptor_model.get("near_duplicate_distance", 2.0))
        )

    def select_with_scene(
        self,
        scene: CooperativeScene,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        accepted: list[tuple[PlanningAwarenessCandidate, float]] = []
        for rank, candidate in enumerate(ordered, start=1):
            probability = self._probability(
                planning_logreg_feature_values(
                    scene,
                    candidate,
                    rank=rank,
                    ordered_candidates=ordered,
                    feature_names=self.feature_names,
                )
            )
            if probability < self.threshold:
                continue
            if self._is_near_duplicate(candidate, tuple(item[0] for item in accepted)):
                continue
            accepted.append((candidate, probability))
        selected = tuple(
            candidate
            for candidate, _probability in sorted(
                accepted,
                key=lambda item: (
                    -item[1],
                    -item[0].score,
                    item[0].distance_to_trajectory,
                    item[0].object_track.object_id,
                ),
            )[: self.max_results]
        )
        return PlanningAwarenessDecision(
            selected_candidates=selected,
            considered_candidates=ordered,
        )

    def _probability(self, raw_features: dict[str, float]) -> float:
        if len(self.weights) != len(self.feature_names):
            return 0.0
        means = self.normalization.get("mean", {}) if isinstance(self.normalization, dict) else {}
        stds = self.normalization.get("std", {}) if isinstance(self.normalization, dict) else {}
        value = self.bias
        for name, weight in zip(self.feature_names, self.weights):
            raw_value = float(raw_features.get(name, 0.0))
            mean = float(means.get(name, 0.0)) if isinstance(means, dict) else 0.0
            std = float(stds.get(name, 1.0)) if isinstance(stds, dict) else 1.0
            if std <= 0.0:
                std = 1.0
            value += weight * ((raw_value - mean) / std)
        return _safe_sigmoid(value)

    def _is_near_duplicate(
        self,
        candidate: PlanningAwarenessCandidate,
        selected: tuple[PlanningAwarenessCandidate, ...],
    ) -> bool:
        candidate_position = candidate.object_track.position
        for existing in selected:
            existing_position = existing.object_track.position
            if (
                dist(
                    (candidate_position.x, candidate_position.y),
                    (existing_position.x, existing_position.y),
                )
                <= self.near_duplicate_distance
            ):
                return True
        return False


class MLPAcceptorDecisionPolicy(LogRegAcceptorDecisionPolicy):
    """Apply a train-frozen one-hidden-layer MLP acceptor to Q4 candidates."""

    def __init__(
        self,
        acceptor_model: dict[str, Any],
        *,
        min_probability: float | None = None,
        max_results: int = 3,
        near_duplicate_distance: float | None = None,
    ) -> None:
        super().__init__(
            acceptor_model=acceptor_model,
            min_probability=min_probability,
            max_results=max_results,
            near_duplicate_distance=near_duplicate_distance,
        )
        self.hidden = int(acceptor_model.get("hidden", 0))
        self.w1 = [
            [float(value) for value in row]
            for row in acceptor_model.get("w1", [])
            if isinstance(row, list)
        ]
        self.b1 = [float(value) for value in acceptor_model.get("b1", [])]
        self.w2 = [float(value) for value in acceptor_model.get("w2", [])]
        self.b2 = float(acceptor_model.get("b2", 0.0))

    def _probability(self, raw_features: dict[str, float]) -> float:
        if (
            self.hidden <= 0
            or len(self.w1) != self.hidden
            or len(self.b1) != self.hidden
            or len(self.w2) != self.hidden
            or any(len(row) != len(self.feature_names) for row in self.w1)
        ):
            return 0.0
        means = self.normalization.get("mean", {}) if isinstance(self.normalization, dict) else {}
        stds = self.normalization.get("std", {}) if isinstance(self.normalization, dict) else {}
        vector: list[float] = []
        for name in self.feature_names:
            raw_value = float(raw_features.get(name, 0.0))
            mean = float(means.get(name, 0.0)) if isinstance(means, dict) else 0.0
            std = float(stds.get(name, 1.0)) if isinstance(stds, dict) else 1.0
            if std <= 0.0:
                std = 1.0
            vector.append((raw_value - mean) / std)
        hidden_values = [
            tanh(
                self.b1[h]
                + sum(weight * value for weight, value in zip(self.w1[h], vector))
            )
            for h in range(self.hidden)
        ]
        return _safe_sigmoid(
            self.b2 + sum(weight * value for weight, value in zip(self.w2, hidden_values))
        )


class TrajectoryCalibratedAcceptorDecisionPolicy(LogRegAcceptorDecisionPolicy):
    """Logistic acceptor with trajectory-aware FP suppression and FN rescue."""

    def __init__(
        self,
        acceptor_model: dict[str, Any],
        *,
        min_probability: float | None = None,
        max_results: int = 3,
        near_duplicate_distance: float | None = None,
    ) -> None:
        super().__init__(
            acceptor_model=acceptor_model,
            min_probability=min_probability,
            max_results=max_results,
            near_duplicate_distance=near_duplicate_distance,
        )
        calibration = acceptor_model.get("trajectory_calibration", {})
        self.far_distance_to_trajectory = (
            float(calibration.get("far_distance_to_trajectory", 10.0))
            if isinstance(calibration, dict)
            else 10.0
        )
        self.far_abs_y = (
            float(calibration.get("far_abs_y", 5.0))
            if isinstance(calibration, dict)
            else 5.0
        )
        self.far_moderate_max_probability = (
            float(calibration.get("far_moderate_max_probability", 0.65))
            if isinstance(calibration, dict)
            else 0.65
        )
        self.rescue_min_probability = (
            float(calibration.get("rescue_min_probability", 0.50))
            if isinstance(calibration, dict)
            else 0.50
        )
        self.rescue_max_rank = (
            int(calibration.get("rescue_max_rank", 6))
            if isinstance(calibration, dict)
            else 6
        )
        self.rescue_max_distance_to_trajectory = (
            float(calibration.get("rescue_max_distance_to_trajectory", 4.0))
            if isinstance(calibration, dict)
            else 4.0
        )

    def select_with_scene(
        self,
        scene: CooperativeScene,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        accepted: list[tuple[PlanningAwarenessCandidate, float]] = []
        for rank, candidate in enumerate(ordered, start=1):
            probability = self._probability(
                planning_logreg_feature_values(
                    scene,
                    candidate,
                    rank=rank,
                    ordered_candidates=ordered,
                    feature_names=self.feature_names,
                )
            )
            if self._suppressed_far_candidate(candidate, probability):
                continue
            if probability < self.threshold and not self._rescued_near_candidate(candidate, probability, rank):
                continue
            if self._is_near_duplicate(candidate, tuple(item[0] for item in accepted)):
                continue
            accepted.append((candidate, probability))
        selected = tuple(
            candidate
            for candidate, _probability in sorted(
                accepted,
                key=lambda item: (
                    -item[1],
                    -item[0].score,
                    item[0].distance_to_trajectory,
                    item[0].object_track.object_id,
                ),
            )[: self.max_results]
        )
        return PlanningAwarenessDecision(
            selected_candidates=selected,
            considered_candidates=ordered,
        )

    def _suppressed_far_candidate(
        self,
        candidate: PlanningAwarenessCandidate,
        probability: float,
    ) -> bool:
        return (
            candidate.distance_to_trajectory > self.far_distance_to_trajectory
            and abs(candidate.object_track.position.y) >= self.far_abs_y
            and probability < self.far_moderate_max_probability
        )

    def _rescued_near_candidate(
        self,
        candidate: PlanningAwarenessCandidate,
        probability: float,
        rank: int,
    ) -> bool:
        return (
            probability >= self.rescue_min_probability
            and rank <= self.rescue_max_rank
            and candidate.distance_to_trajectory <= self.rescue_max_distance_to_trajectory
        )


class CountGatedAcceptorDecisionPolicy(LogRegAcceptorDecisionPolicy):
    """Use candidate acceptance probabilities plus a train-frozen scene count gate."""

    def __init__(
        self,
        acceptor_model: dict[str, Any],
        *,
        min_probability: float | None = None,
        max_results: int = 3,
        near_duplicate_distance: float | None = None,
    ) -> None:
        super().__init__(
            acceptor_model=acceptor_model,
            min_probability=min_probability,
            max_results=max_results,
            near_duplicate_distance=near_duplicate_distance,
        )
        self.candidate_model_type = str(acceptor_model.get("model_type", "logreg"))
        self.hidden = int(acceptor_model.get("hidden", 0))
        self.w1 = [
            [float(value) for value in row]
            for row in acceptor_model.get("w1", [])
            if isinstance(row, list)
        ]
        self.b1 = [float(value) for value in acceptor_model.get("b1", [])]
        self.w2 = [float(value) for value in acceptor_model.get("w2", [])]
        self.b2 = float(acceptor_model.get("b2", 0.0))
        self.count_gate = acceptor_model.get("count_gate", {})

    def _probability(self, raw_features: dict[str, float]) -> float:
        if self.candidate_model_type != "mlp":
            return super()._probability(raw_features)
        if (
            self.hidden <= 0
            or len(self.w1) != self.hidden
            or len(self.b1) != self.hidden
            or len(self.w2) != self.hidden
            or any(len(row) != len(self.feature_names) for row in self.w1)
        ):
            return 0.0
        means = self.normalization.get("mean", {}) if isinstance(self.normalization, dict) else {}
        stds = self.normalization.get("std", {}) if isinstance(self.normalization, dict) else {}
        vector: list[float] = []
        for name in self.feature_names:
            raw_value = float(raw_features.get(name, 0.0))
            mean = float(means.get(name, 0.0)) if isinstance(means, dict) else 0.0
            std = float(stds.get(name, 1.0)) if isinstance(stds, dict) else 1.0
            if std <= 0.0:
                std = 1.0
            vector.append((raw_value - mean) / std)
        hidden_values = [
            tanh(
                self.b1[h]
                + sum(weight * value for weight, value in zip(self.w1[h], vector))
            )
            for h in range(self.hidden)
        ]
        return _safe_sigmoid(
            self.b2 + sum(weight * value for weight, value in zip(self.w2, hidden_values))
        )

    def select_with_scene(
        self,
        scene: CooperativeScene,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        scored: list[tuple[PlanningAwarenessCandidate, float]] = []
        for rank, candidate in enumerate(ordered, start=1):
            probability = self._probability(
                planning_logreg_feature_values(
                    scene,
                    candidate,
                    rank=rank,
                    ordered_candidates=ordered,
                    feature_names=self.feature_names,
                )
            )
            scored.append((candidate, probability))
        non_duplicate = self._non_duplicate_sorted(scored)
        predicted_count = self._predicted_count(non_duplicate, candidate_count=len(ordered))
        eligible = tuple(
            (candidate, probability)
            for candidate, probability in non_duplicate
            if probability >= self.threshold
        )
        selected = tuple(candidate for candidate, _probability in eligible[:predicted_count])
        return PlanningAwarenessDecision(
            selected_candidates=selected[: self.max_results],
            considered_candidates=ordered,
        )

    def _non_duplicate_sorted(
        self,
        scored: list[tuple[PlanningAwarenessCandidate, float]],
    ) -> tuple[tuple[PlanningAwarenessCandidate, float], ...]:
        selected: list[tuple[PlanningAwarenessCandidate, float]] = []
        for candidate, probability in sorted(
            scored,
            key=lambda item: (
                -item[1],
                -item[0].score,
                item[0].distance_to_trajectory,
                item[0].object_track.object_id,
            ),
        ):
            if self._is_near_duplicate(candidate, tuple(item[0] for item in selected)):
                continue
            selected.append((candidate, probability))
        return tuple(selected)

    def _predicted_count(
        self,
        candidate_probabilities: tuple[tuple[PlanningAwarenessCandidate, float], ...],
        *,
        candidate_count: int,
    ) -> int:
        if not isinstance(self.count_gate, dict):
            return min(self.max_results, sum(1 for _candidate, prob in candidate_probabilities if prob >= self.threshold))
        feature_names = tuple(
            str(item)
            for item in self.count_gate.get("feature_names", PLANNING_COUNT_GATE_FEATURE_NAMES)
        )
        features = planning_count_gate_feature_values(
            candidate_probabilities,
            candidate_count=candidate_count,
            threshold=float(self.count_gate.get("candidate_threshold", self.threshold)),
            feature_names=feature_names,
        )
        probabilities = self._count_probabilities(features, feature_names)
        label_values = [int(value) for value in self.count_gate.get("label_values", [0, 1, 2, 3])]
        if not probabilities or len(probabilities) != len(label_values):
            return min(self.max_results, sum(1 for _candidate, prob in candidate_probabilities if prob >= self.threshold))
        predicted = label_values[max(range(len(probabilities)), key=lambda index: probabilities[index])]
        return max(0, min(self.max_results, int(predicted)))

    def _count_probabilities(
        self,
        raw_features: dict[str, float],
        feature_names: tuple[str, ...],
    ) -> list[float]:
        weights = self.count_gate.get("weights", []) if isinstance(self.count_gate, dict) else []
        biases = self.count_gate.get("biases", []) if isinstance(self.count_gate, dict) else []
        if not isinstance(weights, list) or not isinstance(biases, list):
            return []
        normalization = self.count_gate.get("normalization", {})
        means = normalization.get("mean", {}) if isinstance(normalization, dict) else {}
        stds = normalization.get("std", {}) if isinstance(normalization, dict) else {}
        vector: list[float] = []
        for name in feature_names:
            raw_value = float(raw_features.get(name, 0.0))
            mean = float(means.get(name, 0.0)) if isinstance(means, dict) else 0.0
            std = float(stds.get(name, 1.0)) if isinstance(stds, dict) else 1.0
            if std <= 0.0:
                std = 1.0
            vector.append((raw_value - mean) / std)
        logits: list[float] = []
        for row, bias in zip(weights, biases):
            if not isinstance(row, list) or len(row) != len(vector):
                return []
            logits.append(float(bias) + sum(float(weight) * value for weight, value in zip(row, vector)))
        if not logits:
            return []
        max_logit = max(logits)
        exps = [exp(logit - max_logit) for logit in logits]
        total = sum(exps)
        return [value / total for value in exps] if total else []


class SoftCountGatedAcceptorDecisionPolicy(CountGatedAcceptorDecisionPolicy):
    """Count-gated acceptor that can admit one extra strong candidate."""

    def __init__(
        self,
        acceptor_model: dict[str, Any],
        *,
        min_probability: float | None = None,
        max_results: int = 3,
        near_duplicate_distance: float | None = None,
    ) -> None:
        super().__init__(
            acceptor_model=acceptor_model,
            min_probability=min_probability,
            max_results=max_results,
            near_duplicate_distance=near_duplicate_distance,
        )
        count_gate = acceptor_model.get("count_gate", {})
        self.extra_min_probability = (
            float(count_gate.get("soft_extra_min_probability", 0.62))
            if isinstance(count_gate, dict)
            else 0.62
        )
        self.extra_min_relative_to_k = (
            float(count_gate.get("soft_extra_min_relative_to_k", 0.90))
            if isinstance(count_gate, dict)
            else 0.90
        )

    def select_with_scene(
        self,
        scene: CooperativeScene,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        scored: list[tuple[PlanningAwarenessCandidate, float]] = []
        for rank, candidate in enumerate(ordered, start=1):
            probability = self._probability(
                planning_logreg_feature_values(
                    scene,
                    candidate,
                    rank=rank,
                    ordered_candidates=ordered,
                    feature_names=self.feature_names,
                )
            )
            scored.append((candidate, probability))
        non_duplicate = self._non_duplicate_sorted(scored)
        predicted_count = self._predicted_count(non_duplicate, candidate_count=len(ordered))
        eligible = tuple(
            (candidate, probability)
            for candidate, probability in non_duplicate
            if probability >= self.threshold
        )
        final_count = self._soft_count(predicted_count, eligible)
        selected = tuple(candidate for candidate, _probability in eligible[:final_count])
        return PlanningAwarenessDecision(
            selected_candidates=selected[: self.max_results],
            considered_candidates=ordered,
        )

    def _soft_count(
        self,
        predicted_count: int,
        eligible: tuple[tuple[PlanningAwarenessCandidate, float], ...],
    ) -> int:
        capped_count = max(0, min(self.max_results, predicted_count))
        if capped_count >= self.max_results or len(eligible) <= capped_count:
            return capped_count
        next_probability = eligible[capped_count][1]
        if next_probability >= self.extra_min_probability:
            return capped_count + 1
        if capped_count > 0:
            kth_probability = eligible[capped_count - 1][1]
            if kth_probability > 0.0 and next_probability / kth_probability >= self.extra_min_relative_to_k:
                return capped_count + 1
        return capped_count


class EnergyBasedDecisionPolicy:
    """Greedy set selector with pairwise coherence penalties.

    Adaptation note:
    KLDrive's energy module consolidates multi-source scene facts. We reuse the
    same high-level idea for answer-set selection by minimizing a simple
    energy-like objective:

    utility(set) =
        sum(unary score)
        - redundancy penalty for nearby same-type objects
        - visibility-mismatch penalty for low-value visible duplicates
        + small support/cooperation diversity bonus

    This is intentionally lightweight and deterministic so it can serve as a
    strong non-LLM baseline in our current Phase 5A pipeline.
    """

    def __init__(
        self,
        min_score: float = 0.20,
        max_results: int = 3,
        redundancy_weight: float = 1.10,
        diversity_bonus: float = 0.05,
        near_duplicate_distance: float = 2.0,
    ) -> None:
        self.min_score = min_score
        self.max_results = max_results
        self.redundancy_weight = redundancy_weight
        self.diversity_bonus = diversity_bonus
        self.near_duplicate_distance = near_duplicate_distance

    def select(
        self,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        selected: list[PlanningAwarenessCandidate] = []
        for candidate in ordered:
            if candidate.score < self.min_score:
                continue
            marginal_gain = self._marginal_gain(candidate, tuple(selected))
            if marginal_gain <= 0.0:
                continue
            selected.append(candidate)
            if len(selected) >= self.max_results:
                break
        return PlanningAwarenessDecision(
            selected_candidates=tuple(selected),
            considered_candidates=ordered,
        )

    def _marginal_gain(
        self,
        candidate: PlanningAwarenessCandidate,
        selected: tuple[PlanningAwarenessCandidate, ...],
    ) -> float:
        gain = candidate.score
        for existing in selected:
            if existing.object_track.object_type != candidate.object_track.object_type:
                continue
            pair_distance = dist(
                (existing.object_track.position.x, existing.object_track.position.y),
                (candidate.object_track.position.x, candidate.object_track.position.y),
            )
            closeness = max(0.0, 1.0 - min(pair_distance / 12.0, 1.0))
            redundancy_penalty = (
                closeness
                * self.redundancy_weight
                * min(existing.score, candidate.score)
            )
            if pair_distance <= self.near_duplicate_distance:
                redundancy_penalty += self.redundancy_weight * 0.5
            gain -= redundancy_penalty
            if existing.visibility_state != candidate.visibility_state:
                gain += self.diversity_bonus
        return gain


class LLMRerankDecisionPolicy:
    """Rerank a graph-produced shortlist with a local LLM service.

    This follows the intended deployment pattern for Phase 5:

    graph pipeline -> deterministic shortlist -> local LLM rerank -> final set

    We do not let the LLM see raw detections or replace graph construction.
    Instead, the LLM only reorders a short list of already-grounded candidates.
    """

    def __init__(
        self,
        client: PlanningAwarenessBatchLLMClient,
        min_score: float = 0.45,
        max_results: int = 2,
        shortlist_size: int = 5,
        blend_weight: float = 0.70,
    ) -> None:
        self.min_score = min_score
        self.max_results = max_results
        self.shortlist_size = shortlist_size
        self.blend_weight = blend_weight
        self._client = client

    def select_with_scene(
        self,
        scene: CooperativeScene,
        candidates: tuple[PlanningAwarenessCandidate, ...],
    ) -> PlanningAwarenessDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        shortlist = tuple(candidate for candidate in ordered if candidate.score >= self.min_score)[
            : self.shortlist_size
        ]
        if not shortlist:
            return PlanningAwarenessDecision(selected_candidates=(), considered_candidates=ordered)

        rank_items = tuple(
            PlanningAwarenessLLMRankItem(
                object_id=candidate.object_track.object_id,
                object_type=candidate.object_track.object_type,
                visibility_state=candidate.visibility_state.value,
                distance_to_trajectory=candidate.distance_to_trajectory,
                base_score=candidate.score,
                confidence=candidate.object_track.confidence,
                status=candidate.object_track.status.value,
                uncertainty_score=candidate.object_track.uncertainty_score,
                conflict_score=candidate.object_track.conflict_score,
                provenance_agents=candidate.object_track.provenance.source_agent_ids,
            )
            for candidate in shortlist
        )
        reranked = self._client.rerank_candidates(
            asker_agent_id=scene.asker_agent_id,
            raw_question=scene.raw_question,
            candidates=rank_items,
        )
        reranked_by_id = {item.object_id: item for item in reranked}

        combined_candidates: list[PlanningAwarenessCandidate] = []
        for candidate in ordered:
            reranked_item = reranked_by_id.get(candidate.object_track.object_id)
            if reranked_item is None:
                combined_candidates.append(candidate)
                continue
            combined_score = (
                (1.0 - self.blend_weight) * candidate.score
                + self.blend_weight * reranked_item.score
            )
            combined_candidates.append(
                replace(
                    candidate,
                    score=combined_score,
                    rationale=candidate.rationale + tuple(
                        [f"llm_score={reranked_item.score:.2f}", *reranked_item.rationale]
                    ),
                )
            )

        combined_ordered = tuple(
            sorted(
                combined_candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.distance_to_trajectory,
                    candidate.object_track.object_id,
                ),
            )
        )
        selected = tuple(
            candidate for candidate in combined_ordered if candidate.score >= self.min_score
        )[: self.max_results]
        return PlanningAwarenessDecision(
            selected_candidates=selected,
            considered_candidates=combined_ordered,
        )


def build_planning_awareness_scorer(
    ranker: PlanningAwarenessRanker | str = PlanningAwarenessRanker.HEURISTIC,
    llm_client: PlanningAwarenessLLMClient | None = None,
) -> PlanningAwarenessScorer:
    """Create a planning-awareness scorer by registered name."""

    ranker = PlanningAwarenessRanker(ranker)
    if ranker == PlanningAwarenessRanker.HEURISTIC:
        return HeuristicPlanningAwarenessScorer()
    if ranker == PlanningAwarenessRanker.RELATIONAL_IMPORTANCE:
        return RelationalImportancePlanningAwarenessScorer()
    if ranker == PlanningAwarenessRanker.RISK_AWARE:
        return RiskAwarePlanningAwarenessScorer()
    if ranker == PlanningAwarenessRanker.ENERGY_BASED:
        return EnergyBasedPlanningAwarenessScorer()
    if ranker == PlanningAwarenessRanker.LLM:
        # The LLM path uses a deterministic graph shortlist first, then reranks
        # with a local model. We therefore keep RiskAware as the base scorer and
        # let the decision policy perform the LLM reranking step.
        return RiskAwarePlanningAwarenessScorer()
    raise ValueError(f"Unsupported planning-awareness ranker: {ranker}")


def build_planning_awareness_decision_policy(
    ranker: PlanningAwarenessRanker | str,
    scorer: PlanningAwarenessScorer,
    selection_policy: PlanningAwarenessSelectionPolicy | str = PlanningAwarenessSelectionPolicy.DEFAULT,
    llm_client: PlanningAwarenessBatchLLMClient | None = None,
    acceptor_model: dict[str, Any] | None = None,
    max_results: int = 3,
) -> PlanningAwarenessDecisionPolicy:
    """Create the matching selection policy for one planning-awareness ranker."""

    ranker = PlanningAwarenessRanker(ranker)
    selection_policy = PlanningAwarenessSelectionPolicy(selection_policy)
    if ranker == PlanningAwarenessRanker.LLM:
        if llm_client is None:
            raise ValueError("The `llm` planning-awareness ranker requires a batch llm_client.")
        llm_max_results = 2 if selection_policy in (
            PlanningAwarenessSelectionPolicy.TOP2,
            PlanningAwarenessSelectionPolicy.DIVERSE_TOP2,
        ) else max_results
        return LLMRerankDecisionPolicy(
            client=llm_client,
            min_score=scorer.min_score,
            max_results=llm_max_results,
        )
    if selection_policy == PlanningAwarenessSelectionPolicy.TOP2:
        return TopScoreDecisionPolicy(min_score=scorer.min_score, max_results=2)
    if selection_policy == PlanningAwarenessSelectionPolicy.DIVERSE_TOP2:
        return DiverseTopKDecisionPolicy(min_score=scorer.min_score, max_results=2)
    if selection_policy == PlanningAwarenessSelectionPolicy.COUNT_ADAPTIVE:
        return CountAdaptiveDecisionPolicy(min_score=scorer.min_score, max_results=max_results)
    if selection_policy == PlanningAwarenessSelectionPolicy.LOGREG_ACCEPTOR:
        if not acceptor_model:
            raise ValueError("The `logreg_acceptor` planning policy requires an acceptor model.")
        return LogRegAcceptorDecisionPolicy(acceptor_model=acceptor_model, max_results=max_results)
    if selection_policy == PlanningAwarenessSelectionPolicy.MLP_ACCEPTOR:
        if not acceptor_model:
            raise ValueError("The `mlp_acceptor` planning policy requires an acceptor model.")
        return MLPAcceptorDecisionPolicy(acceptor_model=acceptor_model, max_results=max_results)
    if selection_policy == PlanningAwarenessSelectionPolicy.TRAJECTORY_CALIBRATED_ACCEPTOR:
        if not acceptor_model:
            raise ValueError("The `trajectory_calibrated_acceptor` planning policy requires an acceptor model.")
        return TrajectoryCalibratedAcceptorDecisionPolicy(acceptor_model=acceptor_model, max_results=max_results)
    if selection_policy == PlanningAwarenessSelectionPolicy.COUNT_GATED_ACCEPTOR:
        if not acceptor_model:
            raise ValueError("The `count_gated_acceptor` planning policy requires an acceptor model.")
        return CountGatedAcceptorDecisionPolicy(acceptor_model=acceptor_model, max_results=max_results)
    if selection_policy == PlanningAwarenessSelectionPolicy.SOFT_COUNT_GATED_ACCEPTOR:
        if not acceptor_model:
            raise ValueError("The `soft_count_gated_acceptor` planning policy requires an acceptor model.")
        return SoftCountGatedAcceptorDecisionPolicy(acceptor_model=acceptor_model, max_results=max_results)
    if ranker == PlanningAwarenessRanker.ENERGY_BASED:
        return EnergyBasedDecisionPolicy(min_score=scorer.min_score, max_results=max_results)
    return TopScoreDecisionPolicy(min_score=scorer.min_score, max_results=max_results)


class PlanningAwarenessOrchestrator:
    """Staged planner-awareness selector with pluggable scoring and decision policies."""

    def __init__(
        self,
        scorer: PlanningAwarenessScorer | None = None,
        decision_policy: PlanningAwarenessDecisionPolicy | None = None,
        max_distance: float = 30.0,
    ) -> None:
        resolved_scorer = scorer or HeuristicPlanningAwarenessScorer()
        self._scorer = resolved_scorer
        self._decision_policy = decision_policy or TopScoreDecisionPolicy(
            min_score=resolved_scorer.min_score
        )
        self._max_distance = max_distance

    def select(self, scene: CooperativeScene) -> PlanningAwarenessDecision:
        """Collect, score, and rank planning-awareness candidates."""

        candidates = self.collect_candidates(scene)
        if hasattr(self._decision_policy, "select_with_scene"):
            return self._decision_policy.select_with_scene(scene, candidates)  # type: ignore[attr-defined]
        return self._decision_policy.select(candidates)

    def collect_candidates(self, scene: CooperativeScene) -> tuple[PlanningAwarenessCandidate, ...]:
        """Return scored planning-awareness candidates before thresholding."""

        candidates: list[PlanningAwarenessCandidate] = []
        for object_track in scene.object_tracks:
            visibility_state = self._visibility_state_for_asker(scene, object_track.object_id)
            if visibility_state not in (VisibilityState.VISIBLE, VisibilityState.OCCLUDED):
                continue
            distance_to_trajectory = self._distance_to_trajectory(scene, object_track)
            if distance_to_trajectory > self._max_distance:
                continue
            candidates.append(
                self._scorer.score(
                    scene=scene,
                    object_track=object_track,
                    visibility_state=visibility_state,
                    distance_to_trajectory=distance_to_trajectory,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _visibility_state_for_asker(scene: CooperativeScene, object_id: str) -> VisibilityState | None:
        for fact in scene.visibility_facts:
            if fact.agent_id == scene.asker_agent_id and fact.object_id == object_id:
                return fact.state
        return None

    @staticmethod
    def _distance_to_trajectory(scene: CooperativeScene, object_track: ObjectTrack) -> float:
        if not scene.future_trajectory.points:
            return float("inf")
        return min(
            dist(
                (object_track.position.x, object_track.position.y),
                (point.x, point.y),
            )
            for point in scene.future_trajectory.points
        )


def build_planning_awareness_orchestrator(
    ranker: PlanningAwarenessRanker | str = PlanningAwarenessRanker.HEURISTIC,
    llm_client: PlanningAwarenessBatchLLMClient | PlanningAwarenessLLMClient | None = None,
    selection_policy: PlanningAwarenessSelectionPolicy | str = PlanningAwarenessSelectionPolicy.DEFAULT,
    acceptor_model: dict[str, Any] | None = None,
    max_distance: float = 30.0,
    max_results: int = 3,
) -> PlanningAwarenessOrchestrator:
    """Convenience factory used by scripts/router wiring."""

    scorer = build_planning_awareness_scorer(ranker=ranker, llm_client=llm_client)
    decision_policy = build_planning_awareness_decision_policy(
        ranker=ranker,
        scorer=scorer,
        selection_policy=selection_policy,
        llm_client=llm_client,  # type: ignore[arg-type]
        acceptor_model=acceptor_model,
        max_results=max_results,
    )
    return PlanningAwarenessOrchestrator(
        scorer=scorer,
        decision_policy=decision_policy,
        max_distance=max_distance,
    )
