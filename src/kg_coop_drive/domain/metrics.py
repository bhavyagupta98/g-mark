from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneMetrics:
    """Quantitative summary of one single-frame cooperative scene."""

    total_tracks: int
    confirmed_tracks: int
    supported_tracks: int
    candidate_tracks: int
    total_observations: int
    matched_observations: int
    unmatched_observations: int
    support_coverage: float
    average_track_confidence: float
    average_uncertainty_score: float
    average_conflict_score: float
    cross_agent_match_count: int
    relation_count: int
    visibility_fact_count: int


@dataclass(frozen=True)
class TemporalMetrics:
    """Quantitative summary of frame-to-frame identity maintenance."""

    total_tracks: int
    persisted_tracks: int
    new_tracks: int
    retained_stale_tracks: int
    pruned_stale_tracks: int
    persistence_rate: float
    average_track_age: float
    average_miss_count: float
