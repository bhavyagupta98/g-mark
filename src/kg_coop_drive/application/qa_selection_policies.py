from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OccludingSelectionPolicy:
    """Configurable context policy for selecting occluding-object candidates."""

    max_results: int = 3
    min_results_with_visible_fallback: int = 2
    enable_visible_fallback: bool = True
    third_candidate_min_risk: float = 0.45
    third_candidate_min_relative_to_second: float = 0.65
    geometric_weight: float = 0.3
    alignment_weight: float = 0.22
    hidden_relevance_weight: float = 0.18
    provenance_weight: float = 0.12
    model_score_weight: float = 0.18
    candidate_penalty: float = 0.03
    top_two_risk_coverage_target: float = 0.86
    caution_multiplier: float = 1.0


@dataclass(frozen=True)
class InvisibleSelectionPolicy:
    """Configurable policy for precision-aware invisible-object selection."""

    max_results: int = 1
    shortlist_size: int = 6
    min_distance_to_asker: float = 2.0
    max_distance_to_trajectory: float = 5.0
    max_distance_to_asker: float = 80.0
    min_risk: float = 0.58
    min_relative_to_best: float = 0.75
    trajectory_weight: float = 0.34
    asker_weight: float = 0.12
    provenance_weight: float = 0.2
    confidence_weight: float = 0.2
    model_score_weight: float = 0.14
    candidate_penalty: float = 0.18
    conflict_penalty: float = 0.14
    uncertainty_penalty: float = 0.12
    lateral_relevance_min_abs_y: float = 1.0
    lateral_relevance_max_abs_y: float = 8.0
    lateral_relevance_bonus: float = 0.38
    far_centerline_abs_y: float = 1.0
    far_centerline_min_distance_to_asker: float = 15.0
    far_centerline_penalty: float = 0.7
    road_region_min_score: float = 0.0
    far_behind_centerline_abs_y: float = 1.0
    far_behind_min_distance_to_asker: float = 15.0
    far_behind_max_relative_x: float = -1.0
    backtrack_centerline_abs_y: float = 1.0
    backtrack_max_distance_to_trajectory: float = 2.0
    backtrack_max_relative_x: float = -1.0
    rescue_min_relative_x: float = 1.0
    rescue_min_abs_y: float = 1.0
    rescue_min_distance_to_trajectory: float = 2.0
    rescue_min_support_count: int = 2
