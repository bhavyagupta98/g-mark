from __future__ import annotations

from kg_coop_drive.domain.metrics import SceneMetrics, TemporalMetrics
from kg_coop_drive.domain.scene import (
    CooperativeScene,
    CrossAgentAssociationReport,
    ObservationAssociationReport,
    TemporalTrackUpdateReport,
    TrackStatus,
)


class SceneMetricsReporter:
    """Computes progress metrics for one single-frame scene build."""

    def compute(
        self,
        scene: CooperativeScene,
        association_report: ObservationAssociationReport,
        cross_agent_report: CrossAgentAssociationReport | None = None,
    ) -> SceneMetrics:
        """Return a compact quantitative summary for the current scene."""

        total_tracks = len(scene.object_tracks)
        confirmed_tracks = sum(
            1 for track in scene.object_tracks if track.status == TrackStatus.CONFIRMED
        )
        supported_tracks = sum(
            1 for track in scene.object_tracks if track.status == TrackStatus.SUPPORTED
        )
        candidate_tracks = sum(
            1 for track in scene.object_tracks if track.status == TrackStatus.CANDIDATE
        )
        total_observations = len(scene.observations)
        matched_observations = len(association_report.matches)
        unmatched_observations = len(association_report.unmatched_observation_ids)
        support_coverage = (
            supported_tracks / total_tracks if total_tracks else 0.0
        )
        average_track_confidence = (
            sum(track.confidence for track in scene.object_tracks) / total_tracks
            if total_tracks
            else 0.0
        )
        average_uncertainty_score = (
            sum(track.uncertainty_score for track in scene.object_tracks) / total_tracks
            if total_tracks
            else 0.0
        )
        average_conflict_score = (
            sum(track.conflict_score for track in scene.object_tracks) / total_tracks
            if total_tracks
            else 0.0
        )
        return SceneMetrics(
            total_tracks=total_tracks,
            confirmed_tracks=confirmed_tracks,
            supported_tracks=supported_tracks,
            candidate_tracks=candidate_tracks,
            total_observations=total_observations,
            matched_observations=matched_observations,
            unmatched_observations=unmatched_observations,
            support_coverage=support_coverage,
            average_track_confidence=average_track_confidence,
            average_uncertainty_score=average_uncertainty_score,
            average_conflict_score=average_conflict_score,
            cross_agent_match_count=len(cross_agent_report.matches) if cross_agent_report else 0,
            relation_count=len(scene.relations),
            visibility_fact_count=len(scene.visibility_facts),
        )


class TemporalMetricsReporter:
    """Computes progress metrics for frame-to-frame track maintenance."""

    def compute(
        self,
        scene: CooperativeScene,
        report: TemporalTrackUpdateReport,
    ) -> TemporalMetrics:
        """Return a compact temporal summary after one update step."""

        total_tracks = len(scene.object_tracks)
        persisted_tracks = len(report.persisted_track_ids)
        new_tracks = len(report.new_track_ids)
        retained_stale_tracks = len(report.retained_stale_track_ids)
        pruned_stale_tracks = len(report.pruned_stale_track_ids)
        persistence_rate = (
            persisted_tracks / total_tracks if total_tracks else 0.0
        )
        average_track_age = (
            sum(track.age_frames for track in scene.object_tracks) / total_tracks
            if total_tracks
            else 0.0
        )
        average_miss_count = (
            sum(track.miss_count for track in scene.object_tracks) / total_tracks
            if total_tracks
            else 0.0
        )
        return TemporalMetrics(
            total_tracks=total_tracks,
            persisted_tracks=persisted_tracks,
            new_tracks=new_tracks,
            retained_stale_tracks=retained_stale_tracks,
            pruned_stale_tracks=pruned_stale_tracks,
            persistence_rate=persistence_rate,
            average_track_age=average_track_age,
            average_miss_count=average_miss_count,
        )
