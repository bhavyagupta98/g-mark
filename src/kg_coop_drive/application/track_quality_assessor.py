from __future__ import annotations

from dataclasses import replace
from math import dist

from kg_coop_drive.domain.scene import CooperativeScene


class TrackQualityAssessor:
    """Computes uncertainty and conflict signals for current tracks."""

    def assess(self, scene: CooperativeScene) -> CooperativeScene:
        """Return a scene with per-track quality fields updated."""

        updated_tracks = tuple(self._assess_track(track) for track in scene.object_tracks)
        return replace(scene, object_tracks=updated_tracks)

    @staticmethod
    def _assess_track(track):
        support_confidences = tuple(observation.confidence for observation in track.observations)
        last_support_confidence = max(support_confidences, default=0.0)

        if track.observations:
            support_distances = tuple(
                dist(
                    (track.position.x, track.position.y),
                    (observation.position.x, observation.position.y),
                )
                for observation in track.observations
            )
            conflict_score = sum(support_distances) / len(support_distances)
        else:
            conflict_score = 0.0

        is_gt_backed = "GT" in track.provenance.source_agent_ids
        base_uncertainty = 1.0 - track.confidence
        candidate_penalty = 0.15 if not is_gt_backed else 0.0
        miss_penalty = 0.10 * track.miss_count
        conflict_penalty = min(conflict_score / 3.0, 1.0) * 0.20
        support_bonus = 0.10 if track.observations else 0.0
        uncertainty_score = max(
            0.0,
            min(
                1.0,
                base_uncertainty + candidate_penalty + miss_penalty + conflict_penalty - support_bonus,
            ),
        )

        return replace(
            track,
            uncertainty_score=uncertainty_score,
            conflict_score=conflict_score,
            last_support_confidence=last_support_confidence,
        )
