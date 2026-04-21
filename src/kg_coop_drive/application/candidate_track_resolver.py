from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.domain.scene import (
    CandidateResolutionReport,
    CooperativeScene,
    TrackStatus,
)


class CandidateTrackResolver:
    """Applies a conservative keep-or-prune policy to candidate tracks."""

    def resolve(
        self,
        scene: CooperativeScene,
        min_candidate_confidence: float = 0.25,
    ) -> tuple[CooperativeScene, CandidateResolutionReport]:
        """Return a scene with weak candidate tracks pruned."""

        kept_tracks = []
        kept_candidate_ids = []
        pruned_candidate_ids = []

        for object_track in scene.object_tracks:
            if object_track.status != TrackStatus.CANDIDATE:
                kept_tracks.append(object_track)
                continue

            if object_track.confidence >= min_candidate_confidence:
                kept_tracks.append(object_track)
                kept_candidate_ids.append(object_track.object_id)
            else:
                pruned_candidate_ids.append(object_track.object_id)

        resolved_scene = replace(scene, object_tracks=tuple(kept_tracks))
        report = CandidateResolutionReport(
            kept_candidate_ids=tuple(kept_candidate_ids),
            pruned_candidate_ids=tuple(pruned_candidate_ids),
        )
        return resolved_scene, report
