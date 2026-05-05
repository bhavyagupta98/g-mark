from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.domain.scene import CooperativeScene, TrackStatus


class TrackLifecycleManager:
    """Evolves track status over time from conservative lifecycle rules."""

    def update(
        self,
        scene: CooperativeScene,
        promotion_age_frames: int = 2,
        max_supported_miss_count: int = 1,
    ) -> CooperativeScene:
        """Promote stable candidates and downgrade tracks that are repeatedly missed."""

        updated_tracks = tuple(
            self._update_track(
                track,
                promotion_age_frames=promotion_age_frames,
                max_supported_miss_count=max_supported_miss_count,
            )
            for track in scene.object_tracks
        )
        return replace(scene, object_tracks=updated_tracks)

    @staticmethod
    def _update_track(
        track,
        promotion_age_frames: int,
        max_supported_miss_count: int,
    ):
        if (
            track.status == TrackStatus.CANDIDATE
            and track.age_frames >= promotion_age_frames
            and track.miss_count == 0
            and track.observations
        ):
            return replace(track, status=TrackStatus.SUPPORTED)

        if (
            track.status == TrackStatus.SUPPORTED
            and track.miss_count > max_supported_miss_count
        ):
            return replace(track, status=TrackStatus.CANDIDATE)

        return track
