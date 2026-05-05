from __future__ import annotations

from dataclasses import replace
from math import dist

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    ObjectTrack,
    ProvenanceRecord,
    TemporalTrackUpdateReport,
    Vector2D,
)


class TemporalTrackManager:
    """Maintains conservative object identity continuity across frames."""

    def update(
        self,
        previous_scene: CooperativeScene,
        current_scene: CooperativeScene,
        max_distance: float = 2.0,
        max_missed_frames: int = 1,
    ) -> tuple[CooperativeScene, TemporalTrackUpdateReport]:
        """Carry track identities forward when the same object is observed again."""

        candidates = []
        for previous_track in previous_scene.object_tracks:
            for current_track in current_scene.object_tracks:
                if previous_track.object_type != current_track.object_type:
                    continue
                distance_meters = dist(
                    (previous_track.position.x, previous_track.position.y),
                    (current_track.position.x, current_track.position.y),
                )
                if distance_meters <= max_distance:
                    candidates.append(
                        (
                            distance_meters,
                            previous_track.object_id,
                            current_track.object_id,
                        )
                    )

        candidates.sort()
        matched_previous_ids: set[str] = set()
        matched_current_ids: set[str] = set()
        persisted_track_ids: list[str] = []
        previous_by_id = {track.object_id: track for track in previous_scene.object_tracks}
        current_by_id = {track.object_id: track for track in current_scene.object_tracks}
        merged_current_tracks: dict[str, ObjectTrack] = {}

        for _distance_meters, previous_id, current_id in candidates:
            if previous_id in matched_previous_ids or current_id in matched_current_ids:
                continue
            previous_track = previous_by_id[previous_id]
            current_track = current_by_id[current_id]
            merged_current_tracks[current_id] = self._carry_forward(previous_track, current_track)
            matched_previous_ids.add(previous_id)
            matched_current_ids.add(current_id)
            persisted_track_ids.append(previous_id)

        final_tracks = []
        new_track_ids = []
        for current_track in current_scene.object_tracks:
            if current_track.object_id in merged_current_tracks:
                final_tracks.append(merged_current_tracks[current_track.object_id])
            else:
                final_tracks.append(current_track)
                new_track_ids.append(current_track.object_id)

        retained_stale_tracks = []
        retained_stale_track_ids = []
        pruned_stale_track_ids = []
        for track in previous_scene.object_tracks:
            if track.object_id in matched_previous_ids:
                continue
            retained_track = replace(track, miss_count=track.miss_count + 1)
            if retained_track.miss_count <= max_missed_frames:
                retained_stale_tracks.append(retained_track)
                retained_stale_track_ids.append(retained_track.object_id)
            else:
                pruned_stale_track_ids.append(retained_track.object_id)

        final_tracks.extend(retained_stale_tracks)
        report = TemporalTrackUpdateReport(
            persisted_track_ids=tuple(persisted_track_ids),
            new_track_ids=tuple(new_track_ids),
            retained_stale_track_ids=tuple(retained_stale_track_ids),
            pruned_stale_track_ids=tuple(pruned_stale_track_ids),
        )
        return replace(current_scene, object_tracks=tuple(final_tracks)), report

    @staticmethod
    def _carry_forward(previous_track: ObjectTrack, current_track: ObjectTrack) -> ObjectTrack:
        source_agent_ids = list(previous_track.provenance.source_agent_ids)
        observation_ids = list(previous_track.provenance.observation_ids)
        for agent_id in current_track.provenance.source_agent_ids:
            if agent_id not in source_agent_ids:
                source_agent_ids.append(agent_id)
        for observation_id in current_track.provenance.observation_ids:
            if observation_id not in observation_ids:
                observation_ids.append(observation_id)

        provenance = ProvenanceRecord(
            source_agent_ids=tuple(source_agent_ids),
            observation_ids=tuple(observation_ids),
            latest_timestamp_index=max(
                previous_track.provenance.latest_timestamp_index,
                current_track.provenance.latest_timestamp_index,
            ),
        )
        existing_observation_ids = {obs.observation_id for obs in previous_track.observations}
        merged_observations = previous_track.observations + tuple(
            observation
            for observation in current_track.observations
            if observation.observation_id not in existing_observation_ids
        )
        frame_delta = max(
            1,
            current_track.provenance.latest_timestamp_index
            - previous_track.provenance.latest_timestamp_index,
        )
        velocity = Vector2D(
            x=(current_track.position.x - previous_track.position.x) / frame_delta,
            y=(current_track.position.y - previous_track.position.y) / frame_delta,
        )
        return replace(
            current_track,
            object_id=previous_track.object_id,
            provenance=provenance,
            status=current_track.status,
            age_frames=previous_track.age_frames + 1,
            miss_count=0,
            velocity=velocity,
            observations=merged_observations,
        )
