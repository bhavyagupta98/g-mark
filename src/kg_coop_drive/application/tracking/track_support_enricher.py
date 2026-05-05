from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    ObjectTrack,
    ObservationAssociationReport,
    ObservationEvidence,
    ProvenanceRecord,
    TrackStatus,
)


class TrackSupportEnricher:
    """Attaches matched observations as support evidence onto object tracks."""

    def enrich(
        self,
        scene: CooperativeScene,
        association_report: ObservationAssociationReport,
    ) -> CooperativeScene:
        """Return a scene whose tracks carry matched observation support."""

        observation_by_id = {
            observation.observation_id: observation for observation in scene.observations
        }
        matches_by_track_id: dict[str, list[ObservationEvidence]] = {}
        for match in association_report.matches:
            observation = observation_by_id.get(match.observation_id)
            if observation is None:
                continue
            matches_by_track_id.setdefault(match.track_id, []).append(observation)

        enriched_tracks = tuple(
            self._enrich_track(object_track, matches_by_track_id.get(object_track.object_id, []))
            for object_track in scene.object_tracks
        )
        return replace(scene, object_tracks=enriched_tracks)

    @staticmethod
    def _enrich_track(
        object_track: ObjectTrack,
        matched_observations: list[ObservationEvidence],
    ) -> ObjectTrack:
        if not matched_observations:
            return object_track

        existing_agents = list(object_track.provenance.source_agent_ids)
        existing_observation_ids = list(object_track.provenance.observation_ids)
        latest_timestamp_index = object_track.provenance.latest_timestamp_index

        for observation in matched_observations:
            if observation.source_agent_id not in existing_agents:
                existing_agents.append(observation.source_agent_id)
            if observation.observation_id not in existing_observation_ids:
                existing_observation_ids.append(observation.observation_id)
            latest_timestamp_index = max(latest_timestamp_index, observation.timestamp_index)

        provenance = ProvenanceRecord(
            source_agent_ids=tuple(existing_agents),
            observation_ids=tuple(existing_observation_ids),
            latest_timestamp_index=latest_timestamp_index,
        )
        merged_observations = object_track.observations + tuple(matched_observations)
        return replace(
            object_track,
            provenance=provenance,
            status=TrackStatus.SUPPORTED,
            observations=merged_observations,
        )
