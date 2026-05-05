from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    CrossAgentAssociationReport,
    CrossAgentSupportAttachmentReport,
    ObjectTrack,
    ObservationEvidence,
    ProvenanceRecord,
)


class CrossAgentSupportEnricher:
    """Attaches cross-agent matched observations onto already-supported tracks."""

    def enrich(
        self,
        scene: CooperativeScene,
        cross_agent_report: CrossAgentAssociationReport,
    ) -> tuple[CooperativeScene, CrossAgentSupportAttachmentReport]:
        """Return a scene whose matching tracks carry additional cross-agent evidence."""

        if not cross_agent_report.matches:
            return (
                scene,
                CrossAgentSupportAttachmentReport(
                    attached_match_count=0,
                    enriched_track_ids=tuple(),
                ),
            )

        observation_by_id = {
            observation.observation_id: observation for observation in scene.observations
        }
        observation_track_ids: dict[str, list[str]] = {}
        for object_track in scene.object_tracks:
            for observation in object_track.observations:
                observation_track_ids.setdefault(observation.observation_id, []).append(
                    object_track.object_id
                )

        additional_by_track_id: dict[str, list[ObservationEvidence]] = {}
        attached_match_count = 0
        for match in cross_agent_report.matches:
            left_track_ids = observation_track_ids.get(match.left_observation_id, [])
            right_track_ids = observation_track_ids.get(match.right_observation_id, [])

            if left_track_ids and not right_track_ids:
                counterpart = observation_by_id.get(match.right_observation_id)
                if counterpart is not None:
                    for track_id in left_track_ids:
                        additional_by_track_id.setdefault(track_id, []).append(counterpart)
                    attached_match_count += 1
                continue

            if right_track_ids and not left_track_ids:
                counterpart = observation_by_id.get(match.left_observation_id)
                if counterpart is not None:
                    for track_id in right_track_ids:
                        additional_by_track_id.setdefault(track_id, []).append(counterpart)
                    attached_match_count += 1

        if not additional_by_track_id:
            return (
                scene,
                CrossAgentSupportAttachmentReport(
                    attached_match_count=0,
                    enriched_track_ids=tuple(),
                ),
            )

        enriched_track_ids: list[str] = []
        enriched_tracks = []
        for object_track in scene.object_tracks:
            additional = additional_by_track_id.get(object_track.object_id, [])
            if additional:
                enriched_tracks.append(self._enrich_track(object_track, additional))
                enriched_track_ids.append(object_track.object_id)
            else:
                enriched_tracks.append(object_track)

        return (
            replace(scene, object_tracks=tuple(enriched_tracks)),
            CrossAgentSupportAttachmentReport(
                attached_match_count=attached_match_count,
                enriched_track_ids=tuple(enriched_track_ids),
            ),
        )

    @staticmethod
    def _enrich_track(
        object_track: ObjectTrack,
        additional_observations: list[ObservationEvidence],
    ) -> ObjectTrack:
        existing_agents = list(object_track.provenance.source_agent_ids)
        existing_observation_ids = list(object_track.provenance.observation_ids)
        latest_timestamp_index = object_track.provenance.latest_timestamp_index
        existing_attached_ids = {observation.observation_id for observation in object_track.observations}
        merged_observations = list(object_track.observations)

        for observation in additional_observations:
            if observation.source_agent_id not in existing_agents:
                existing_agents.append(observation.source_agent_id)
            if observation.observation_id not in existing_observation_ids:
                existing_observation_ids.append(observation.observation_id)
            if observation.observation_id not in existing_attached_ids:
                merged_observations.append(observation)
                existing_attached_ids.add(observation.observation_id)
            latest_timestamp_index = max(latest_timestamp_index, observation.timestamp_index)

        return replace(
            object_track,
            provenance=ProvenanceRecord(
                source_agent_ids=tuple(existing_agents),
                observation_ids=tuple(existing_observation_ids),
                latest_timestamp_index=latest_timestamp_index,
            ),
            observations=tuple(merged_observations),
        )
