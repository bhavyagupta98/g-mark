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


class CandidateTrackCreator:
    """Promotes unmatched observations into prediction-only candidate tracks."""

    def promote(
        self,
        scene: CooperativeScene,
        association_report: ObservationAssociationReport,
    ) -> CooperativeScene:
        """Return a scene with candidate tracks appended for unmatched observations."""

        if not association_report.unmatched_observation_ids:
            return scene

        observation_by_id = {
            observation.observation_id: observation for observation in scene.observations
        }
        existing_ids = {object_track.object_id for object_track in scene.object_tracks}
        candidate_tracks: list[ObjectTrack] = []

        for index, observation_id in enumerate(association_report.unmatched_observation_ids):
            observation = observation_by_id.get(observation_id)
            if observation is None:
                continue
            candidate_id = self._next_candidate_id(
                existing_ids=existing_ids,
                timestamp_index=scene.global_timestamp_index,
                index=index,
            )
            existing_ids.add(candidate_id)
            candidate_tracks.append(
                self._build_candidate_track(candidate_id=candidate_id, observation=observation)
            )

        return replace(scene, object_tracks=scene.object_tracks + tuple(candidate_tracks))

    @staticmethod
    def _next_candidate_id(
        existing_ids: set[str],
        timestamp_index: int,
        index: int,
    ) -> str:
        base_id = f"pred_candidate_{timestamp_index}_{index}"
        if base_id not in existing_ids:
            return base_id

        suffix = 1
        while f"{base_id}_{suffix}" in existing_ids:
            suffix += 1
        return f"{base_id}_{suffix}"

    @staticmethod
    def _build_candidate_track(
        candidate_id: str,
        observation: ObservationEvidence,
    ) -> ObjectTrack:
        provenance = ProvenanceRecord(
            source_agent_ids=(observation.source_agent_id,),
            observation_ids=(observation.observation_id,),
            latest_timestamp_index=observation.timestamp_index,
        )
        return ObjectTrack(
            object_id=candidate_id,
            object_type=observation.object_type,
            position=observation.position,
            confidence=observation.confidence,
            provenance=provenance,
            status=TrackStatus.CANDIDATE,
            observations=(observation,),
        )
