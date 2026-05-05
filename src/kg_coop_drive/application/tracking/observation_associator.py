from __future__ import annotations

from math import dist

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    ObservationAssociationReport,
    ObservationTrackAssociation,
)


class ObservationAssociator:
    """Associates detector-backed observations to current object tracks."""

    def associate(
        self,
        scene: CooperativeScene,
        max_distance: float = 3.0,
    ) -> ObservationAssociationReport:
        """Greedily match nearest compatible observations to object tracks."""

        candidates = []
        for object_track in scene.object_tracks:
            for observation in scene.observations:
                if observation.object_type != object_track.object_type:
                    continue
                distance_meters = dist(
                    (object_track.position.x, object_track.position.y),
                    (observation.position.x, observation.position.y),
                )
                if distance_meters <= max_distance:
                    candidates.append(
                        (
                            distance_meters,
                            -observation.confidence,
                            object_track.object_id,
                            observation.observation_id,
                            observation.source_agent_id,
                            observation.confidence,
                        )
                    )

        candidates.sort()

        matched_track_ids: set[str] = set()
        matched_observation_ids: set[str] = set()
        matches: list[ObservationTrackAssociation] = []

        for (
            distance_meters,
            _negative_confidence,
            track_id,
            observation_id,
            source_agent_id,
            observation_confidence,
        ) in candidates:
            if track_id in matched_track_ids or observation_id in matched_observation_ids:
                continue
            matched_track_ids.add(track_id)
            matched_observation_ids.add(observation_id)
            matches.append(
                ObservationTrackAssociation(
                    track_id=track_id,
                    observation_id=observation_id,
                    source_agent_id=source_agent_id,
                    distance_meters=distance_meters,
                    observation_confidence=observation_confidence,
                )
            )

        unmatched_track_ids = tuple(
            object_track.object_id
            for object_track in scene.object_tracks
            if object_track.object_id not in matched_track_ids
        )
        unmatched_observation_ids = tuple(
            observation.observation_id
            for observation in scene.observations
            if observation.observation_id not in matched_observation_ids
        )

        return ObservationAssociationReport(
            matches=tuple(matches),
            unmatched_track_ids=unmatched_track_ids,
            unmatched_observation_ids=unmatched_observation_ids,
        )
