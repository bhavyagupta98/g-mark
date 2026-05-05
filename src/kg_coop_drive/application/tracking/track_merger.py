from __future__ import annotations

from dataclasses import replace
from math import dist

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    ObjectTrack,
    ProvenanceRecord,
    TrackMerge,
    TrackMergeReport,
    TrackStatus,
)


class TrackMerger:
    """Conservatively merges candidate tracks into stronger existing tracks."""

    def merge(
        self,
        scene: CooperativeScene,
        max_distance: float = 1.0,
    ) -> tuple[CooperativeScene, TrackMergeReport]:
        """Return a scene with mergeable candidate tracks folded into stronger tracks."""

        candidates = [track for track in scene.object_tracks if track.status == TrackStatus.CANDIDATE]
        anchors = [track for track in scene.object_tracks if track.status != TrackStatus.CANDIDATE]
        if not candidates or not anchors:
            return scene, TrackMergeReport(merges=tuple(), remaining_candidate_ids=tuple(track.object_id for track in candidates))

        candidate_by_id = {track.object_id: track for track in candidates}
        anchor_by_id = {track.object_id: track for track in anchors}
        merge_candidates = []
        for candidate in candidates:
            for anchor in anchors:
                if candidate.object_type != anchor.object_type:
                    continue
                distance_meters = dist(
                    (candidate.position.x, candidate.position.y),
                    (anchor.position.x, anchor.position.y),
                )
                if distance_meters <= max_distance:
                    merge_candidates.append((distance_meters, candidate.object_id, anchor.object_id))

        merge_candidates.sort()
        merged_candidate_ids: set[str] = set()
        touched_anchor_ids: set[str] = set()
        merge_records: list[TrackMerge] = []

        for distance_meters, candidate_id, anchor_id in merge_candidates:
            if candidate_id in merged_candidate_ids or anchor_id in touched_anchor_ids:
                continue
            candidate_track = candidate_by_id[candidate_id]
            anchor_track = anchor_by_id[anchor_id]
            anchor_by_id[anchor_id] = self._merge_tracks(anchor_track, candidate_track)
            merged_candidate_ids.add(candidate_id)
            touched_anchor_ids.add(anchor_id)
            merge_records.append(
                TrackMerge(
                    source_track_id=candidate_id,
                    target_track_id=anchor_id,
                    distance_meters=distance_meters,
                )
            )

        merged_scene_tracks = []
        for object_track in scene.object_tracks:
            if object_track.object_id in merged_candidate_ids:
                continue
            if object_track.object_id in anchor_by_id:
                merged_scene_tracks.append(anchor_by_id[object_track.object_id])
            else:
                merged_scene_tracks.append(object_track)

        remaining_candidate_ids = tuple(
            track.object_id
            for track in merged_scene_tracks
            if track.status == TrackStatus.CANDIDATE
        )
        return replace(scene, object_tracks=tuple(merged_scene_tracks)), TrackMergeReport(
            merges=tuple(merge_records),
            remaining_candidate_ids=remaining_candidate_ids,
        )

    @staticmethod
    def _merge_tracks(anchor_track: ObjectTrack, candidate_track: ObjectTrack) -> ObjectTrack:
        source_agent_ids = list(anchor_track.provenance.source_agent_ids)
        observation_ids = list(anchor_track.provenance.observation_ids)
        for agent_id in candidate_track.provenance.source_agent_ids:
            if agent_id not in source_agent_ids:
                source_agent_ids.append(agent_id)
        for observation_id in candidate_track.provenance.observation_ids:
            if observation_id not in observation_ids:
                observation_ids.append(observation_id)

        provenance = ProvenanceRecord(
            source_agent_ids=tuple(source_agent_ids),
            observation_ids=tuple(observation_ids),
            latest_timestamp_index=max(
                anchor_track.provenance.latest_timestamp_index,
                candidate_track.provenance.latest_timestamp_index,
            ),
        )
        merged_observations = anchor_track.observations + tuple(
            observation
            for observation in candidate_track.observations
            if observation.observation_id
            not in {existing.observation_id for existing in anchor_track.observations}
        )
        return replace(
            anchor_track,
            provenance=provenance,
            observations=merged_observations,
            status=TrackStatus.SUPPORTED,
            confidence=max(anchor_track.confidence, candidate_track.confidence),
        )
