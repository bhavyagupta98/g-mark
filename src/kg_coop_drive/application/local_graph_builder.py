from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.application.candidate_track_creator import CandidateTrackCreator
from kg_coop_drive.application.candidate_track_resolver import CandidateTrackResolver
from kg_coop_drive.application.observation_associator import ObservationAssociator
from kg_coop_drive.application.processed_scene_service import ProcessedSceneEnricher
from kg_coop_drive.application.relation_builder import RelationBuilder
from kg_coop_drive.application.track_quality_assessor import TrackQualityAssessor
from kg_coop_drive.application.track_merger import TrackMerger
from kg_coop_drive.application.track_support_enricher import TrackSupportEnricher
from kg_coop_drive.application.visibility_reasoner import VisibilityReasoner
from kg_coop_drive.domain.processed_scene import ProcessedFrameSceneData
from kg_coop_drive.domain.scene import CooperativeScene


class LocalGraphBuilder:
    """Builds a single-agent local graph using one agent's evidence only."""

    def __init__(self) -> None:
        self._processed_scene_enricher = ProcessedSceneEnricher()
        self._observation_associator = ObservationAssociator()
        self._track_support_enricher = TrackSupportEnricher()
        self._candidate_track_creator = CandidateTrackCreator()
        self._candidate_track_resolver = CandidateTrackResolver()
        self._track_merger = TrackMerger()
        self._track_quality_assessor = TrackQualityAssessor()
        self._visibility_reasoner = VisibilityReasoner()
        self._relation_builder = RelationBuilder()

    def build(
        self,
        scene: CooperativeScene,
        processed_data: ProcessedFrameSceneData | None,
        agent_id: str,
    ) -> CooperativeScene:
        """Return a local graph for one agent from a global scene seed plus processed data."""

        local_scene = self._project_seed_to_local_scene(scene, agent_id)
        if processed_data is None:
            return local_scene

        local_processed_data = self._filter_processed_data_for_agent(processed_data, agent_id)
        local_scene = self._processed_scene_enricher.enrich(local_scene, local_processed_data)
        association_report = self._observation_associator.associate(local_scene, max_distance=3.0)
        local_scene = self._track_support_enricher.enrich(local_scene, association_report)
        local_scene = self._candidate_track_creator.promote(local_scene, association_report)
        local_scene, _candidate_resolution_report = self._candidate_track_resolver.resolve(
            local_scene,
            min_candidate_confidence=0.25,
        )
        local_scene, _track_merge_report = self._track_merger.merge(local_scene, max_distance=1.0)
        local_scene = self._track_quality_assessor.assess(local_scene)
        local_scene, _visibility_reasoning_report = self._visibility_reasoner.infer(
            local_scene,
            uncertain_distance=30.0,
            min_candidate_visible_confidence=0.5,
        )
        local_scene = self._relation_builder.build(local_scene)
        return local_scene

    @staticmethod
    def _project_seed_to_local_scene(scene: CooperativeScene, agent_id: str) -> CooperativeScene:
        local_agents = tuple(agent for agent in scene.agents if agent.agent_id == agent_id)
        return replace(
            scene,
            asker_agent_id=agent_id,
            agents=local_agents,
            observations=tuple(),
            object_tracks=tuple(),
            relations=tuple(),
            visibility_facts=tuple(),
        )

    @staticmethod
    def _filter_processed_data_for_agent(
        processed_data: ProcessedFrameSceneData,
        agent_id: str,
    ) -> ProcessedFrameSceneData:
        filtered_observations = tuple(
            observation
            for observation in processed_data.observations
            if observation.source_agent_id == agent_id
        )
        filtered_visibility_facts = tuple(
            fact
            for fact in processed_data.visibility_facts
            if fact.agent_id == agent_id
        )
        return ProcessedFrameSceneData(
            timestamp_index=processed_data.timestamp_index,
            observations=filtered_observations,
            object_tracks=processed_data.object_tracks,
            visibility_facts=filtered_visibility_facts,
            source_paths=processed_data.source_paths,
        )
