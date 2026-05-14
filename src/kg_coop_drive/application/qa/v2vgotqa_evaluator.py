from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum

from kg_coop_drive.application.candidate_track_creator import CandidateTrackCreator
from kg_coop_drive.application.candidate_track_resolver import CandidateTrackResolver
from kg_coop_drive.application.cross_agent_associator import CrossAgentAssociator
from kg_coop_drive.application.cross_agent_support_enricher import CrossAgentSupportEnricher
from kg_coop_drive.application.observation_associator import ObservationAssociator
from kg_coop_drive.application.processed_scene_service import ProcessedSceneEnricher
from kg_coop_drive.application.relation_builder import RelationBuilder
from kg_coop_drive.application.temporal_track_manager import TemporalTrackManager
from kg_coop_drive.application.track_merger import TrackMerger
from kg_coop_drive.application.track_quality_assessor import TrackQualityAssessor
from kg_coop_drive.application.track_support_enricher import TrackSupportEnricher
from kg_coop_drive.application.v2vgotqa_router import V2VGoTQARouter
from kg_coop_drive.application.visibility_reasoner import VisibilityReasoner
from kg_coop_drive.domain.benchmark import (
    BenchmarkEvaluationSummary,
    BenchmarkPrediction,
    BenchmarkSample,
    BenchmarkTaskType,
)
from kg_coop_drive.domain.processed_scene import ProcessedFrameSceneData
from kg_coop_drive.domain.scene import CooperativeScene, VisibilityFact
from kg_coop_drive.domain.scene import ObjectTrack, ProvenanceRecord, RelationType, TrackStatus, VisibilityState
from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader


class GraphAblationMode(str, Enum):
    """Scene-graph ablations for G-MARK validation and train/validation studies."""

    FULL = "full"
    NO_PROVENANCE = "no_provenance"
    NO_CANDIDATE_RETENTION = "no_candidate_retention"
    NO_UNCERTAINTY_CONFLICT = "no_uncertainty_conflict"
    NO_GRAPH_RELATIONS = "no_graph_relations"
    FLAT_NON_GRAPH_READOUT = "flat_non_graph_readout"


@dataclass(frozen=True)
class GraphAblationConfig:
    """Controls additive graph ablations without changing the default pipeline."""

    mode: GraphAblationMode = GraphAblationMode.FULL

    @classmethod
    def from_value(cls, value: str | GraphAblationMode = GraphAblationMode.FULL) -> "GraphAblationConfig":
        if isinstance(value, GraphAblationMode):
            return cls(mode=value)
        return cls(mode=GraphAblationMode(value))


class V2VGoTQAPhase5AEvaluator:
    """Evaluates supported Phase 5A V2V-GoT-QA tasks through the graph pipeline."""

    def __init__(
        self,
        repository_root: str,
        router: V2VGoTQARouter | None = None,
        processed_loader: V2VGoTProcessedAssetLoader | None = None,
        graph_ablation: GraphAblationConfig | str | None = None,
    ) -> None:
        self._router = router or V2VGoTQARouter()
        self._graph_ablation = (
            graph_ablation
            if isinstance(graph_ablation, GraphAblationConfig)
            else GraphAblationConfig.from_value(graph_ablation or GraphAblationMode.FULL)
        )
        self._processed_loader = processed_loader or V2VGoTProcessedAssetLoader(
            repository_root,
            asset_profile="cooperative",
        )
        self._processed_enricher = ProcessedSceneEnricher()
        self._observation_associator = ObservationAssociator()
        self._track_support_enricher = TrackSupportEnricher()
        self._candidate_track_creator = CandidateTrackCreator()
        self._candidate_track_resolver = CandidateTrackResolver()
        self._cross_agent_associator = CrossAgentAssociator()
        self._cross_agent_support_enricher = CrossAgentSupportEnricher()
        self._track_merger = TrackMerger()
        self._track_quality_assessor = TrackQualityAssessor()
        self._visibility_reasoner = VisibilityReasoner()
        self._relation_builder = RelationBuilder()
        self._temporal_track_manager = TemporalTrackManager()

    def evaluate_samples(
        self,
        samples: tuple[BenchmarkSample, ...],
        baseline_mode: str = "cooperative",
        progress_every: int = 0,
    ) -> tuple[BenchmarkPrediction, ...]:
        """Evaluate one batch of benchmark samples."""

        predictions: list[BenchmarkPrediction] = []
        total_samples = len(samples)
        for index, sample in enumerate(samples, start=1):
            prepared_scene = self.prepare_sample(
                sample=sample,
                baseline_mode=baseline_mode,
            )
            prepared_sample = replace(sample, scene=prepared_scene)
            answer = self._router.answer(prepared_sample)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=sample.sample_id,
                    dataset_name=sample.dataset_name,
                    split_name=sample.split_name,
                    task_type=sample.task_type,
                    qa_type_id=sample.qa_type_id,
                    supported=answer.supported,
                    answer_text=answer.answer_text,
                    object_ids=answer.object_ids,
                    baseline_mode=baseline_mode,
                )
            )
            if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total_samples):
                print(
                    f"progress: {index}/{total_samples} "
                    f"task={sample.task_type.value} sample_id={sample.sample_id}",
                    flush=True,
                )
        return tuple(predictions)

    @staticmethod
    def summarize(
        predictions: tuple[BenchmarkPrediction, ...],
    ) -> BenchmarkEvaluationSummary:
        """Summarize one prediction batch."""

        if not predictions:
            return BenchmarkEvaluationSummary(
                dataset_name="V2V-GoT-QA",
                split_name="unknown",
                baseline_mode="unknown",
                total_samples=0,
                evaluated_samples=0,
                supported_predictions=0,
                unsupported_predictions=0,
            )

        first = predictions[0]
        supported_predictions = sum(1 for prediction in predictions if prediction.supported)
        return BenchmarkEvaluationSummary(
            dataset_name=first.dataset_name,
            split_name=first.split_name,
            baseline_mode=first.baseline_mode,
            total_samples=len(predictions),
            evaluated_samples=len(predictions),
            supported_predictions=supported_predictions,
            unsupported_predictions=len(predictions) - supported_predictions,
        )

    def prepare_sample(
        self,
        sample: BenchmarkSample,
        baseline_mode: str,
    ) -> CooperativeScene:
        """Prepare one benchmark sample scene for evaluation under one baseline mode."""

        processed_data = self._processed_loader.load_frame_scene_data(
            timestamp_index=sample.scene.global_timestamp_index,
            split_name=sample.split_name,
        )
        if processed_data is None:
            return sample.scene

        processed_data = self._apply_baseline_mode_to_processed_data(
            sample=sample,
            processed_data=processed_data,
            baseline_mode=baseline_mode,
        )
        scene = self._prepare_single_frame_scene(sample.scene, processed_data)
        if self._should_temporally_enrich(sample):
            scene = self._temporally_enrich_motion_scene(
                sample=sample,
                current_scene=scene,
                baseline_mode=baseline_mode,
            )
        if baseline_mode == "ego_only":
            scene = self._keep_asker_visibility_only(scene)
        return self._apply_graph_ablation(scene)

    def _prepare_single_frame_scene(
        self,
        scene: CooperativeScene,
        processed_data: ProcessedFrameSceneData,
    ) -> CooperativeScene:
        scene = self._processed_enricher.enrich(scene, processed_data)
        association_report = self._observation_associator.associate(scene, max_distance=3.0)
        if self._uses_provenance():
            scene = self._track_support_enricher.enrich(scene, association_report)
        else:
            scene = self._without_provenance(scene)

        if self._retains_candidates():
            scene = self._candidate_track_creator.promote(scene, association_report)
            scene, _candidate_resolution_report = self._candidate_track_resolver.resolve(
                scene,
                min_candidate_confidence=0.25,
            )
            if not self._uses_provenance():
                scene = self._without_provenance(scene)

        cross_agent_report = self._cross_agent_associator.associate(scene, max_distance=3.0)
        if self._uses_provenance():
            scene, _cross_agent_support_report = self._cross_agent_support_enricher.enrich(
                scene,
                cross_agent_report,
            )
        if self._retains_candidates():
            scene, _track_merge_report = self._track_merger.merge(scene, max_distance=1.0)
            if not self._uses_provenance():
                scene = self._without_provenance(scene)

        if self._uses_uncertainty_conflict():
            scene = self._track_quality_assessor.assess(scene)
        else:
            scene = self._without_uncertainty_conflict(scene)

        if self._infers_visibility():
            scene, _visibility_reasoning_report = self._visibility_reasoner.infer(
                scene,
                uncertain_distance=30.0,
                min_candidate_visible_confidence=0.5,
            )
        if self._builds_graph_relations():
            scene = self._relation_builder.build(scene)
        return self._apply_graph_ablation(scene)

    def _temporally_enrich_motion_scene(
        self,
        sample: BenchmarkSample,
        current_scene: CooperativeScene,
        baseline_mode: str,
    ) -> CooperativeScene:
        previous_timestamp_index = sample.scene.global_timestamp_index - 1
        if previous_timestamp_index < 0:
            return current_scene

        previous_processed_data = self._processed_loader.load_frame_scene_data(
            timestamp_index=previous_timestamp_index,
            split_name=sample.split_name,
        )
        if previous_processed_data is None:
            return current_scene

        previous_scene_seed = replace(
            sample.scene,
            local_timestamp_index=previous_timestamp_index,
            global_timestamp_index=previous_timestamp_index,
        )
        previous_processed_data = self._apply_baseline_mode_to_processed_data(
            sample=replace(sample, scene=previous_scene_seed),
            processed_data=previous_processed_data,
            baseline_mode=baseline_mode,
        )
        previous_scene = self._prepare_single_frame_scene(
            previous_scene_seed,
            previous_processed_data,
        )
        temporal_scene, _temporal_report = self._temporal_track_manager.update(
            previous_scene,
            current_scene,
            max_distance=8.0,
            max_missed_frames=0,
        )
        if not self._uses_provenance():
            temporal_scene = self._without_provenance(temporal_scene)
        if self._uses_uncertainty_conflict():
            temporal_scene = self._track_quality_assessor.assess(temporal_scene)
        else:
            temporal_scene = self._without_uncertainty_conflict(temporal_scene)
        if self._builds_graph_relations():
            temporal_scene = self._relation_builder.build(temporal_scene)
        return self._apply_graph_ablation(temporal_scene)

    def _uses_provenance(self) -> bool:
        return self._graph_ablation.mode not in {
            GraphAblationMode.NO_PROVENANCE,
            GraphAblationMode.FLAT_NON_GRAPH_READOUT,
        }

    def _retains_candidates(self) -> bool:
        return self._graph_ablation.mode not in {
            GraphAblationMode.NO_CANDIDATE_RETENTION,
            GraphAblationMode.FLAT_NON_GRAPH_READOUT,
        }

    def _uses_uncertainty_conflict(self) -> bool:
        return self._graph_ablation.mode not in {
            GraphAblationMode.NO_UNCERTAINTY_CONFLICT,
            GraphAblationMode.FLAT_NON_GRAPH_READOUT,
        }

    def _builds_graph_relations(self) -> bool:
        return self._graph_ablation.mode not in {
            GraphAblationMode.NO_GRAPH_RELATIONS,
            GraphAblationMode.FLAT_NON_GRAPH_READOUT,
        }

    def _infers_visibility(self) -> bool:
        return self._graph_ablation.mode != GraphAblationMode.FLAT_NON_GRAPH_READOUT

    def _apply_graph_ablation(self, scene: CooperativeScene) -> CooperativeScene:
        mode = self._graph_ablation.mode
        if mode == GraphAblationMode.FULL:
            return scene
        if mode == GraphAblationMode.NO_PROVENANCE:
            return self._without_provenance(scene)
        if mode == GraphAblationMode.NO_CANDIDATE_RETENTION:
            return self._without_candidates(scene)
        if mode == GraphAblationMode.NO_UNCERTAINTY_CONFLICT:
            return self._without_uncertainty_conflict(scene)
        if mode == GraphAblationMode.NO_GRAPH_RELATIONS:
            return replace(scene, relations=tuple())
        if mode == GraphAblationMode.FLAT_NON_GRAPH_READOUT:
            return self._flat_non_graph_scene(scene)
        raise ValueError(f"Unsupported graph ablation mode: {mode}")

    @staticmethod
    def _without_provenance(scene: CooperativeScene) -> CooperativeScene:
        object_tracks = tuple(
            replace(
                track,
                provenance=ProvenanceRecord(
                    source_agent_ids=tuple(),
                    observation_ids=tuple(),
                    latest_timestamp_index=track.provenance.latest_timestamp_index,
                ),
                observations=tuple(),
                last_support_confidence=0.0,
            )
            for track in scene.object_tracks
        )
        provenance_relation_types = {
            RelationType.COOPERATIVELY_SUPPORTED,
            RelationType.OBSERVED_BY,
        }
        return replace(
            scene,
            object_tracks=object_tracks,
            relations=tuple(
                relation
                for relation in scene.relations
                if relation.relation_type not in provenance_relation_types
            ),
        )

    @staticmethod
    def _without_candidates(scene: CooperativeScene) -> CooperativeScene:
        kept_tracks = tuple(
            track
            for track in scene.object_tracks
            if track.status != TrackStatus.CANDIDATE and not track.object_id.startswith("pred_candidate_")
        )
        kept_ids = {track.object_id for track in kept_tracks}
        return replace(
            scene,
            object_tracks=kept_tracks,
            visibility_facts=tuple(fact for fact in scene.visibility_facts if fact.object_id in kept_ids),
            relations=tuple(
                relation
                for relation in scene.relations
                if relation.subject_id in kept_ids and relation.object_id in kept_ids
            ),
        )

    @staticmethod
    def _without_uncertainty_conflict(scene: CooperativeScene) -> CooperativeScene:
        object_tracks = tuple(
            replace(track, uncertainty_score=0.0, conflict_score=0.0)
            for track in scene.object_tracks
        )
        return replace(
            scene,
            object_tracks=object_tracks,
            relations=tuple(
                relation
                for relation in scene.relations
                if relation.relation_type != RelationType.LOW_CONFLICT
            ),
        )

    def _flat_non_graph_scene(self, scene: CooperativeScene) -> CooperativeScene:
        scene = self._without_candidates(scene)
        scene = self._without_provenance(scene)
        scene = self._without_uncertainty_conflict(scene)
        object_tracks: tuple[ObjectTrack, ...] = tuple(
            replace(track, status=TrackStatus.CONFIRMED)
            for track in scene.object_tracks
        )
        return replace(
            scene,
            object_tracks=object_tracks,
            relations=tuple(),
        )

    @staticmethod
    def _keep_asker_visibility_only(scene: CooperativeScene) -> CooperativeScene:
        return replace(
            scene,
            visibility_facts=tuple(
                fact for fact in scene.visibility_facts if fact.agent_id == scene.asker_agent_id
            ),
        )

    @staticmethod
    def _should_temporally_enrich(sample: BenchmarkSample) -> bool:
        return sample.task_type in {
            BenchmarkTaskType.OBJECT_MOTION_PREDICTION,
            BenchmarkTaskType.AGENT_MOTION_PREDICTION,
        }

    @staticmethod
    def _apply_baseline_mode_to_processed_data(
        sample: BenchmarkSample,
        processed_data: ProcessedFrameSceneData,
        baseline_mode: str,
    ) -> ProcessedFrameSceneData:
        if baseline_mode == "cooperative":
            return processed_data
        if baseline_mode != "ego_only":
            raise ValueError(f"Unsupported baseline mode: {baseline_mode}")

        asker_agent_id = sample.scene.asker_agent_id
        observations = tuple(
            observation
            for observation in processed_data.observations
            if observation.source_agent_id == asker_agent_id
        )
        visible_object_ids = {
            fact.object_id
            for fact in processed_data.visibility_facts
            if fact.agent_id == asker_agent_id and fact.state == VisibilityState.VISIBLE
        }
        object_tracks = tuple(
            track
            for track in processed_data.object_tracks
            if track.object_id in visible_object_ids
        )
        visibility_facts = tuple(
            fact
            for fact in processed_data.visibility_facts
            if fact.agent_id == asker_agent_id and fact.object_id in visible_object_ids
        )
        source_paths = tuple(
            path
            for path in processed_data.source_paths
            if "/ego/" in path
            or "gt_object_id" in path
            or path.endswith("_gt.npy")
            or any(
                suffix in path
                for suffix in (
                    f"_visible_to_{'ego' if asker_agent_id == 'CAV_EGO' else '1'}.npy",
                    f"_invisible_to_{'ego' if asker_agent_id == 'CAV_EGO' else '1'}.npy",
                )
            )
        )
        return ProcessedFrameSceneData(
            timestamp_index=processed_data.timestamp_index,
            observations=observations,
            object_tracks=object_tracks,
            visibility_facts=visibility_facts,
            source_paths=source_paths,
        )
