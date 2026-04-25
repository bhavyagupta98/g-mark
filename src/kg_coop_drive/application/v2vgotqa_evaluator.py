from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.application.candidate_track_creator import CandidateTrackCreator
from kg_coop_drive.application.candidate_track_resolver import CandidateTrackResolver
from kg_coop_drive.application.cross_agent_associator import CrossAgentAssociator
from kg_coop_drive.application.cross_agent_support_enricher import CrossAgentSupportEnricher
from kg_coop_drive.application.observation_associator import ObservationAssociator
from kg_coop_drive.application.processed_scene_service import ProcessedSceneEnricher
from kg_coop_drive.application.relation_builder import RelationBuilder
from kg_coop_drive.application.track_merger import TrackMerger
from kg_coop_drive.application.track_quality_assessor import TrackQualityAssessor
from kg_coop_drive.application.track_support_enricher import TrackSupportEnricher
from kg_coop_drive.application.v2vgotqa_router import V2VGoTQARouter
from kg_coop_drive.application.visibility_reasoner import VisibilityReasoner
from kg_coop_drive.domain.benchmark import (
    BenchmarkEvaluationSummary,
    BenchmarkPrediction,
    BenchmarkSample,
)
from kg_coop_drive.domain.processed_scene import ProcessedFrameSceneData
from kg_coop_drive.domain.scene import CooperativeScene, VisibilityFact
from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader


class V2VGoTQAPhase5AEvaluator:
    """Evaluates supported Phase 5A V2V-GoT-QA tasks through the graph pipeline."""

    def __init__(
        self,
        repository_root: str,
        router: V2VGoTQARouter | None = None,
        processed_loader: V2VGoTProcessedAssetLoader | None = None,
    ) -> None:
        self._router = router or V2VGoTQARouter()
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

    def evaluate_samples(
        self,
        samples: tuple[BenchmarkSample, ...],
        baseline_mode: str = "cooperative",
    ) -> tuple[BenchmarkPrediction, ...]:
        """Evaluate one batch of benchmark samples."""

        predictions: list[BenchmarkPrediction] = []
        for sample in samples:
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
        scene = self._processed_enricher.enrich(sample.scene, processed_data)
        association_report = self._observation_associator.associate(scene, max_distance=3.0)
        scene = self._track_support_enricher.enrich(scene, association_report)
        scene = self._candidate_track_creator.promote(scene, association_report)
        scene, _candidate_resolution_report = self._candidate_track_resolver.resolve(
            scene,
            min_candidate_confidence=0.25,
        )
        cross_agent_report = self._cross_agent_associator.associate(scene, max_distance=3.0)
        scene, _cross_agent_support_report = self._cross_agent_support_enricher.enrich(
            scene,
            cross_agent_report,
        )
        scene, _track_merge_report = self._track_merger.merge(scene, max_distance=1.0)
        scene = self._track_quality_assessor.assess(scene)
        scene, _visibility_reasoning_report = self._visibility_reasoner.infer(
            scene,
            uncertain_distance=30.0,
            min_candidate_visible_confidence=0.5,
        )
        scene = self._relation_builder.build(scene)
        return scene

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
        visibility_facts = tuple(
            fact
            for fact in processed_data.visibility_facts
            if fact.agent_id == asker_agent_id
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
            object_tracks=processed_data.object_tracks,
            visibility_facts=visibility_facts,
            source_paths=source_paths,
        )
