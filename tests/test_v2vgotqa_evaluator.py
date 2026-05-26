from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.application.qa.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator
from kg_coop_drive.application.qa.v2vgotqa_router import V2VGoTQARouter
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType
from kg_coop_drive.domain.processed_scene import ProcessedFrameSceneData
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObservationEvidence,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    RelationFact,
    RelationType,
    Trajectory,
    TrackStatus,
    VisibilityFact,
    VisibilityState,
)


class _StubProcessedLoader:
    def __init__(self, processed_data: ProcessedFrameSceneData) -> None:
        self._processed_data = processed_data

    def load_frame_scene_data(
        self,
        timestamp_index: int,
        split_name: str = "val",
    ) -> ProcessedFrameSceneData | None:
        return self._processed_data


class _TimestampedStubProcessedLoader:
    def __init__(self, processed_data_by_timestamp: dict[int, ProcessedFrameSceneData]) -> None:
        self._processed_data_by_timestamp = processed_data_by_timestamp

    def load_frame_scene_data(
        self,
        timestamp_index: int,
        split_name: str = "val",
    ) -> ProcessedFrameSceneData | None:
        del split_name
        return self._processed_data_by_timestamp.get(timestamp_index)


def _seed_scene() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0))
        ),
    )


def _processed_data() -> ProcessedFrameSceneData:
    return ProcessedFrameSceneData(
        timestamp_index=0,
        observations=(
            ObservationEvidence(
                observation_id=".../ego/0000_pred.npy::obs_ego_0",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=0.9,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id=".../1/0000_pred.npy::obs_1_0",
                source_agent_id="CAV_1",
                object_type="car",
                position=Point2D(12.0, -0.5),
                confidence=0.8,
                timestamp_index=0,
            ),
        ),
        object_tracks=(
            ObjectTrack(
                object_id="gt-1",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt-1",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="gt-1",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_1",
                object_id="gt-1",
                state=VisibilityState.OCCLUDED,
            ),
        ),
        source_paths=(
            "/tmp/0000_gt.npy",
            "/tmp/0000_gt_object_id.npy",
            "/tmp/ego/0000_pred.npy",
            "/tmp/1/0000_pred.npy",
        ),
    )


def _processed_data_with_partner_only_track() -> ProcessedFrameSceneData:
    return ProcessedFrameSceneData(
        timestamp_index=0,
        observations=(
            ObservationEvidence(
                observation_id=".../ego/0000_pred.npy::obs_ego_0",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=0.9,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id=".../1/0000_pred.npy::obs_1_0",
                source_agent_id="CAV_1",
                object_type="car",
                position=Point2D(17.0, -1.0),
                confidence=0.8,
                timestamp_index=0,
            ),
        ),
        object_tracks=(
            ObjectTrack(
                object_id="ego-visible",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt-visible",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="partner-only",
                object_type="car",
                position=Point2D(17.0, -1.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt-partner-only",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="ego-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="partner-only",
                state=VisibilityState.OCCLUDED,
            ),
            VisibilityFact(
                agent_id="CAV_1",
                object_id="partner-only",
                state=VisibilityState.VISIBLE,
            ),
        ),
    )


def _processed_data_with_unmatched_observation() -> ProcessedFrameSceneData:
    return ProcessedFrameSceneData(
        timestamp_index=0,
        observations=(
            ObservationEvidence(
                observation_id=".../ego/0000_pred.npy::obs_ego_0",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=0.9,
                timestamp_index=0,
            ),
            ObservationEvidence(
                observation_id=".../1/0000_pred.npy::unmatched_partner_0",
                source_agent_id="CAV_1",
                object_type="car",
                position=Point2D(40.0, 5.0),
                confidence=0.8,
                timestamp_index=0,
            ),
        ),
        object_tracks=(
            ObjectTrack(
                object_id="ego-visible",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt-visible",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="ego-visible",
                state=VisibilityState.VISIBLE,
            ),
        ),
    )


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="sample-1",
        dataset_name="V2V-GoT-QA",
        split_name="val",
        file_name="demo.json",
        task_type=BenchmarkTaskType.NOTABLE_OBJECTS,
        scene=_seed_scene(),
        raw_record={},
        qa_type_id=11,
    )


def _sample_at_timestamp(
    task_type: BenchmarkTaskType,
    timestamp_index: int,
    qa_type_id: int,
) -> BenchmarkSample:
    scene = _seed_scene()
    return BenchmarkSample(
        sample_id=f"sample-{task_type.value}-{timestamp_index}",
        dataset_name="V2V-GoT-QA",
        split_name="val",
        file_name="demo.json",
        task_type=task_type,
        scene=replace(
            scene,
            local_timestamp_index=timestamp_index,
            global_timestamp_index=timestamp_index,
        ),
        raw_record={},
        qa_type_id=qa_type_id,
    )


def _motion_processed_data(timestamp_index: int, x_position: float) -> ProcessedFrameSceneData:
    return ProcessedFrameSceneData(
        timestamp_index=timestamp_index,
        observations=(),
        object_tracks=(
            ObjectTrack(
                object_id="moving-car",
                object_type="car",
                position=Point2D(x_position, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=(f"gt-moving-car-{timestamp_index}",),
                    latest_timestamp_index=timestamp_index,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="moving-car",
                state=VisibilityState.VISIBLE,
            ),
        ),
    )


def test_v2vgotqa_evaluator_supports_cooperative_predictions() -> None:
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        router=V2VGoTQARouter(),
        processed_loader=_StubProcessedLoader(_processed_data()),
    )

    predictions = evaluator.evaluate_samples((_sample(),), baseline_mode="cooperative")

    assert len(predictions) == 1
    assert predictions[0].supported is True
    assert predictions[0].baseline_mode == "cooperative"


def test_v2vgotqa_evaluator_filters_to_ego_only_mode() -> None:
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        router=V2VGoTQARouter(),
        processed_loader=_StubProcessedLoader(_processed_data()),
    )

    cooperative = evaluator.evaluate_samples((_sample(),), baseline_mode="cooperative")
    ego_only = evaluator.evaluate_samples((_sample(),), baseline_mode="ego_only")

    assert cooperative[0].supported is True
    assert ego_only[0].supported is True
    assert ego_only[0].baseline_mode == "ego_only"


def test_v2vgotqa_evaluator_ego_only_removes_partner_only_tracks() -> None:
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        router=V2VGoTQARouter(),
        processed_loader=_StubProcessedLoader(_processed_data_with_partner_only_track()),
    )

    cooperative_scene = evaluator.prepare_sample(_sample(), baseline_mode="cooperative")
    ego_only_scene = evaluator.prepare_sample(_sample(), baseline_mode="ego_only")

    assert {track.object_id for track in cooperative_scene.object_tracks} >= {
        "ego-visible",
        "partner-only",
    }
    assert "partner-only" not in {track.object_id for track in ego_only_scene.object_tracks}
    assert {track.object_id for track in ego_only_scene.object_tracks} == {"ego-visible"}
    assert all(fact.agent_id == "CAV_EGO" for fact in ego_only_scene.visibility_facts)
    assert all(fact.state == VisibilityState.VISIBLE for fact in ego_only_scene.visibility_facts)


def test_graph_ablation_removes_provenance_and_provenance_relations() -> None:
    scene = _seed_scene()
    track = ObjectTrack(
        object_id="supported",
        object_type="car",
        position=Point2D(10.0, 0.0),
        confidence=1.0,
        provenance=ProvenanceRecord(
            source_agent_ids=("CAV_EGO", "CAV_1"),
            observation_ids=("obs-1", "obs-2"),
            latest_timestamp_index=0,
        ),
    )
    scene = replace(
        scene,
        object_tracks=(track,),
        relations=(
            RelationFact(
                subject_id="supported",
                relation_type=RelationType.COOPERATIVELY_SUPPORTED,
                object_id="CAV_EGO",
                confidence=1.0,
            ),
            RelationFact(
                subject_id="supported",
                relation_type=RelationType.PATH_RELEVANT,
                object_id="CAV_EGO",
                confidence=1.0,
            ),
        ),
    )
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        graph_ablation="no_provenance",
    )

    ablated = evaluator._apply_graph_ablation(scene)

    assert ablated.object_tracks[0].provenance.source_agent_ids == ()
    assert ablated.object_tracks[0].provenance.observation_ids == ()
    assert [relation.relation_type for relation in ablated.relations] == [RelationType.PATH_RELEVANT]


def test_graph_ablation_removes_conflict_relation_trace() -> None:
    scene = _seed_scene()
    track = ObjectTrack(
        object_id="low-conflict",
        object_type="car",
        position=Point2D(10.0, 0.0),
        confidence=1.0,
        provenance=ProvenanceRecord(
            source_agent_ids=("CAV_EGO",),
            observation_ids=("obs-1",),
            latest_timestamp_index=0,
        ),
        uncertainty_score=0.8,
        conflict_score=0.4,
    )
    scene = replace(
        scene,
        object_tracks=(track,),
        relations=(
            RelationFact(
                subject_id="low-conflict",
                relation_type=RelationType.LOW_CONFLICT,
                object_id="CAV_EGO",
                confidence=0.6,
            ),
            RelationFact(
                subject_id="low-conflict",
                relation_type=RelationType.PATH_RELEVANT,
                object_id="CAV_EGO",
                confidence=1.0,
            ),
        ),
    )
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        graph_ablation="no_uncertainty_conflict",
    )

    ablated = evaluator._apply_graph_ablation(scene)

    assert ablated.object_tracks[0].uncertainty_score == 0.0
    assert ablated.object_tracks[0].conflict_score == 0.0
    assert [relation.relation_type for relation in ablated.relations] == [RelationType.PATH_RELEVANT]


def test_no_candidate_retention_skips_candidate_construction() -> None:
    full_evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        processed_loader=_StubProcessedLoader(_processed_data_with_unmatched_observation()),
    )
    no_candidate_evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        processed_loader=_StubProcessedLoader(_processed_data_with_unmatched_observation()),
        graph_ablation="no_candidate_retention",
    )

    full_scene = full_evaluator.prepare_sample(_sample(), baseline_mode="cooperative")
    no_candidate_scene = no_candidate_evaluator.prepare_sample(_sample(), baseline_mode="cooperative")

    assert any(track.status == TrackStatus.CANDIDATE for track in full_scene.object_tracks)
    assert all(track.status != TrackStatus.CANDIDATE for track in no_candidate_scene.object_tracks)
    assert all(
        not track.object_id.startswith("pred_candidate_")
        for track in no_candidate_scene.object_tracks
    )
    retained_track_ids = {track.object_id for track in no_candidate_scene.object_tracks}
    assert all(
        relation.subject_id in retained_track_ids and relation.object_id in retained_track_ids
        for relation in no_candidate_scene.relations
    )


def test_no_graph_relations_skips_relation_construction() -> None:
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        processed_loader=_StubProcessedLoader(_processed_data()),
        graph_ablation="no_graph_relations",
    )

    scene = evaluator.prepare_sample(_sample(), baseline_mode="cooperative")

    assert scene.object_tracks
    assert scene.relations == ()


def test_v2vgotqa_evaluator_temporally_enriches_motion_predictions() -> None:
    evaluator = V2VGoTQAPhase5AEvaluator(
        repository_root="/tmp",
        router=V2VGoTQARouter(),
        processed_loader=_TimestampedStubProcessedLoader(
            {
                0: _motion_processed_data(timestamp_index=0, x_position=10.0),
                1: _motion_processed_data(timestamp_index=1, x_position=12.0),
            }
        ),
    )

    predictions = evaluator.evaluate_samples(
        (
            _sample_at_timestamp(
                BenchmarkTaskType.OBJECT_MOTION_PREDICTION,
                timestamp_index=1,
                qa_type_id=15,
            ),
        ),
        baseline_mode="cooperative",
    )

    assert predictions[0].supported is True
    assert predictions[0].object_ids == ("moving-car",)
    assert "moving-car=moving forward from (12.0, 0.0) to (14.0, 0.0)" in predictions[0].answer_text
