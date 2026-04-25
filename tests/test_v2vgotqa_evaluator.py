from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.application.v2vgotqa_evaluator import V2VGoTQAPhase5AEvaluator
from kg_coop_drive.application.v2vgotqa_router import V2VGoTQARouter
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
    Trajectory,
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
