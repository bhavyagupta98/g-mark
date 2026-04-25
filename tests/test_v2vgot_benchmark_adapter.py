from __future__ import annotations

from kg_coop_drive.domain.benchmark import BenchmarkTaskType
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter


def _record(question: str, qa_type_id: int | None = None) -> dict[str, object]:
    record = {
        "scenario_index": 1,
        "local_timestamp_index": 0,
        "global_timestamp_index": 0,
        "asker_cav_id": "ego",
        "future_trajectory_str_in_ego": "[(0.0, 0.0)]",
        "cav_ego_lidar_pose": [0, 0, 0, 0, 0, 0],
        "cav_1_lidar_pose": [0, 0, 0, 0, 0, 0],
        "conversations": [{"from": "human", "value": question}],
    }
    if qa_type_id is not None:
        record["qa_type_id"] = qa_type_id
    return record


def test_v2vgot_benchmark_adapter_classifies_by_qa_type_id() -> None:
    adapter = V2VGoTQABenchmarkAdapter("/tmp")

    assert adapter.classify_record(_record("ignored", qa_type_id=11)) == BenchmarkTaskType.NOTABLE_OBJECTS
    assert adapter.classify_record(_record("ignored", qa_type_id=12)) == BenchmarkTaskType.OCCLUDING_OBJECTS
    assert adapter.classify_record(_record("ignored", qa_type_id=13)) == BenchmarkTaskType.INVISIBLE_OBJECTS
    assert adapter.classify_record(_record("ignored", qa_type_id=14)) == BenchmarkTaskType.PLANNING_AWARENESS
    assert adapter.classify_record(_record("ignored", qa_type_id=18)) == BenchmarkTaskType.CONTROL_SETTINGS
    assert adapter.classify_record(_record("ignored", qa_type_id=19)) == BenchmarkTaskType.FUTURE_TRAJECTORY


def test_v2vgot_benchmark_adapter_classifies_from_question_text() -> None:
    adapter = V2VGoTQABenchmarkAdapter("/tmp")

    assert adapter.classify_record(_record("What objects are visible to me?")) == BenchmarkTaskType.VISIBLE_OBJECTS
    assert adapter.classify_record(_record("Which objects are occluding my view?")) == BenchmarkTaskType.OCCLUDING_OBJECTS
    assert adapter.classify_record(_record("What objects might obstruct my view?")) == BenchmarkTaskType.OCCLUDING_OBJECTS
    assert adapter.classify_record(_record("What invisible objects are near me?")) == BenchmarkTaskType.INVISIBLE_OBJECTS
    assert adapter.classify_record(_record("What notable objects are near my planned future trajectory?")) == BenchmarkTaskType.NOTABLE_OBJECTS
    assert adapter.classify_record(_record("Is there anything I need to be aware of if my planned future trajectory is [...]?")) == BenchmarkTaskType.PLANNING_AWARENESS
    assert adapter.classify_record(_record("Where might those notable objects move in the future?")) == BenchmarkTaskType.OBJECT_MOTION_PREDICTION
    assert adapter.classify_record(_record("Where might other CAVs move in the future given their planned future trajectories?")) == BenchmarkTaskType.AGENT_MOTION_PREDICTION
    assert adapter.classify_record(_record("What are the suggested speed and steering settings to avoid collision with nearby objects?")) == BenchmarkTaskType.CONTROL_SETTINGS
    assert adapter.classify_record(_record("What is my future trajectory?")) == BenchmarkTaskType.FUTURE_TRAJECTORY


def test_v2vgot_benchmark_adapter_load_samples_builds_task_labeled_samples(tmp_path) -> None:
    root = (
        tmp_path
        / "DMSTrack"
        / "V2V4Real"
        / "official_models"
        / "no_fusion_keep_all"
        / "npy"
        / "co_llm"
    )
    root.mkdir(parents=True)
    dataset_path = root / "v2v4real_3d_grounding_qa_dataset_v2vgot.json"
    dataset_path.write_text(
        '[{"scenario_index": 1, "global_timestamp_index": 0, "local_timestamp_index": 0, '
        '"asker_cav_id": "ego", "qa_type_id": 11, '
        '"future_trajectory_str_in_ego": "[(0.0, 0.0)]", '
        '"cav_ego_lidar_pose": [0, 0, 0, 0, 0, 0], "cav_1_lidar_pose": [0, 0, 0, 0, 0, 0], '
        '"conversations": [{"from": "human", "value": "What notable objects are visible?"}]}]',
        encoding="utf-8",
    )

    samples = V2VGoTQABenchmarkAdapter(str(tmp_path)).load_samples()

    assert len(samples) == 1
    assert samples[0].dataset_name == "V2V-GoT-QA"
    assert samples[0].task_type == BenchmarkTaskType.NOTABLE_OBJECTS
    assert samples[0].scene.asker_agent_id == "CAV_EGO"


def test_v2vgot_benchmark_adapter_load_samples_attaches_adjacent_agent_velocity(tmp_path) -> None:
    root = (
        tmp_path
        / "DMSTrack"
        / "V2V4Real"
        / "official_models"
        / "no_fusion_keep_all"
        / "npy"
        / "co_llm"
    )
    root.mkdir(parents=True)
    dataset_path = root / "v2v4real_3d_grounding_qa_dataset_v2vgot.json"
    dataset_path.write_text(
        "["
        '{"id": "prev", "scenario_index": 1, "global_timestamp_index": 0, "local_timestamp_index": 0, '
        '"asker_cav_id": "ego", "qa_type_id": 16, '
        '"future_trajectory_str_in_ego": "[(0.0, 0.0)]", '
        '"cav_ego_lidar_pose": [0, 0, 0, 0, 0, 0], "cav_1_lidar_pose": [5, 1, 0, 0, 0, 0], '
        '"conversations": [{"from": "human", "value": "Where might other CAVs move?"}]},'
        '{"id": "curr", "scenario_index": 1, "global_timestamp_index": 1, "local_timestamp_index": 1, '
        '"asker_cav_id": "ego", "qa_type_id": 16, '
        '"future_trajectory_str_in_ego": "[(0.0, 0.0)]", '
        '"cav_ego_lidar_pose": [1, 0, 0, 0, 0, 0], "cav_1_lidar_pose": [5, 3, 0, 0, 0, 0], '
        '"conversations": [{"from": "human", "value": "Where might other CAVs move?"}]}'
        "]",
        encoding="utf-8",
    )

    samples = V2VGoTQABenchmarkAdapter(str(tmp_path)).load_samples()

    assert samples[0].scene.agents[1].velocity is None
    cav1_velocity = samples[1].scene.agents[1].velocity
    assert cav1_velocity is not None
    assert cav1_velocity.x == 0.0
    assert cav1_velocity.y == 2.0
