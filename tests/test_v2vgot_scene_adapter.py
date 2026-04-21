from kg_coop_drive.infrastructure.v2vgot_scene_adapter import V2VGoTSceneAdapter


def test_v2vgot_scene_adapter_builds_scene_from_record() -> None:
    adapter = V2VGoTSceneAdapter("/tmp")
    record = {
        "scenario_index": 42,
        "local_timestamp_index": 7,
        "global_timestamp_index": 70,
        "asker_cav_id": "CAV_EGO",
        "future_trajectory_str_in_ego": "[(1.0, 2.0), (3.0, 4.0)]",
        "cav_ego_lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cav_1_lidar_pose": [5.0, 1.0, 0.0, 0.0, 0.2, 0.0],
        "conversations": [
            {"from": "human", "value": "What is visible?"},
            {"from": "gpt", "value": "A car is visible."},
        ],
    }

    scene = adapter.build_scene(record)

    assert scene.scene_id == "42"
    assert scene.asker_agent_id == "CAV_EGO"
    assert len(scene.agents) == 2
    assert scene.agents[1].pose.position.x == 5.0
    assert scene.future_trajectory.points[1].y == 4.0
    assert scene.raw_question == "What is visible?"
    assert scene.raw_answer == "A car is visible."


def test_v2vgot_scene_adapter_normalizes_asker_agent_id() -> None:
    adapter = V2VGoTSceneAdapter("/tmp")
    record = {
        "scenario_index": 1,
        "local_timestamp_index": 0,
        "global_timestamp_index": 0,
        "asker_cav_id": "ego",
        "future_trajectory_str_in_ego": "[(0.0, 0.0)]",
        "cav_ego_lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cav_1_lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "conversations": [],
    }

    scene = adapter.build_scene(record)

    assert scene.asker_agent_id == "CAV_EGO"


def test_v2vgot_scene_adapter_load_records_reads_json_file(tmp_path) -> None:
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
        '"asker_cav_id": "ego", "future_trajectory_str_in_ego": "[(0.0, 0.0)]", '
        '"cav_ego_lidar_pose": [0, 0, 0, 0, 0, 0], "cav_1_lidar_pose": [0, 0, 0, 0, 0, 0], '
        '"conversations": []}]',
        encoding="utf-8",
    )

    records = V2VGoTSceneAdapter(str(tmp_path)).load_records()

    assert len(records) == 1
    assert records[0]["scenario_index"] == 1
