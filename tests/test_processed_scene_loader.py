from pathlib import Path

import numpy as np

from kg_coop_drive.infrastructure.v2vgot_processed_assets import (
    V2VGoTProcessedAssetLoader,
)


def test_processed_scene_loader_builds_tracks_and_visibility(tmp_path: Path) -> None:
    npy_root = (
        tmp_path
        / "DMSTrack"
        / "V2V4Real"
        / "official_models"
        / "no_fusion_keep_all"
        / "npy"
    )
    npy_root.mkdir(parents=True)

    corners = np.array(
        [
            [
                [9.0, 4.0, 1.0],
                [11.0, 4.0, 1.0],
                [11.0, 4.0, -1.0],
                [9.0, 4.0, -1.0],
                [9.0, 6.0, 1.0],
                [11.0, 6.0, 1.0],
                [11.0, 6.0, -1.0],
                [9.0, 6.0, -1.0],
            ]
        ]
    )
    np.save(npy_root / "0000_gt.npy", corners)
    np.save(npy_root / "0000_gt_object_id.npy", np.array([101]))
    np.save(npy_root / "0000_gt_object_id_visible_to_ego.npy", np.array([101]))
    np.save(npy_root / "0000_gt_object_id_invisible_to_ego.npy", np.array([]))
    np.save(npy_root / "0000_gt_object_id_visible_to_1.npy", np.array([]))
    np.save(npy_root / "0000_gt_object_id_invisible_to_1.npy", np.array([101]))
    (npy_root / "co_llm" / "ego").mkdir(parents=True)
    np.save(
        npy_root / "co_llm" / "ego" / "0000_detection_box_score.npy",
        np.array([[1.7, 2.0, 4.0, 10.0, 3.0, -8.0, 0.0, 0.9]]),
    )

    loader = V2VGoTProcessedAssetLoader(str(tmp_path))
    availability = loader.inspect_availability(timestamp_index=0, split_name="val")
    data = loader.load_frame_scene_data(timestamp_index=0, split_name="val")

    assert availability.has_gt_boxes
    assert availability.has_pred_for_ego
    assert data is not None
    assert len(data.observations) == 1
    assert len(data.object_tracks) == 1
    assert data.object_tracks[0].object_id == "101"
    assert data.object_tracks[0].position.x == 10.0
    assert data.object_tracks[0].position.y == 5.0
    assert data.observations[0].position.x == 10.0
    assert data.observations[0].position.y == 3.0
    assert len(data.visibility_facts) == 2


def test_processed_scene_loader_prefers_cooperative_root_by_default(tmp_path: Path) -> None:
    cobevt_root = tmp_path / "cobevt" / "npy"
    cobevt_root.mkdir(parents=True)
    cooperative_root = (
        tmp_path
        / "DMSTrack"
        / "V2V4Real"
        / "official_models"
        / "no_fusion_keep_all"
        / "npy"
    )
    cooperative_root.mkdir(parents=True)

    corners = np.array(
        [
            [
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )
    np.save(cobevt_root / "0008_gt.npy", corners)
    np.save(cobevt_root / "0008_gt_object_id.npy", np.array([7]))
    np.save(cooperative_root / "0008_gt.npy", corners)
    np.save(cooperative_root / "0008_gt_object_id.npy", np.array([11]))

    loader = V2VGoTProcessedAssetLoader(str(tmp_path))
    availability = loader.inspect_availability(timestamp_index=8, split_name="val")
    data = loader.load_frame_scene_data(timestamp_index=8, split_name="val")

    assert availability.npy_root.endswith("no_fusion_keep_all/npy")
    assert availability.has_gt_boxes
    assert data is not None
    assert data.object_tracks[0].object_id == "11"


def test_processed_scene_loader_supports_benchmark_profile_priority(tmp_path: Path) -> None:
    cobevt_root = tmp_path / "cobevt" / "npy"
    cobevt_root.mkdir(parents=True)
    cooperative_root = (
        tmp_path
        / "DMSTrack"
        / "V2V4Real"
        / "official_models"
        / "no_fusion_keep_all"
        / "npy"
    )
    cooperative_root.mkdir(parents=True)

    corners = np.array(
        [
            [
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )
    np.save(cobevt_root / "0008_gt.npy", corners)
    np.save(cobevt_root / "0008_gt_object_id.npy", np.array([7]))
    np.save(cooperative_root / "0008_gt.npy", corners)
    np.save(cooperative_root / "0008_gt_object_id.npy", np.array([11]))

    loader = V2VGoTProcessedAssetLoader(str(tmp_path), asset_profile="benchmark")
    availability = loader.inspect_availability(timestamp_index=8, split_name="val")
    data = loader.load_frame_scene_data(timestamp_index=8, split_name="val")

    assert availability.npy_root.endswith("cobevt/npy")
    assert availability.has_gt_boxes
    assert data is not None
    assert data.object_tracks[0].object_id == "7"


def test_processed_scene_loader_reads_per_agent_prediction_files(tmp_path: Path) -> None:
    npy_root = tmp_path / "cobevt" / "npy"
    (npy_root / "ego").mkdir(parents=True)
    np.save(npy_root / "0003_gt.npy", np.zeros((0, 8, 3)))
    np.save(npy_root / "0003_gt_object_id.npy", np.array([]))
    pred_corners = np.array(
        [
            [
                [9.0, 7.0, 1.0],
                [11.0, 7.0, 1.0],
                [11.0, 7.0, -1.0],
                [9.0, 7.0, -1.0],
                [9.0, 9.0, 1.0],
                [11.0, 9.0, 1.0],
                [11.0, 9.0, -1.0],
                [9.0, 9.0, -1.0],
            ]
        ]
    )
    np.save(npy_root / "ego" / "0003_pred.npy", pred_corners)
    np.save(npy_root / "ego" / "0003_pred_score.npy", np.array([0.75]))

    loader = V2VGoTProcessedAssetLoader(str(tmp_path))
    availability = loader.inspect_availability(timestamp_index=3, split_name="val")
    data = loader.load_frame_scene_data(timestamp_index=3, split_name="val")

    assert availability.has_pred_for_ego
    assert data is not None
    assert len(data.observations) == 1
    assert data.observations[0].source_agent_id == "CAV_EGO"
    assert data.observations[0].confidence == 0.75
    assert data.observations[0].position.x == 10.0
    assert data.observations[0].position.y == 8.0
