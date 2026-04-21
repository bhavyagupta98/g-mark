from pathlib import Path

import numpy as np

from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader


def test_processed_asset_loader_lists_timestamps_from_root_and_agent_dirs(tmp_path: Path) -> None:
    npy_root = tmp_path / "cobevt" / "npy"
    ego_dir = npy_root / "ego"
    cav1_dir = npy_root / "1"
    ego_dir.mkdir(parents=True)
    cav1_dir.mkdir(parents=True)

    np.save(npy_root / "0000_gt.npy", np.zeros((1, 8, 3)))
    np.save(npy_root / "0001_gt_object_id.npy", np.array([1]))
    np.save(ego_dir / "0002_pred.npy", np.zeros((1, 8, 3)))
    np.save(cav1_dir / "0003_pred.npy", np.zeros((1, 8, 3)))

    loader = V2VGoTProcessedAssetLoader(str(tmp_path))
    timestamps = loader.list_available_timestamps()
    root_specific_timestamps = loader.list_available_timestamps_for_root(npy_root)

    assert timestamps == (0, 1, 2, 3)
    assert root_specific_timestamps == (0, 1, 2, 3)
