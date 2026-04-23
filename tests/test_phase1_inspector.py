import os
from pathlib import Path

import pytest

from kg_coop_drive.application.phase1 import Phase1DatasetInspectionService
from kg_coop_drive.infrastructure.v2vgot_dataset import V2VGoTDatasetInspector


def _resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    candidates = [
        Path(env_value).expanduser().resolve() if env_value else None,
        Path("/workspace/repos/V2V-GoT"),
        Path("/Users/bhavya/Desktop/ms_projects/V2V-GoT"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    pytest.skip("V2V-GoT repository is not available in this environment.")


def test_v2vgot_dataset_inspector_reports_expected_splits() -> None:
    repository_root = _resolve_v2vgot_root()
    expected_metadata = repository_root / "LLaVA" / "playground" / "data" / "V2V4Real" / "data.json"
    if not expected_metadata.exists():
        pytest.skip("Phase 1 LLaVA metadata files are not available in this environment.")

    inspector = V2VGoTDatasetInspector(str(repository_root))
    service = Phase1DatasetInspectionService(inspector)

    summary = service.inspect()

    assert "train" in summary.available_split_names
    assert "val" in summary.available_split_names
    assert summary.report.sample_summaries
