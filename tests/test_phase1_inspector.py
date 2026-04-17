from kg_coop_drive.application.phase1 import Phase1DatasetInspectionService
from kg_coop_drive.infrastructure.v2vgot_dataset import V2VGoTDatasetInspector


def test_v2vgot_dataset_inspector_reports_expected_splits() -> None:
    inspector = V2VGoTDatasetInspector("/Users/bhavya/Desktop/ms_projects/V2V-GoT")
    service = Phase1DatasetInspectionService(inspector)

    summary = service.inspect()

    assert "train" in summary.available_split_names
    assert "val" in summary.available_split_names
    assert summary.report.sample_summaries
