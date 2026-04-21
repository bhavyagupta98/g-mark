from kg_coop_drive.application.phase1_completion import Phase1CompletionService
from kg_coop_drive.infrastructure.v2vgot_collm import V2VGoTCoLLMInspector


def test_phase1_completion_service_reads_real_collm_files() -> None:
    service = Phase1CompletionService(
        V2VGoTCoLLMInspector("/Users/bhavya/Desktop/ms_projects/V2V-GoT")
    )
    summary = service.complete()

    assert summary.report.file_summaries
    assert summary.report.recommended_task_slice
