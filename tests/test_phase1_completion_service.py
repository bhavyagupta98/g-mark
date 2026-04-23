import os
from pathlib import Path

import pytest

from kg_coop_drive.application.phase1_completion import Phase1CompletionService
from kg_coop_drive.infrastructure.v2vgot_collm import V2VGoTCoLLMInspector


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


def test_phase1_completion_service_reads_real_collm_files() -> None:
    repository_root = _resolve_v2vgot_root()
    service = Phase1CompletionService(
        V2VGoTCoLLMInspector(str(repository_root))
    )
    summary = service.complete()

    if not summary.report.file_summaries:
        pytest.skip("co_llm benchmark files are not available in this environment.")

    assert summary.report.file_summaries
    assert summary.report.recommended_task_slice
