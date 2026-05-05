from __future__ import annotations

from dataclasses import dataclass

from kg_coop_drive.domain.phase1_completion import Phase1CompletionReport


@dataclass(frozen=True)
class Phase1CompletionSummary:
    """High-level result of the Phase 1 closeout inspection."""

    report: Phase1CompletionReport

    @property
    def is_complete(self) -> bool:
        return bool(self.report.file_summaries) and bool(self.report.recommended_task_slice)


class Phase1CompletionService:
    """Coordinates the final Phase 1 dataset understanding step."""

    def __init__(self, inspector: object) -> None:
        self._inspector = inspector

    def complete(self) -> Phase1CompletionSummary:
        return Phase1CompletionSummary(report=self._inspector.inspect())
