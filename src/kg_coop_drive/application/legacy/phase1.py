from __future__ import annotations

from dataclasses import dataclass

from kg_coop_drive.domain.dataset import DatasetInspectionReport


@dataclass(frozen=True)
class Phase1Summary:
    """A concise application-level summary of Phase 1 findings."""

    report: DatasetInspectionReport

    @property
    def has_full_processed_assets(self) -> bool:
        return all(asset.exists for asset in self.report.required_assets)

    @property
    def available_split_names(self) -> tuple[str, ...]:
        return tuple(split.name for split in self.report.split_summaries)


class Phase1DatasetInspectionService:
    """Coordinates the dataset inspection use case for Phase 1."""

    def __init__(self, inspector: object) -> None:
        self._inspector = inspector

    def inspect(self) -> Phase1Summary:
        report = self._inspector.inspect()
        return Phase1Summary(report=report)
