from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSplit:
    """Describes one dataset split and the metadata currently known about it."""

    name: str
    sequence_ids: tuple[str, ...]
    cumulative_lengths: tuple[int, ...]
    llm_data_paths: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleFieldSummary:
    """Summarizes the fields available in one representative sample."""

    top_level_keys: tuple[str, ...]
    conversation_roles: tuple[str, ...]
    scenario_index: int | None
    local_timestamp_index: int | None
    global_timestamp_index: int | None
    prompt_preview: str
    response_preview: str


@dataclass(frozen=True)
class DataAssetAvailability:
    """Captures whether an expected local dataset path is currently available."""

    name: str
    path: str
    exists: bool


@dataclass(frozen=True)
class DatasetInspectionReport:
    """Aggregates the findings from the Phase 1 dataset inspection."""

    dataset_name: str
    repository_root: str
    split_summaries: tuple[DatasetSplit, ...]
    sample_summaries: tuple[SampleFieldSummary, ...]
    required_assets: tuple[DataAssetAvailability, ...]
    observations: tuple[str, ...]
