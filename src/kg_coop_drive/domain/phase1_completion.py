from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CoLLMFileSummary:
    """Summarizes one real co_llm dataset file."""

    split_name: str
    file_name: str
    record_count: int
    top_level_keys: tuple[str, ...]
    conversation_roles: tuple[str, ...]
    question_preview: str
    answer_preview: str
    qa_type_id: int | None
    qa_source: str | None


@dataclass(frozen=True)
class BootstrapArtifactSummary:
    """Explains one bootstrap artifact and its expected role."""

    name: str
    kind: str
    purpose: str
    expected_location: str


@dataclass(frozen=True)
class Phase1CompletionReport:
    """Aggregates the final Phase 1 closeout findings."""

    repository_root: str
    collm_root: str
    available_splits: tuple[str, ...]
    file_summaries: tuple[CoLLMFileSummary, ...]
    recommended_task_slice: tuple[str, ...]
    bootstrap_artifacts: tuple[BootstrapArtifactSummary, ...]
    conclusions: tuple[str, ...] = field(default_factory=tuple)
