from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class SampleLatencyRecord:
    """Per-sample latency breakdown captured during one evaluation pass."""

    sample_id: str
    split_name: str
    task_type: str
    qa_type_id: int | None
    baseline_mode: str
    timings_ms: dict[str, float]


class SampleLatencyRecorder:
    """Accumulates stage timings for one sample."""

    def __init__(
        self,
        *,
        sample_id: str,
        split_name: str,
        task_type: str,
        qa_type_id: int | None,
        baseline_mode: str,
    ) -> None:
        self._sample_id = sample_id
        self._split_name = split_name
        self._task_type = task_type
        self._qa_type_id = qa_type_id
        self._baseline_mode = baseline_mode
        self._timings_ms: dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str):
        started = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started) * 1000.0
            self._timings_ms[stage_name] = self._timings_ms.get(stage_name, 0.0) + elapsed_ms

    def snapshot(self) -> SampleLatencyRecord:
        return SampleLatencyRecord(
            sample_id=self._sample_id,
            split_name=self._split_name,
            task_type=self._task_type,
            qa_type_id=self._qa_type_id,
            baseline_mode=self._baseline_mode,
            timings_ms=dict(self._timings_ms),
        )


class EvaluationLatencyCollector:
    """Owns sample recorders and stores completed latency snapshots."""

    def __init__(self) -> None:
        self._records: list[SampleLatencyRecord] = []

    def start_sample(
        self,
        *,
        sample_id: str,
        split_name: str,
        task_type: str,
        qa_type_id: int | None,
        baseline_mode: str,
    ) -> SampleLatencyRecorder:
        return SampleLatencyRecorder(
            sample_id=sample_id,
            split_name=split_name,
            task_type=task_type,
            qa_type_id=qa_type_id,
            baseline_mode=baseline_mode,
        )

    def finish_sample(self, recorder: SampleLatencyRecorder) -> None:
        self._records.append(recorder.snapshot())

    def records(self) -> tuple[SampleLatencyRecord, ...]:
        return tuple(self._records)
