from __future__ import annotations

"""Published benchmark reference numbers used in Phase 5 summaries.

These are paper-reported reference values, not locally reproduced results.

Primary source:
- Hsu-kuang Chiu et al., "V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous
  Driving with Multimodal Large Language Models and Graph-of-Thoughts",
  arXiv:2509.18053
  PDF: https://arxiv.org/pdf/2509.18053

Numbers below are taken from:
- Table I (planning task on V2V-GoT-QA)
- Table II (question-answering tasks on V2V-GoT-QA)
"""

from dataclasses import dataclass

from kg_coop_drive.domain.benchmark import BenchmarkTaskType


@dataclass(frozen=True)
class PaperMetricReference:
    """One published reference metric from a paper table."""

    source_paper: str
    source_table: str
    dataset_name: str
    task_type: BenchmarkTaskType
    method_name: str
    metric_name: str
    metric_value: float
    higher_is_better: bool
    notes: str = ""


V2VGOT_PAPER_REFERENCES: tuple[PaperMetricReference, ...] = (
    # Table I: planning task.
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table I",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.FUTURE_TRAJECTORY,
        method_name="V2V-LLM",
        metric_name="Q9 L2 (m)",
        metric_value=4.93,
        higher_is_better=False,
        notes="Final suggested trajectory error reported in Table I.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table I",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.FUTURE_TRAJECTORY,
        method_name="V2V-LLM",
        metric_name="Avg CR (%)",
        metric_value=2.85,
        higher_is_better=False,
        notes="Average collision rate reported in Table I.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table I",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.FUTURE_TRAJECTORY,
        method_name="V2V-GoT",
        metric_name="Q9 L2 (m)",
        metric_value=2.62,
        higher_is_better=False,
        notes="Final suggested trajectory error reported in Table I.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table I",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.FUTURE_TRAJECTORY,
        method_name="V2V-GoT",
        metric_name="Avg CR (%)",
        metric_value=1.83,
        higher_is_better=False,
        notes="Average collision rate reported in Table I.",
    ),
    # Table II: QA tasks directly relevant to Phase 5A.
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.VISIBLE_OBJECTS,
        method_name="V2V-GoT",
        metric_name="Q1 F1",
        metric_value=52.5,
        higher_is_better=True,
        notes="Visible notable objects F1.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.OCCLUDING_OBJECTS,
        method_name="V2V-GoT",
        metric_name="Q2 F1",
        metric_value=30.1,
        higher_is_better=True,
        notes="Occluding objects F1.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.INVISIBLE_OBJECTS,
        method_name="V2V-GoT",
        metric_name="Q3 F1",
        metric_value=44.0,
        higher_is_better=True,
        notes="Invisible notable objects F1.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.PLANNING_AWARENESS,
        method_name="V2V-GoT",
        metric_name="Q4 F1",
        metric_value=60.8,
        higher_is_better=True,
        notes="Overall notable objects F1. This is the closest published QA metric to our current planning-awareness workstream.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.OBJECT_MOTION_PREDICTION,
        method_name="V2V-GoT",
        metric_name="Q5 L2 (m)",
        metric_value=8.05,
        higher_is_better=False,
        notes="Prediction-by-perception error.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.AGENT_MOTION_PREDICTION,
        method_name="V2V-GoT",
        metric_name="Q6 Accuracy",
        metric_value=87.4,
        higher_is_better=True,
        notes="Prediction-by-planning accuracy.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.OBJECT_MOTION_PREDICTION,
        method_name="V2V-GoT",
        metric_name="Q7 L2 (m)",
        metric_value=7.61,
        higher_is_better=False,
        notes="Overall prediction error.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.CONTROL_SETTINGS,
        method_name="V2V-GoT",
        metric_name="Q8 L1",
        metric_value=0.0876,
        higher_is_better=False,
        notes="Suggested action classification/regression error.",
    ),
    PaperMetricReference(
        source_paper="V2V-GoT",
        source_table="Table II",
        dataset_name="V2V-GoT-QA",
        task_type=BenchmarkTaskType.FUTURE_TRAJECTORY,
        method_name="V2V-GoT",
        metric_name="Q9 L2 (m)",
        metric_value=2.62,
        higher_is_better=False,
        notes="Suggested trajectory error.",
    ),
)


def references_for_task(task_type: BenchmarkTaskType) -> tuple[PaperMetricReference, ...]:
    """Return the published references relevant to one benchmark task."""

    alias_map = {
        BenchmarkTaskType.NOTABLE_OBJECTS: BenchmarkTaskType.VISIBLE_OBJECTS,
    }
    resolved_task_type = alias_map.get(task_type, task_type)
    return tuple(
        reference
        for reference in V2VGOT_PAPER_REFERENCES
        if reference.task_type == resolved_task_type
    )
