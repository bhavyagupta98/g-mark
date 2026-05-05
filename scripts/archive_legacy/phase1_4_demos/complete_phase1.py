#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.phase1_completion import Phase1CompletionService
from kg_coop_drive.infrastructure.v2vgot_collm import V2VGoTCoLLMInspector

DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def resolve_v2vgot_root() -> Path:
    """Resolve the V2V-GoT root in either pod or local-dev environments."""

    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_V2VGOT_ROOTS[0]


def format_report() -> str:
    v2vgot_root = resolve_v2vgot_root()
    service = Phase1CompletionService(V2VGoTCoLLMInspector(str(v2vgot_root)))
    summary = service.complete()
    report = summary.report

    lines: list[str] = []
    lines.append("# Phase 1 Completion Summary")
    lines.append("")
    lines.append(f"- Repository root inspected: `{report.repository_root}`")
    lines.append(f"- co_llm root family: `{report.collm_root}`")
    lines.append(f"- Available splits: {', '.join(report.available_splits)}")
    lines.append("")
    lines.append("## Real File Inspection")
    lines.append("")
    for file_summary in report.file_summaries:
        lines.append(f"- `{file_summary.split_name}` / `{file_summary.file_name}`")
        lines.append(f"  records: {file_summary.record_count}")
        lines.append(f"  keys: {', '.join(file_summary.top_level_keys)}")
        lines.append(f"  conversation roles: {', '.join(file_summary.conversation_roles)}")
        if file_summary.qa_type_id is not None:
            lines.append(f"  qa_type_id: {file_summary.qa_type_id}")
        if file_summary.qa_source is not None:
            lines.append(f"  qa_source: {file_summary.qa_source}")
        lines.append(f"  question preview: {file_summary.question_preview}")
        lines.append(f"  answer preview: {file_summary.answer_preview}")
        lines.append("")
    lines.append("## Recommended First Task Slice")
    lines.append("")
    for task in report.recommended_task_slice:
        lines.append(f"- {task}")
    lines.append("")
    lines.append("## Bootstrap Artifacts")
    lines.append("")
    for artifact in report.bootstrap_artifacts:
        lines.append(f"- `{artifact.name}`")
        lines.append(f"  kind: {artifact.kind}")
        lines.append(f"  purpose: {artifact.purpose}")
        lines.append(f"  expected location: `{artifact.expected_location}`")
        lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    for conclusion in report.conclusions:
        lines.append(f"- {conclusion}")
    lines.append("")
    lines.append("## Phase 1 Status")
    lines.append("")
    if summary.is_complete:
        lines.append("Phase 1 can be considered complete after reviewing and accepting the recommended first task slice above.")
    else:
        lines.append("Phase 1 is not yet complete because the real co_llm inspection did not yield the expected summaries.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report_text = format_report()
    print(report_text)

    output_path = REPO_ROOT / "docs" / "phase1_completion_summary.md"
    output_path.write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
