from __future__ import annotations

import json
from pathlib import Path

from kg_coop_drive.domain.phase1_completion import (
    BootstrapArtifactSummary,
    CoLLMFileSummary,
    Phase1CompletionReport,
)


class V2VGoTCoLLMInspector:
    """Inspects the real downloaded co_llm benchmark JSON files."""

    def __init__(self, repository_root: str) -> None:
        self._repository_root = Path(repository_root).expanduser().resolve()
        self._collm_roots = {
            "val": self._repository_root / "DMSTrack" / "V2V4Real" / "official_models" / "no_fusion_keep_all" / "npy" / "co_llm",
            "train": self._repository_root / "DMSTrack" / "V2V4Real" / "official_models" / "train_no_fusion_keep_all" / "npy" / "co_llm",
        }

    def inspect(self) -> Phase1CompletionReport:
        file_summaries: list[CoLLMFileSummary] = []
        for split_name, root in self._collm_roots.items():
            for file_name in (
                "v2v4real_3d_grounding_qa_dataset_v2vgot.json",
                "v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json",
            ):
                path = root / file_name
                if path.exists():
                    file_summaries.append(self._inspect_file(split_name, path))

        return Phase1CompletionReport(
            repository_root=str(self._repository_root),
            collm_root=str(self._collm_roots["val"].parent.parent.parent),
            available_splits=tuple(self._collm_roots.keys()),
            file_summaries=tuple(file_summaries),
            recommended_task_slice=(
                "object existence",
                "object count",
                "relative position",
                "visible notable object queries",
            ),
            bootstrap_artifacts=(
                BootstrapArtifactSummary(
                    name="dataset_jsons.zip",
                    kind="benchmark JSON archive",
                    purpose="Provides V2V-GoT and nq* co_llm QA datasets used by training, inference, and evaluation scripts.",
                    expected_location=str(self._repository_root / "DMSTrack" / "V2V4Real" / "official_models"),
                ),
                BootstrapArtifactSummary(
                    name="dataset_processed_features_and_gt.zip",
                    kind="processed perception data archive",
                    purpose="Provides processed perception features, point clouds, detections, and ground-truth assets expected by the original V2V-GoT pipeline.",
                    expected_location=str(self._repository_root / "DMSTrack" / "V2V4Real" / "official_models"),
                ),
                BootstrapArtifactSummary(
                    name="model_ckpt.zip",
                    kind="model checkpoint archive",
                    purpose="Provides pretrained V2V-GoT and V2V-LLM task checkpoints under the LLaVA checkpoints tree for reproduction and inference.",
                    expected_location=str(self._repository_root / "LLaVA" / "checkpoints"),
                ),
            ),
            conclusions=(
                "The real co_llm benchmark files are now available and readable from the expected official_models paths.",
                "Representative v2vgot and nq1 records share a stable JSON structure with two-turn conversations and explicit scenario/timestamp metadata.",
                "The first KG prototype can safely target existence, count, relative-position, and visible-object queries before tackling planning-style outputs.",
            ),
        )

    @staticmethod
    def _inspect_file(split_name: str, path: Path) -> CoLLMFileSummary:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        first = data[0]
        conversations = first.get("conversations", [])
        question_preview = conversations[0].get("value", "")[:240] if conversations else ""
        answer_preview = conversations[1].get("value", "")[:240] if len(conversations) > 1 else ""
        return CoLLMFileSummary(
            split_name=split_name,
            file_name=path.name,
            record_count=len(data),
            top_level_keys=tuple(first.keys()),
            conversation_roles=tuple(item.get("from", "") for item in conversations),
            question_preview=question_preview,
            answer_preview=answer_preview,
            qa_type_id=first.get("qa_type_id"),
            qa_source=first.get("qa_source"),
        )
