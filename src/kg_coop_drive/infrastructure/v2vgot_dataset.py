from __future__ import annotations

import json
from pathlib import Path

from kg_coop_drive.domain.dataset import (
    DataAssetAvailability,
    DatasetInspectionReport,
    DatasetSplit,
    SampleFieldSummary,
)


class V2VGoTDatasetInspector:
    """Inspects locally available V2V-GoT assets for Phase 1 planning."""

    def __init__(self, repository_root: str) -> None:
        self._repository_root = Path(repository_root).expanduser().resolve()

    def inspect(self) -> DatasetInspectionReport:
        metadata = self._load_json(
            self._repository_root / "LLaVA" / "playground" / "data" / "V2V4Real" / "data.json"
        )
        split_summaries = self._build_split_summaries(metadata)
        sample_summaries = self._build_sample_summaries()
        required_assets = self._build_required_assets(metadata)
        observations = self._build_observations(required_assets, sample_summaries)

        return DatasetInspectionReport(
            dataset_name="V2V-GoT / V2V4Real",
            repository_root=str(self._repository_root),
            split_summaries=split_summaries,
            sample_summaries=sample_summaries,
            required_assets=required_assets,
            observations=observations,
        )

    def _build_split_summaries(self, metadata: dict) -> tuple[DatasetSplit, ...]:
        splits: list[DatasetSplit] = []
        for split_name, split_data in metadata.items():
            splits.append(
                DatasetSplit(
                    name=split_name,
                    sequence_ids=tuple(split_data.get("seq_eval", [])),
                    cumulative_lengths=tuple(split_data.get("len_record", [])),
                    llm_data_paths=dict(split_data.get("llm_data_path", {})),
                )
            )
        return tuple(splits)

    def _build_sample_summaries(self) -> tuple[SampleFieldSummary, ...]:
        sample_files = (
            self._repository_root / "LLaVA" / "playground" / "data" / "V2V4Real" / "v2v4real_dataset_for_llava_train.json",
            self._repository_root / "LLaVA" / "playground" / "data" / "V2V4Real" / "v2v4real_dataset_for_llava_val.json",
        )
        summaries: list[SampleFieldSummary] = []
        for sample_file in sample_files:
            samples = self._load_json(sample_file)
            if not isinstance(samples, list) or not samples:
                continue
            representative = samples[0]
            conversations = representative.get("conversations", [])
            prompt_preview = conversations[0]["value"][:240] if conversations else ""
            response_preview = conversations[1]["value"][:240] if len(conversations) > 1 else ""
            summaries.append(
                SampleFieldSummary(
                    top_level_keys=tuple(representative.keys()),
                    conversation_roles=tuple(item.get("from", "") for item in conversations),
                    scenario_index=representative.get("scenario_index"),
                    local_timestamp_index=representative.get("local_timestamp_index"),
                    global_timestamp_index=representative.get("global_timestamp_index"),
                    prompt_preview=prompt_preview,
                    response_preview=response_preview,
                )
            )
        return tuple(summaries)

    def _build_required_assets(self, metadata: dict) -> tuple[DataAssetAvailability, ...]:
        required_assets: list[DataAssetAvailability] = []
        for split_name, split_data in metadata.items():
            llm_paths = split_data.get("llm_data_path", {})
            for variant_name, relative_path in llm_paths.items():
                # The relative path in metadata is rooted from LLaVA, not from the V2V4Real folder.
                corrected_path = (self._repository_root / "LLaVA" / relative_path).resolve()
                required_assets.append(
                    DataAssetAvailability(
                        name=f"{split_name}:{variant_name}",
                        path=str(corrected_path),
                        exists=corrected_path.exists(),
                    )
                )
        return tuple(required_assets)

    def _build_observations(
        self,
        required_assets: tuple[DataAssetAvailability, ...],
        sample_summaries: tuple[SampleFieldSummary, ...],
    ) -> tuple[str, ...]:
        observations: list[str] = []
        if any(not asset.exists for asset in required_assets):
            observations.append(
                "The repository contains split metadata and LLaVA-style cooperative detection samples, "
                "but the expected processed `official_models/.../npy/co_llm` assets are not currently present locally."
            )
        if sample_summaries:
            observations.append(
                "The locally available sample JSON files are instruction-style records describing multi-agent detection "
                "inputs and cooperative detection outputs, which are useful for adapter design and field inspection."
            )
            observations.append(
                "Representative sample fields currently visible are `id`, `conversations`, `scenario_index`, "
                "`local_timestamp_index`, and `global_timestamp_index`."
            )
        observations.append(
            "Phase 1 should therefore target a loader/adapter that supports both metadata-only inspection now and "
            "full processed asset loading later when the missing dataset folders are available."
        )
        return tuple(observations)

    @staticmethod
    def _load_json(path: Path) -> dict | list:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
