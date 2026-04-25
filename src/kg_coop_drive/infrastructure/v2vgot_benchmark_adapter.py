from __future__ import annotations

from collections import Counter
from pathlib import Path

from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType
from kg_coop_drive.infrastructure.v2vgot_scene_adapter import V2VGoTSceneAdapter


class V2VGoTQABenchmarkAdapter:
    """Loads V2V-GoT-QA samples and classifies them into stable task types."""

    _DEFAULT_FILE_NAME = "v2v4real_3d_grounding_qa_dataset_v2vgot.json"
    _TASK_BY_QA_TYPE_ID = {
        11: BenchmarkTaskType.NOTABLE_OBJECTS,
        12: BenchmarkTaskType.OCCLUDING_OBJECTS,
        13: BenchmarkTaskType.INVISIBLE_OBJECTS,
        14: BenchmarkTaskType.PLANNING_AWARENESS,
        15: BenchmarkTaskType.OBJECT_MOTION_PREDICTION,
        16: BenchmarkTaskType.AGENT_MOTION_PREDICTION,
        17: BenchmarkTaskType.OBJECT_MOTION_PREDICTION,
        18: BenchmarkTaskType.CONTROL_SETTINGS,
        9: BenchmarkTaskType.FUTURE_TRAJECTORY,
        19: BenchmarkTaskType.FUTURE_TRAJECTORY,
    }

    def __init__(
        self,
        repository_root: str,
        scene_adapter: V2VGoTSceneAdapter | None = None,
    ) -> None:
        self._repository_root = Path(repository_root).expanduser().resolve()
        self._scene_adapter = scene_adapter or V2VGoTSceneAdapter(str(self._repository_root))

    def load_samples(
        self,
        split_name: str = "val",
        file_name: str = _DEFAULT_FILE_NAME,
    ) -> tuple[BenchmarkSample, ...]:
        """Load benchmark samples together with canonical scenes and task labels."""

        records = self._scene_adapter.load_records(split_name=split_name, file_name=file_name)
        previous_record_by_scene_timestamp = self._index_previous_records(records)
        samples: list[BenchmarkSample] = []
        for index, record in enumerate(records):
            previous_record = previous_record_by_scene_timestamp.get(
                self._previous_record_key(record)
            )
            scene = self._scene_adapter.build_scene(
                record,
                previous_record=previous_record,
            )
            qa_type_id = self._read_qa_type_id(record)
            sample_id = self._read_sample_id(record, index=index)
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    dataset_name="V2V-GoT-QA",
                    split_name=split_name,
                    file_name=file_name,
                    task_type=self.classify_record(record),
                    scene=scene,
                    raw_record=record,
                    qa_type_id=qa_type_id,
                )
            )
        return tuple(samples)

    @classmethod
    def _index_previous_records(
        cls,
        records: list[dict[str, object]],
    ) -> dict[tuple[str, int], dict[str, object]]:
        return {
            cls._record_key(record): record
            for record in records
        }

    @classmethod
    def _previous_record_key(cls, record: dict[str, object]) -> tuple[str, int]:
        return (
            str(record.get("scenario_index", "unknown_scene")),
            cls._read_timestamp(record) - 1,
        )

    @classmethod
    def _record_key(cls, record: dict[str, object]) -> tuple[str, int]:
        return (
            str(record.get("scenario_index", "unknown_scene")),
            cls._read_timestamp(record),
        )

    def summarize_task_inventory(
        self,
        split_name: str = "val",
        file_name: str = _DEFAULT_FILE_NAME,
    ) -> dict[BenchmarkTaskType, int]:
        """Return task counts for one benchmark file."""

        counts: Counter[BenchmarkTaskType] = Counter()
        for sample in self.load_samples(split_name=split_name, file_name=file_name):
            counts[sample.task_type] += 1
        return dict(sorted(counts.items(), key=lambda item: item[0].value))

    def classify_record(self, record: dict[str, object]) -> BenchmarkTaskType:
        """Classify one raw benchmark record into a stable task category."""

        qa_type_id = self._read_qa_type_id(record)
        if qa_type_id in self._TASK_BY_QA_TYPE_ID:
            return self._TASK_BY_QA_TYPE_ID[qa_type_id]

        question = self._extract_question_text(record).lower()
        if "occluding" in question:
            return BenchmarkTaskType.OCCLUDING_OBJECTS
        if "obstruct my view" in question:
            return BenchmarkTaskType.OCCLUDING_OBJECTS
        if "invisible" in question or "not visible" in question:
            return BenchmarkTaskType.INVISIBLE_OBJECTS
        if "where might those notable objects move" in question:
            return BenchmarkTaskType.OBJECT_MOTION_PREDICTION
        if "where might other cavs move" in question:
            return BenchmarkTaskType.AGENT_MOTION_PREDICTION
        if "suggested speed and steering settings" in question:
            return BenchmarkTaskType.CONTROL_SETTINGS
        if "need to be aware of" in question:
            return BenchmarkTaskType.PLANNING_AWARENESS
        if "future trajectory" in question or "planned future trajectory" in question:
            if (
                "what is the suggested future trajectory" in question
                or "what is my future trajectory" in question
                or question.startswith("what is the future trajectory")
            ):
                return BenchmarkTaskType.FUTURE_TRAJECTORY
        if "visible" in question:
            if "notable" in question:
                return BenchmarkTaskType.NOTABLE_OBJECTS
            return BenchmarkTaskType.VISIBLE_OBJECTS
        if "notable" in question:
            return BenchmarkTaskType.NOTABLE_OBJECTS
        if "plan" in question or "trajectory" in question:
            return BenchmarkTaskType.PLANNING_AWARENESS
        return BenchmarkTaskType.UNKNOWN

    def supported_task_types(self) -> tuple[BenchmarkTaskType, ...]:
        """Return the currently recognized task categories."""

        return tuple(task_type for task_type in BenchmarkTaskType if task_type != BenchmarkTaskType.UNKNOWN)

    @staticmethod
    def _extract_question_text(record: dict[str, object]) -> str:
        conversations = record.get("conversations", [])
        if not isinstance(conversations, list):
            return ""
        for item in conversations:
            if isinstance(item, dict) and item.get("from") == "human":
                value = item.get("value", "")
                return value if isinstance(value, str) else ""
        return ""

    @staticmethod
    def _read_qa_type_id(record: dict[str, object]) -> int | None:
        value = record.get("qa_type_id")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_timestamp(record: dict[str, object]) -> int:
        return int(record.get("global_timestamp_index", record.get("local_timestamp_index", -1)))

    @staticmethod
    def _read_sample_id(record: dict[str, object], index: int) -> str:
        preferred_keys = (
            "sample_id",
            "id",
            "question_id",
        )
        for key in preferred_keys:
            value = record.get(key)
            if value is not None:
                return str(value)
        scenario_index = record.get("scenario_index", "unknown_scene")
        global_timestamp_index = record.get("global_timestamp_index", "unknown_time")
        return f"{scenario_index}:{global_timestamp_index}:{index}"
