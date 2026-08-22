from __future__ import annotations

import ast
import json
from pathlib import Path

from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    Point2D,
    Pose2D,
    Trajectory,
    Vector2D,
)


class V2VGoTSceneAdapter:
    """Converts one V2V-GoT QA record into the canonical Phase 2 scene seed."""

    def __init__(self, repository_root: str) -> None:
        self._repository_root = Path(repository_root).expanduser().resolve()

    def load_first_scene(
        self,
        split_name: str = "val",
        file_name: str = "v2v4real_3d_grounding_qa_dataset_v2vgot.json",
    ) -> CooperativeScene:
        """Load the first record from one co_llm file as a scene seed."""

        records = self.load_records(split_name=split_name, file_name=file_name)
        return self.build_scene(records[0])

    def load_records(
        self,
        split_name: str = "val",
        file_name: str = "v2v4real_3d_grounding_qa_dataset_v2vgot.json",
    ) -> list[dict[str, object]]:
        """Load all records from one co_llm file."""

        return self._load_records(split_name=split_name, file_name=file_name)

    def build_scene(
        self,
        record: dict[str, object],
        previous_record: dict[str, object] | None = None,
    ) -> CooperativeScene:
        """Build a scene seed from one raw V2V-GoT QA record."""

        conversations = record.get("conversations", [])
        question = self._conversation_value(conversations, 0)
        answer = self._conversation_value(conversations, 1)
        ego_pose = self._parse_pose(record.get("cav_ego_lidar_pose"))
        cav1_pose = self._parse_pose(record.get("cav_1_lidar_pose"))
        ego_trajectory = self._parse_trajectory(
            str(record.get("future_trajectory_str_in_ego", "[]"))
        )
        cav1_trajectory = self._parse_trajectory(
            str(record.get("future_trajectory_str_in_self", "[]"))
        )

        return CooperativeScene(
            scene_id=str(record.get("scenario_index", "unknown_scene")),
            local_timestamp_index=int(record.get("local_timestamp_index", -1)),
            global_timestamp_index=int(record.get("global_timestamp_index", -1)),
            asker_agent_id=self._normalize_agent_id(
                str(record.get("asker_cav_id", "unknown_agent"))
            ),
            agents=(
                AgentContext(
                    agent_id="CAV_EGO",
                    pose=ego_pose,
                    velocity=self._derive_agent_velocity(
                        record=record,
                        previous_record=previous_record,
                        pose=ego_pose,
                        pose_key="cav_ego_lidar_pose",
                    ),
                    planned_trajectory=ego_trajectory,
                ),
                AgentContext(
                    agent_id="CAV_1",
                    pose=cav1_pose,
                    velocity=self._derive_agent_velocity(
                        record=record,
                        previous_record=previous_record,
                        pose=cav1_pose,
                        pose_key="cav_1_lidar_pose",
                    ),
                    planned_trajectory=cav1_trajectory,
                ),
            ),
            future_trajectory=ego_trajectory,
            raw_question=question,
            raw_answer=answer,
        )

    def _load_records(self, split_name: str, file_name: str) -> list[dict[str, object]]:
        path = self._resolve_file_path(split_name=split_name, file_name=file_name)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_file_path(self, split_name: str, file_name: str) -> Path:
        file_path = Path(file_name).expanduser()
        if file_path.is_absolute():
            return file_path
        split_dir = (
            "no_fusion_keep_all"
            if split_name == "val"
            else "train_no_fusion_keep_all"
        )
        return (
            self._repository_root
            / "DMSTrack"
            / "V2V4Real"
            / "official_models"
            / split_dir
            / "npy"
            / "co_llm"
            / file_name
        )

    @staticmethod
    def _conversation_value(
        conversations: object,
        index: int,
    ) -> str:
        if not isinstance(conversations, list) or len(conversations) <= index:
            return ""
        item = conversations[index]
        if not isinstance(item, dict):
            return ""
        value = item.get("value", "")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _parse_pose(raw_pose: object) -> Pose2D:
        values = list(raw_pose) if isinstance(raw_pose, list) else [0.0, 0.0]
        x = float(values[0]) if len(values) > 0 else 0.0
        y = float(values[1]) if len(values) > 1 else 0.0
        yaw = float(values[4]) if len(values) > 4 else 0.0
        return Pose2D(position=Point2D(x=x, y=y), yaw_radians=yaw)

    @classmethod
    def _derive_agent_velocity(
        cls,
        record: dict[str, object],
        previous_record: dict[str, object] | None,
        pose: Pose2D,
        pose_key: str,
    ) -> Vector2D | None:
        if previous_record is None:
            return None
        if str(record.get("scenario_index", "")) != str(previous_record.get("scenario_index", "")):
            return None

        current_timestamp = cls._read_timestamp(record)
        previous_timestamp = cls._read_timestamp(previous_record)
        frame_delta = current_timestamp - previous_timestamp
        if frame_delta <= 0:
            return None

        previous_pose = cls._parse_pose(previous_record.get(pose_key))
        return Vector2D(
            x=(pose.position.x - previous_pose.position.x) / frame_delta,
            y=(pose.position.y - previous_pose.position.y) / frame_delta,
        )

    @staticmethod
    def _read_timestamp(record: dict[str, object]) -> int:
        return int(record.get("global_timestamp_index", record.get("local_timestamp_index", -1)))

    @staticmethod
    def _parse_trajectory(raw_trajectory: str) -> Trajectory:
        points_literal = ast.literal_eval(raw_trajectory)
        points = tuple(
            Point2D(x=float(point[0]), y=float(point[1]))
            for point in points_literal
        )
        return Trajectory(points=points)

    @staticmethod
    def _normalize_agent_id(raw_agent_id: str) -> str:
        mapping = {
            "ego": "CAV_EGO",
            "1": "CAV_1",
            "CAV_EGO": "CAV_EGO",
            "CAV_1": "CAV_1",
        }
        return mapping.get(raw_agent_id, raw_agent_id)
