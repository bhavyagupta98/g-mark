from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    TrackStatus,
    Trajectory,
    Vector2D,
)


@dataclass(frozen=True)
class OPV2VFrameRef:
    """Pointer to one OPV2V scenario/timestamp."""

    split: str
    scenario_id: str
    timestamp: str
    ego_agent_id: str
    agent_ids: tuple[str, ...]


class OPV2VSceneAdapter:
    """Read-only adapter from OPV2V YAML records into the G-MARK scene schema."""

    def __init__(self, data_root: str | Path) -> None:
        self._data_root = Path(data_root).expanduser().resolve()

    @property
    def data_root(self) -> Path:
        return self._data_root

    def list_scenarios(self, split: str = "test") -> list[str]:
        split_dir = self._split_dir(split)
        if not split_dir.exists():
            return []
        return sorted(path.name for path in split_dir.iterdir() if path.is_dir())

    def list_agents(self, split: str, scenario_id: str) -> list[str]:
        scenario_dir = self._scenario_dir(split, scenario_id)
        if not scenario_dir.exists():
            return []
        return sorted(
            (path.name for path in scenario_dir.iterdir() if path.is_dir()),
            key=self._agent_sort_key,
        )

    def list_timestamps(self, split: str, scenario_id: str, agent_id: str | None = None) -> list[str]:
        agents = self.list_agents(split, scenario_id)
        if not agents:
            return []
        selected_agent = agent_id or agents[0]
        agent_dir = self._scenario_dir(split, scenario_id) / selected_agent
        return sorted(
            path.stem
            for path in agent_dir.glob("*.yaml")
            if "additional" not in path.name
        )

    def first_frame(self, split: str = "test") -> OPV2VFrameRef:
        scenarios = self.list_scenarios(split)
        if not scenarios:
            raise FileNotFoundError(f"No OPV2V scenarios found under {self._split_dir(split)}")
        scenario_id = scenarios[0]
        agents = self.list_agents(split, scenario_id)
        if not agents:
            raise FileNotFoundError(f"No CAV folders found under {self._scenario_dir(split, scenario_id)}")
        timestamps = self.list_timestamps(split, scenario_id, agents[0])
        if not timestamps:
            raise FileNotFoundError(f"No YAML timestamps found for {split}/{scenario_id}/{agents[0]}")
        return OPV2VFrameRef(
            split=split,
            scenario_id=scenario_id,
            timestamp=timestamps[0],
            ego_agent_id=agents[0],
            agent_ids=tuple(agents),
        )

    def build_scene(
        self,
        frame_ref: OPV2VFrameRef,
        mode: str = "cooperative",
    ) -> CooperativeScene:
        """Build a G-MARK scene from one OPV2V frame.

        `mode="ego_only"` keeps only ego observations. `mode="cooperative"` keeps all agents.
        """

        if mode not in {"cooperative", "ego_only"}:
            raise ValueError(f"Unsupported OPV2V scene mode: {mode}")

        agent_ids = (
            (frame_ref.ego_agent_id,)
            if mode == "ego_only"
            else frame_ref.agent_ids
        )
        records = {
            agent_id: self.load_agent_yaml(
                frame_ref.split,
                frame_ref.scenario_id,
                agent_id,
                frame_ref.timestamp,
            )
            for agent_id in agent_ids
        }

        agents = tuple(
            AgentContext(
                agent_id=agent_id,
                pose=self._parse_pose(record.get("lidar_pose")),
                velocity=self._parse_speed(record.get("ego_speed")),
                planned_trajectory=None,
            )
            for agent_id, record in records.items()
        )
        observations: list[ObservationEvidence] = []
        tracks: list[ObjectTrack] = []

        for agent_id, record in records.items():
            vehicles = record.get("vehicles", {})
            if not isinstance(vehicles, dict):
                continue
            for object_id, vehicle in sorted(vehicles.items(), key=lambda item: str(item[0])):
                if not isinstance(vehicle, dict):
                    continue
                position = self._vehicle_world_position(vehicle)
                observation_id = f"{frame_ref.scenario_id}:{frame_ref.timestamp}:{agent_id}:{object_id}"
                observation = ObservationEvidence(
                    observation_id=observation_id,
                    source_agent_id=agent_id,
                    object_type="vehicle",
                    position=position,
                    confidence=1.0,
                    timestamp_index=self._timestamp_index(frame_ref.timestamp),
                )
                observations.append(observation)
                tracks.append(
                    ObjectTrack(
                        object_id=f"opv2v_{agent_id}_{object_id}",
                        object_type="vehicle",
                        position=position,
                        confidence=1.0,
                        provenance=ProvenanceRecord(
                            source_agent_ids=(agent_id,),
                            observation_ids=(observation_id,),
                            latest_timestamp_index=self._timestamp_index(frame_ref.timestamp),
                        ),
                        status=TrackStatus.SUPPORTED,
                        last_support_confidence=1.0,
                        observations=(observation,),
                    )
                )

        return CooperativeScene(
            scene_id=f"opv2v/{frame_ref.split}/{frame_ref.scenario_id}",
            local_timestamp_index=self._timestamp_index(frame_ref.timestamp),
            global_timestamp_index=self._timestamp_index(frame_ref.timestamp),
            asker_agent_id=frame_ref.ego_agent_id,
            agents=agents,
            future_trajectory=Trajectory(points=()),
            observations=tuple(observations),
            object_tracks=tuple(tracks),
            raw_question="OPV2V graph inspection frame",
            raw_answer="",
        )

    def inspect_frame(self, frame_ref: OPV2VFrameRef) -> dict[str, Any]:
        """Return JSON-serializable inspection stats for one frame."""

        per_agent: dict[str, dict[str, Any]] = {}
        object_sets: dict[str, set[str]] = {}
        for agent_id in frame_ref.agent_ids:
            record = self.load_agent_yaml(
                frame_ref.split,
                frame_ref.scenario_id,
                agent_id,
                frame_ref.timestamp,
            )
            vehicles = record.get("vehicles", {})
            vehicle_ids = set(str(key) for key in vehicles) if isinstance(vehicles, dict) else set()
            object_sets[agent_id] = vehicle_ids
            per_agent[agent_id] = {
                "yaml_path": str(
                    self._agent_yaml_path(
                        frame_ref.split,
                        frame_ref.scenario_id,
                        agent_id,
                        frame_ref.timestamp,
                    )
                ),
                "lidar_pose": self._jsonable(record.get("lidar_pose")),
                "ego_speed": self._jsonable(record.get("ego_speed")),
                "vehicle_count": len(vehicle_ids),
                "vehicle_ids_sample": sorted(vehicle_ids)[:10],
            }

        ego_objects = object_sets.get(frame_ref.ego_agent_id, set())
        all_objects = set().union(*object_sets.values()) if object_sets else set()
        partner_objects = set().union(
            *(ids for agent_id, ids in object_sets.items() if agent_id != frame_ref.ego_agent_id)
        ) if len(object_sets) > 1 else set()
        partner_only = partner_objects - ego_objects

        return {
            "data_root": str(self._data_root),
            "split": frame_ref.split,
            "scenario_id": frame_ref.scenario_id,
            "timestamp": frame_ref.timestamp,
            "ego_agent_id": frame_ref.ego_agent_id,
            "agent_ids": list(frame_ref.agent_ids),
            "agent_count": len(frame_ref.agent_ids),
            "all_unique_vehicle_count": len(all_objects),
            "ego_vehicle_count": len(ego_objects),
            "partner_only_vehicle_count": len(partner_only),
            "partner_only_vehicle_ids_sample": sorted(partner_only)[:20],
            "per_agent": per_agent,
        }

    def load_agent_yaml(
        self,
        split: str,
        scenario_id: str,
        agent_id: str,
        timestamp: str,
    ) -> dict[str, Any]:
        path = self._agent_yaml_path(split, scenario_id, agent_id, timestamp)
        with path.open("r", encoding="utf-8") as handle:
            try:
                loaded = yaml.safe_load(handle)
            except yaml.constructor.ConstructorError:
                handle.seek(0)
                # OPV2V/OpenCOOD YAML files can contain NumPy Python tags.
                # These are trusted local dataset files, so fall back to the
                # loader OpenCOOD-style data requires while keeping this path
                # isolated to the OPV2V adapter.
                loaded = yaml.unsafe_load(handle)
        return loaded if isinstance(loaded, dict) else {}

    def write_scene_json(self, scene: CooperativeScene, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene_id": scene.scene_id,
            "local_timestamp_index": scene.local_timestamp_index,
            "asker_agent_id": scene.asker_agent_id,
            "agent_count": len(scene.agents),
            "observation_count": len(scene.observations),
            "object_track_count": len(scene.object_tracks),
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "x": agent.pose.position.x,
                    "y": agent.pose.position.y,
                    "yaw_radians": agent.pose.yaw_radians,
                    "velocity": (
                        {"x": agent.velocity.x, "y": agent.velocity.y}
                        if agent.velocity is not None
                        else None
                    ),
                }
                for agent in scene.agents
            ],
            "object_tracks_sample": [
                {
                    "object_id": track.object_id,
                    "object_type": track.object_type,
                    "x": track.position.x,
                    "y": track.position.y,
                    "source_agent_ids": list(track.provenance.source_agent_ids),
                    "observation_ids": list(track.provenance.observation_ids),
                }
                for track in scene.object_tracks[:50]
            ],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _split_dir(self, split: str) -> Path:
        return self._data_root / split

    def _scenario_dir(self, split: str, scenario_id: str) -> Path:
        return self._split_dir(split) / scenario_id

    def _agent_yaml_path(self, split: str, scenario_id: str, agent_id: str, timestamp: str) -> Path:
        return self._scenario_dir(split, scenario_id) / agent_id / f"{timestamp}.yaml"

    @staticmethod
    def _agent_sort_key(agent_id: str) -> tuple[int, int | str]:
        try:
            return (0 if int(agent_id) >= 0 else 1, int(agent_id))
        except ValueError:
            return (2, agent_id)

    @staticmethod
    def _parse_pose(raw_pose: object) -> Pose2D:
        values = raw_pose if isinstance(raw_pose, list) else []
        x = float(values[0]) if len(values) > 0 else 0.0
        y = float(values[1]) if len(values) > 1 else 0.0
        yaw_degrees = float(values[4]) if len(values) > 4 else 0.0
        return Pose2D(position=Point2D(x=x, y=y), yaw_radians=math.radians(yaw_degrees))

    @staticmethod
    def _parse_speed(raw_speed: object) -> Vector2D | None:
        if isinstance(raw_speed, (int, float)):
            return Vector2D(x=float(raw_speed), y=0.0)
        return None

    @staticmethod
    def _vehicle_world_position(vehicle: dict[str, Any]) -> Point2D:
        location = vehicle.get("location", [0.0, 0.0, 0.0])
        center = vehicle.get("center", [0.0, 0.0, 0.0])
        lx = float(location[0]) if isinstance(location, list) and len(location) > 0 else 0.0
        ly = float(location[1]) if isinstance(location, list) and len(location) > 1 else 0.0
        cx = float(center[0]) if isinstance(center, list) and len(center) > 0 else 0.0
        cy = float(center[1]) if isinstance(center, list) and len(center) > 1 else 0.0
        return Point2D(x=lx + cx, y=ly + cy)

    @staticmethod
    def _timestamp_index(timestamp: str) -> int:
        try:
            return int(timestamp)
        except ValueError:
            digits = "".join(ch for ch in timestamp if ch.isdigit())
            return int(digits) if digits else -1

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if hasattr(value, "tolist"):
            return cls._jsonable(value.tolist())
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
