from __future__ import annotations

import re
from dataclasses import dataclass
from math import cos, sin

from kg_coop_drive.domain.benchmark import BenchmarkSample
from kg_coop_drive.domain.scene import Point2D


_POSITION_RE = re.compile(
    r"I am\s+(?P<agent>[A-Za-z0-9_]+)\s+at\s+"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
)
_SPEED_RE = re.compile(r"suggested speed setting is:\s*(?P<label>[^.]+)", re.IGNORECASE)
_STEERING_RE = re.compile(r"suggested steering setting is:\s*(?P<label>[^.]+)", re.IGNORECASE)


@dataclass(frozen=True)
class FutureTrajectoryPlan:
    """Predicted future trajectory plus the source used to produce it."""

    points: tuple[Point2D, ...]
    source: str


class ControlConditionedFutureTrajectoryPlanner:
    """Predict Q9 future waypoints without replaying the ground-truth trajectory.

    The V2V-GoT Q9 records contain `future_trajectory_str_in_ego`, which is also
    the official answer. This planner intentionally ignores that field. It uses
    the question-visible current position and control context, optionally with a
    frozen train-derived delta model.
    """

    _default_speed_step = {
        "fast": 8.5,
        "normal": 5.8,
        "medium": 5.8,
        "slow": 3.0,
        "stop": 0.0,
        "stopped": 0.0,
    }
    _default_steering_angle = {
        "hard left": 0.28,
        "left": 0.18,
        "slightly left": 0.06,
        "straight": 0.02,
        "slightly right": -0.06,
        "right": -0.18,
        "hard right": -0.28,
    }

    def __init__(self, model: dict[str, object] | None = None, waypoint_count: int = 6) -> None:
        self._model = model or {}
        self._waypoint_count = waypoint_count

    def plan(self, sample: BenchmarkSample) -> FutureTrajectoryPlan:
        current = self._current_position(sample)
        absolute_points = self._predict_absolute_points(sample)
        if absolute_points:
            source = (
                "frozen_control_metadata_linear_tail_residual_model"
                if self._model.get("model_type")
                == "phase9_q9_control_metadata_linear_tail_residual_v1"
                else "frozen_control_metadata_linear_model"
            )
            return FutureTrajectoryPlan(
                points=absolute_points[: self._waypoint_count],
                source=source,
            )

        relative_points = self._lookup_relative_points(sample)
        if relative_points:
            return FutureTrajectoryPlan(
                points=tuple(
                    Point2D(x=current.x + dx, y=current.y + dy)
                    for dx, dy in relative_points[: self._waypoint_count]
                ),
                source="frozen_control_delta_model",
            )

        speed_label = self._speed_label(sample)
        steering_label = self._steering_label(sample)
        step = self._default_speed_step.get(speed_label, self._default_speed_step["normal"])
        angle = self._default_steering_angle.get(steering_label, 0.02)
        return FutureTrajectoryPlan(
            points=tuple(
                Point2D(
                    x=current.x + step * index * cos(angle),
                    y=current.y + step * index * sin(angle),
                )
                for index in range(1, self._waypoint_count + 1)
            ),
            source="control_kinematic_prior",
        )

    def _predict_absolute_points(self, sample: BenchmarkSample) -> tuple[Point2D, ...]:
        model_type = self._model.get("model_type")
        if model_type not in (
            "phase9_q9_control_metadata_linear_v1",
            "phase9_q9_control_metadata_linear_tail_residual_v1",
        ):
            return ()
        coefficients = self._model.get("coefficients")
        if not isinstance(coefficients, list):
            return ()
        features = self._linear_features(sample)
        outputs: list[float] = []
        for row in coefficients:
            if not isinstance(row, list) or len(row) != len(features):
                return ()
            outputs.append(
                sum(float(weight) * feature for weight, feature in zip(row, features))
            )
        if model_type == "phase9_q9_control_metadata_linear_tail_residual_v1":
            outputs = self._apply_tail_residual(sample, outputs)
        if len(outputs) < self._waypoint_count * 2:
            return ()
        return tuple(
            Point2D(x=outputs[index * 2], y=outputs[index * 2 + 1])
            for index in range(self._waypoint_count)
        )

    def _apply_tail_residual(self, sample: BenchmarkSample, outputs: list[float]) -> list[float]:
        tail_coefficients = self._model.get("tail_residual_coefficients")
        if not isinstance(tail_coefficients, list):
            return outputs
        tail_start_index = _safe_int(self._model.get("tail_start_index"))
        if tail_start_index < 0:
            return outputs
        tail_start_column = tail_start_index * 2
        if tail_start_column >= len(outputs):
            return outputs

        tail_features = self._linear_tail_features(sample)
        residuals: list[float] = []
        for row in tail_coefficients:
            if not isinstance(row, list) or len(row) != len(tail_features):
                return outputs
            residuals.append(
                sum(float(weight) * feature for weight, feature in zip(row, tail_features))
            )
        if len(residuals) != len(outputs[tail_start_column:]):
            return outputs
        adjusted = outputs[:]
        for index, residual in enumerate(residuals):
            adjusted[tail_start_column + index] += residual
        return adjusted

    @staticmethod
    def _linear_features(sample: BenchmarkSample) -> tuple[float, ...]:
        raw = sample.raw_record
        current = ControlConditionedFutureTrajectoryPlanner._current_position(sample)
        asker_is_cav1 = 1.0 if str(raw.get("asker_cav_id", "")) == "1" else 0.0
        speed_idx = _safe_int(raw.get("suggested_speed_idx"))
        steering_idx = _safe_int(raw.get("suggested_steering_idx"))
        distance = _safe_float(raw.get("dist"))
        angle = _safe_float(raw.get("angle"))
        return (
            1.0,
            current.x,
            current.y,
            asker_is_cav1,
            *(1.0 if speed_idx == index else 0.0 for index in range(5)),
            *(1.0 if steering_idx == index else 0.0 for index in range(5)),
            distance,
            sin(angle),
            cos(angle),
            distance * sin(angle),
            distance * cos(angle),
        )

    @staticmethod
    def _linear_tail_features(sample: BenchmarkSample) -> tuple[float, ...]:
        raw = sample.raw_record
        base = ControlConditionedFutureTrajectoryPlanner._linear_features(sample)
        current = ControlConditionedFutureTrajectoryPlanner._current_position(sample)
        distance = _safe_float(raw.get("dist"))
        angle = _safe_float(raw.get("angle"))
        speed_idx = _safe_int(raw.get("suggested_speed_idx"))
        steering_idx = _safe_int(raw.get("suggested_steering_idx"))
        return base + (
            current.x * distance,
            current.y * distance,
            distance * distance,
            sin(2.0 * angle),
            cos(2.0 * angle),
            distance * distance * sin(angle),
            distance * distance * cos(angle),
            float(speed_idx * steering_idx) if speed_idx >= 0 and steering_idx >= 0 else 0.0,
        )

    def _lookup_relative_points(self, sample: BenchmarkSample) -> tuple[tuple[float, float], ...]:
        lookup = self._model.get("relative_waypoints_by_key", {})
        if not isinstance(lookup, dict):
            return ()

        raw = sample.raw_record
        keys = (
            self._model_key(
                asker_cav_id=str(raw.get("asker_cav_id", "")),
                speed_idx=str(raw.get("suggested_speed_idx", "")),
                steering_idx=str(raw.get("suggested_steering_idx", "")),
            ),
            self._model_key(
                asker_cav_id="*",
                speed_idx=str(raw.get("suggested_speed_idx", "")),
                steering_idx=str(raw.get("suggested_steering_idx", "")),
            ),
            "__fallback__",
        )
        for key in keys:
            value = lookup.get(key)
            if not isinstance(value, list):
                continue
            parsed = self._parse_relative_points(value)
            if parsed:
                return parsed
        return ()

    @staticmethod
    def _parse_relative_points(value: list[object]) -> tuple[tuple[float, float], ...]:
        points: list[tuple[float, float]] = []
        for item in value:
            if (
                isinstance(item, list)
                and len(item) >= 2
                and isinstance(item[0], (int, float))
                and isinstance(item[1], (int, float))
            ):
                points.append((float(item[0]), float(item[1])))
        return tuple(points)

    @classmethod
    def _model_key(cls, *, asker_cav_id: str, speed_idx: str, steering_idx: str) -> str:
        return f"asker={asker_cav_id}|speed={speed_idx}|steering={steering_idx}"

    @staticmethod
    def _current_position(sample: BenchmarkSample) -> Point2D:
        match = _POSITION_RE.search(sample.scene.raw_question)
        if match:
            return Point2D(x=float(match.group("x")), y=float(match.group("y")))

        asker = next(
            (agent for agent in sample.scene.agents if agent.agent_id == sample.scene.asker_agent_id),
            None,
        )
        if asker is not None:
            return asker.pose.position
        return Point2D(x=0.0, y=0.0)

    @staticmethod
    def _speed_label(sample: BenchmarkSample) -> str:
        match = _SPEED_RE.search(sample.scene.raw_question)
        if match:
            return match.group("label").strip().lower()
        raw_value = sample.raw_record.get("suggested_speed_idx")
        return {
            0: "fast",
            1: "normal",
            2: "slow",
            3: "stop",
        }.get(raw_value, "normal")

    @staticmethod
    def _steering_label(sample: BenchmarkSample) -> str:
        match = _STEERING_RE.search(sample.scene.raw_question)
        if match:
            return match.group("label").strip().lower()
        raw_value = sample.raw_record.get("suggested_steering_idx")
        return {
            0: "left",
            1: "slightly left",
            2: "straight",
            3: "slightly right",
            4: "right",
        }.get(raw_value, "straight")


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
