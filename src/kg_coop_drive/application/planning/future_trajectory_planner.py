from __future__ import annotations

import re
from dataclasses import dataclass
from math import atan2, cos, sin, sqrt

from kg_coop_drive.domain.benchmark import BenchmarkSample
from kg_coop_drive.domain.scene import Point2D


_POSITION_RE = re.compile(
    r"I am\s+(?P<agent>[A-Za-z0-9_]+)\s+at\s+"
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
)
_COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
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
    _speed_labels = ("fast", "moderate", "slow", "very slow", "stop")
    _steering_labels = ("left", "slightly left", "straight", "slightly right", "right")
    _q8_speed_control_values = {
        "fast": 1.0,
        "moderate": 0.65,
        "slow": 0.35,
        "very slow": 0.15,
        "stop": 0.0,
    }
    _q8_steering_control_values = {
        "left": -1.0,
        "slightly left": -0.5,
        "straight": 0.0,
        "slightly right": 0.5,
        "right": 1.0,
    }

    def __init__(self, model: dict[str, object] | None = None, waypoint_count: int = 6) -> None:
        resolved_model = model or {}
        if isinstance(resolved_model.get("model_payload"), dict):
            payload = resolved_model.get("model_payload")
            assert isinstance(payload, dict)
            resolved_model = payload
        self._model = resolved_model
        self._waypoint_count = waypoint_count

    def plan(self, sample: BenchmarkSample) -> FutureTrajectoryPlan:
        current = self._current_position(sample)
        clean_points = self._predict_clean_linear_points(sample)
        if clean_points:
            return FutureTrajectoryPlan(
                points=clean_points[: self._waypoint_count],
                source="clean_q9_linear_no_oracle_metadata_model",
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

    def _predict_clean_linear_points(self, sample: BenchmarkSample) -> tuple[Point2D, ...]:
        feature_mean = self._model.get("feature_mean")
        feature_scale = self._model.get("feature_scale")
        coefficients = self._model.get("coefficients")
        if not isinstance(feature_mean, list) or not isinstance(feature_scale, list):
            return ()
        feature_row = self._clean_q9_feature_row(sample)
        if not feature_row:
            return ()
        if len(feature_mean) != len(feature_row) or len(feature_scale) != len(feature_row):
            return ()
        row_count = self._waypoint_count * 2

        normalized = []
        for value, mean, scale in zip(feature_row, feature_mean, feature_scale):
            safe_scale = float(scale) if abs(float(scale)) > 1e-8 else 1.0
            normalized.append((float(value) - float(mean)) / safe_scale)

        outputs: list[float] = []
        if isinstance(coefficients, list):
            if len(coefficients) < row_count:
                return ()
            for row in coefficients[:row_count]:
                if not isinstance(row, list) or len(row) != len(normalized):
                    return ()
                outputs.append(sum(float(weight) * feature for weight, feature in zip(row, normalized)))
        else:
            estimators = self._model.get("estimators")
            if not isinstance(estimators, list) or len(estimators) < row_count:
                return ()
            for estimator in estimators[:row_count]:
                if not isinstance(estimator, dict):
                    return ()
                coef = estimator.get("coef")
                intercept = estimator.get("intercept")
                if not isinstance(coef, list) or not isinstance(intercept, (int, float)):
                    return ()
                if len(coef) != len(normalized):
                    return ()
                outputs.append(float(intercept) + sum(float(weight) * feature for weight, feature in zip(coef, normalized)))
        if len(outputs) < row_count:
            return ()
        return tuple(
            Point2D(x=outputs[index * 2], y=outputs[index * 2 + 1])
            for index in range(self._waypoint_count)
        )

    def _clean_q9_feature_row(self, sample: BenchmarkSample) -> tuple[float, ...]:
        current = self._current_position(sample)
        asker_raw = str(sample.raw_record.get("asker_cav_id", "")).strip()
        asker_from_q = "1" if "I am CAV_1" in sample.scene.raw_question else ""
        asker_is_cav1 = 1.0 if (asker_raw == "1" or asker_from_q == "1") else 0.0
        row = [1.0, current.x, current.y, asker_is_cav1]
        row.extend(self._trajectory_geometry_features(sample, current))
        row.extend(self._q8_context_features(sample))
        return tuple(row)

    @classmethod
    def _trajectory_geometry_features(cls, sample: BenchmarkSample, current: Point2D) -> tuple[float, ...]:
        points = tuple((float(x), float(y)) for x, y in _COORD_RE.findall(sample.scene.raw_question)[:6])
        if not points:
            return (0.0,) * 10
        rel_points = tuple((px - current.x, py - current.y) for px, py in points)
        dists = tuple(sqrt(dx * dx + dy * dy) for dx, dy in rel_points)
        first_dx, first_dy = rel_points[0]
        last_dx, last_dy = rel_points[-1]
        first_dist = dists[0]
        last_dist = dists[-1]
        if len(points) >= 2:
            step_lengths = tuple(
                sqrt((points[idx][0] - points[idx - 1][0]) ** 2 + (points[idx][1] - points[idx - 1][1]) ** 2)
                for idx in range(1, len(points))
            )
            mean_step = sum(step_lengths) / len(step_lengths)
            variance = sum((value - mean_step) ** 2 for value in step_lengths) / len(step_lengths)
            std_step = sqrt(variance)
        else:
            mean_step = 0.0
            std_step = 0.0
        heading = atan2(first_dy, first_dx) if first_dist > 1e-6 else 0.0
        return (
            first_dx,
            first_dy,
            first_dist,
            last_dx,
            last_dy,
            last_dist,
            mean_step,
            std_step,
            sin(heading),
            cos(heading),
        )

    @classmethod
    def _q8_context_features(cls, sample: BenchmarkSample) -> tuple[float, ...]:
        speed_label = cls._speed_label(sample)
        steering_label = cls._steering_label(sample)
        speed_one_hot = tuple(1.0 if speed_label == label else 0.0 for label in cls._speed_labels)
        steering_one_hot = tuple(1.0 if steering_label == label else 0.0 for label in cls._steering_labels)
        speed_value = cls._q8_speed_control_values.get(speed_label, 0.0)
        steering_value = cls._q8_steering_control_values.get(steering_label, 0.0)
        return speed_one_hot + steering_one_hot + (float(speed_value), float(steering_value))

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
