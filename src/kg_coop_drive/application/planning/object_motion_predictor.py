from __future__ import annotations

import base64
from dataclasses import dataclass
from math import dist
import pickle

from kg_coop_drive.domain.scene import VisibilityState


@dataclass(frozen=True)
class ObjectMotionPredictionResult:
    """One predicted end-point and action label for an object track."""

    end_x: float
    end_y: float
    future_points: tuple[tuple[float, float], ...]
    motion_label: str
    source: str


class LearnedObjectMotionPredictor:
    """Predict one-step object motion for Q5 from a frozen train model.

    Model formats:
    - linear: phase9_q5_object_motion_linear_v1
    - piecewise linear experts: phase9_q5_object_motion_piecewise_linear_v1
    - regression tree: phase9_q5_object_motion_regression_tree_v1
    - gradient boosting: phase9_q5_object_motion_gradient_boosting_v1
    - sklearn MLP: phase9_q5_object_motion_mlp_v1
    """

    _default_horizon_seconds = 1.0

    def __init__(self, model: dict[str, object] | None = None) -> None:
        self._model = model or {}

    def predict(
        self,
        *,
        scene,
        object_track,
        visibility_state: VisibilityState | None,
    ) -> ObjectMotionPredictionResult:
        learned = self._predict_from_model(
            scene=scene,
            object_track=object_track,
            visibility_state=visibility_state,
        )
        if learned is not None:
            future_points = learned
            end_x, end_y = future_points[-1]
            return ObjectMotionPredictionResult(
                end_x=end_x,
                end_y=end_y,
                future_points=future_points,
                motion_label=self._q5_motion_label(
                    dx=end_x - object_track.position.x,
                    dy=end_y - object_track.position.y,
                ),
                source="frozen_q5_linear_model",
            )

        velocity = object_track.velocity
        if velocity is None:
            end_x = object_track.position.x
            end_y = object_track.position.y
        else:
            end_x = object_track.position.x + velocity.x * self._default_horizon_seconds
            end_y = object_track.position.y + velocity.y * self._default_horizon_seconds
        return ObjectMotionPredictionResult(
            end_x=end_x,
            end_y=end_y,
            future_points=((end_x, end_y),),
            motion_label=self._q5_motion_label(
                dx=end_x - object_track.position.x,
                dy=end_y - object_track.position.y,
            ),
            source="velocity_projection_fallback",
        )

    def _predict_from_model(
        self,
        *,
        scene,
        object_track,
        visibility_state: VisibilityState | None,
    ) -> tuple[tuple[float, float], ...] | None:
        model_type = str(self._model.get("model_type", ""))
        feature_map = self._feature_map(
            scene=scene,
            object_track=object_track,
            visibility_state=visibility_state,
        )
        if model_type == "phase9_q5_object_motion_linear_v1":
            feature_names = self._model.get("feature_names")
            coefficients = self._model.get("coefficients")
            if not isinstance(feature_names, list) or not isinstance(coefficients, list):
                return None
            if len(coefficients) < 2 or len(coefficients) % 2 != 0:
                return None
            if any(not isinstance(row, list) or len(row) != len(feature_names) for row in coefficients):
                return None
            features = [float(feature_map.get(str(name), 0.0)) for name in feature_names]
            target_values = [
                sum(float(weight) * value for weight, value in zip(row, features))
                for row in coefficients
            ]
        elif model_type == "phase9_q5_object_motion_piecewise_linear_v1":
            feature_names = self._model.get("feature_names")
            default_coefficients = self._model.get("default_coefficients")
            experts = self._model.get("piecewise_experts")
            if (
                not isinstance(feature_names, list)
                or not isinstance(default_coefficients, list)
                or not isinstance(experts, dict)
            ):
                return None
            key = self._piecewise_key(feature_map, visibility_state, object_track)
            coeff_list = experts.get(key, default_coefficients)
            if not isinstance(coeff_list, list) or len(coeff_list) < 2 or len(coeff_list) % 2 != 0:
                return None
            features = [float(feature_map.get(str(name), 0.0)) for name in feature_names]
            target_values = [
                sum(float(weight) * value for weight, value in zip(row, features))
                for row in coeff_list
            ]
        elif model_type == "phase9_q5_object_motion_regression_tree_v1":
            split_feature_names = self._model.get("split_feature_names")
            tree = self._model.get("tree")
            if not isinstance(split_feature_names, list) or not isinstance(tree, dict):
                return None
            split_features = [float(feature_map.get(str(name), 0.0)) for name in split_feature_names]
            target_values = self._tree_predict(tree, split_features)
        elif model_type == "phase9_q5_object_motion_gradient_boosting_v1":
            feature_names = self._model.get("feature_names")
            payload = self._model.get("sklearn_pickle_b64")
            if not isinstance(feature_names, list) or not isinstance(payload, str) or not payload:
                return None
            try:
                regressors = pickle.loads(base64.b64decode(payload.encode("ascii")))
            except Exception:
                return None
            if isinstance(regressors, tuple):
                regressors = list(regressors)
            if not isinstance(regressors, list) or len(regressors) < 2:
                return None
            features = [[float(feature_map.get(str(name), 0.0)) for name in feature_names]]
            try:
                target_values = [float(regressor.predict(features)[0]) for regressor in regressors]
            except Exception:
                return None
        elif model_type == "phase9_q5_object_motion_mlp_v1":
            feature_names = self._model.get("feature_names")
            payload = self._model.get("sklearn_pickle_b64")
            if not isinstance(feature_names, list) or not isinstance(payload, str) or not payload:
                return None
            try:
                regressor = pickle.loads(base64.b64decode(payload.encode("ascii")))
            except Exception:
                return None
            features = [[float(feature_map.get(str(name), 0.0)) for name in feature_names]]
            try:
                raw_prediction = regressor.predict(features)[0]
            except Exception:
                return None
            try:
                target_values = [float(value) for value in raw_prediction]
            except TypeError:
                target_values = [float(raw_prediction)]
        else:
            return None

        max_abs_delta = self._safe_float(self._model.get("max_abs_delta"), default=120.0)
        future_points: list[tuple[float, float]] = []
        for index in range(0, len(target_values) - 1, 2):
            dx = max(-max_abs_delta, min(max_abs_delta, float(target_values[index])))
            dy = max(-max_abs_delta, min(max_abs_delta, float(target_values[index + 1])))
            future_points.append((object_track.position.x + dx, object_track.position.y + dy))
        return tuple(future_points) if future_points else None

    @staticmethod
    def _piecewise_key(
        feature_map: dict[str, float],
        visibility_state: VisibilityState | None,
        object_track,
    ) -> str:
        speed = float(feature_map.get("speed", 0.0))
        if speed < 1.0:
            speed_bucket = "slow"
        elif speed < 5.0:
            speed_bucket = "mid"
        else:
            speed_bucket = "fast"
        traj = float(feature_map.get("distance_to_trajectory", 999.0))
        if traj < 5.0:
            traj_bucket = "near"
        elif traj < 15.0:
            traj_bucket = "mid"
        else:
            traj_bucket = "far"
        if visibility_state == VisibilityState.VISIBLE:
            vis_bucket = "visible"
        elif visibility_state == VisibilityState.OCCLUDED:
            vis_bucket = "occluded"
        elif visibility_state == VisibilityState.UNCERTAIN:
            vis_bucket = "uncertain"
        else:
            vis_bucket = "unknown"
        status = str(object_track.status.value)
        status_bucket = "supported" if status == "supported" else "candidate" if status == "candidate" else "other"
        return f"speed={speed_bucket}|traj={traj_bucket}|visibility={vis_bucket}|status={status_bucket}"

    @classmethod
    def _tree_predict(cls, node: dict[str, object], features: list[float]) -> list[float]:
        current = node
        while not bool(current.get("leaf", True)):
            idx = int(current.get("feature_index", -1))
            threshold = cls._safe_float(current.get("threshold"), default=0.0)
            if idx < 0 or idx >= len(features):
                break
            go_left = features[idx] <= threshold
            next_node = current.get("left") if go_left else current.get("right")
            if not isinstance(next_node, dict):
                break
            current = next_node
        prediction = current.get("prediction", [0.0, 0.0])
        if not isinstance(prediction, list) or len(prediction) < 2:
            return [0.0, 0.0]
        return [cls._safe_float(value, default=0.0) for value in prediction]

    @staticmethod
    def _q5_motion_label(*, dx: float, dy: float) -> str:
        speed = (dx * dx + dy * dy) ** 0.5
        if speed < 0.1:
            return "staying at the same location"
        if abs(dy) > abs(dx):
            return "moving right" if dy >= 0.0 else "moving left"
        if dx >= 0.0:
            return "moving forward"
        return "turning right" if dy >= 0.0 else "turning left"

    @classmethod
    def _feature_map(
        cls,
        *,
        scene,
        object_track,
        visibility_state: VisibilityState | None,
    ) -> dict[str, float]:
        velocity = object_track.velocity
        vx = 0.0 if velocity is None else float(velocity.x)
        vy = 0.0 if velocity is None else float(velocity.y)
        speed = (vx * vx + vy * vy) ** 0.5

        if scene.future_trajectory.points:
            dist_to_traj = min(
                dist((object_track.position.x, object_track.position.y), (point.x, point.y))
                for point in scene.future_trajectory.points
            )
        else:
            dist_to_traj = 999.0

        asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
        if asker is None:
            dist_to_asker = 999.0
        else:
            dist_to_asker = dist(
                (object_track.position.x, object_track.position.y),
                (asker.pose.position.x, asker.pose.position.y),
            )

        status = str(object_track.status.value)
        support_count = cls._support_count(object_track)
        values = {
            "bias": 1.0,
            "x": float(object_track.position.x),
            "y": float(object_track.position.y),
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "distance_to_trajectory": float(dist_to_traj),
            "distance_to_asker": float(dist_to_asker),
            "confidence": float(object_track.confidence),
            "support_count": support_count,
            "conflict_score": float(object_track.conflict_score),
            "uncertainty_score": float(object_track.uncertainty_score),
            "status_supported": 1.0 if status == "supported" else 0.0,
            "status_candidate": 1.0 if status == "candidate" else 0.0,
            "visibility_visible": 1.0 if visibility_state == VisibilityState.VISIBLE else 0.0,
            "visibility_occluded": 1.0 if visibility_state == VisibilityState.OCCLUDED else 0.0,
            "visibility_uncertain": 1.0 if visibility_state == VisibilityState.UNCERTAIN else 0.0,
        }
        values.update(
            cls._q7_path_relative_features(
                scene=scene,
                object_track=object_track,
                asker=asker,
                vx=vx,
                vy=vy,
            )
        )
        return values

    @staticmethod
    def _q7_path_relative_features(*, scene, object_track, asker, vx: float, vy: float) -> dict[str, float]:
        if asker is None:
            asker_x = object_track.position.x
            asker_y = object_track.position.y
        else:
            asker_x = asker.pose.position.x
            asker_y = asker.pose.position.y

        future_points = tuple(scene.future_trajectory.points)
        if future_points:
            final_x = float(future_points[-1].x)
            final_y = float(future_points[-1].y)
            closest_idx = min(
                range(len(future_points)),
                key=lambda idx: (object_track.position.x - future_points[idx].x) ** 2
                + (object_track.position.y - future_points[idx].y) ** 2,
            )
            closest_point = future_points[closest_idx]
            closest_norm = float(closest_idx) / float(max(1, len(future_points) - 1))
        else:
            final_x = asker_x
            final_y = asker_y
            closest_point = None
            closest_norm = 0.0

        goal_dx = float(final_x - object_track.position.x)
        goal_dy = float(final_y - object_track.position.y)
        goal_dist = (goal_dx * goal_dx + goal_dy * goal_dy) ** 0.5
        if goal_dist > 1e-6:
            unit_x = goal_dx / goal_dist
            unit_y = goal_dy / goal_dist
            vel_toward_goal = vx * unit_x + vy * unit_y
            vel_lateral_goal = -vx * unit_y + vy * unit_x
        else:
            vel_toward_goal = 0.0
            vel_lateral_goal = 0.0

        if closest_point is None:
            closest_dx = 0.0
            closest_dy = 0.0
        else:
            closest_dx = float(closest_point.x - object_track.position.x)
            closest_dy = float(closest_point.y - object_track.position.y)

        return {
            "relative_x_to_asker": float(object_track.position.x - asker_x),
            "relative_y_to_asker": float(object_track.position.y - asker_y),
            "asker_goal_x_from_object": goal_dx,
            "asker_goal_y_from_object": goal_dy,
            "distance_to_asker_goal": goal_dist,
            "closest_asker_path_x_from_object": closest_dx,
            "closest_asker_path_y_from_object": closest_dy,
            "closest_asker_path_index_norm": closest_norm,
            "object_vx_along_asker_goal": vel_toward_goal,
            "object_vy_lateral_to_asker_goal": vel_lateral_goal,
            "object_velocity_toward_asker_goal": vel_toward_goal,
        }

    @staticmethod
    def _safe_float(value: object, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _support_count(object_track) -> float:
        if hasattr(object_track, "supporting_agent_ids") and getattr(object_track, "supporting_agent_ids") is not None:
            return float(len(getattr(object_track, "supporting_agent_ids")))
        provenance = getattr(object_track, "provenance", None)
        if provenance is not None:
            source_agent_ids = getattr(provenance, "source_agent_ids", ())
            return float(len(source_agent_ids))
        observations = getattr(object_track, "observations", None)
        if observations is not None:
            return float(len(observations))
        return 0.0
