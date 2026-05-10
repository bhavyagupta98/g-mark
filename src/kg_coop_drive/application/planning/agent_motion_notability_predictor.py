from __future__ import annotations

import base64
from math import exp
import pickle


class LearnedAgentMotionNotabilityPredictor:
    """Predict Q6 notable/not-notable label from a frozen linear-logistic model."""

    def __init__(self, model: dict[str, object] | None = None) -> None:
        self._model = model or {}

    def predict_is_notable(self, *, scene, other_agent) -> bool | None:
        model_type = str(self._model.get("model_type", ""))
        features = self._feature_map(scene=scene, other_agent=other_agent)
        threshold = self._safe_float(self._model.get("decision_threshold"), 0.5)
        if model_type == "phase9_q6_agent_motion_notability_logistic_v1":
            feature_names = self._model.get("feature_names")
            weights = self._model.get("weights")
            if not isinstance(feature_names, list) or not isinstance(weights, list):
                return None
            if len(feature_names) != len(weights):
                return None
            z = 0.0
            for name, weight in zip(feature_names, weights):
                z += self._safe_float(weight, 0.0) * self._safe_float(features.get(str(name), 0.0), 0.0)
            prob = 1.0 / (1.0 + exp(-max(-40.0, min(40.0, z))))
            return prob >= threshold
        if model_type == "phase9_q6_agent_motion_notability_tree_v1":
            split_feature_names = self._model.get("split_feature_names")
            tree = self._model.get("tree")
            if not isinstance(split_feature_names, list) or not isinstance(tree, dict):
                return None
            split_features = [self._safe_float(features.get(str(name), 0.0), 0.0) for name in split_feature_names]
            prob = self._tree_predict_prob(tree, split_features)
            return prob >= threshold
        if model_type == "phase9_q6_agent_motion_notability_gbdt_v1":
            payload = self._model.get("sklearn_pickle_b64")
            if not isinstance(payload, str) or not payload:
                payload = self._model.get("xgboost_pickle_b64")
            if not isinstance(payload, str) or not payload:
                return None
            feature_names = self._model.get("feature_names")
            if not isinstance(feature_names, list):
                return None
            try:
                clf = pickle.loads(base64.b64decode(payload.encode("ascii")))
            except Exception:
                return None
            row = [[self._safe_float(features.get(str(name), 0.0), 0.0) for name in feature_names]]
            try:
                prob = float(clf.predict_proba(row)[0][1])
            except Exception:
                return None
            return prob >= threshold
        return None

    @classmethod
    def _feature_map(cls, *, scene, other_agent) -> dict[str, float]:
        asker = next((a for a in scene.agents if a.agent_id == scene.asker_agent_id), None)
        if asker is None:
            return {"bias": 1.0}

        other_points = getattr(getattr(other_agent, "planned_trajectory", None), "points", ()) or ()
        if other_points:
            final_dx = cls._safe_float(other_points[-1].x, 0.0)
            final_dy = cls._safe_float(other_points[-1].y, 0.0)
        else:
            final_dx = 0.0
            final_dy = 0.0
        final_dist = (final_dx * final_dx + final_dy * final_dy) ** 0.5
        other_max_step = cls._max_abs_step(other_points)

        asker_points = scene.future_trajectory.points
        min_dist_to_asker_path = cls._distance_to_future_path(
            asker_points,
            other_agent.pose.position.x,
            other_agent.pose.position.y,
        )
        asker_path_length = 0.0
        if len(asker_points) >= 2:
            for i in range(1, len(asker_points)):
                pdx = asker_points[i].x - asker_points[i - 1].x
                pdy = asker_points[i].y - asker_points[i - 1].y
                asker_path_length += (pdx * pdx + pdy * pdy) ** 0.5
        asker_path_max_step = cls._max_abs_step(asker_points)

        other_path_overlap_ratio_2m = 0.0
        if other_points:
            overlap = 0
            for p in other_points:
                if cls._distance_to_future_path(asker_points, p.x, p.y) <= 2.0:
                    overlap += 1
            other_path_overlap_ratio_2m = overlap / len(other_points)

        if asker_points:
            asker_final = asker_points[-1]
            endpoint_distance_to_asker_final = (
                ((other_agent.pose.position.x + final_dx - asker_final.x) ** 2)
                + ((other_agent.pose.position.y + final_dy - asker_final.y) ** 2)
            ) ** 0.5
            goal_vec_x = asker_final.x - asker.pose.position.x
            goal_vec_y = asker_final.y - asker.pose.position.y
            other_vec_x = final_dx
            other_vec_y = final_dy
            goal_norm = (goal_vec_x * goal_vec_x + goal_vec_y * goal_vec_y) ** 0.5
            other_norm = (other_vec_x * other_vec_x + other_vec_y * other_vec_y) ** 0.5
            if goal_norm > 1e-6 and other_norm > 1e-6:
                other_heading_alignment_with_asker_goal = (
                    (goal_vec_x * other_vec_x + goal_vec_y * other_vec_y) / (goal_norm * other_norm)
                )
            else:
                other_heading_alignment_with_asker_goal = 0.0
            rel_x = other_agent.pose.position.x - asker.pose.position.x
            rel_y = other_agent.pose.position.y - asker.pose.position.y
            proj = rel_x * goal_vec_x + rel_y * goal_vec_y
            other_ahead_of_asker_goal_flag = 1.0 if proj > 0.0 else 0.0
        else:
            endpoint_distance_to_asker_final = 999.0
            other_heading_alignment_with_asker_goal = 0.0
            other_ahead_of_asker_goal_flag = 0.0

        if len(other_points) >= 3:
            total_turn = 0.0
            turn_count = 0
            for i in range(1, len(other_points) - 1):
                ax = other_points[i].x - other_points[i - 1].x
                ay = other_points[i].y - other_points[i - 1].y
                bx = other_points[i + 1].x - other_points[i].x
                by = other_points[i + 1].y - other_points[i].y
                an = (ax * ax + ay * ay) ** 0.5
                bn = (bx * bx + by * by) ** 0.5
                if an > 1e-6 and bn > 1e-6:
                    cosv = max(-1.0, min(1.0, (ax * bx + ay * by) / (an * bn)))
                    total_turn += (1.0 - cosv)
                    turn_count += 1
            other_path_curvature_proxy = (total_turn / turn_count) if turn_count > 0 else 0.0
        else:
            other_path_curvature_proxy = 0.0

        nearby_5 = 0.0
        nearby_10 = 0.0
        nearby_dynamic_10 = 0.0
        for obj in scene.object_tracks:
            odx = obj.position.x - asker.pose.position.x
            ody = obj.position.y - asker.pose.position.y
            dist_to_asker = (odx * odx + ody * ody) ** 0.5
            if dist_to_asker <= 5.0:
                nearby_5 += 1.0
            if dist_to_asker <= 10.0:
                nearby_10 += 1.0
                v = obj.velocity
                if v is not None and ((v.x * v.x + v.y * v.y) ** 0.5) > 0.1:
                    nearby_dynamic_10 += 1.0

        return {
            "bias": 1.0,
            "other_planned_final_dx": final_dx,
            "other_planned_final_dy": final_dy,
            "other_planned_final_dist": final_dist,
            "other_planned_max_step": other_max_step,
            "other_min_distance_to_asker_path": min_dist_to_asker_path,
            "asker_path_length": asker_path_length,
            "asker_path_max_step": asker_path_max_step,
            "asker_nearby_object_count_5m": nearby_5,
            "asker_nearby_object_count_10m": nearby_10,
            "asker_nearby_dynamic_count_10m": nearby_dynamic_10,
            "other_path_overlap_ratio_2m": other_path_overlap_ratio_2m,
            "endpoint_distance_to_asker_final": endpoint_distance_to_asker_final,
            "other_heading_alignment_with_asker_goal": other_heading_alignment_with_asker_goal,
            "other_path_curvature_proxy": other_path_curvature_proxy,
            "other_ahead_of_asker_goal_flag": other_ahead_of_asker_goal_flag,
        }

    @staticmethod
    def _distance_to_future_path(points, point_x: float, point_y: float) -> float:
        if not points:
            return 999.0
        return min((((point_x - p.x) ** 2) + ((point_y - p.y) ** 2)) ** 0.5 for p in points)

    @staticmethod
    def _max_abs_step(points) -> float:
        if not points or len(points) < 2:
            return 0.0
        best = 0.0
        for i in range(1, len(points)):
            dx = points[i].x - points[i - 1].x
            dy = points[i].y - points[i - 1].y
            best = max(best, (dx * dx + dy * dy) ** 0.5)
        return best

    @staticmethod
    def _safe_float(v: object, default: float) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _tree_predict_prob(cls, node: dict[str, object], features: list[float]) -> float:
        current = node
        while not bool(current.get("leaf", True)):
            idx = int(current.get("feature_index", -1))
            threshold = cls._safe_float(current.get("threshold"), 0.0)
            if idx < 0 or idx >= len(features):
                break
            go_left = features[idx] <= threshold
            next_node = current.get("left") if go_left else current.get("right")
            if not isinstance(next_node, dict):
                break
            current = next_node
        return cls._safe_float(current.get("probability"), 0.0)
