from __future__ import annotations

import logging
from dataclasses import dataclass
from math import atan2, cos, dist, hypot, sin
from typing import Any

from kg_coop_drive.application.planning.control_settings_policy import (
    build_control_feature_vector,
    control_feature_names,
    rank_control_candidates,
)
from kg_coop_drive.domain.benchmark import BenchmarkSample
from kg_coop_drive.domain.scene import CooperativeScene, ObjectTrack, RelationType, VisibilityState

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedFeatureBank:
    sample_id: str
    qa_type_id: int | None
    task_type: str
    asker_id: str
    object_rows: tuple[dict[str, Any], ...]
    object_pair_rows: tuple[dict[str, Any], ...]
    motion_rows: tuple[dict[str, Any], ...]
    scene_row: dict[str, Any]
    missing_counts: dict[str, int]


class UnifiedFeatureBankBuilder:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        features_cfg = self._config.get("features", {}) if isinstance(self._config.get("features"), dict) else {}
        self._missing_policy = str(features_cfg.get("missing_value_policy", "zero_with_indicator"))
        exclude = features_cfg.get("exclude_leakage_fields", [])
        self._exclude_fields = set(str(v) for v in exclude) if isinstance(exclude, list) else set()

    def build(self, sample: BenchmarkSample, kg: CooperativeScene) -> UnifiedFeatureBank:
        missing_counts: dict[str, int] = {}
        visibility = {
            fact.object_id: fact.state
            for fact in kg.visibility_facts
            if fact.agent_id == kg.asker_agent_id
        }
        object_type_code: dict[str, int] = {}
        object_rows: list[dict[str, Any]] = []
        motion_rows: list[dict[str, Any]] = []
        for object_track in kg.object_tracks:
            row = self._object_row(sample=sample, kg=kg, object_track=object_track, visibility=visibility, missing_counts=missing_counts, object_type_code=object_type_code)
            object_rows.append(row)
            motion_rows.append(dict(row))

        object_pair_rows = self._object_pair_rows(sample=sample, kg=kg, visibility=visibility, missing_counts=missing_counts)
        scene_row = self._scene_action_row(sample=sample, kg=kg, missing_counts=missing_counts)

        return UnifiedFeatureBank(
            sample_id=sample.sample_id,
            qa_type_id=sample.qa_type_id,
            task_type=sample.task_type.value,
            asker_id=kg.asker_agent_id,
            object_rows=tuple(object_rows),
            object_pair_rows=tuple(object_pair_rows),
            motion_rows=tuple(motion_rows),
            scene_row=scene_row,
            missing_counts=missing_counts,
        )

    def _set(self, base: dict[str, Any], missing_counts: dict[str, int], key: str, value: Any, *, numeric: bool = True) -> None:
        if value is None:
            base[key] = 0.0 if numeric else ""
            missing_counts[key] = missing_counts.get(key, 0) + 1
            if self._missing_policy == "zero_with_indicator":
                base[f"{key}_missing"] = 1.0
            return
        if numeric:
            try:
                base[key] = float(value)
            except (TypeError, ValueError):
                base[key] = 0.0
                missing_counts[key] = missing_counts.get(key, 0) + 1
                if self._missing_policy == "zero_with_indicator":
                    base[f"{key}_missing"] = 1.0
                return
        else:
            base[key] = value
        if self._missing_policy == "zero_with_indicator" and f"{key}_missing" not in base:
            base[f"{key}_missing"] = 0.0

    def _object_row(
        self,
        *,
        sample: BenchmarkSample,
        kg: CooperativeScene,
        object_track: ObjectTrack,
        visibility: dict[str, VisibilityState],
        missing_counts: dict[str, int],
        object_type_code: dict[str, int],
    ) -> dict[str, Any]:
        asker = next((agent for agent in kg.agents if agent.agent_id == kg.asker_agent_id), None)
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "qa_type_id": sample.qa_type_id,
            "task_type": sample.task_type.value,
            "asker_id": kg.asker_agent_id,
            "candidate_id": object_track.object_id,
            "track_id": object_track.object_id,
        }
        x = float(object_track.position.x)
        y = float(object_track.position.y)
        asker_x = float(asker.pose.position.x) if asker is not None else 0.0
        asker_y = float(asker.pose.position.y) if asker is not None else 0.0
        rel_x = x - asker_x
        rel_y = y - asker_y
        self._set(row, missing_counts, "x", x)
        self._set(row, missing_counts, "y", y)
        self._set(row, missing_counts, "relative_x", rel_x)
        self._set(row, missing_counts, "relative_y", rel_y)
        self._set(row, missing_counts, "abs_relative_x", abs(rel_x))
        self._set(row, missing_counts, "abs_relative_y", abs(rel_y))

        dist_asker = hypot(rel_x, rel_y)
        self._set(row, missing_counts, "distance_to_asker", dist_asker)

        points = tuple(kg.future_trajectory.points)
        if points:
            nearest_idx = min(range(len(points)), key=lambda idx: (x - float(points[idx].x)) ** 2 + (y - float(points[idx].y)) ** 2)
            nearest_point = points[nearest_idx]
            nearest_dist = dist((x, y), (float(nearest_point.x), float(nearest_point.y)))
            first_dist = dist((x, y), (float(points[0].x), float(points[0].y)))
            nearest_norm = float(nearest_idx) / float(max(1, len(points) - 1))
            cumulative = [0.0]
            for i in range(1, len(points)):
                cumulative.append(cumulative[-1] + dist((float(points[i - 1].x), float(points[i - 1].y)), (float(points[i].x), float(points[i].y))))
            total_length = cumulative[-1]
            along_progress = 0.0 if total_length <= 1e-6 else cumulative[nearest_idx] / total_length
            if nearest_idx < len(points) - 1:
                dx = float(points[nearest_idx + 1].x) - float(points[nearest_idx].x)
                dy = float(points[nearest_idx + 1].y) - float(points[nearest_idx].y)
            elif nearest_idx > 0:
                dx = float(points[nearest_idx].x) - float(points[nearest_idx - 1].x)
                dy = float(points[nearest_idx].y) - float(points[nearest_idx - 1].y)
            else:
                dx = 1.0
                dy = 0.0
            norm = hypot(dx, dy)
            track_vx = float(object_track.velocity.x) if object_track.velocity is not None else 0.0
            track_vy = float(object_track.velocity.y) if object_track.velocity is not None else 0.0
            vnorm = hypot(track_vx, track_vy)
            heading_alignment = 0.0
            if norm > 1e-8 and vnorm > 1e-8:
                heading_alignment = (dx / norm) * (track_vx / vnorm) + (dy / norm) * (track_vy / vnorm)
        else:
            nearest_idx = 0
            nearest_dist = 999.0
            first_dist = 999.0
            nearest_norm = 0.0
            along_progress = 0.0
            heading_alignment = 0.0

        self._set(row, missing_counts, "distance_to_trajectory", nearest_dist)
        self._set(row, missing_counts, "distance_to_first_waypoint", first_dist)
        self._set(row, missing_counts, "nearest_waypoint_index", float(nearest_idx))
        self._set(row, missing_counts, "normalized_nearest_waypoint_index", nearest_norm)
        self._set(row, missing_counts, "along_path_progress", along_progress)
        self._set(row, missing_counts, "heading_alignment", heading_alignment)

        track_vx = float(object_track.velocity.x) if object_track.velocity is not None else 0.0
        track_vy = float(object_track.velocity.y) if object_track.velocity is not None else 0.0
        self._set(row, missing_counts, "vx", track_vx)
        self._set(row, missing_counts, "vy", track_vy)
        self._set(row, missing_counts, "speed", hypot(track_vx, track_vy))

        source_agent_ids = tuple(object_track.provenance.source_agent_ids)
        vis = visibility.get(object_track.object_id)
        status = str(object_track.status.value)
        self._set(row, missing_counts, "confidence", object_track.confidence)
        self._set(row, missing_counts, "support_count", len(source_agent_ids))
        self._set(row, missing_counts, "supporting_agent_count", len(source_agent_ids))
        self._set(row, missing_counts, "conflict_score", object_track.conflict_score)
        self._set(row, missing_counts, "uncertainty_score", object_track.uncertainty_score)
        self._set(row, missing_counts, "status_confirmed", 1.0 if status == "confirmed" else 0.0)
        self._set(row, missing_counts, "status_supported", 1.0 if status == "supported" else 0.0)
        self._set(row, missing_counts, "status_candidate", 1.0 if status == "candidate" else 0.0)
        self._set(row, missing_counts, "visibility_visible", 1.0 if vis == VisibilityState.VISIBLE else 0.0)
        self._set(row, missing_counts, "visibility_occluded", 1.0 if vis == VisibilityState.OCCLUDED else 0.0)
        self._set(row, missing_counts, "visibility_uncertain", 1.0 if vis == VisibilityState.UNCERTAIN else 0.0)
        self._set(row, missing_counts, "ego_visible", 1.0 if vis == VisibilityState.VISIBLE else 0.0)
        partner_only = 1.0 if source_agent_ids and kg.asker_agent_id not in source_agent_ids else 0.0
        self._set(row, missing_counts, "partner_only_supported", partner_only)

        path_relevant = 0.0
        near_trajectory = 0.0
        for relation in kg.relations:
            if relation.subject_id != object_track.object_id:
                continue
            if relation.relation_type == RelationType.PATH_RELEVANT:
                path_relevant = max(path_relevant, float(relation.confidence))
            if relation.relation_type == RelationType.NEAR_TRAJECTORY:
                near_trajectory = max(near_trajectory, float(relation.confidence))
        self._set(row, missing_counts, "path_relevant", path_relevant)
        self._set(row, missing_counts, "near_trajectory", near_trajectory)

        if object_track.object_type not in object_type_code:
            object_type_code[object_track.object_type] = len(object_type_code) + 1
        self._set(row, missing_counts, "object_type_code", object_type_code[object_track.object_type])
        row[f"object_type_is_{object_track.object_type.lower()}"] = 1.0
        return row

    def _object_pair_rows(
        self,
        *,
        sample: BenchmarkSample,
        kg: CooperativeScene,
        visibility: dict[str, VisibilityState],
        missing_counts: dict[str, int],
    ) -> list[dict[str, Any]]:
        asker = next((agent for agent in kg.agents if agent.agent_id == kg.asker_agent_id), None)
        if asker is None:
            return []
        hidden = [obj for obj in kg.object_tracks if visibility.get(obj.object_id) in {VisibilityState.OCCLUDED, VisibilityState.UNCERTAIN}]
        visible = [obj for obj in kg.object_tracks if visibility.get(obj.object_id) == VisibilityState.VISIBLE]
        rows: list[dict[str, Any]] = []
        for blocker in visible:
            for target in hidden:
                if blocker.object_id == target.object_id:
                    continue
                asker_pos = (asker.pose.position.x, asker.pose.position.y)
                blocker_vec = (blocker.position.x - asker_pos[0], blocker.position.y - asker_pos[1])
                target_vec = (target.position.x - asker_pos[0], target.position.y - asker_pos[1])
                blocker_norm = hypot(blocker_vec[0], blocker_vec[1])
                target_norm = hypot(target_vec[0], target_vec[1])
                if blocker_norm > 1e-6 and target_norm > 1e-6:
                    line_alignment = (blocker_vec[0] * target_vec[0] + blocker_vec[1] * target_vec[1]) / (blocker_norm * target_norm)
                else:
                    line_alignment = 0.0
                depth_order = 1.0 if blocker_norm < target_norm else 0.0
                row: dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "qa_type_id": sample.qa_type_id,
                    "task_type": sample.task_type.value,
                    "asker_id": kg.asker_agent_id,
                    "blocker_candidate_id": blocker.object_id,
                    "hidden_candidate_id": target.object_id,
                }
                self._set(row, missing_counts, "line_of_sight_alignment", line_alignment)
                self._set(row, missing_counts, "depth_order_score", depth_order)
                self._set(row, missing_counts, "visible_hidden_distance", dist((blocker.position.x, blocker.position.y), (target.position.x, target.position.y)))
                self._set(row, missing_counts, "hidden_distance_to_trajectory", self._distance_to_trajectory(kg=kg, x=target.position.x, y=target.position.y))
                self._set(row, missing_counts, "hidden_support_count", len(target.provenance.source_agent_ids))
                self._set(row, missing_counts, "occlusion_relation_score", max(0.0, line_alignment * depth_order))
                self._set(row, missing_counts, "blocker_risk_score", self._simple_risk_score(kg=kg, track=blocker, visibility=visibility.get(blocker.object_id)))
                self._set(row, missing_counts, "blocker_confidence", blocker.confidence)
                self._set(row, missing_counts, "blocker_support_count", len(blocker.provenance.source_agent_ids))
                self._set(row, missing_counts, "blocker_conflict_score", blocker.conflict_score)
                self._set(row, missing_counts, "blocker_uncertainty_score", blocker.uncertainty_score)
                rows.append(row)
        return rows

    def _scene_action_row(
        self,
        *,
        sample: BenchmarkSample,
        kg: CooperativeScene,
        missing_counts: dict[str, int],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "qa_type_id": sample.qa_type_id,
            "task_type": sample.task_type.value,
            "asker_id": kg.asker_agent_id,
        }
        asker = next((a for a in kg.agents if a.agent_id == kg.asker_agent_id), None)
        asker_points = tuple(kg.future_trajectory.points)
        path_length = 0.0
        path_max_step = 0.0
        for i in range(1, len(asker_points)):
            step = dist((asker_points[i - 1].x, asker_points[i - 1].y), (asker_points[i].x, asker_points[i].y))
            path_length += step
            path_max_step = max(path_max_step, step)
        self._set(row, missing_counts, "asker_path_length", path_length)
        self._set(row, missing_counts, "asker_path_max_step", path_max_step)

        nearby_asker_5 = 0.0
        nearby_asker_10 = 0.0
        nearby_dynamic_10 = 0.0
        if asker is not None:
            for obj in kg.object_tracks:
                d = dist((obj.position.x, obj.position.y), (asker.pose.position.x, asker.pose.position.y))
                if d <= 5.0:
                    nearby_asker_5 += 1.0
                if d <= 10.0:
                    nearby_asker_10 += 1.0
                    if obj.velocity is not None and hypot(obj.velocity.x, obj.velocity.y) > 0.1:
                        nearby_dynamic_10 += 1.0
        self._set(row, missing_counts, "nearby_object_count_5m", nearby_asker_5)
        self._set(row, missing_counts, "nearby_object_count_10m", nearby_asker_10)
        self._set(row, missing_counts, "nearby_dynamic_object_count_10m", nearby_dynamic_10)

        other = next((a for a in kg.agents if a.agent_id != kg.asker_agent_id), None)
        if other is not None and other.planned_trajectory is not None and other.planned_trajectory.points:
            last = other.planned_trajectory.points[-1]
            self._set(row, missing_counts, "other_agent_planned_final_dx", float(last.x))
            self._set(row, missing_counts, "other_agent_planned_final_dy", float(last.y))
            self._set(row, missing_counts, "other_agent_planned_final_dist", hypot(float(last.x), float(last.y)))
            if asker_points:
                overlap = sum(1 for p in other.planned_trajectory.points if self._distance_to_trajectory(kg=kg, x=p.x, y=p.y) <= 2.0)
                self._set(row, missing_counts, "path_overlap_ratio", overlap / float(len(other.planned_trajectory.points)))
                self._set(row, missing_counts, "other_min_distance_to_asker_path", min(self._distance_to_trajectory(kg=kg, x=p.x, y=p.y) for p in other.planned_trajectory.points))
                last_asker = asker_points[-1]
                self._set(
                    row,
                    missing_counts,
                    "endpoint_distance_to_asker_final",
                    dist((other.pose.position.x + last.x, other.pose.position.y + last.y), (last_asker.x, last_asker.y)),
                )
            else:
                self._set(row, missing_counts, "path_overlap_ratio", 0.0)
                self._set(row, missing_counts, "other_min_distance_to_asker_path", 999.0)
                self._set(row, missing_counts, "endpoint_distance_to_asker_final", 999.0)
        else:
            self._set(row, missing_counts, "other_agent_planned_final_dx", 0.0)
            self._set(row, missing_counts, "other_agent_planned_final_dy", 0.0)
            self._set(row, missing_counts, "other_agent_planned_final_dist", 0.0)
            self._set(row, missing_counts, "path_overlap_ratio", 0.0)
            self._set(row, missing_counts, "other_min_distance_to_asker_path", 999.0)
            self._set(row, missing_counts, "endpoint_distance_to_asker_final", 999.0)

        if asker is not None and asker_points:
            goal = asker_points[-1]
            goal_vec = (float(goal.x - asker.pose.position.x), float(goal.y - asker.pose.position.y))
            goal_norm = hypot(goal_vec[0], goal_vec[1])
            heading_unit = (cos(float(asker.pose.yaw_radians)), sin(float(asker.pose.yaw_radians)))
            goal_unit = (goal_vec[0] / goal_norm, goal_vec[1] / goal_norm) if goal_norm > 1e-6 else (1.0, 0.0)
            self._set(row, missing_counts, "heading_alignment_with_asker_goal", heading_unit[0] * goal_unit[0] + heading_unit[1] * goal_unit[1])
            curvature = 0.0
            turns = 0
            for i in range(1, len(asker_points) - 1):
                ax = asker_points[i].x - asker_points[i - 1].x
                ay = asker_points[i].y - asker_points[i - 1].y
                bx = asker_points[i + 1].x - asker_points[i].x
                by = asker_points[i + 1].y - asker_points[i].y
                an = hypot(ax, ay)
                bn = hypot(bx, by)
                if an > 1e-6 and bn > 1e-6:
                    curvature += abs(atan2(ax * by - ay * bx, ax * bx + ay * by))
                    turns += 1
            self._set(row, missing_counts, "curvature_proxy", curvature / float(turns) if turns > 0 else 0.0)
        else:
            self._set(row, missing_counts, "heading_alignment_with_asker_goal", 0.0)
            self._set(row, missing_counts, "curvature_proxy", 0.0)

        ranked = rank_control_candidates(kg)
        top = ranked[0] if ranked else None
        if top is not None:
            top_obj, _ = top
            self._set(row, missing_counts, "top_risk_object_distance_to_trajectory", self._distance_to_trajectory(kg=kg, x=top_obj.position.x, y=top_obj.position.y))
            self._set(row, missing_counts, "top_risk_object_confidence", top_obj.confidence)
            self._set(row, missing_counts, "top_risk_object_support_count", len(top_obj.provenance.source_agent_ids))
            self._set(row, missing_counts, "top_risk_object_conflict_score", top_obj.conflict_score)
            self._set(row, missing_counts, "top_risk_object_uncertainty_score", top_obj.uncertainty_score)
        else:
            self._set(row, missing_counts, "top_risk_object_distance_to_trajectory", 999.0)
            self._set(row, missing_counts, "top_risk_object_confidence", 0.0)
            self._set(row, missing_counts, "top_risk_object_support_count", 0.0)
            self._set(row, missing_counts, "top_risk_object_conflict_score", 0.0)
            self._set(row, missing_counts, "top_risk_object_uncertainty_score", 0.0)

        extended_names = control_feature_names("extended_v1")
        extended_values = build_control_feature_vector(kg, feature_set="extended_v1")
        for name, value in zip(extended_names, extended_values):
            self._set(row, missing_counts, f"q8_{name}", value)

        for blocked in self._exclude_fields:
            if blocked in row:
                LOGGER.warning("Excluded leakage field in scene row: %s", blocked)
                row.pop(blocked, None)
        return row

    @staticmethod
    def _distance_to_trajectory(*, kg: CooperativeScene, x: float, y: float) -> float:
        points = tuple(kg.future_trajectory.points)
        if not points:
            return 999.0
        return min(dist((x, y), (float(point.x), float(point.y))) for point in points)

    @staticmethod
    def _simple_risk_score(*, kg: CooperativeScene, track: ObjectTrack, visibility: VisibilityState | None) -> float:
        risk = 1.0 / (1.0 + UnifiedFeatureBankBuilder._distance_to_trajectory(kg=kg, x=track.position.x, y=track.position.y))
        risk += 0.5 * float(track.confidence)
        risk += 0.2 * float(len(track.provenance.source_agent_ids))
        risk -= 0.4 * float(track.conflict_score)
        risk -= 0.3 * float(track.uncertainty_score)
        if visibility == VisibilityState.OCCLUDED:
            risk += 0.15
        elif visibility == VisibilityState.UNCERTAIN:
            risk += 0.08
        return float(risk)


def build_unified_feature_bank(sample: BenchmarkSample, kg: CooperativeScene, config: dict[str, Any] | None = None) -> UnifiedFeatureBank:
    return UnifiedFeatureBankBuilder(config=config).build(sample=sample, kg=kg)
