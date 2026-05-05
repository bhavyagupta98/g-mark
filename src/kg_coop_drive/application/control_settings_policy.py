from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, dist, exp, hypot, sin

from kg_coop_drive.domain.scene import TrackStatus, VisibilityState

SPEED_CLASSES = ("fast", "moderate", "slow", "very slow", "stop")
STEERING_CLASSES = ("left", "slightly left", "straight", "slightly right", "right")
CONTROL_FEATURE_NAMES_BASE = (
    "top1_risk",
    "top2_risk",
    "risk_gap",
    "top1_distance_to_trajectory",
    "top1_distance_to_asker",
    "top1_confidence",
    "top1_uncertainty",
    "top1_conflict",
    "top1_support_count",
    "top1_visible",
    "top1_uncertain",
    "top1_occluded",
    "top1_is_candidate",
    "top1_asker_supported",
    "top1_multi_supported",
    "top1_lateral_offset",
    "top1_abs_lateral_offset",
    "num_top_objects",
)
CONTROL_FEATURE_NAMES_EXTENDED_V1 = CONTROL_FEATURE_NAMES_BASE + (
    "top1_distance_to_first_waypoint",
    "top1_nearest_waypoint_index_norm",
    "top1_along_path_progress_norm",
    "top1_local_path_curvature_rad",
    "top1_heading_alignment_cos",
)


@dataclass(frozen=True)
class ControlSettingsDecision:
    """Structured output for control-settings recommendations."""

    speed_instruction: str
    steering_instruction: str
    object_ids: tuple[str, ...]


def parse_speed_steering_idx(answer: str) -> tuple[int, int]:
    text = answer.lower()
    if "very slow" in text:
        speed_idx = 3
    elif "moderate" in text:
        speed_idx = 1
    elif "fast" in text:
        speed_idx = 0
    elif "slow" in text:
        speed_idx = 2
    elif "stop" in text:
        speed_idx = 4
    else:
        speed_idx = 4

    if "slightly left" in text:
        steering_idx = 1
    elif "slightly right" in text:
        steering_idx = 3
    elif "straight" in text:
        steering_idx = 2
    elif "left" in text:
        steering_idx = 0
    elif "right" in text:
        steering_idx = 4
    else:
        steering_idx = 2
    return speed_idx, steering_idx


def visibility_lookup(scene, asker_agent_id: str) -> dict[str, VisibilityState]:
    return {
        fact.object_id: fact.state
        for fact in scene.visibility_facts
        if fact.agent_id == asker_agent_id
    }


def distance_to_trajectory(scene, object_track) -> float:
    if not scene.future_trajectory.points:
        return float("inf")
    return min(
        dist(
            (object_track.position.x, object_track.position.y),
            (point.x, point.y),
        )
        for point in scene.future_trajectory.points
    )


def distance_to_asker(scene, object_track) -> float:
    asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
    if asker is None:
        return float("inf")
    return dist(
        (object_track.position.x, object_track.position.y),
        (asker.pose.position.x, asker.pose.position.y),
    )


def control_risk_score(scene, object_track, visibility_by_object: dict[str, VisibilityState]) -> float:
    distance_to_traj = distance_to_trajectory(scene, object_track)
    distance_to_asker_value = distance_to_asker(scene, object_track)
    visibility_state = visibility_by_object.get(object_track.object_id)
    score = 1.0 / (1.0 + distance_to_traj)
    score += 0.5 / (1.0 + distance_to_asker_value)
    if visibility_state == VisibilityState.OCCLUDED:
        score += 0.20
    elif visibility_state == VisibilityState.UNCERTAIN:
        score += 0.10
    if object_track.status == TrackStatus.CANDIDATE:
        score -= 0.05
    if scene.asker_agent_id in object_track.provenance.source_agent_ids:
        score += 0.05
    if len(object_track.provenance.source_agent_ids) >= 2:
        score += 0.05
    return score


def rank_control_candidates(scene) -> tuple[tuple[object, float], ...]:
    visibility_by_object = visibility_lookup(scene, scene.asker_agent_id)
    ranked = sorted(
        scene.object_tracks,
        key=lambda object_track: (
            -control_risk_score(scene, object_track, visibility_by_object),
            distance_to_trajectory(scene, object_track),
            object_track.object_id,
        ),
    )
    return tuple(
        (object_track, control_risk_score(scene, object_track, visibility_by_object))
        for object_track in ranked
    )


def select_top_control_objects(scene) -> tuple[object, ...]:
    ranked = rank_control_candidates(scene)
    top = tuple(
        object_track for object_track, score in ranked if score >= 0.35
    )[:2]
    if not top:
        fallback = tuple(object_track for object_track, _ in ranked[:1])
        return fallback
    return top


def _base_control_feature_vector(scene) -> tuple[float, ...]:
    visibility_by_object = visibility_lookup(scene, scene.asker_agent_id)
    ranked = rank_control_candidates(scene)
    top = select_top_control_objects(scene)
    asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
    if asker is None or not top:
        return (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    top1 = top[0]
    top1_risk = control_risk_score(scene, top1, visibility_by_object)
    top2_risk = 0.0
    if len(top) > 1:
        top2 = top[1]
        top2_risk = control_risk_score(scene, top2, visibility_by_object)
    elif len(ranked) > 1:
        top2_risk = ranked[1][1]

    top1_dist_traj = distance_to_trajectory(scene, top1)
    top1_dist_asker = distance_to_asker(scene, top1)
    top1_visibility = visibility_by_object.get(top1.object_id)
    lateral_offset = top1.position.y - asker.pose.position.y
    source_agents = top1.provenance.source_agent_ids
    support_count = float(len(source_agents))
    return (
        top1_risk,
        top2_risk,
        top1_risk - top2_risk,
        top1_dist_traj,
        top1_dist_asker,
        float(top1.confidence),
        float(top1.uncertainty_score),
        float(top1.conflict_score),
        support_count,
        1.0 if top1_visibility == VisibilityState.VISIBLE else 0.0,
        1.0 if top1_visibility == VisibilityState.UNCERTAIN else 0.0,
        1.0 if top1_visibility == VisibilityState.OCCLUDED else 0.0,
        1.0 if top1.status == TrackStatus.CANDIDATE else 0.0,
        1.0 if scene.asker_agent_id in source_agents else 0.0,
        1.0 if len(source_agents) >= 2 else 0.0,
        lateral_offset,
        abs(lateral_offset),
        float(len(top)),
    )


def _trajectory_geometry_features(scene, asker, top_object) -> tuple[float, float, float, float, float]:
    points = scene.future_trajectory.points
    if not points:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    first_wp = points[0]
    dist_to_first = dist((top_object.position.x, top_object.position.y), (first_wp.x, first_wp.y))

    nearest_idx = 0
    nearest_dist = float("inf")
    for idx, point in enumerate(points):
        candidate_dist = dist((top_object.position.x, top_object.position.y), (point.x, point.y))
        if candidate_dist < nearest_dist:
            nearest_dist = candidate_dist
            nearest_idx = idx

    index_norm = float(nearest_idx) / float(max(len(points) - 1, 1))

    cumulative = [0.0]
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        curr = points[idx]
        cumulative.append(cumulative[-1] + dist((prev.x, prev.y), (curr.x, curr.y)))
    total_length = cumulative[-1]
    progress_norm = 0.0 if total_length <= 1e-6 else cumulative[nearest_idx] / total_length

    curvature = 0.0
    if 0 < nearest_idx < len(points) - 1:
        prev = points[nearest_idx - 1]
        curr = points[nearest_idx]
        nxt = points[nearest_idx + 1]
        v1x = curr.x - prev.x
        v1y = curr.y - prev.y
        v2x = nxt.x - curr.x
        v2y = nxt.y - curr.y
        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y
        curvature = abs(atan2(cross, dot))

    vx = top_object.position.x - asker.pose.position.x
    vy = top_object.position.y - asker.pose.position.y
    norm = hypot(vx, vy)
    if norm <= 1e-6:
        heading_alignment = 1.0
    else:
        ux = vx / norm
        uy = vy / norm
        heading_alignment = cos(asker.pose.yaw_radians) * ux + sin(asker.pose.yaw_radians) * uy

    return (
        float(dist_to_first),
        float(index_norm),
        float(progress_norm),
        float(curvature),
        float(heading_alignment),
    )


def build_control_feature_vector(scene, feature_set: str = "base") -> tuple[float, ...]:
    base = _base_control_feature_vector(scene)
    if feature_set == "base":
        return base
    if feature_set == "extended_v1":
        top = select_top_control_objects(scene)
        asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
        if asker is None or not top:
            return base + (0.0, 0.0, 0.0, 0.0, 0.0)
        geo = _trajectory_geometry_features(scene, asker, top[0])
        return base + geo
    raise ValueError(f"Unsupported control feature set: {feature_set}")


def control_feature_names(feature_set: str = "base") -> tuple[str, ...]:
    if feature_set == "base":
        return CONTROL_FEATURE_NAMES_BASE
    if feature_set == "extended_v1":
        return CONTROL_FEATURE_NAMES_EXTENDED_V1
    raise ValueError(f"Unsupported control feature set: {feature_set}")


def decide_control_settings(scene, selection_policy: str = "rule", model: dict[str, object] | None = None) -> ControlSettingsDecision:
    if selection_policy == "linear_classifier":
        model_decision = _decide_control_with_linear_model(scene, model or {})
        if model_decision is not None:
            return model_decision
    return _decide_control_with_rule(scene)


def _decide_control_with_rule(scene) -> ControlSettingsDecision:
    asker = next((agent for agent in scene.agents if agent.agent_id == scene.asker_agent_id), None)
    if asker is None:
        return ControlSettingsDecision(
            speed_instruction="moderate",
            steering_instruction="straight",
            object_ids=(),
        )

    visibility_by_object = visibility_lookup(scene, scene.asker_agent_id)
    top_objects = select_top_control_objects(scene)
    if not top_objects:
        return ControlSettingsDecision(
            speed_instruction="moderate",
            steering_instruction="straight",
            object_ids=(),
        )

    top_object = top_objects[0]
    min_distance_to_trajectory = distance_to_trajectory(scene, top_object)
    visibility_state = visibility_by_object.get(top_object.object_id)

    if min_distance_to_trajectory <= 4.0 or visibility_state == VisibilityState.OCCLUDED:
        speed_instruction = "very slow"
    elif min_distance_to_trajectory <= 8.0 or visibility_state == VisibilityState.UNCERTAIN:
        speed_instruction = "slow"
    else:
        speed_instruction = "moderate"

    steering_instruction = _steering_label(asker, top_object)
    return ControlSettingsDecision(
        speed_instruction=speed_instruction,
        steering_instruction=steering_instruction,
        object_ids=tuple(object_track.object_id for object_track in top_objects),
    )


def _decide_control_with_linear_model(scene, model: dict[str, object]) -> ControlSettingsDecision | None:
    if model.get("model_type") not in {
        "phase9_q8_control_linear_classifier_v1",
        "phase9_q8_control_linear_classifier_v2",
    }:
        return None
    speed_weights = model.get("speed_weights")
    speed_ordinal_weights = model.get("speed_ordinal_weights")
    speed_head_type = str(model.get("speed_head_type", "multiclass"))
    speed_ordinal_threshold = float(model.get("speed_ordinal_threshold", 0.5))
    speed_ordinal_threshold_policy = str(model.get("speed_ordinal_threshold_policy", "global"))
    speed_ordinal_thresholds = model.get("speed_ordinal_thresholds", {})
    speed_risk_split_low = float(model.get("speed_risk_split_low", 0.2))
    speed_risk_split_high = float(model.get("speed_risk_split_high", 0.5))
    feature_set = str(model.get("feature_set", "base"))
    steering_weights = model.get("steering_weights")
    feature_mean = model.get("feature_mean")
    feature_std = model.get("feature_std")
    if not (
        isinstance(steering_weights, list)
        and isinstance(feature_mean, list)
        and isinstance(feature_std, list)
    ):
        return None

    try:
        expected_feature_names = control_feature_names(feature_set)
    except ValueError:
        return None
    raw_features = build_control_feature_vector(scene, feature_set=feature_set)
    if "feature_names" in model:
        model_feature_names = model.get("feature_names")
        if isinstance(model_feature_names, list) and tuple(model_feature_names) != expected_feature_names:
            return None
    if len(raw_features) != len(feature_mean) or len(raw_features) != len(feature_std):
        return None
    normalized = tuple(
        (value - float(mean)) / max(float(std), 1e-6)
        for value, mean, std in zip(raw_features, feature_mean, feature_std)
    )
    model_input = (1.0,) + normalized
    steering_scores = _linear_head_scores(model_input, steering_weights)
    if steering_scores is None:
        return None

    if speed_head_type == "ordinal":
        if not isinstance(speed_ordinal_weights, list):
            return None
        ordinal_scores = _linear_head_scores(model_input, speed_ordinal_weights)
        if ordinal_scores is None:
            return None
        if (
            speed_ordinal_threshold_policy == "risk3"
            and isinstance(speed_ordinal_thresholds, dict)
            and raw_features
        ):
            top1_risk = float(raw_features[0])
            if top1_risk < speed_risk_split_low:
                speed_ordinal_threshold = float(
                    speed_ordinal_thresholds.get("low", speed_ordinal_threshold)
                )
            elif top1_risk < speed_risk_split_high:
                speed_ordinal_threshold = float(
                    speed_ordinal_thresholds.get("mid", speed_ordinal_threshold)
                )
            else:
                speed_ordinal_threshold = float(
                    speed_ordinal_thresholds.get("high", speed_ordinal_threshold)
                )
        ordinal_probs = tuple(1.0 / (1.0 + exp(-max(-30.0, min(30.0, score)))) for score in ordinal_scores)
        speed_idx = sum(prob > speed_ordinal_threshold for prob in ordinal_probs)
        speed_idx = max(0, min(speed_idx, len(SPEED_CLASSES) - 1))
    else:
        if not isinstance(speed_weights, list):
            return None
        speed_scores = _linear_head_scores(model_input, speed_weights)
        if speed_scores is None:
            return None
        speed_idx = max(range(len(speed_scores)), key=speed_scores.__getitem__)

    steering_idx = max(range(len(steering_scores)), key=steering_scores.__getitem__)
    top_objects = select_top_control_objects(scene)
    return ControlSettingsDecision(
        speed_instruction=SPEED_CLASSES[speed_idx],
        steering_instruction=STEERING_CLASSES[steering_idx],
        object_ids=tuple(object_track.object_id for object_track in top_objects[:2]),
    )


def _linear_head_scores(features: tuple[float, ...], weights: list[object]) -> list[float] | None:
    scores: list[float] = []
    for row in weights:
        if not isinstance(row, list) or len(row) != len(features):
            return None
        scores.append(sum(float(weight) * value for weight, value in zip(row, features)))
    return scores


def _steering_label(asker, object_track) -> str:
    lateral_offset = object_track.position.y - asker.pose.position.y
    if lateral_offset > 1.0:
        return "right"
    if lateral_offset > 0.2:
        return "slightly right"
    if lateral_offset < -1.0:
        return "left"
    if lateral_offset < -0.2:
        return "slightly left"
    return "straight"
