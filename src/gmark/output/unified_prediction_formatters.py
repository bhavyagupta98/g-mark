from __future__ import annotations

from typing import Any


def format_object_grounding_answer(qa_type_id: int, object_ids: list[str]) -> str:
    if not object_ids:
        if qa_type_id == 12:
            return "There is no object obstructing your view."
        if qa_type_id == 13:
            return "There is no notable object invisible to you."
        if qa_type_id == 14:
            return "There is no notable object."
        return "There is no notable object visible to you."
    rendered = ", ".join(object_ids)
    if qa_type_id == 12:
        return f"Objects obstructing your view: {rendered}."
    if qa_type_id == 13:
        return f"Invisible objects to you: {rendered}."
    if qa_type_id == 14:
        return f"Objects to be aware of: {rendered}."
    return f"Visible notable objects: {rendered}."


def format_motion_answer(qa_type_id: int, object_id: str, pred: list[float], start_xy: tuple[float, float] | None = None) -> str:
    points = []
    for i in range(0, len(pred), 2):
        if i + 1 >= len(pred):
            break
        points.append((float(pred[i]), float(pred[i + 1])))

    # Official evaluator paths for motion/future tasks are sensitive to waypoint count.
    # Emit exactly 6 waypoints by truncating or repeating the last point.
    if not points:
        points = [(0.0, 0.0)]
    if len(points) > 6:
        points = points[:6]
    if len(points) < 6:
        points = points + [points[-1]] * (6 - len(points))

    if qa_type_id == 19:
        rendered = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in points)
        return f"The suggested future trajectory is [{rendered}]."

    if start_xy is None:
        start_xy = (0.0, 0.0)
    rendered_points = ",".join(f"({x:.1f},{y:.1f})" for x, y in points)
    if points:
        end_x, end_y = points[-1]
        dx = end_x - float(start_xy[0])
        dy = end_y - float(start_xy[1])
    else:
        dx = 0.0
        dy = 0.0
    speed = (dx * dx + dy * dy) ** 0.5
    if speed < 0.1:
        motion = "staying at the same location"
    elif abs(dy) > abs(dx):
        motion = "turning right" if dy >= 0.0 else "turning left"
    elif dx >= 0.0:
        motion = "moving forward"
    else:
        motion = "turning right" if dy >= 0.0 else "turning left"
    return (
        f"Predicted object motion: {object_id}={motion} "
        f"from ({start_xy[0]:.1f}, {start_xy[1]:.1f}) trajectory [{rendered_points}]."
    )


def format_q6_answer(prob_or_label: float, threshold: float = 0.5) -> str:
    positive = float(prob_or_label) >= float(threshold)
    if positive:
        return "The other agent is a notable object based on its trajectory interaction."
    return "The other agent is a not notable object based on its trajectory interaction."


def format_q8_answer(speed_label: str, steering_label: str) -> str:
    return (
        f"The suggested speed setting is: {speed_label}. "
        f"The suggested steering setting is: {steering_label}."
    )


def base_prediction_record(
    *,
    sample_id: str,
    qa_type_id: int,
    task_type: str,
    answer_text: str,
    object_ids: list[str] | tuple[str, ...],
    supported: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "sample_id": str(sample_id),
        "qa_type_id": int(qa_type_id),
        "task_type": str(task_type),
        "supported": bool(supported),
        "answer_text": str(answer_text),
        "object_ids": [str(x) for x in object_ids],
    }
    if extra:
        record.update(extra)
    return record
