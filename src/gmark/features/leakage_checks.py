from __future__ import annotations

from typing import Iterable

Q9_LEAKAGE_FIELDS = {
    "dist",
    "angle",
    "suggested_speed_idx",
    "suggested_steering_idx",
    "future_trajectory_str_in_ego",
    "future_trajectory_str_in_self",
}

GENERIC_TARGET_LEAKAGE_TOKENS = (
    "label",
    "target",
    "ground_truth",
    "gt_",
    "answer",
    "future_trajectory",
)


def assert_no_leakage_features(
    feature_names: Iterable[str],
    qa_type_id: int | None,
    strict: bool = True,
) -> None:
    names = [str(name) for name in feature_names]
    lower_names = {name.lower() for name in names}

    if int(qa_type_id or -1) == 19:
        present_q9 = sorted(name for name in Q9_LEAKAGE_FIELDS if name in lower_names)
        if present_q9:
            message = f"Q9 leakage-risk features found in model input: {present_q9}"
            if strict:
                raise ValueError(message)

    present_generic: list[str] = []
    for name in lower_names:
        if any(token in name for token in GENERIC_TARGET_LEAKAGE_TOKENS):
            if name not in Q9_LEAKAGE_FIELDS:
                present_generic.append(name)
    if present_generic:
        message = (
            "Potential target/reference leakage feature names found in model input: "
            f"{sorted(set(present_generic))}"
        )
        if strict:
            raise ValueError(message)
