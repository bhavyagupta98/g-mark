#!/usr/bin/env python3

from __future__ import annotations

try:
    from scripts.train_q5_object_motion_predictor import ObjectMotionTrainerEntrypoint
except ModuleNotFoundError:  # pragma: no cover
    from train_q5_object_motion_predictor import ObjectMotionTrainerEntrypoint


class Q7ObjectMotionTrainer(ObjectMotionTrainerEntrypoint):
    qa_type_id = 17
    task_label = "Q7"


def main() -> None:
    Q7ObjectMotionTrainer().main()


if __name__ == "__main__":
    main()
