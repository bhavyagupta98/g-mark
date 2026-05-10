#!/usr/bin/env python3

from __future__ import annotations

try:
    from scripts.train_q5_object_motion_predictor import main
except ModuleNotFoundError:
    from train_q5_object_motion_predictor import main


if __name__ == "__main__":
    main()
