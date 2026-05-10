#!/usr/bin/env python3

from __future__ import annotations

try:
    from scripts.train_q6_agent_motion_notability import main
except ModuleNotFoundError:
    from train_q6_agent_motion_notability import main


if __name__ == "__main__":
    main()

