#!/usr/bin/env python3

try:
    from scripts.configure_q4_trajectory_calibration import main
except ModuleNotFoundError:
    from configure_q4_trajectory_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
