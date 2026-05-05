#!/usr/bin/env python3

try:
    from scripts.train_q9_future_trajectory_regressor import main
except ModuleNotFoundError:
    from train_q9_future_trajectory_regressor import main


if __name__ == "__main__":
    main()
