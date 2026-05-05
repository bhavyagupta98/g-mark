#!/usr/bin/env python3

try:
    from scripts.train_q8_control_policy import main
except ModuleNotFoundError:
    from train_q8_control_policy import main


if __name__ == "__main__":
    main()
