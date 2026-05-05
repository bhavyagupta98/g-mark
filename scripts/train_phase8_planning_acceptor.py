#!/usr/bin/env python3

try:
    from scripts.train_q4_planning_acceptor import main
except ModuleNotFoundError:
    from train_q4_planning_acceptor import main


if __name__ == "__main__":
    main()
