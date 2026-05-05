#!/usr/bin/env python3

try:
    from scripts.evaluate_qa_router import *  # type: ignore[F401,F403]
    from scripts.evaluate_qa_router import main
except ModuleNotFoundError:
    from evaluate_qa_router import *  # type: ignore[F401,F403]
    from evaluate_qa_router import main


if __name__ == "__main__":
    main()
