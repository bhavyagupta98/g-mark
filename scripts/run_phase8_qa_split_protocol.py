#!/usr/bin/env python3

try:
    from scripts.run_qa_split_pipeline import main
except ModuleNotFoundError:
    from run_qa_split_pipeline import main


if __name__ == "__main__":
    main()
