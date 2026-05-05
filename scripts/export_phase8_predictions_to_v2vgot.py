#!/usr/bin/env python3

try:
    from scripts.export_qa_predictions import main
except ModuleNotFoundError:
    from export_qa_predictions import main


if __name__ == "__main__":
    main()
