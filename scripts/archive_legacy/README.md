# Legacy Script Archive

This folder stores scripts that are no longer part of the frozen train/val pipeline.

Rules:

- Scripts here are retained for reproducibility and historical inspection.
- Frozen baseline workflows should use the canonical top-level entrypoints:
  - `scripts/run_qa_split_pipeline.py`
  - `scripts/evaluate_qa_router.py`
  - `scripts/export_qa_predictions.py`
  - `scripts/evaluate_official_qa.py`
  - `scripts/train_q3_invisible_acceptor.py`
  - `scripts/train_q4_planning_acceptor.py`
  - `scripts/configure_q4_trajectory_calibration.py`
  - `scripts/train_q8_control_policy.py`
  - `scripts/train_q9_future_trajectory_regressor.py`

Archive layout:

- `phase1_4_demos/`: early-phase demo and validation scripts.
- `phase8_experiments/`: phase-8 experiment sweeps and analysis scripts not used by frozen baselines.
