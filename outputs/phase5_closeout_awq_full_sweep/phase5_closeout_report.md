# Phase 5 Closeout

- `repository_root`: `/workspace/repos/V2V-GoT`
- `split`: `val`
- `limit`: `100`
- `full_sweep_all_tasks`: `True`
- `tasks`: `notable_objects, occluding_objects, invisible_objects, planning_awareness`

Published references below are target context from the V2V-GoT paper. Our current outputs are structural prediction comparisons, not reproduced benchmark F1/L2 scores.

## notable_objects

Full scenario sweep requested. Under the current router, this task does not consume the planning-awareness ranker/policy, so scenario-to-scenario differences should mainly reflect `cooperative` vs `ego_only` preparation rather than the ranking method.

Published target context:
- `V2V-GoT` `Q1 F1` = `52.5` (higher is better, Table II)
  Visible notable objects F1.

Baseline scenario: `risk_diverse_top2_cooperative`

| Scenario | Mode | Ranker | Policy | Supported | Exact vs Baseline | Set vs Baseline | Semantic Diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| risk_diverse_top2_cooperative | cooperative | risk_aware | diverse_top2 | 100/100 | 100/100 | 100/100 | 0 |
| energy_cooperative | cooperative | energy_based | default | 100/100 | 100/100 | 100/100 | 0 |
| heuristic_cooperative | cooperative | heuristic | default | 100/100 | 100/100 | 100/100 | 0 |
| heuristic_ego_only | ego_only | heuristic | default | 100/100 | 85/100 | 85/100 | 15 |
| llm_top2_cooperative | cooperative | llm | top2 | 100/100 | 100/100 | 100/100 | 0 |
| llm_top2_ego_only | ego_only | llm | top2 | 100/100 | 85/100 | 85/100 | 15 |
| relational_cooperative | cooperative | relational_importance | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_default_cooperative | cooperative | risk_aware | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_default_ego_only | ego_only | risk_aware | default | 100/100 | 85/100 | 85/100 | 15 |
| risk_diverse_top2_ego_only | ego_only | risk_aware | diverse_top2 | 100/100 | 85/100 | 85/100 | 15 |
| risk_top2_cooperative | cooperative | risk_aware | top2 | 100/100 | 100/100 | 100/100 | 0 |
| risk_top2_ego_only | ego_only | risk_aware | top2 | 100/100 | 85/100 | 85/100 | 15 |

## occluding_objects

Full scenario sweep requested. Under the current router, this task does not consume the planning-awareness ranker/policy, so scenario-to-scenario differences should mainly reflect `cooperative` vs `ego_only` preparation rather than the ranking method.

Published target context:
- `V2V-GoT` `Q2 F1` = `30.1` (higher is better, Table II)
  Occluding objects F1.

Baseline scenario: `risk_diverse_top2_cooperative`

| Scenario | Mode | Ranker | Policy | Supported | Exact vs Baseline | Set vs Baseline | Semantic Diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| risk_diverse_top2_cooperative | cooperative | risk_aware | diverse_top2 | 100/100 | 100/100 | 100/100 | 0 |
| energy_cooperative | cooperative | energy_based | default | 100/100 | 100/100 | 100/100 | 0 |
| heuristic_cooperative | cooperative | heuristic | default | 100/100 | 100/100 | 100/100 | 0 |
| heuristic_ego_only | ego_only | heuristic | default | 100/100 | 100/100 | 100/100 | 0 |
| llm_top2_cooperative | cooperative | llm | top2 | 100/100 | 100/100 | 100/100 | 0 |
| llm_top2_ego_only | ego_only | llm | top2 | 100/100 | 100/100 | 100/100 | 0 |
| relational_cooperative | cooperative | relational_importance | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_default_cooperative | cooperative | risk_aware | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_default_ego_only | ego_only | risk_aware | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_diverse_top2_ego_only | ego_only | risk_aware | diverse_top2 | 100/100 | 100/100 | 100/100 | 0 |
| risk_top2_cooperative | cooperative | risk_aware | top2 | 100/100 | 100/100 | 100/100 | 0 |
| risk_top2_ego_only | ego_only | risk_aware | top2 | 100/100 | 100/100 | 100/100 | 0 |

## invisible_objects

Full scenario sweep requested. Under the current router, this task does not consume the planning-awareness ranker/policy, so scenario-to-scenario differences should mainly reflect `cooperative` vs `ego_only` preparation rather than the ranking method.

Published target context:
- `V2V-GoT` `Q3 F1` = `44.0` (higher is better, Table II)
  Invisible notable objects F1.

Baseline scenario: `risk_diverse_top2_cooperative`

| Scenario | Mode | Ranker | Policy | Supported | Exact vs Baseline | Set vs Baseline | Semantic Diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| risk_diverse_top2_cooperative | cooperative | risk_aware | diverse_top2 | 100/100 | 100/100 | 100/100 | 0 |
| energy_cooperative | cooperative | energy_based | default | 100/100 | 100/100 | 100/100 | 0 |
| heuristic_cooperative | cooperative | heuristic | default | 100/100 | 100/100 | 100/100 | 0 |
| heuristic_ego_only | ego_only | heuristic | default | 100/100 | 100/100 | 100/100 | 0 |
| llm_top2_cooperative | cooperative | llm | top2 | 100/100 | 100/100 | 100/100 | 0 |
| llm_top2_ego_only | ego_only | llm | top2 | 100/100 | 100/100 | 100/100 | 0 |
| relational_cooperative | cooperative | relational_importance | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_default_cooperative | cooperative | risk_aware | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_default_ego_only | ego_only | risk_aware | default | 100/100 | 100/100 | 100/100 | 0 |
| risk_diverse_top2_ego_only | ego_only | risk_aware | diverse_top2 | 100/100 | 100/100 | 100/100 | 0 |
| risk_top2_cooperative | cooperative | risk_aware | top2 | 100/100 | 100/100 | 100/100 | 0 |
| risk_top2_ego_only | ego_only | risk_aware | top2 | 100/100 | 100/100 | 100/100 | 0 |

## planning_awareness

This task uses the planning-awareness orchestrator, so ranker and selection policy are active variables.

Published target context:
- `V2V-GoT` `Q4 F1` = `60.8` (higher is better, Table II)
  Overall notable objects F1. This is the closest published QA metric to our current planning-awareness workstream.

Baseline scenario: `risk_diverse_top2_cooperative`

| Scenario | Mode | Ranker | Policy | Supported | Exact vs Baseline | Set vs Baseline | Semantic Diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| risk_diverse_top2_cooperative | cooperative | risk_aware | diverse_top2 | 100/100 | 100/100 | 100/100 | 0 |
| energy_cooperative | cooperative | energy_based | default | 100/100 | 71/100 | 71/100 | 29 |
| heuristic_cooperative | cooperative | heuristic | default | 100/100 | 74/100 | 74/100 | 26 |
| heuristic_ego_only | ego_only | heuristic | default | 100/100 | 74/100 | 74/100 | 26 |
| llm_top2_cooperative | cooperative | llm | top2 | 100/100 | 100/100 | 100/100 | 0 |
| llm_top2_ego_only | ego_only | llm | top2 | 100/100 | 85/100 | 85/100 | 15 |
| relational_cooperative | cooperative | relational_importance | default | 100/100 | 68/100 | 68/100 | 32 |
| risk_default_cooperative | cooperative | risk_aware | default | 100/100 | 74/100 | 74/100 | 26 |
| risk_default_ego_only | ego_only | risk_aware | default | 100/100 | 74/100 | 74/100 | 26 |
| risk_diverse_top2_ego_only | ego_only | risk_aware | diverse_top2 | 100/100 | 85/100 | 85/100 | 15 |
| risk_top2_cooperative | cooperative | risk_aware | top2 | 100/100 | 100/100 | 100/100 | 0 |
| risk_top2_ego_only | ego_only | risk_aware | top2 | 100/100 | 85/100 | 85/100 | 15 |
