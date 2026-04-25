# Phase 5 Scored Report

- `manifest`: `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/phase8_baselines/phase8_qa_baseline_manifest.json`
- `repository_root`: `/Users/bhavya/Desktop/ms_projects/V2V-GoT`
- `distance_threshold`: `3.0`

Scores below are object-level metrics derived by resolving benchmark reference answer coordinates back to cooperative-scene object IDs. They are closer to the paper's QA F1 framing than the earlier structural diff summaries, but they are still our local reproduction layer rather than an official benchmark script.

## notable_objects

Published target context:
- `V2V-GoT` `Q1 F1` = `52.5` (higher is better, Table II)
  Visible notable objects F1.

Best local scenario by object-level F1: `phase8_cooperative_baseline`

| Scenario | Mode | Ranker | Policy | Exact | Precision | Recall | F1 | TP | FP | FN | Resolved Ref | Unresolved Ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase8_cooperative_baseline | cooperative | heuristic | default | 67/100 | 0.000 | 0.000 | 0.000 | 0 | 57 | 0 | 0 | 50 |

## occluding_objects

Published target context:
- `V2V-GoT` `Q2 F1` = `30.1` (higher is better, Table II)
  Occluding objects F1.

Best local scenario by object-level F1: `phase8_cooperative_baseline`

| Scenario | Mode | Ranker | Policy | Exact | Precision | Recall | F1 | TP | FP | FN | Resolved Ref | Unresolved Ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase8_cooperative_baseline | cooperative | heuristic | default | 4/100 | 0.000 | 0.000 | 0.000 | 0 | 186 | 0 | 0 | 252 |

## invisible_objects

Published target context:
- `V2V-GoT` `Q3 F1` = `44.0` (higher is better, Table II)
  Invisible notable objects F1.

Best local scenario by object-level F1: `phase8_cooperative_baseline`

| Scenario | Mode | Ranker | Policy | Exact | Precision | Recall | F1 | TP | FP | FN | Resolved Ref | Unresolved Ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase8_cooperative_baseline | cooperative | heuristic | default | 94/100 | 0.000 | 0.000 | 0.000 | 0 | 6 | 0 | 0 | 7 |

## planning_awareness

Published target context:
- `V2V-GoT` `Q4 F1` = `60.8` (higher is better, Table II)
  Overall notable objects F1. This is the closest published QA metric to our current planning-awareness workstream.

Best local scenario by object-level F1: `phase8_cooperative_baseline`

| Scenario | Mode | Ranker | Policy | Exact | Precision | Recall | F1 | TP | FP | FN | Resolved Ref | Unresolved Ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase8_cooperative_baseline | cooperative | heuristic | default | 9/100 | 0.000 | 0.000 | 0.000 | 0 | 169 | 0 | 0 | 57 |
