# Phase 8 QA Proxy Analysis

- `manifest`: `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/phase8_baselines/phase8_qa_baseline_manifest.json`
- `repository_root`: `/Users/bhavya/Desktop/ms_projects/V2V-GoT`
- `split`: `val`

This is a local proxy analysis, not an official benchmark scorer. It uses coordinate mentions in benchmark reference answers plus empty-vs-positive matching to estimate which QA tasks are most promising for improvement.

## notable_objects

| Task | Presence F1 | Precision | Recall | Ref Positive | Pred Positive | FN | FP | Presence Match | Count Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| notable_objects | 0.795 | 1.000 | 0.660 | 50 | 33 | 17 | 0 | 83/100 | 59/100 |

Representative false negatives:
- `sample_id=1` reference=`There is a car at (-21.1,1.5) visible to you. `
- `sample_id=3` reference=`There is a car at (-20.4,-0.1) visible to you. `
- `sample_id=5` reference=`There is a car at (-20.3,-0.0) visible to you. `
- `sample_id=7` reference=`There is a car at (-20.2,-0.0) visible to you. `
- `sample_id=9` reference=`There is a car at (-20.2,-0.0) visible to you. `

## occluding_objects

| Task | Presence F1 | Precision | Recall | Ref Positive | Pred Positive | FN | FP | Presence Match | Count Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occluding_objects | 0.980 | 1.000 | 0.960 | 100 | 96 | 4 | 0 | 96/100 | 39/100 |

Representative false negatives:
- `sample_id=28` reference=`There is a car at (-47.8,20.7) obstructing your view. There is a car at (-19.6,-0.0) obstructing your view. `
- `sample_id=29` reference=`There is a car at (-47.8,20.7) obstructing your view. There is a car at (-19.6,-0.0) obstructing your view. `
- `sample_id=32` reference=`There is a car at (-46.7,20.0) obstructing your view. There is a car at (-19.5,-0.0) obstructing your view. `
- `sample_id=33` reference=`There is a car at (-46.7,20.0) obstructing your view. There is a car at (-19.5,-0.0) obstructing your view. `

## invisible_objects

| Task | Presence F1 | Precision | Recall | Ref Positive | Pred Positive | FN | FP | Presence Match | Count Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| invisible_objects | 0.923 | 1.000 | 0.857 | 7 | 6 | 1 | 0 | 99/100 | 99/100 |

Representative false negatives:
- `sample_id=1` reference=`There is a car at (-20.5,-0.1) invisible to you. `

## planning_awareness

| Task | Presence F1 | Precision | Recall | Ref Positive | Pred Positive | FN | FP | Presence Match | Count Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planning_awareness | 0.709 | 0.549 | 1.000 | 50 | 91 | 0 | 41 | 59/100 | 10/100 |

Representative false positives:
- `sample_id=0` predicted_ids=`['107', '1']`
- `sample_id=2` predicted_ids=`['1']`
- `sample_id=4` predicted_ids=`['1']`
- `sample_id=6` predicted_ids=`['1']`
- `sample_id=8` predicted_ids=`['1']`
