# Phase 8 Occluding Mismatch Inspection

- `manifest`: `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/phase8_baselines/phase8_qa_baseline_manifest.json`
- `prediction_path`: `/Users/bhavya/Desktop/ms_projects/kg_coop_drive/outputs/phase8_baselines/occluding_objects_cooperative_limit100.jsonl`
- `repository_root`: `/Users/bhavya/Desktop/ms_projects/V2V-GoT`
- `split`: `val`

This report buckets archived occluding-object predictions by reference/predicted answer count. It is an inspection aid for Phase 8 selector work, not an official benchmark score.

## Summary

| Metric | Value |
| --- | --- |
| Inspected samples | 100 |
| Reference coordinate mentions | 252 |
| Predicted object mentions | 186 |
| Exact count matches | 39 |
| Under-predicted counts | 61 |
| Over-predicted counts | 0 |
| Empty predictions with reference | 4 |
| Predictions without reference | 0 |

## Count Histograms

| Reference Count | Samples |
| --- | --- |
| 2 | 48 |
| 3 | 52 |

| Predicted Count | Samples |
| --- | --- |
| 0 | 4 |
| 1 | 6 |
| 2 | 90 |

## under_predicted_count

- `sample_id=0` ref_count=`2` pred_count=`1` pred_ids=`['1']`
  - reference: There is a car at (-21.1,1.5) obstructing your view. There is a car at (-20.5,-0.1) obstructing your view. 
  - predicted: Potentially occluding objects: 1.
- `sample_id=1` ref_count=`2` pred_count=`1` pred_ids=`['107']`
  - reference: There is a car at (-21.1,1.5) obstructing your view. There is a car at (-20.5,-0.1) obstructing your view. 
  - predicted: Potentially occluding objects: 107.
- `sample_id=2` ref_count=`2` pred_count=`1` pred_ids=`['1']`
  - reference: There is a car at (-55.4,24.9) obstructing your view. There is a car at (-20.4,-0.1) obstructing your view. 
  - predicted: Potentially occluding objects: 1.
- `sample_id=5` ref_count=`2` pred_count=`1` pred_ids=`['1']`
  - reference: There is a car at (-55.4,23.9) obstructing your view. There is a car at (-20.3,-0.0) obstructing your view. 
  - predicted: Potentially occluding objects: 1.
- `sample_id=7` ref_count=`2` pred_count=`1` pred_ids=`['1']`
  - reference: There is a car at (-54.7,23.6) obstructing your view. There is a car at (-20.2,-0.0) obstructing your view. 
  - predicted: Potentially occluding objects: 1.
- `sample_id=28` ref_count=`2` pred_count=`0` pred_ids=`[]`
  - reference: There is a car at (-47.8,20.7) obstructing your view. There is a car at (-19.6,-0.0) obstructing your view. 
  - predicted: There is no object currently marked as obstructing your view.
- `sample_id=29` ref_count=`2` pred_count=`0` pred_ids=`[]`
  - reference: There is a car at (-47.8,20.7) obstructing your view. There is a car at (-19.6,-0.0) obstructing your view. 
  - predicted: There is no object currently marked as obstructing your view.
- `sample_id=32` ref_count=`2` pred_count=`0` pred_ids=`[]`
  - reference: There is a car at (-46.7,20.0) obstructing your view. There is a car at (-19.5,-0.0) obstructing your view. 
  - predicted: There is no object currently marked as obstructing your view.

## over_predicted_count

No examples captured.

## empty_prediction_with_reference

- `sample_id=28` ref_count=`2` pred_count=`0` pred_ids=`[]`
  - reference: There is a car at (-47.8,20.7) obstructing your view. There is a car at (-19.6,-0.0) obstructing your view. 
  - predicted: There is no object currently marked as obstructing your view.
- `sample_id=29` ref_count=`2` pred_count=`0` pred_ids=`[]`
  - reference: There is a car at (-47.8,20.7) obstructing your view. There is a car at (-19.6,-0.0) obstructing your view. 
  - predicted: There is no object currently marked as obstructing your view.
- `sample_id=32` ref_count=`2` pred_count=`0` pred_ids=`[]`
  - reference: There is a car at (-46.7,20.0) obstructing your view. There is a car at (-19.5,-0.0) obstructing your view. 
  - predicted: There is no object currently marked as obstructing your view.
- `sample_id=33` ref_count=`2` pred_count=`0` pred_ids=`[]`
  - reference: There is a car at (-46.7,20.0) obstructing your view. There is a car at (-19.5,-0.0) obstructing your view. 
  - predicted: There is no object currently marked as obstructing your view.

## prediction_without_reference

No examples captured.
