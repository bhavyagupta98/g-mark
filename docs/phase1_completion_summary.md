# Phase 1 Completion Summary

- Repository root inspected: `/Users/bhavya/Desktop/ms_projects/V2V-GoT`
- co_llm root family: `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models`
- Available splits: val, train

## Real File Inspection

- `val` / `v2v4real_3d_grounding_qa_dataset_v2vgot.json`
  records: 31014
  keys: id, conversations, scenario_index, local_timestamp_index, global_timestamp_index, qa_sub_type, distance_to_waypoint, future_time, future_trajectory_str_in_ego, future_trajectory_str_in_self, asker_cav_id, cav_ego_lidar_pose, cav_1_lidar_pose, qa_source, qa_type_id
  conversation roles: human, gpt
  qa_type_id: 11
  qa_source: nq1sm3w0d
  question preview: I am CAV_EGO at (0.0,0.0). What are the notable objects visible to me near my planned future trajectory [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]? 
  answer preview: There is no notable object visible to you.

- `val` / `v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json`
  records: 3446
  keys: id, conversations, scenario_index, local_timestamp_index, global_timestamp_index, qa_sub_type, distance_to_waypoint, future_time, future_trajectory_str_in_ego, future_trajectory_str_in_self, asker_cav_id, cav_ego_lidar_pose, cav_1_lidar_pose
  conversation roles: human, gpt
  question preview: I am CAV_EGO at (0.0,0.0). What are the notable objects visible to me near my planned future trajectory [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]? 
  answer preview: There is no notable object visible to you.

- `train` / `v2v4real_3d_grounding_qa_dataset_v2vgot.json`
  records: 110610
  keys: id, conversations, scenario_index, local_timestamp_index, global_timestamp_index, qa_sub_type, distance_to_waypoint, future_time, future_trajectory_str_in_ego, future_trajectory_str_in_self, asker_cav_id, cav_ego_lidar_pose, cav_1_lidar_pose, qa_source, qa_type_id
  conversation roles: human, gpt
  qa_type_id: 11
  qa_source: nq1sm3w0d
  question preview: I am CAV_EGO at (0.0,0.0). What are the notable objects visible to me near my planned future trajectory [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]? 
  answer preview: There is no notable object visible to you.

- `train` / `v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json`
  records: 12290
  keys: id, conversations, scenario_index, local_timestamp_index, global_timestamp_index, qa_sub_type, distance_to_waypoint, future_time, future_trajectory_str_in_ego, future_trajectory_str_in_self, asker_cav_id, cav_ego_lidar_pose, cav_1_lidar_pose
  conversation roles: human, gpt
  question preview: I am CAV_EGO at (0.0,0.0). What are the notable objects visible to me near my planned future trajectory [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]? 
  answer preview: There is no notable object visible to you.

## Recommended First Task Slice

- object existence
- object count
- relative position
- visible notable object queries

## Bootstrap Artifacts

- `dataset_jsons.zip`
  kind: benchmark JSON archive
  purpose: Provides V2V-GoT and nq* co_llm QA datasets used by training, inference, and evaluation scripts.
  expected location: `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models`

- `dataset_processed_features_and_gt.zip`
  kind: processed perception data archive
  purpose: Provides processed perception features, point clouds, detections, and ground-truth assets expected by the original V2V-GoT pipeline.
  expected location: `/Users/bhavya/Desktop/ms_projects/V2V-GoT/DMSTrack/V2V4Real/official_models`

- `model_ckpt.zip`
  kind: model checkpoint archive
  purpose: Provides pretrained V2V-GoT and V2V-LLM task checkpoints under the LLaVA checkpoints tree for reproduction and inference.
  expected location: `/Users/bhavya/Desktop/ms_projects/V2V-GoT/LLaVA/checkpoints`

## Conclusions

- The real co_llm benchmark files are now available and readable from the expected official_models paths.
- Representative v2vgot and nq1 records share a stable JSON structure with two-turn conversations and explicit scenario/timestamp metadata.
- The first KG prototype can safely target existence, count, relative-position, and visible-object queries before tackling planning-style outputs.

## Phase 1 Status

Phase 1 can be considered complete after reviewing and accepting the recommended first task slice above.

