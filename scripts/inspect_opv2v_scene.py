#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.infrastructure.opv2v_scene_adapter import OPV2VFrameRef, OPV2VSceneAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one OPV2V frame and export read-only G-MARK scene snapshots."
    )
    parser.add_argument(
        "--data-root",
        default="/workspace/repos/OpenCOOD/opv2v_data_dumping",
        help="OPV2V data root containing split folders such as test/validate/train.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--ego-agent-id", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/opv2v_inspection",
        help="Directory for inspection JSON outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = OPV2VSceneAdapter(args.data_root)

    if args.scenario_id is None or args.timestamp is None:
        frame_ref = adapter.first_frame(split=args.split)
    else:
        agent_ids = tuple(adapter.list_agents(args.split, args.scenario_id))
        if not agent_ids:
            raise FileNotFoundError(f"No agents found for {args.split}/{args.scenario_id}")
        frame_ref = OPV2VFrameRef(
            split=args.split,
            scenario_id=args.scenario_id,
            timestamp=args.timestamp,
            ego_agent_id=args.ego_agent_id or agent_ids[0],
            agent_ids=agent_ids,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inspection = adapter.inspect_frame(frame_ref)
    cooperative_scene = adapter.build_scene(frame_ref, mode="cooperative")
    ego_scene = adapter.build_scene(frame_ref, mode="ego_only")

    stem = f"{frame_ref.split}_{frame_ref.scenario_id}_{frame_ref.timestamp}"
    inspection_path = output_dir / f"{stem}_inspection.json"
    cooperative_path = output_dir / f"{stem}_cooperative_scene.json"
    ego_path = output_dir / f"{stem}_ego_only_scene.json"

    inspection_path.write_text(
        json.dumps(inspection, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    adapter.write_scene_json(cooperative_scene, cooperative_path)
    adapter.write_scene_json(ego_scene, ego_path)

    print(f"data_root: {adapter.data_root}")
    print(f"split: {frame_ref.split}")
    print(f"scenario_id: {frame_ref.scenario_id}")
    print(f"timestamp: {frame_ref.timestamp}")
    print(f"ego_agent_id: {frame_ref.ego_agent_id}")
    print(f"agent_ids: {', '.join(frame_ref.agent_ids)}")
    print(f"agent_count: {inspection['agent_count']}")
    print(f"ego_vehicle_count: {inspection['ego_vehicle_count']}")
    print(f"all_unique_vehicle_count: {inspection['all_unique_vehicle_count']}")
    print(f"partner_only_vehicle_count: {inspection['partner_only_vehicle_count']}")
    print(f"cooperative_observation_count: {len(cooperative_scene.observations)}")
    print(f"ego_observation_count: {len(ego_scene.observations)}")
    print(f"saved_inspection: {inspection_path}")
    print(f"saved_cooperative_scene: {cooperative_path}")
    print(f"saved_ego_only_scene: {ego_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
