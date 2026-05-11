#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.infrastructure.opv2v_scene_adapter import OPV2VFrameRef, OPV2VSceneAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-inspect OPV2V frames for ego/cooperative/partner-only graph signal."
    )
    parser.add_argument(
        "--data-root",
        default="/workspace/repos/OpenCOOD/opv2v_data_dumping",
        help="OPV2V data root containing split folders such as test/validate/train.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Inspect every Nth timestamp within each scenario.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/opv2v_inspection/opv2v_batch_inspection_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = OPV2VSceneAdapter(args.data_root)
    scenarios = adapter.list_scenarios(args.split)
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]

    rows: list[dict[str, object]] = []
    for scenario_id in scenarios:
        agent_ids = tuple(adapter.list_agents(args.split, scenario_id))
        if not agent_ids:
            continue
        timestamps = adapter.list_timestamps(args.split, scenario_id, agent_ids[0])
        for timestamp in timestamps[:: max(args.stride, 1)]:
            frame_ref = OPV2VFrameRef(
                split=args.split,
                scenario_id=scenario_id,
                timestamp=timestamp,
                ego_agent_id=agent_ids[0],
                agent_ids=agent_ids,
            )
            inspection = adapter.inspect_frame(frame_ref)
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "timestamp": timestamp,
                    "agent_count": inspection["agent_count"],
                    "ego_vehicle_count": inspection["ego_vehicle_count"],
                    "all_unique_vehicle_count": inspection["all_unique_vehicle_count"],
                    "partner_only_vehicle_count": inspection["partner_only_vehicle_count"],
                    "cooperative_gain": (
                        int(inspection["all_unique_vehicle_count"])
                        - int(inspection["ego_vehicle_count"])
                    ),
                }
            )
            if len(rows) >= args.max_frames:
                break
        if len(rows) >= args.max_frames:
            break

    if not rows:
        raise FileNotFoundError(
            f"No OPV2V frames inspected under {adapter.data_root / args.split}"
        )

    partner_only_counts = [int(row["partner_only_vehicle_count"]) for row in rows]
    cooperative_gains = [int(row["cooperative_gain"]) for row in rows]
    ego_counts = [int(row["ego_vehicle_count"]) for row in rows]
    coop_counts = [int(row["all_unique_vehicle_count"]) for row in rows]
    agent_counts = [int(row["agent_count"]) for row in rows]

    summary = {
        "data_root": str(adapter.data_root),
        "split": args.split,
        "max_frames": args.max_frames,
        "max_scenarios": args.max_scenarios,
        "stride": args.stride,
        "frames_inspected": len(rows),
        "scenarios_seen": len(set(str(row["scenario_id"]) for row in rows)),
        "avg_agents": mean(agent_counts),
        "avg_ego_objects": mean(ego_counts),
        "avg_unique_coop_objects": mean(coop_counts),
        "avg_partner_only_objects": mean(partner_only_counts),
        "avg_cooperative_gain": mean(cooperative_gains),
        "partner_only_frame_count": sum(1 for value in partner_only_counts if value > 0),
        "partner_only_frame_rate": sum(1 for value in partner_only_counts if value > 0) / len(rows),
        "cooperative_gain_frame_count": sum(1 for value in cooperative_gains if value > 0),
        "cooperative_gain_frame_rate": sum(1 for value in cooperative_gains if value > 0) / len(rows),
        "max_partner_only_objects": max(partner_only_counts),
        "max_cooperative_gain": max(cooperative_gains),
        "rows_sample": rows[:20],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"data_root: {summary['data_root']}")
    print(f"split: {summary['split']}")
    print(f"frames_inspected: {summary['frames_inspected']}")
    print(f"scenarios_seen: {summary['scenarios_seen']}")
    print(f"avg_agents: {summary['avg_agents']:.3f}")
    print(f"avg_ego_objects: {summary['avg_ego_objects']:.3f}")
    print(f"avg_unique_coop_objects: {summary['avg_unique_coop_objects']:.3f}")
    print(f"avg_partner_only_objects: {summary['avg_partner_only_objects']:.3f}")
    print(f"avg_cooperative_gain: {summary['avg_cooperative_gain']:.3f}")
    print(f"partner_only_frame_rate: {summary['partner_only_frame_rate']:.3f}")
    print(f"cooperative_gain_frame_rate: {summary['cooperative_gain_frame_rate']:.3f}")
    print(f"max_partner_only_objects: {summary['max_partner_only_objects']}")
    print(f"max_cooperative_gain: {summary['max_cooperative_gain']}")
    print(f"saved_json: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
