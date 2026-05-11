#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.infrastructure.opv2v_scene_adapter import OPV2VFrameRef, OPV2VSceneAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a first OPV2V structural visibility/recovery ablation."
    )
    parser.add_argument(
        "--data-root",
        default="/workspace/repos/OpenCOOD/opv2v_data_dumping",
        help="OPV2V data root containing split folders such as test/validate/train.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default="outputs/opv2v_inspection/opv2v_visibility_ablation_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = OPV2VSceneAdapter(args.data_root)
    rows = collect_rows(adapter, args.split, args.max_frames, args.max_scenarios, args.stride)
    if not rows:
        raise FileNotFoundError(f"No OPV2V rows found under {adapter.data_root / args.split}")

    methods = ["ego_only", "naive_late_fusion", "conservative_graph"]
    summaries = {method: summarize_method(rows, method) for method in methods}
    payload = {
        "data_root": str(adapter.data_root),
        "split": args.split,
        "frames": len(rows),
        "scenarios_seen": len(set(row["scenario_id"] for row in rows)),
        "max_frames": args.max_frames,
        "max_scenarios": args.max_scenarios,
        "stride": args.stride,
        "method_summaries": summaries,
        "rows_sample": rows[:20],
        "interpretation": {
            "gt_definition": "Union of per-agent OPV2V vehicle annotation ids in a frame.",
            "ego_only": "Predicts only objects annotated for the ego CAV.",
            "naive_late_fusion": "Concatenates all per-agent observations; recall is high but duplicate observations are retained.",
            "conservative_graph": "Groups all observations by OPV2V vehicle id; this is the first structural proxy for duplicate-suppressed graph fusion.",
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"frames: {payload['frames']}")
    print(f"scenarios_seen: {payload['scenarios_seen']}")
    print("method\trecall\tprecision\tf1\tpartner_only_recall\tduplicate_rate\tavg_predictions")
    for method in methods:
        summary = summaries[method]
        print(
            "\t".join(
                [
                    method,
                    f"{summary['recall']:.6f}",
                    f"{summary['precision']:.6f}",
                    f"{summary['f1']:.6f}",
                    f"{summary['partner_only_recall']:.6f}",
                    f"{summary['duplicate_rate']:.6f}",
                    f"{summary['avg_predictions']:.3f}",
                ]
            )
        )
    print(f"saved_json: {output_path}")
    return 0


def collect_rows(
    adapter: OPV2VSceneAdapter,
    split: str,
    max_frames: int,
    max_scenarios: int | None,
    stride: int,
) -> list[dict[str, object]]:
    scenarios = adapter.list_scenarios(split)
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]

    rows: list[dict[str, object]] = []
    for scenario_id in scenarios:
        agent_ids = tuple(adapter.list_agents(split, scenario_id))
        if not agent_ids:
            continue
        timestamps = adapter.list_timestamps(split, scenario_id, agent_ids[0])
        for timestamp in timestamps[:: max(stride, 1)]:
            frame_ref = OPV2VFrameRef(
                split=split,
                scenario_id=scenario_id,
                timestamp=timestamp,
                ego_agent_id=agent_ids[0],
                agent_ids=agent_ids,
            )
            per_agent_ids = load_per_agent_vehicle_ids(adapter, frame_ref)
            ego_ids = per_agent_ids.get(frame_ref.ego_agent_id, set())
            union_ids = set().union(*per_agent_ids.values()) if per_agent_ids else set()
            partner_ids = set().union(
                *(ids for agent_id, ids in per_agent_ids.items() if agent_id != frame_ref.ego_agent_id)
            ) if len(per_agent_ids) > 1 else set()
            partner_only_ids = partner_ids - ego_ids
            observation_count_by_id: dict[str, int] = defaultdict(int)
            for ids in per_agent_ids.values():
                for vehicle_id in ids:
                    observation_count_by_id[vehicle_id] += 1

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "timestamp": timestamp,
                    "agent_count": len(agent_ids),
                    "gt_count": len(union_ids),
                    "ego_count": len(ego_ids),
                    "partner_only_count": len(partner_only_ids),
                    "all_observation_count": sum(len(ids) for ids in per_agent_ids.values()),
                    "duplicated_observation_count": sum(
                        max(0, count - 1) for count in observation_count_by_id.values()
                    ),
                    "ego_tp": len(ego_ids & union_ids),
                    "naive_tp": len(union_ids),
                    "graph_tp": len(union_ids),
                    "ego_pred_count": len(ego_ids),
                    "naive_pred_count": sum(len(ids) for ids in per_agent_ids.values()),
                    "graph_pred_count": len(union_ids),
                    "ego_partner_only_tp": len(ego_ids & partner_only_ids),
                    "naive_partner_only_tp": len(partner_only_ids),
                    "graph_partner_only_tp": len(partner_only_ids),
                }
            )
            if len(rows) >= max_frames:
                return rows
    return rows


def load_per_agent_vehicle_ids(
    adapter: OPV2VSceneAdapter,
    frame_ref: OPV2VFrameRef,
) -> dict[str, set[str]]:
    per_agent: dict[str, set[str]] = {}
    for agent_id in frame_ref.agent_ids:
        record = adapter.load_agent_yaml(
            frame_ref.split,
            frame_ref.scenario_id,
            agent_id,
            frame_ref.timestamp,
        )
        vehicles = record.get("vehicles", {})
        per_agent[agent_id] = set(str(key) for key in vehicles) if isinstance(vehicles, dict) else set()
    return per_agent


def summarize_method(rows: list[dict[str, object]], method: str) -> dict[str, float]:
    prefix = {
        "ego_only": "ego",
        "naive_late_fusion": "naive",
        "conservative_graph": "graph",
    }[method]
    gt_total = sum(int(row["gt_count"]) for row in rows)
    partner_total = sum(int(row["partner_only_count"]) for row in rows)
    tp_total = sum(int(row[f"{prefix}_tp"]) for row in rows)
    pred_total = sum(int(row[f"{prefix}_pred_count"]) for row in rows)
    partner_tp_total = sum(int(row[f"{prefix}_partner_only_tp"]) for row in rows)
    recall = tp_total / gt_total if gt_total else 0.0
    precision = tp_total / pred_total if pred_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    partner_only_recall = partner_tp_total / partner_total if partner_total else 0.0
    duplicate_total = 0
    if method == "naive_late_fusion":
        duplicate_total = sum(int(row["duplicated_observation_count"]) for row in rows)
    duplicate_rate = duplicate_total / pred_total if pred_total else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "partner_only_recall": partner_only_recall,
        "duplicate_rate": duplicate_rate,
        "avg_predictions": mean(int(row[f"{prefix}_pred_count"]) for row in rows),
        "avg_gt_objects": mean(int(row["gt_count"]) for row in rows),
        "avg_partner_only_objects": mean(int(row["partner_only_count"]) for row in rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
