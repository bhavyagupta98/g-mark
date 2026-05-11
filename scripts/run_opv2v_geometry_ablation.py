#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.domain.scene import Point2D
from kg_coop_drive.infrastructure.opv2v_scene_adapter import OPV2VFrameRef, OPV2VSceneAdapter


@dataclass(frozen=True)
class Observation:
    vehicle_id: str
    agent_id: str
    position: Point2D


@dataclass(frozen=True)
class Prediction:
    position: Point2D
    source_agent_ids: tuple[str, ...]
    source_vehicle_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OPV2V geometry-based graph fusion ablation without oracle ID grouping."
    )
    parser.add_argument("--data-root", default="/workspace/repos/OpenCOOD/opv2v_data_dumping")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--association-radius",
        type=float,
        default=1.0,
        help="Distance gate for geometry-only cross-agent observation clustering.",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=1.0,
        help="Distance gate for matching predictions to GT object centers.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/opv2v_inspection/opv2v_geometry_ablation_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = OPV2VSceneAdapter(args.data_root)
    rows = collect_rows(
        adapter=adapter,
        split=args.split,
        max_frames=args.max_frames,
        max_scenarios=args.max_scenarios,
        stride=args.stride,
        association_radius=args.association_radius,
        match_radius=args.match_radius,
    )
    if not rows:
        raise FileNotFoundError(f"No OPV2V rows found under {adapter.data_root / args.split}")

    methods = ["ego_only", "naive_late_fusion", "geometry_graph"]
    summaries = {method: summarize_method(rows, method) for method in methods}
    payload = {
        "data_root": str(adapter.data_root),
        "split": args.split,
        "frames": len(rows),
        "scenarios_seen": len(set(row["scenario_id"] for row in rows)),
        "max_frames": args.max_frames,
        "max_scenarios": args.max_scenarios,
        "stride": args.stride,
        "association_radius": args.association_radius,
        "match_radius": args.match_radius,
        "method_summaries": summaries,
        "diagnostics": summarize_diagnostics(rows),
        "rows_sample": rows[:20],
        "interpretation": {
            "gt_definition": "GT object centers are built from the union of per-agent OPV2V annotation IDs.",
            "important_caveat": "Annotation IDs are used only for evaluation and false-merge diagnosis, not for geometry_graph fusion.",
            "geometry_graph": "Clusters cross-agent observations using only Euclidean center distance, then evaluates clusters against annotation IDs.",
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"frames: {payload['frames']}")
    print(f"scenarios_seen: {payload['scenarios_seen']}")
    print(f"association_radius: {args.association_radius}")
    print(f"match_radius: {args.match_radius}")
    print(
        "method\trecall\tprecision\tf1\tpartner_only_recall\tduplicate_rate\tfalse_merge_rate\tavg_predictions"
    )
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
                    f"{summary['false_merge_rate']:.6f}",
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
    association_radius: float,
    match_radius: float,
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
            observations = load_observations(adapter, frame_ref)
            gt_centers = build_gt_centers(observations)
            ego_vehicle_ids = {obs.vehicle_id for obs in observations if obs.agent_id == frame_ref.ego_agent_id}
            partner_vehicle_ids = {
                obs.vehicle_id for obs in observations if obs.agent_id != frame_ref.ego_agent_id
            }
            partner_only_ids = partner_vehicle_ids - ego_vehicle_ids

            predictions_by_method = {
                "ego": [
                    Prediction(
                        position=obs.position,
                        source_agent_ids=(obs.agent_id,),
                        source_vehicle_ids=(obs.vehicle_id,),
                    )
                    for obs in observations
                    if obs.agent_id == frame_ref.ego_agent_id
                ],
                "naive": [
                    Prediction(
                        position=obs.position,
                        source_agent_ids=(obs.agent_id,),
                        source_vehicle_ids=(obs.vehicle_id,),
                    )
                    for obs in observations
                ],
                "geometry": cluster_observations(observations, association_radius),
            }

            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "timestamp": timestamp,
                "agent_count": len(agent_ids),
                "gt_count": len(gt_centers),
                "partner_only_count": len(partner_only_ids),
                "same_id_pair_distances": same_id_pair_distances(observations),
            }
            for key, predictions in predictions_by_method.items():
                metrics = evaluate_predictions(predictions, gt_centers, partner_only_ids, match_radius)
                row.update(
                    {
                        f"{key}_tp": metrics["tp"],
                        f"{key}_pred_count": len(predictions),
                        f"{key}_partner_only_tp": metrics["partner_only_tp"],
                        f"{key}_duplicate_count": metrics["duplicate_count"],
                        f"{key}_false_merge_count": metrics["false_merge_count"],
                    }
                )
            rows.append(row)
            if len(rows) >= max_frames:
                return rows
    return rows


def load_observations(adapter: OPV2VSceneAdapter, frame_ref: OPV2VFrameRef) -> list[Observation]:
    observations: list[Observation] = []
    for agent_id in frame_ref.agent_ids:
        record = adapter.load_agent_yaml(
            frame_ref.split,
            frame_ref.scenario_id,
            agent_id,
            frame_ref.timestamp,
        )
        vehicles = record.get("vehicles", {})
        if not isinstance(vehicles, dict):
            continue
        for vehicle_id, vehicle in vehicles.items():
            if not isinstance(vehicle, dict):
                continue
            observations.append(
                Observation(
                    vehicle_id=str(vehicle_id),
                    agent_id=agent_id,
                    position=OPV2VSceneAdapter._vehicle_world_position(vehicle),
                )
            )
    return observations


def build_gt_centers(observations: list[Observation]) -> dict[str, Point2D]:
    grouped: dict[str, list[Point2D]] = defaultdict(list)
    for observation in observations:
        grouped[observation.vehicle_id].append(observation.position)
    return {
        vehicle_id: average_point(points)
        for vehicle_id, points in grouped.items()
    }


def cluster_observations(observations: list[Observation], association_radius: float) -> list[Prediction]:
    clusters: list[list[Observation]] = []
    for observation in sorted(observations, key=lambda obs: (obs.agent_id, obs.vehicle_id)):
        best_index = None
        best_distance = float("inf")
        for index, cluster in enumerate(clusters):
            center = average_point([obs.position for obs in cluster])
            distance = point_distance(observation.position, center)
            if distance <= association_radius and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            clusters.append([observation])
        else:
            clusters[best_index].append(observation)
    return [
        Prediction(
            position=average_point([obs.position for obs in cluster]),
            source_agent_ids=tuple(sorted({obs.agent_id for obs in cluster})),
            source_vehicle_ids=tuple(sorted({obs.vehicle_id for obs in cluster})),
        )
        for cluster in clusters
    ]


def evaluate_predictions(
    predictions: list[Prediction],
    gt_centers: dict[str, Point2D],
    partner_only_ids: set[str],
    match_radius: float,
) -> dict[str, int]:
    matched_gt: dict[str, list[int]] = defaultdict(list)
    matched_prediction_indices: set[int] = set()
    for pred_index, prediction in enumerate(predictions):
        best_gt_id = None
        best_distance = float("inf")
        for gt_id, gt_center in gt_centers.items():
            distance = point_distance(prediction.position, gt_center)
            if distance <= match_radius and distance < best_distance:
                best_gt_id = gt_id
                best_distance = distance
        if best_gt_id is not None:
            matched_gt[best_gt_id].append(pred_index)
            matched_prediction_indices.add(pred_index)

    tp = len(matched_gt)
    partner_only_tp = sum(1 for gt_id in partner_only_ids if gt_id in matched_gt)
    duplicate_count = sum(max(0, len(indices) - 1) for indices in matched_gt.values())
    false_merge_count = sum(
        1
        for prediction in predictions
        if len(set(prediction.source_vehicle_ids)) > 1
    )
    return {
        "tp": tp,
        "partner_only_tp": partner_only_tp,
        "duplicate_count": duplicate_count,
        "false_merge_count": false_merge_count,
    }


def summarize_method(rows: list[dict[str, object]], method: str) -> dict[str, float]:
    prefix = {
        "ego_only": "ego",
        "naive_late_fusion": "naive",
        "geometry_graph": "geometry",
    }[method]
    gt_total = sum(int(row["gt_count"]) for row in rows)
    partner_total = sum(int(row["partner_only_count"]) for row in rows)
    tp_total = sum(int(row[f"{prefix}_tp"]) for row in rows)
    pred_total = sum(int(row[f"{prefix}_pred_count"]) for row in rows)
    partner_tp_total = sum(int(row[f"{prefix}_partner_only_tp"]) for row in rows)
    duplicate_total = sum(int(row[f"{prefix}_duplicate_count"]) for row in rows)
    false_merge_total = sum(int(row[f"{prefix}_false_merge_count"]) for row in rows)
    recall = tp_total / gt_total if gt_total else 0.0
    precision = tp_total / pred_total if pred_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "partner_only_recall": partner_tp_total / partner_total if partner_total else 0.0,
        "duplicate_rate": duplicate_total / pred_total if pred_total else 0.0,
        "false_merge_rate": false_merge_total / pred_total if pred_total else 0.0,
        "avg_predictions": mean(int(row[f"{prefix}_pred_count"]) for row in rows),
        "avg_gt_objects": mean(int(row["gt_count"]) for row in rows),
        "avg_partner_only_objects": mean(int(row["partner_only_count"]) for row in rows),
    }


def summarize_diagnostics(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    distances: list[float] = []
    for row in rows:
        distances.extend(float(value) for value in row.get("same_id_pair_distances", []))
    distances.sort()
    return {
        "same_id_cross_agent_pair_count": len(distances),
        "same_id_cross_agent_distance_mean": mean(distances) if distances else None,
        "same_id_cross_agent_distance_p50": percentile(distances, 50),
        "same_id_cross_agent_distance_p75": percentile(distances, 75),
        "same_id_cross_agent_distance_p90": percentile(distances, 90),
        "same_id_cross_agent_distance_p95": percentile(distances, 95),
        "same_id_cross_agent_distance_max": max(distances) if distances else None,
    }


def same_id_pair_distances(observations: list[Observation]) -> list[float]:
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.vehicle_id].append(observation)
    distances: list[float] = []
    for group in grouped.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if left.agent_id == right.agent_id:
                    continue
                distances.append(point_distance(left.position, right.position))
    return distances


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    index = (len(values) - 1) * pct / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def average_point(points: list[Point2D]) -> Point2D:
    return Point2D(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def point_distance(left: Point2D, right: Point2D) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


if __name__ == "__main__":
    raise SystemExit(main())
