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
    agent_id: str
    local_vehicle_id: str
    position: Point2D


@dataclass(frozen=True)
class Cluster:
    center: Point2D
    source_agent_ids: tuple[str, ...]
    local_vehicle_ids: tuple[str, ...]
    observation_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OPV2V geometry-derived structural ablation without using annotation IDs "
            "as object identity."
        )
    )
    parser.add_argument("--data-root", default="/workspace/repos/OpenCOOD/opv2v_data_dumping")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--gt-cluster-radius",
        type=float,
        default=1.0,
        help="Geometry radius used to derive the target object set from all observations.",
    )
    parser.add_argument(
        "--association-radius",
        type=float,
        default=1.0,
        help="Geometry radius used by graph fusion. Sweep this independently from GT radius.",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=1.0,
        help="Distance gate for matching method predictions to geometry-derived targets.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/opv2v_inspection/opv2v_geometry_gt_ablation_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = OPV2VSceneAdapter(args.data_root)
    rows = collect_rows(args, adapter)
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
        "gt_cluster_radius": args.gt_cluster_radius,
        "association_radius": args.association_radius,
        "match_radius": args.match_radius,
        "method_summaries": summaries,
        "diagnostics": summarize_diagnostics(rows),
        "rows_sample": rows[:20],
        "interpretation": {
            "target_definition": (
                "Targets are geometry-derived clusters over the union of all per-agent OPV2V "
                "vehicle observations in a frame."
            ),
            "important_caveat": (
                "OPV2V local vehicle IDs are not used as object identity for the main metrics. "
                "They are retained only for diagnostics."
            ),
            "ego_only": "Predicts only ego-agent observations.",
            "naive_late_fusion": "Predicts all per-agent observations without duplicate suppression.",
            "geometry_graph": (
                "Clusters observations using only geometry at association_radius, preserving "
                "source-agent provenance."
            ),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"frames: {payload['frames']}")
    print(f"scenarios_seen: {payload['scenarios_seen']}")
    print(f"gt_cluster_radius: {args.gt_cluster_radius}")
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


def collect_rows(args: argparse.Namespace, adapter: OPV2VSceneAdapter) -> list[dict[str, object]]:
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
            observations = load_observations(adapter, frame_ref)
            gt_clusters = cluster_observations(observations, args.gt_cluster_radius)
            ego_predictions = [
                Cluster(
                    center=observation.position,
                    source_agent_ids=(observation.agent_id,),
                    local_vehicle_ids=(observation.local_vehicle_id,),
                    observation_indices=(index,),
                )
                for index, observation in enumerate(observations)
                if observation.agent_id == frame_ref.ego_agent_id
            ]
            naive_predictions = [
                Cluster(
                    center=observation.position,
                    source_agent_ids=(observation.agent_id,),
                    local_vehicle_ids=(observation.local_vehicle_id,),
                    observation_indices=(index,),
                )
                for index, observation in enumerate(observations)
            ]
            graph_predictions = cluster_observations(observations, args.association_radius)

            partner_only_gt = {
                index
                for index, cluster in enumerate(gt_clusters)
                if frame_ref.ego_agent_id not in cluster.source_agent_ids
            }

            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "timestamp": timestamp,
                "agent_count": len(agent_ids),
                "gt_count": len(gt_clusters),
                "partner_only_count": len(partner_only_gt),
                "raw_observation_count": len(observations),
                "same_local_id_pair_distances": same_local_id_pair_distances(observations),
            }
            for prefix, predictions in (
                ("ego", ego_predictions),
                ("naive", naive_predictions),
                ("geometry", graph_predictions),
            ):
                metrics = evaluate_predictions(
                    predictions=predictions,
                    gt_clusters=gt_clusters,
                    partner_only_gt=partner_only_gt,
                    match_radius=args.match_radius,
                )
                row.update(
                    {
                        f"{prefix}_tp": metrics["tp"],
                        f"{prefix}_pred_count": len(predictions),
                        f"{prefix}_partner_only_tp": metrics["partner_only_tp"],
                        f"{prefix}_duplicate_count": metrics["duplicate_count"],
                        f"{prefix}_false_merge_count": metrics["false_merge_count"],
                    }
                )
            rows.append(row)
            if len(rows) >= args.max_frames:
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
                    agent_id=agent_id,
                    local_vehicle_id=str(vehicle_id),
                    position=OPV2VSceneAdapter._vehicle_world_position(vehicle),
                )
            )
    return observations


def cluster_observations(observations: list[Observation], radius: float) -> list[Cluster]:
    clusters: list[list[tuple[int, Observation]]] = []
    for index, observation in sorted(
        enumerate(observations),
        key=lambda item: (item[1].agent_id, item[1].local_vehicle_id),
    ):
        best_index = None
        best_distance = float("inf")
        for cluster_index, cluster in enumerate(clusters):
            center = average_point([obs.position for _idx, obs in cluster])
            distance = point_distance(observation.position, center)
            if distance <= radius and distance < best_distance:
                best_index = cluster_index
                best_distance = distance
        if best_index is None:
            clusters.append([(index, observation)])
        else:
            clusters[best_index].append((index, observation))

    return [
        Cluster(
            center=average_point([obs.position for _idx, obs in cluster]),
            source_agent_ids=tuple(sorted({obs.agent_id for _idx, obs in cluster})),
            local_vehicle_ids=tuple(sorted({obs.local_vehicle_id for _idx, obs in cluster})),
            observation_indices=tuple(sorted(idx for idx, _obs in cluster)),
        )
        for cluster in clusters
    ]


def evaluate_predictions(
    predictions: list[Cluster],
    gt_clusters: list[Cluster],
    partner_only_gt: set[int],
    match_radius: float,
) -> dict[str, int]:
    matched_gt: dict[int, list[int]] = defaultdict(list)
    for pred_index, prediction in enumerate(predictions):
        best_gt_index = None
        best_distance = float("inf")
        for gt_index, gt_cluster in enumerate(gt_clusters):
            distance = point_distance(prediction.center, gt_cluster.center)
            if distance <= match_radius and distance < best_distance:
                best_gt_index = gt_index
                best_distance = distance
        if best_gt_index is not None:
            matched_gt[best_gt_index].append(pred_index)

    false_merge_count = 0
    for prediction in predictions:
        covered_gt = {
            gt_index
            for gt_index, gt_cluster in enumerate(gt_clusters)
            if any(index in gt_cluster.observation_indices for index in prediction.observation_indices)
        }
        if len(covered_gt) > 1:
            false_merge_count += 1

    return {
        "tp": len(matched_gt),
        "partner_only_tp": sum(1 for gt_index in partner_only_gt if gt_index in matched_gt),
        "duplicate_count": sum(max(0, len(pred_indices) - 1) for pred_indices in matched_gt.values()),
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
        distances.extend(float(value) for value in row.get("same_local_id_pair_distances", []))
    distances.sort()
    return {
        "same_local_id_cross_agent_pair_count": len(distances),
        "same_local_id_cross_agent_distance_mean": mean(distances) if distances else None,
        "same_local_id_cross_agent_distance_p50": percentile(distances, 50),
        "same_local_id_cross_agent_distance_p90": percentile(distances, 90),
        "same_local_id_cross_agent_distance_p95": percentile(distances, 95),
        "same_local_id_cross_agent_distance_max": max(distances) if distances else None,
    }


def same_local_id_pair_distances(observations: list[Observation]) -> list[float]:
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.local_vehicle_id].append(observation)
    distances: list[float] = []
    for group in grouped.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if left.agent_id != right.agent_id:
                    distances.append(point_distance(left.position, right.position))
    return distances


def average_point(points: list[Point2D]) -> Point2D:
    return Point2D(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def point_distance(left: Point2D, right: Point2D) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


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


if __name__ == "__main__":
    raise SystemExit(main())
