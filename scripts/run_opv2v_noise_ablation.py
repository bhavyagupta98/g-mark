#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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
    observation_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OPV2V synthetic partner-noise robustness ablation."
    )
    parser.add_argument("--data-root", default="/workspace/repos/OpenCOOD/opv2v_data_dumping")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gt-cluster-radius", type=float, default=1.0)
    parser.add_argument("--match-radius", type=float, default=1.0)
    parser.add_argument(
        "--noise-stds",
        default="0.0,0.25,0.5,1.0,1.5",
        help="Comma-separated Gaussian xy noise std values in meters for non-ego observations.",
    )
    parser.add_argument(
        "--graph-radii",
        default="1.0,3.0",
        help="Comma-separated graph association radii. Use 1.0 as conservative, 3.0 as aggressive.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-json",
        default="outputs/opv2v_inspection/opv2v_noise_ablation_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    noise_stds = parse_float_list(args.noise_stds)
    graph_radii = parse_float_list(args.graph_radii)
    adapter = OPV2VSceneAdapter(args.data_root)
    frame_refs = collect_frame_refs(adapter, args.split, args.max_frames, args.max_scenarios, args.stride)
    if not frame_refs:
        raise FileNotFoundError(f"No OPV2V frames found under {adapter.data_root / args.split}")

    all_results: list[dict[str, object]] = []
    for noise_std in noise_stds:
        rows = evaluate_noise_level(
            adapter=adapter,
            frame_refs=frame_refs,
            noise_std=noise_std,
            graph_radii=graph_radii,
            gt_cluster_radius=args.gt_cluster_radius,
            match_radius=args.match_radius,
            seed=args.seed,
        )
        all_results.extend(rows)

    summaries = summarize_results(all_results)
    payload = {
        "data_root": str(adapter.data_root),
        "split": args.split,
        "frames": len(frame_refs),
        "scenarios_seen": len(set(frame.scenario_id for frame in frame_refs)),
        "gt_cluster_radius": args.gt_cluster_radius,
        "match_radius": args.match_radius,
        "noise_stds": noise_stds,
        "graph_radii": graph_radii,
        "seed": args.seed,
        "summaries": summaries,
        "interpretation": {
            "target_definition": "Geometry-derived target clusters from clean all-agent observations.",
            "noise_model": "Gaussian xy noise applied only to non-ego observations before method prediction.",
            "conservative_graph": "Graph fusion using a small association radius, usually 1.0m.",
            "aggressive_graph": "Graph fusion using a large association radius, usually 3.0m.",
            "non_regression_guardrail": "Standalone OPV2V ablation script; does not modify V2V-GoT task/e2e code.",
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"frames: {payload['frames']}")
    print(f"scenarios_seen: {payload['scenarios_seen']}")
    print(f"gt_cluster_radius: {args.gt_cluster_radius}")
    print(f"match_radius: {args.match_radius}")
    print("noise_std\tmethod\trecall\tprecision\tf1\tpartner_only_recall\tduplicate_rate\tfalse_merge_rate\tavg_predictions")
    for summary in summaries:
        print(
            "\t".join(
                [
                    f"{summary['noise_std']:.3f}",
                    str(summary["method"]),
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


def collect_frame_refs(
    adapter: OPV2VSceneAdapter,
    split: str,
    max_frames: int,
    max_scenarios: int | None,
    stride: int,
) -> list[OPV2VFrameRef]:
    scenarios = adapter.list_scenarios(split)
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
    frame_refs: list[OPV2VFrameRef] = []
    for scenario_id in scenarios:
        agent_ids = tuple(adapter.list_agents(split, scenario_id))
        if not agent_ids:
            continue
        timestamps = adapter.list_timestamps(split, scenario_id, agent_ids[0])
        for timestamp in timestamps[:: max(stride, 1)]:
            frame_refs.append(
                OPV2VFrameRef(
                    split=split,
                    scenario_id=scenario_id,
                    timestamp=timestamp,
                    ego_agent_id=agent_ids[0],
                    agent_ids=agent_ids,
                )
            )
            if len(frame_refs) >= max_frames:
                return frame_refs
    return frame_refs


def evaluate_noise_level(
    adapter: OPV2VSceneAdapter,
    frame_refs: list[OPV2VFrameRef],
    noise_std: float,
    graph_radii: list[float],
    gt_cluster_radius: float,
    match_radius: float,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = random.Random(seed + int(noise_std * 1000))
    for frame_ref in frame_refs:
        clean_observations = load_observations(adapter, frame_ref)
        noisy_observations = perturb_partner_observations(
            clean_observations,
            ego_agent_id=frame_ref.ego_agent_id,
            noise_std=noise_std,
            rng=rng,
        )
        gt_clusters = cluster_observations(clean_observations, gt_cluster_radius)
        partner_only_gt = {
            index
            for index, cluster in enumerate(gt_clusters)
            if frame_ref.ego_agent_id not in cluster.source_agent_ids
        }
        method_predictions: dict[str, list[Cluster]] = {
            "ego_only": singleton_predictions(
                noisy_observations,
                lambda obs: obs.agent_id == frame_ref.ego_agent_id,
            ),
            "naive_late_fusion": singleton_predictions(noisy_observations, lambda _obs: True),
        }
        for radius in graph_radii:
            label = f"geometry_graph_r{radius:g}"
            method_predictions[label] = cluster_observations(noisy_observations, radius)

        for method, predictions in method_predictions.items():
            metrics = evaluate_predictions(predictions, gt_clusters, partner_only_gt, match_radius)
            rows.append(
                {
                    "noise_std": noise_std,
                    "method": method,
                    "scenario_id": frame_ref.scenario_id,
                    "timestamp": frame_ref.timestamp,
                    "gt_count": len(gt_clusters),
                    "partner_only_count": len(partner_only_gt),
                    "pred_count": len(predictions),
                    **metrics,
                }
            )
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


def perturb_partner_observations(
    observations: list[Observation],
    ego_agent_id: str,
    noise_std: float,
    rng: random.Random,
) -> list[Observation]:
    perturbed: list[Observation] = []
    for observation in observations:
        if observation.agent_id == ego_agent_id or noise_std <= 0.0:
            perturbed.append(observation)
            continue
        perturbed.append(
            Observation(
                agent_id=observation.agent_id,
                local_vehicle_id=observation.local_vehicle_id,
                position=Point2D(
                    x=observation.position.x + rng.gauss(0.0, noise_std),
                    y=observation.position.y + rng.gauss(0.0, noise_std),
                ),
            )
        )
    return perturbed


def singleton_predictions(
    observations: list[Observation],
    predicate: object,
) -> list[Cluster]:
    predictions: list[Cluster] = []
    for index, observation in enumerate(observations):
        if predicate(observation):
            predictions.append(
                Cluster(
                    center=observation.position,
                    source_agent_ids=(observation.agent_id,),
                    observation_indices=(index,),
                )
            )
    return predictions


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


def summarize_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["noise_std"]), str(row["method"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (noise_std, method), method_rows in sorted(grouped.items()):
        gt_total = sum(int(row["gt_count"]) for row in method_rows)
        partner_total = sum(int(row["partner_only_count"]) for row in method_rows)
        tp_total = sum(int(row["tp"]) for row in method_rows)
        pred_total = sum(int(row["pred_count"]) for row in method_rows)
        partner_tp_total = sum(int(row["partner_only_tp"]) for row in method_rows)
        duplicate_total = sum(int(row["duplicate_count"]) for row in method_rows)
        false_merge_total = sum(int(row["false_merge_count"]) for row in method_rows)
        recall = tp_total / gt_total if gt_total else 0.0
        precision = tp_total / pred_total if pred_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summaries.append(
            {
                "noise_std": noise_std,
                "method": method,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "partner_only_recall": partner_tp_total / partner_total if partner_total else 0.0,
                "duplicate_rate": duplicate_total / pred_total if pred_total else 0.0,
                "false_merge_rate": false_merge_total / pred_total if pred_total else 0.0,
                "avg_predictions": mean(int(row["pred_count"]) for row in method_rows),
            }
        )
    return summaries


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def average_point(points: list[Point2D]) -> Point2D:
    return Point2D(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def point_distance(left: Point2D, right: Point2D) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


if __name__ == "__main__":
    raise SystemExit(main())
