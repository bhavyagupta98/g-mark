#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.candidate_track_creator import CandidateTrackCreator
from kg_coop_drive.application.candidate_track_resolver import CandidateTrackResolver
from kg_coop_drive.application.observation_associator import ObservationAssociator
from kg_coop_drive.application.processed_scene_service import ProcessedSceneEnricher
from kg_coop_drive.application.relation_builder import RelationBuilder
from kg_coop_drive.application.track_merger import TrackMerger
from kg_coop_drive.application.track_quality_assessor import TrackQualityAssessor
from kg_coop_drive.application.track_support_enricher import TrackSupportEnricher
from kg_coop_drive.application.visibility_reasoner import VisibilityReasoner
from kg_coop_drive.domain.benchmark import BenchmarkTaskType
from kg_coop_drive.domain.scene import CooperativeScene
from kg_coop_drive.domain.scene import TrackStatus
from kg_coop_drive.domain.scene import VisibilityState
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter
from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader


@dataclass(frozen=True)
class StagePack:
    stage0_seed: CooperativeScene
    stage1_graph: CooperativeScene
    stage2_association: CooperativeScene
    stage3_enriched: CooperativeScene
    chosen_sample_id: str
    split_name: str
    qa_type_id: int | None
    task_type: str
    timestamp_index: int
    scenario_index: object
    occluded_count_to_asker: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a clean motivation figure from one real V2V4Real/V2V-GoT sample, "
            "showing seed evidence -> cooperative graph -> conservative association -> enrichment."
        )
    )
    parser.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    parser.add_argument(
        "--prefer-qa-types",
        default="12,13,14",
        help="Comma-separated QA types to prefer while searching occlusion-relevant scenes.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/paper_figures/occlusion_motivation")
    parser.add_argument("--base-name", default="v2v4real_occlusion_motivation")
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def _resolve_output_dir(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _parse_prefer_qa_types(raw: str) -> set[int]:
    result: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError:
            continue
    return result


def _occluded_count_to_asker(scene: CooperativeScene) -> int:
    return sum(
        1
        for fact in scene.visibility_facts
        if fact.agent_id == scene.asker_agent_id and fact.state == VisibilityState.OCCLUDED
    )


def _choose_sample(
    *,
    adapter: V2VGoTQABenchmarkAdapter,
    loader: V2VGoTProcessedAssetLoader,
    split_name: str,
    file_name: str,
    prefer_qa_types: set[int],
    limit: int,
):
    samples = adapter.load_samples(split_name=split_name, file_name=file_name)
    if limit > 0:
        samples = samples[:limit]

    # Pass 1: strictly prefer requested QA types.
    best = None
    best_key = None
    preferred_pool = [s for s in samples if int(s.qa_type_id or -1) in prefer_qa_types]
    candidate_pools = (preferred_pool, list(samples))
    for pool in candidate_pools:
        for sample in pool:
            processed = loader.load_frame_scene_data(
                timestamp_index=sample.scene.global_timestamp_index,
                split_name=split_name,
            )
            if processed is None:
                continue

            qa_type_id = int(sample.qa_type_id or -1)
            asker = sample.scene.asker_agent_id
            occ = sum(1 for fact in processed.visibility_facts if fact.agent_id == asker and fact.state == VisibilityState.OCCLUDED)
            vis = sum(1 for fact in processed.visibility_facts if fact.agent_id == asker and fact.state == VisibilityState.VISIBLE)
            support = len(processed.observations)
            if occ <= 0:
                continue

            prefer_bonus = 1 if qa_type_id in prefer_qa_types else 0
            # prioritize real occlusion-heavy, still having visible context + observations
            key = (prefer_bonus, occ, vis, support)
            if best is None or key > best_key:
                best = (sample, processed)
                best_key = key
        if best is not None:
            break

    if best is None:
        raise SystemExit("No occlusion-relevant sample found. Try another split or lower filters.")
    return best


def _build_stages(sample, processed) -> StagePack:
    seed = sample.scene

    enricher = ProcessedSceneEnricher()
    associator = ObservationAssociator()
    support_enricher = TrackSupportEnricher()
    cand_creator = CandidateTrackCreator()
    cand_resolver = CandidateTrackResolver()
    merger = TrackMerger()
    quality = TrackQualityAssessor()
    vis_reasoner = VisibilityReasoner()
    relation_builder = RelationBuilder()

    stage1 = enricher.enrich(seed, processed)

    association_report = associator.associate(stage1, max_distance=3.0)
    stage2 = support_enricher.enrich(stage1, association_report)
    stage2 = cand_creator.promote(stage2, association_report)
    stage2, _ = cand_resolver.resolve(stage2, min_candidate_confidence=0.25)
    stage2, _ = merger.merge(stage2, max_distance=1.0)

    stage3 = quality.assess(stage2)
    stage3, _ = vis_reasoner.infer(stage3, uncertain_distance=30.0, min_candidate_visible_confidence=0.5)
    stage3 = relation_builder.build(stage3)

    return StagePack(
        stage0_seed=seed,
        stage1_graph=stage1,
        stage2_association=stage2,
        stage3_enriched=stage3,
        chosen_sample_id=str(sample.sample_id),
        split_name=sample.split_name,
        qa_type_id=sample.qa_type_id,
        task_type=sample.task_type.value if isinstance(sample.task_type, BenchmarkTaskType) else str(sample.task_type),
        timestamp_index=int(sample.scene.global_timestamp_index),
        scenario_index=sample.raw_record.get("scenario_index"),
        occluded_count_to_asker=_occluded_count_to_asker(stage3),
    )


def _iter_points(scene: CooperativeScene) -> Iterable[tuple[float, float]]:
    for agent in scene.agents:
        yield (agent.pose.position.x, agent.pose.position.y)
    for ob in scene.observations:
        yield (ob.position.x, ob.position.y)
    for tr in scene.object_tracks:
        yield (tr.position.x, tr.position.y)
    for p in scene.future_trajectory.points:
        yield (p.x, p.y)


def _scene_bounds(scene: CooperativeScene) -> tuple[float, float, float, float]:
    points = list(_iter_points(scene))
    if not points:
        return (-10.0, 10.0, -10.0, 10.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(4.0, (max_x - min_x) * 0.12)
    pad_y = max(4.0, (max_y - min_y) * 0.12)
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


def _draw_scene(ax, scene: CooperativeScene, title: str, subtitle: str = "") -> None:
    min_x, max_x, min_y, max_y = _scene_bounds(scene)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)

    # ego future trajectory
    if scene.future_trajectory.points:
        tx = [p.x for p in scene.future_trajectory.points]
        ty = [p.y for p in scene.future_trajectory.points]
        ax.plot(tx, ty, color="#1f77b4", linewidth=2.0, alpha=0.85, label="ego trajectory")

    # agents
    for agent in scene.agents:
        color = "#d62728" if agent.agent_id == scene.asker_agent_id else "#ff7f0e"
        ax.scatter(
            [agent.pose.position.x],
            [agent.pose.position.y],
            s=90,
            marker="^",
            c=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )

    # observations (source agent colored)
    obs_color = {"CAV_EGO": "#9467bd", "CAV_1": "#8c564b", "ego": "#9467bd", "1": "#8c564b"}
    for ob in scene.observations:
        c = obs_color.get(ob.source_agent_id, "#7f7f7f")
        ax.scatter([ob.position.x], [ob.position.y], s=26, marker="x", c=c, alpha=0.9, zorder=4)

    # object tracks
    vis_by_object = {fact.object_id: fact.state for fact in scene.visibility_facts if fact.agent_id == scene.asker_agent_id}
    for tr in scene.object_tracks:
        vis = vis_by_object.get(tr.object_id)
        if vis == VisibilityState.OCCLUDED:
            face = "#e74c3c"
        elif vis == VisibilityState.UNCERTAIN:
            face = "#f1c40f"
        elif vis == VisibilityState.VISIBLE:
            face = "#2ecc71"
        else:
            face = "#95a5a6"

        marker = "o"
        if tr.status == TrackStatus.CANDIDATE:
            marker = "D"
        elif tr.status == TrackStatus.SUPPORTED:
            marker = "s"

        ax.scatter(
            [tr.position.x],
            [tr.position.y],
            s=54,
            marker=marker,
            c=face,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.95,
            zorder=5,
        )

    ax.set_title(title, fontsize=11, pad=6)
    if subtitle:
        ax.text(0.01, 0.01, subtitle, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")


def _write_metadata(path: Path, stage_pack: StagePack) -> None:
    payload = asdict(stage_pack)
    # remove heavy scene objects from metadata output
    payload.pop("stage0_seed", None)
    payload.pop("stage1_graph", None)
    payload.pop("stage2_association", None)
    payload.pop("stage3_enriched", None)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_image_paths_from_record(sample) -> list[Path]:
    candidates: list[Path] = []
    raw = sample.raw_record
    keys = (
        "image",
        "image_path",
        "img_path",
        "ego_image",
        "ego_image_path",
        "cav1_image",
        "cav1_image_path",
    )
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(Path(value.strip()))
    return candidates


def _candidate_image_paths_from_processed_source_paths(processed_source_paths: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in processed_source_paths:
        p = Path(str(raw))
        # Observations store path::obsid, keep only path prefix.
        if "::" in str(raw):
            p = Path(str(raw).split("::", 1)[0])
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            paths.append(p)
    return paths


def _timestamp_view_candidates(v2vgot_root: Path, split_name: str, timestamp_index: int) -> dict[str, list[Path]]:
    # Heuristic search over common folder patterns in V2V/DMSTrack style layouts.
    split_dir = "train_no_fusion_keep_all" if split_name == "train" else "no_fusion_keep_all"
    roots = [
        v2vgot_root / "DMSTrack" / "V2V4Real" / "official_models" / split_dir / "npy",
        v2vgot_root / "DMSTrack" / "V2V4Real" / "official_models" / split_dir / "npy" / "co_llm",
    ]
    ts = str(timestamp_index)
    patterns = {
        "ego": [
            f"{ts}_ego.png",
            f"{ts}_ego.jpg",
            f"{ts}_rgb.png",
            f"{ts}_rgb.jpg",
            f"{ts}_camera.png",
            f"{ts}_camera.jpg",
            f"{ts}.png",
            f"{ts}.jpg",
        ],
        "partner": [
            f"{ts}_1.png",
            f"{ts}_1.jpg",
            f"{ts}_cav1.png",
            f"{ts}_cav1.jpg",
            f"{ts}_partner.png",
            f"{ts}_partner.jpg",
            f"{ts}.png",
            f"{ts}.jpg",
        ],
        "bev": [
            f"{ts}_bev.png",
            f"{ts}_bev.jpg",
            f"{ts}_birdseye.png",
            f"{ts}_birdseye.jpg",
            f"{ts}_topdown.png",
            f"{ts}_topdown.jpg",
        ],
    }
    out = {"ego": [], "partner": [], "bev": []}
    for root in roots:
        if not root.exists():
            continue
        for view_name, pats in patterns.items():
            for pat in pats:
                out[view_name].extend(root.rglob(pat))
    return out


def _read_image_or_none(path: Path) -> np.ndarray | None:
    try:
        if path.exists():
            img = mpimg.imread(path)
            if img is not None and img.size > 0:
                # normalize if grayscale
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                return img
    except Exception:
        return None
    return None


def _find_stage1_views(sample, processed_source_paths: Iterable[str], v2vgot_root: Path) -> dict[str, np.ndarray | None]:
    # 1) Direct references from record
    direct_paths = _candidate_image_paths_from_record(sample)
    direct_paths.extend(_candidate_image_paths_from_processed_source_paths(processed_source_paths))
    direct_imgs = [_read_image_or_none(p) for p in direct_paths]
    direct_imgs = [img for img in direct_imgs if img is not None]

    # 2) Heuristic timestamp search
    guessed = _timestamp_view_candidates(
        v2vgot_root=v2vgot_root,
        split_name=sample.split_name,
        timestamp_index=sample.scene.global_timestamp_index,
    )
    ego_img = None
    partner_img = None
    bev_img = None
    for p in guessed["ego"]:
        ego_img = _read_image_or_none(p)
        if ego_img is not None:
            break
    for p in guessed["partner"]:
        partner_img = _read_image_or_none(p)
        if partner_img is not None:
            break
    for p in guessed["bev"]:
        bev_img = _read_image_or_none(p)
        if bev_img is not None:
            break

    # 3) If direct images exist but view-specific aren't found, use them in order.
    if direct_imgs:
        if ego_img is None:
            ego_img = direct_imgs[0]
        if partner_img is None and len(direct_imgs) > 1:
            partner_img = direct_imgs[1]
        if bev_img is None and len(direct_imgs) > 2:
            bev_img = direct_imgs[2]

    return {"ego": ego_img, "partner": partner_img, "bev": bev_img}


def _blank_tile(text: str, h: int = 320, w: int = 480) -> np.ndarray:
    tile = np.ones((h, w, 3), dtype=np.float32)
    tile[:] = np.array([0.93, 0.93, 0.93], dtype=np.float32)
    # simple center-ish text marker using matplotlib later (we draw text in axis instead)
    return tile


def _draw_stage1_multiview(ax, sample, processed_source_paths: Iterable[str], v2vgot_root: Path) -> None:
    views = _find_stage1_views(sample, processed_source_paths=processed_source_paths, v2vgot_root=v2vgot_root)
    ego = views["ego"] if views["ego"] is not None else _blank_tile("ego")
    partner = views["partner"] if views["partner"] is not None else _blank_tile("partner")
    bev = views["bev"] if views["bev"] is not None else _blank_tile("bev")

    # normalize heights for top row
    top_h = min(ego.shape[0], partner.shape[0], 360)
    def _resize_nn(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        y_idx = (np.linspace(0, img.shape[0] - 1, target_h)).astype(int)
        x_idx = (np.linspace(0, img.shape[1] - 1, target_w)).astype(int)
        return img[y_idx][:, x_idx]

    ego_w = int(ego.shape[1] * (top_h / ego.shape[0]))
    partner_w = int(partner.shape[1] * (top_h / partner.shape[0]))
    ego_r = _resize_nn(ego, top_h, max(64, ego_w))
    partner_r = _resize_nn(partner, top_h, max(64, partner_w))
    pad = np.ones((top_h, 12, 3), dtype=np.float32)
    pad[:] = 1.0
    top = np.concatenate([ego_r, pad, partner_r], axis=1)

    # BEV spans bottom
    bot_h = 260
    bev_w = top.shape[1]
    bev_r = _resize_nn(bev, bot_h, bev_w)
    pad_h = np.ones((12, top.shape[1], 3), dtype=np.float32)
    pad_h[:] = 1.0
    canvas = np.concatenate([top, pad_h, bev_r], axis=0)

    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Stage 1: Real Multi-View Context", fontsize=11, pad=6)
    ax.text(0.01, 0.99, "Ego view", transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0.0))
    ax.text(0.70, 0.99, "Partner view", transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0.0))
    ax.text(0.01, 0.37, "Top-down / BEV context", transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0.0))

    missing = []
    if views["ego"] is None:
        missing.append("ego")
    if views["partner"] is None:
        missing.append("partner")
    if views["bev"] is None:
        missing.append("bev")
    if missing:
        ax.text(
            0.5,
            0.02,
            f"missing real assets: {', '.join(missing)} (placeholder tile shown)",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3cd", alpha=0.95, linewidth=0.0),
        )


def _graph_state(scene: CooperativeScene) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []

    # Agent nodes
    for agent in scene.agents:
        node_id = f"agent::{agent.agent_id}"
        nodes.append(
            {
                "id": node_id,
                "type": "agent",
                "agent_id": agent.agent_id,
                "x": float(agent.pose.position.x),
                "y": float(agent.pose.position.y),
            }
        )

    # Observation nodes + observed-by edges
    for ob in scene.observations:
        node_id = f"obs::{ob.observation_id}"
        nodes.append(
            {
                "id": node_id,
                "type": "observation",
                "observation_id": ob.observation_id,
                "source_agent_id": ob.source_agent_id,
                "object_type": ob.object_type,
                "x": float(ob.position.x),
                "y": float(ob.position.y),
                "confidence": float(ob.confidence),
            }
        )
        edges.append(
            {
                "source": node_id,
                "target": f"agent::{ob.source_agent_id}",
                "type": "observed_by",
                "confidence": float(ob.confidence),
            }
        )

    # Object-hypothesis(track) nodes + support/provenance edges
    for tr in scene.object_tracks:
        node_id = f"track::{tr.object_id}"
        nodes.append(
            {
                "id": node_id,
                "type": "object_hypothesis",
                "object_id": tr.object_id,
                "object_type": tr.object_type,
                "x": float(tr.position.x),
                "y": float(tr.position.y),
                "status": tr.status.value,
                "confidence": float(tr.confidence),
                "uncertainty_score": float(tr.uncertainty_score),
                "conflict_score": float(tr.conflict_score),
                "source_agents": list(tr.provenance.source_agent_ids),
                "observation_ids": list(tr.provenance.observation_ids),
            }
        )
        for obs_id in tr.provenance.observation_ids:
            edges.append(
                {
                    "source": f"obs::{obs_id}",
                    "target": node_id,
                    "type": "supports",
                    "confidence": 1.0,
                }
            )
        for agent_id in tr.provenance.source_agent_ids:
            edges.append(
                {
                    "source": node_id,
                    "target": f"agent::{agent_id}",
                    "type": "provenance_from_agent",
                    "confidence": 1.0,
                }
            )

    # Visibility edges (agent -> track)
    for fact in scene.visibility_facts:
        edges.append(
            {
                "source": f"agent::{fact.agent_id}",
                "target": f"track::{fact.object_id}",
                "type": "visibility",
                "state": fact.state.value,
                "confidence": 1.0,
            }
        )

    # Relation edges (track -> object_id)
    for rel in scene.relations:
        edges.append(
            {
                "source": f"track::{rel.subject_id}",
                "target": f"track::{rel.object_id}" if rel.object_id.startswith("pred_candidate_") or rel.object_id.startswith("track_") else rel.object_id,
                "type": rel.relation_type.value,
                "confidence": float(rel.confidence),
            }
        )

    return {
        "scene_id": scene.scene_id,
        "timestamp_index": int(scene.global_timestamp_index),
        "asker_agent_id": scene.asker_agent_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    args = build_parser().parse_args()
    out_dir = _resolve_output_dir(args.output_dir)
    prefer = _parse_prefer_qa_types(args.prefer_qa_types)

    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    loader = V2VGoTProcessedAssetLoader(args.v2vgot_root, asset_profile="cooperative")

    sample, processed = _choose_sample(
        adapter=adapter,
        loader=loader,
        split_name=args.split,
        file_name=args.file_name,
        prefer_qa_types=prefer,
        limit=args.limit,
    )
    stage_pack = _build_stages(sample, processed)

    v2vgot_root_path = Path(args.v2vgot_root).expanduser().resolve()

    # Individual stage exports
    stages = [
        ("stage1_seed", stage_pack.stage0_seed, "Stage 1: Scene Seed", "agents + local observations + ego trajectory"),
        ("stage2_cooperative_graph", stage_pack.stage1_graph, "Stage 2: Cooperative Graph", "processed tracks + provenance/visibility inputs"),
        ("stage3_conservative_association", stage_pack.stage2_association, "Stage 3: Conservative Association", "matched support + retained candidates"),
        ("stage4_enrichment", stage_pack.stage3_enriched, "Stage 4: Graph Enrichment", "visibility + uncertainty/conflict + relations"),
    ]

    for key, scene, title, subtitle in stages:
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        if key == "stage1_seed":
            _draw_stage1_multiview(
                ax,
                sample,
                processed_source_paths=processed.source_paths,
                v2vgot_root=v2vgot_root_path,
            )
        else:
            _draw_scene(ax, scene, title=title, subtitle=subtitle)
        base = out_dir / f"{args.base_name}_{key}"
        fig.tight_layout()
        fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        # Export graph state JSON for each stage.
        graph_state_path = out_dir / f"{args.base_name}_{key}_graph_state.json"
        graph_state_path.write_text(json.dumps(_graph_state(scene), indent=2), encoding="utf-8")

    # Combined 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (_key, scene, title, subtitle) in zip(axes.flatten(), stages):
        if _key == "stage1_seed":
            _draw_stage1_multiview(
                ax,
                sample,
                processed_source_paths=processed.source_paths,
                v2vgot_root=v2vgot_root_path,
            )
        else:
            _draw_scene(ax, scene, title=title, subtitle=subtitle)
    fig.suptitle(
        (
            f"Occlusion Motivation Example (sample_id={stage_pack.chosen_sample_id}, "
            f"qa_type={stage_pack.qa_type_id}, split={stage_pack.split_name})"
        ),
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    combo_base = out_dir / f"{args.base_name}_4panel"
    fig.savefig(combo_base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(combo_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    _write_metadata(out_dir / f"{args.base_name}_metadata.json", stage_pack)

    print("=" * 72)
    print("Motivation Figure Export Complete")
    print("=" * 72)
    print(f"output_dir: {out_dir}")
    print(f"sample_id: {stage_pack.chosen_sample_id}")
    print(f"split: {stage_pack.split_name}")
    print(f"qa_type_id: {stage_pack.qa_type_id}")
    print(f"task_type: {stage_pack.task_type}")
    print(f"timestamp_index: {stage_pack.timestamp_index}")
    print(f"scenario_index: {stage_pack.scenario_index}")
    print(f"occluded_count_to_asker: {stage_pack.occluded_count_to_asker}")
    print(f"combined_png: {combo_base.with_suffix('.png')}")
    print(f"combined_pdf: {combo_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
