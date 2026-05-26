#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

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
from kg_coop_drive.domain.scene import VisibilityState
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter
from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader


@dataclass(frozen=True)
class StagePack:
    seed: object
    association: object
    enriched: object
    sample: object


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export JSON-only KG states for occlusion example (3 stages).")
    p.add_argument("--v2vgot-root", default="/workspace/repos/V2V-GoT")
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--file-name", default="v2v4real_3d_grounding_qa_dataset_v2vgot.json")
    p.add_argument("--prefer-qa-types", default="12,13,14")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--output-dir", default="outputs/paper_figures/occlusion_motivation")
    p.add_argument("--base-name", default="v2v4real_occlusion_kg_states")
    return p


def _parse_qa_types(raw: str) -> set[int]:
    out = set()
    for t in raw.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.add(int(t))
        except ValueError:
            pass
    return out


def _graph_state(scene) -> dict[str, object]:
    nodes = []
    edges = []

    for agent in scene.agents:
        nodes.append({
            "id": f"agent::{agent.agent_id}",
            "type": "agent",
            "agent_id": agent.agent_id,
            "position": {"x": float(agent.pose.position.x), "y": float(agent.pose.position.y)},
            "yaw_radians": float(agent.pose.yaw_radians),
        })

    for ob in scene.observations:
        nid = f"obs::{ob.observation_id}"
        nodes.append({
            "id": nid,
            "type": "observation",
            "observation_id": ob.observation_id,
            "source_agent_id": ob.source_agent_id,
            "object_type": ob.object_type,
            "position": {"x": float(ob.position.x), "y": float(ob.position.y)},
            "confidence": float(ob.confidence),
            "timestamp_index": int(ob.timestamp_index),
        })
        edges.append({
            "source": nid,
            "target": f"agent::{ob.source_agent_id}",
            "type": "observed_by",
            "confidence": float(ob.confidence),
        })

    for tr in scene.object_tracks:
        nid = f"track::{tr.object_id}"
        nodes.append({
            "id": nid,
            "type": "object_hypothesis",
            "object_id": tr.object_id,
            "object_type": tr.object_type,
            "position": {"x": float(tr.position.x), "y": float(tr.position.y)},
            "status": tr.status.value,
            "confidence": float(tr.confidence),
            "uncertainty_score": float(tr.uncertainty_score),
            "conflict_score": float(tr.conflict_score),
            "provenance": {
                "source_agent_ids": list(tr.provenance.source_agent_ids),
                "observation_ids": list(tr.provenance.observation_ids),
                "latest_timestamp_index": int(tr.provenance.latest_timestamp_index),
            },
        })
        for obs_id in tr.provenance.observation_ids:
            edges.append({
                "source": f"obs::{obs_id}",
                "target": nid,
                "type": "supports",
                "confidence": 1.0,
            })
        for src in tr.provenance.source_agent_ids:
            edges.append({
                "source": nid,
                "target": f"agent::{src}",
                "type": "provenance_from_agent",
                "confidence": 1.0,
            })

    for fact in scene.visibility_facts:
        edges.append({
            "source": f"agent::{fact.agent_id}",
            "target": f"track::{fact.object_id}",
            "type": "visibility",
            "state": fact.state.value,
            "confidence": 1.0,
        })

    for rel in scene.relations:
        edges.append({
            "source": f"track::{rel.subject_id}",
            "target": rel.object_id if rel.object_id.startswith("agent::") else f"track::{rel.object_id}",
            "type": rel.relation_type.value,
            "confidence": float(rel.confidence),
        })

    return {
        "scene_id": scene.scene_id,
        "global_timestamp_index": int(scene.global_timestamp_index),
        "local_timestamp_index": int(scene.local_timestamp_index),
        "asker_agent_id": scene.asker_agent_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _choose_sample(adapter, loader, split_name: str, file_name: str, prefer: set[int], limit: int):
    samples = adapter.load_samples(split_name=split_name, file_name=file_name)
    if limit > 0:
        samples = samples[:limit]

    best = None
    best_key = None
    for sample in samples:
        processed = loader.load_frame_scene_data(timestamp_index=sample.scene.global_timestamp_index, split_name=split_name)
        if processed is None:
            continue
        qa = int(sample.qa_type_id or -1)
        asker = sample.scene.asker_agent_id
        occ = sum(1 for f in processed.visibility_facts if f.agent_id == asker and f.state == VisibilityState.OCCLUDED)
        vis = sum(1 for f in processed.visibility_facts if f.agent_id == asker and f.state == VisibilityState.VISIBLE)
        if occ <= 0:
            continue
        key = (1 if qa in prefer else 0, occ, vis, len(processed.observations))
        if best is None or key > best_key:
            best = (sample, processed)
            best_key = key
    if best is None:
        raise SystemExit("No suitable occlusion sample found.")
    return best


def _build_stages(sample, processed) -> StagePack:
    seed = ProcessedSceneEnricher().enrich(sample.scene, processed)

    assoc = ObservationAssociator().associate(seed, max_distance=3.0)
    s2 = TrackSupportEnricher().enrich(seed, assoc)
    s2 = CandidateTrackCreator().promote(s2, assoc)
    s2, _ = CandidateTrackResolver().resolve(s2, min_candidate_confidence=0.25)
    s2, _ = TrackMerger().merge(s2, max_distance=1.0)

    s3 = TrackQualityAssessor().assess(s2)
    s3, _ = VisibilityReasoner().infer(s3, uncertain_distance=30.0, min_candidate_visible_confidence=0.5)
    s3 = RelationBuilder().build(s3)

    return StagePack(seed=seed, association=s2, enriched=s3, sample=sample)


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefer = _parse_qa_types(args.prefer_qa_types)
    adapter = V2VGoTQABenchmarkAdapter(args.v2vgot_root)
    loader = V2VGoTProcessedAssetLoader(args.v2vgot_root, asset_profile="cooperative")

    sample, processed = _choose_sample(adapter, loader, args.split, args.file_name, prefer, args.limit)
    stages = _build_stages(sample, processed)

    payload_meta = {
        "chosen_sample_id": str(sample.sample_id),
        "split_name": sample.split_name,
        "qa_type_id": sample.qa_type_id,
        "task_type": sample.task_type.value,
        "timestamp_index": int(sample.scene.global_timestamp_index),
        "scenario_index": sample.raw_record.get("scenario_index"),
    }
    _write_json(out_dir / f"{args.base_name}_metadata.json", payload_meta)

    _write_json(out_dir / f"{args.base_name}_stage1_seed_graph_state.json", _graph_state(stages.seed))
    _write_json(out_dir / f"{args.base_name}_stage2_association_graph_state.json", _graph_state(stages.association))
    _write_json(out_dir / f"{args.base_name}_stage3_enrichment_graph_state.json", _graph_state(stages.enriched))

    print("=" * 72)
    print("JSON KG Export Complete")
    print("=" * 72)
    print(f"output_dir: {out_dir}")
    print(f"sample_id: {sample.sample_id}")
    print(f"qa_type_id: {sample.qa_type_id}")
    print(f"stage1_json: {out_dir / f'{args.base_name}_stage1_seed_graph_state.json'}")
    print(f"stage2_json: {out_dir / f'{args.base_name}_stage2_association_graph_state.json'}")
    print(f"stage3_json: {out_dir / f'{args.base_name}_stage3_enrichment_graph_state.json'}")


if __name__ == "__main__":
    main()
