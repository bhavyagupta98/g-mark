#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.local_graph_builder import LocalGraphBuilder
from kg_coop_drive.application.local_graph_serializer import LocalGraphSerializer
from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader
from kg_coop_drive.infrastructure.v2vgot_scene_adapter import V2VGoTSceneAdapter


DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def resolve_v2vgot_root() -> Path:
    """Resolve the local V2V-GoT root for either pod or local development."""

    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_V2VGOT_ROOTS[0]


def print_section(title: str) -> None:
    """Print a readable section boundary."""

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for Phase 3 local validation."""

    parser = argparse.ArgumentParser(description="Validate Phase 3 local graphs over multiple timestamps.")
    parser.add_argument("--agent-id", default="CAV_EGO", help="Agent to build the local graph for.")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum number of frames to validate.")
    parser.add_argument(
        "--export-dir",
        default="",
        help="Optional directory to write serialized local graph JSON snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTSceneAdapter(str(repository_root))
    processed_loader = V2VGoTProcessedAssetLoader(str(repository_root))
    local_graph_builder = LocalGraphBuilder()
    serializer = LocalGraphSerializer()

    records = adapter.load_records(split_name="val")
    timestamps = processed_loader.list_available_timestamps(split_name="val")
    record_by_timestamp: dict[int, dict[str, object]] = {}
    for record in records:
        timestamp_index = int(record.get("global_timestamp_index", -1))
        if timestamp_index not in record_by_timestamp:
            record_by_timestamp[timestamp_index] = record

    export_dir = Path(args.export_dir).expanduser().resolve() if args.export_dir else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    print_section("Phase 3 Local Validation")
    print(f"repository_root: {repository_root}")
    print(f"agent_id: {args.agent_id}")
    print(f"candidate_timestamps: {len(timestamps)}")

    validated_count = 0
    total_objects = 0
    total_relations = 0
    total_visibility_facts = 0
    total_supported_tracks = 0
    total_candidate_tracks = 0
    total_visible_objects = 0
    for timestamp_index in timestamps:
        record = record_by_timestamp.get(timestamp_index)
        if record is None:
            continue
        processed_data = processed_loader.load_frame_scene_data(
            timestamp_index=timestamp_index,
            split_name="val",
        )
        if processed_data is None:
            continue

        scene = adapter.build_scene(record)
        local_scene = local_graph_builder.build(
            scene=scene,
            processed_data=processed_data,
            agent_id=args.agent_id,
        )

        print(
            f"- timestamp={timestamp_index}, objects={len(local_scene.object_tracks)}, "
            f"relations={len(local_scene.relations)}, visibility_facts={len(local_scene.visibility_facts)}"
        )

        total_objects += len(local_scene.object_tracks)
        total_relations += len(local_scene.relations)
        total_visibility_facts += len(local_scene.visibility_facts)
        total_supported_tracks += sum(
            1 for track in local_scene.object_tracks if track.status.value == "supported"
        )
        total_candidate_tracks += sum(
            1 for track in local_scene.object_tracks if track.status.value == "candidate"
        )
        visible_object_ids = {
            fact.object_id
            for fact in local_scene.visibility_facts
            if fact.agent_id == args.agent_id and fact.state.value == "visible"
        }
        total_visible_objects += len(visible_object_ids)

        if export_dir is not None:
            output_path = export_dir / f"local_graph_{args.agent_id.lower()}_{timestamp_index:04d}.json"
            output_path.write_text(serializer.to_json(local_scene), encoding="utf-8")

        validated_count += 1
        if validated_count >= args.max_frames:
            break

    print()
    print(f"validated_frames: {validated_count}")
    if validated_count:
        print(f"average_objects_per_frame: {total_objects / validated_count:.2f}")
        print(f"average_relations_per_frame: {total_relations / validated_count:.2f}")
        print(f"average_visibility_facts_per_frame: {total_visibility_facts / validated_count:.2f}")
        print(f"average_supported_tracks_per_frame: {total_supported_tracks / validated_count:.2f}")
        print(f"average_candidate_tracks_per_frame: {total_candidate_tracks / validated_count:.2f}")
        print(f"average_visible_objects_per_frame: {total_visible_objects / validated_count:.2f}")
    if export_dir is not None:
        print(f"export_dir: {export_dir}")


if __name__ == "__main__":
    main()
