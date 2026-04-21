#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.local_graph_builder import LocalGraphBuilder
from kg_coop_drive.application.local_graph_serializer import LocalGraphSerializer
from kg_coop_drive.application.query_engine import SceneQueryEngine
from kg_coop_drive.application.scene_builder import QueryInterpreter
from kg_coop_drive.domain.scene import RelationType, VisibilityState
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


def main() -> None:
    repository_root = resolve_v2vgot_root()
    scene = V2VGoTSceneAdapter(str(repository_root)).load_first_scene()
    processed_loader = V2VGoTProcessedAssetLoader(str(repository_root))
    query_engine = SceneQueryEngine()
    query_interpreter = QueryInterpreter()
    processed_data = processed_loader.load_frame_scene_data(
        timestamp_index=scene.global_timestamp_index,
        split_name="val",
    )

    local_scene = LocalGraphBuilder().build(
        scene=scene,
        processed_data=processed_data,
        agent_id="CAV_EGO",
    )

    print_section("Local Graph Summary")
    print(f"scene_id: {local_scene.scene_id}")
    print(f"agent_id: {local_scene.asker_agent_id}")
    print(f"local_timestamp_index: {local_scene.local_timestamp_index}")
    print(f"global_timestamp_index: {local_scene.global_timestamp_index}")
    print(f"objects: {len(local_scene.object_tracks)}")
    print(f"relations: {len(local_scene.relations)}")
    print(f"visibility_facts: {len(local_scene.visibility_facts)}")

    print_section("Local Objects")
    if not local_scene.object_tracks:
        print("No local object tracks were built.")
    else:
        for track in local_scene.object_tracks:
            print(
                f"- object_id={track.object_id}, status={track.status.value}, "
                f"position=({track.position.x:.2f}, {track.position.y:.2f}), "
                f"confidence={track.confidence:.2f}, support_count={len(track.observations)}"
            )

    print_section("Serialized Local Graph")
    print(LocalGraphSerializer().to_json(local_scene))

    print_section("Local Query Walkthrough")
    selection = query_engine.select_objects(local_scene)
    selection_explanation = query_interpreter.explain_selection(selection)
    print(f"{selection_explanation.title}:")
    for step in selection_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {selection_explanation.outcome}")

    visible = query_engine.filter_by_visibility(
        selection,
        agent_id=local_scene.asker_agent_id,
        visibility=VisibilityState.VISIBLE,
    )
    visible_explanation = query_interpreter.explain_visibility_filter(
        agent_id=local_scene.asker_agent_id,
        visibility=VisibilityState.VISIBLE,
        result=visible,
    )
    print()
    print(f"{visible_explanation.title}:")
    for step in visible_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {visible_explanation.outcome}")

    near_trajectory = query_engine.filter_near_trajectory(selection, max_distance=3.0)
    trajectory_explanation = query_interpreter.explain_trajectory_filter(
        max_distance=3.0,
        result=near_trajectory,
    )
    print()
    print(f"{trajectory_explanation.title}:")
    for step in trajectory_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {trajectory_explanation.outcome}")

    behind = query_engine.filter_by_relation(
        selection,
        relation_type=RelationType.BEHIND,
        reference_id=local_scene.asker_agent_id,
    )
    relation_explanation = query_interpreter.explain_relation_filter(
        relation_name=RelationType.BEHIND.value,
        reference_id=local_scene.asker_agent_id,
        result=behind,
    )
    print()
    print(f"{relation_explanation.title}:")
    for step in relation_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {relation_explanation.outcome}")


if __name__ == "__main__":
    main()
