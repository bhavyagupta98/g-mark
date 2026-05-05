#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.query_engine import SceneQueryEngine
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    RelationFact,
    RelationType,
    TrackStatus,
    Trajectory,
    VisibilityFact,
    VisibilityState,
)


def build_demo_scene() -> CooperativeScene:
    return CooperativeScene(
        scene_id="phase4-demo",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),),
        future_trajectory=Trajectory(points=(Point2D(8.0, 0.0), Point2D(16.0, 0.0))),
        object_tracks=(
            ObjectTrack(
                object_id="track-1",
                object_type="car",
                position=Point2D(9.0, 1.0),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_EGO"),
                    observation_ids=("gt_track-1_0", "ego-obs-1"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
            ObjectTrack(
                object_id="track-2",
                object_type="car",
                position=Point2D(20.0, -1.0),
                confidence=0.4,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("ego-obs-2",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
            ),
        ),
        relations=(
            RelationFact(subject_id="track-1", relation_type=RelationType.FRONT_OF, object_id="CAV_EGO", confidence=1.0),
            RelationFact(subject_id="track-1", relation_type=RelationType.LEFT_OF, object_id="CAV_EGO", confidence=1.0),
            RelationFact(subject_id="track-2", relation_type=RelationType.FRONT_OF, object_id="CAV_EGO", confidence=1.0),
            RelationFact(subject_id="track-2", relation_type=RelationType.RIGHT_OF, object_id="CAV_EGO", confidence=1.0),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="track-1", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="track-2", state=VisibilityState.UNCERTAIN),
        ),
    )


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    engine = SceneQueryEngine()
    scene = build_demo_scene()
    selected = engine.select_objects(scene)

    print_section("Phase 4 Query Examples")
    print(f"scene_id: {scene.scene_id}")

    visible = engine.filter_by_visibility(selected, agent_id="CAV_EGO", visibility=VisibilityState.VISIBLE)
    print(f"visible_count: {engine.count(visible)}")
    print(f"visible_exists: {engine.exists(visible)}")
    print(f"visible_ids: {[track.object_id for track in visible.objects]}")

    cars = engine.filter_by_type(selected, object_type="car")
    print(f"car_ids: {[track.object_id for track in cars.objects]}")

    front_of = engine.filter_by_relation(
        selected,
        relation_type=RelationType.FRONT_OF,
        reference_id="CAV_EGO",
    )
    print(f"front_of_ids: {[track.object_id for track in front_of.objects]}")

    ego_sourced = engine.filter_by_source_agent(selected, source_agent_id="CAV_EGO")
    print(f"source_agent_ids: {[track.object_id for track in ego_sourced.objects]}")

    confidence_values = engine.get_attribute(selected, attribute_name="confidence")
    print(
        "confidence_values: "
        + str([(item.object_id, item.value) for item in confidence_values])
    )

    confidence_comparisons = engine.compare(selected, attribute_name="confidence")
    print(
        "confidence_comparisons: "
        + str(
            [
                (item.left_object_id, item.relation, item.right_object_id)
                for item in confidence_comparisons
            ]
        )
    )

    object_type_comparisons = engine.compare(selected, attribute_name="object_type")
    print(
        "object_type_comparisons: "
        + str(
            [
                (item.left_object_id, item.relation, item.right_object_id)
                for item in object_type_comparisons
            ]
        )
    )

    unknown_attribute_values = engine.get_attribute(selected, attribute_name="unknown_field")
    print(
        "unknown_attribute_values: "
        + str([(item.object_id, item.value) for item in unknown_attribute_values])
    )

    provenance_traces = engine.trace_provenance(selected)
    print(
        "provenance_traces: "
        + str(
            [
                (item.object_id, item.source_agent_ids, item.observation_ids)
                for item in provenance_traces
            ]
        )
    )

    empty = engine.filter_by_type(selected, object_type="pedestrian")
    print(f"empty_count: {engine.count(empty)}")
    print(f"empty_exists: {engine.exists(empty)}")


if __name__ == "__main__":
    main()
