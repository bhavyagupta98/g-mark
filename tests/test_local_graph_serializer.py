import json

from kg_coop_drive.application.local_graph_serializer import LocalGraphSerializer
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    RelationFact,
    RelationType,
    Trajectory,
    VisibilityFact,
    VisibilityState,
)


def test_local_graph_serializer_outputs_deterministic_json() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        object_tracks=(
            ObjectTrack(
                object_id="track-1",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT",),
                    observation_ids=("gt_track-1_0",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        relations=(
            RelationFact(
                subject_id="track-1",
                relation_type=RelationType.FRONT_OF,
                object_id="CAV_EGO",
                confidence=1.0,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="track-1",
                state=VisibilityState.VISIBLE,
            ),
        ),
    )

    payload = json.loads(LocalGraphSerializer().to_json(scene))

    assert payload["scene_id"] == "scene-1"
    assert payload["asker_agent_id"] == "CAV_EGO"
    assert payload["object_tracks"][0]["object_id"] == "track-1"
    assert payload["relations"][0]["relation_type"] == "front_of"
    assert payload["visibility_facts"][0]["state"] == "visible"
