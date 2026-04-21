from kg_coop_drive.application.cross_agent_associator import CrossAgentAssociator
from kg_coop_drive.application.cross_agent_support_enricher import (
    CrossAgentSupportEnricher,
)
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    TrackStatus,
    Trajectory,
)


def test_cross_agent_support_enricher_attaches_missing_counterpart_observation() -> None:
    ego_observation = ObservationEvidence(
        observation_id="ego-obs-1",
        source_agent_id="CAV_EGO",
        object_type="car",
        position=Point2D(10.0, 0.0),
        confidence=0.8,
        timestamp_index=0,
    )
    cav1_observation = ObservationEvidence(
        observation_id="cav1-obs-1",
        source_agent_id="CAV_1",
        object_type="car",
        position=Point2D(10.5, 0.2),
        confidence=0.7,
        timestamp_index=0,
    )
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(1.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(ego_observation, cav1_observation),
        object_tracks=(
            ObjectTrack(
                object_id="track-1",
                object_type="car",
                position=Point2D(10.0, 0.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_EGO"),
                    observation_ids=("gt_track-1_0", "ego-obs-1"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
                observations=(ego_observation,),
            ),
        ),
    )

    cross_agent_report = CrossAgentAssociator().associate(scene, max_distance=3.0)
    enriched_scene, support_report = CrossAgentSupportEnricher().enrich(
        scene,
        cross_agent_report,
    )

    track = enriched_scene.object_tracks[0]
    assert support_report.attached_match_count == 1
    assert support_report.enriched_track_ids == ("track-1",)
    assert track.provenance.source_agent_ids == ("GT", "CAV_EGO", "CAV_1")
    assert track.provenance.observation_ids == (
        "gt_track-1_0",
        "ego-obs-1",
        "cav1-obs-1",
    )
    assert {observation.observation_id for observation in track.observations} == {
        "ego-obs-1",
        "cav1-obs-1",
    }
