from kg_coop_drive.application.tracking.observation_associator import ObservationAssociator
from kg_coop_drive.application.tracking.track_support_enricher import TrackSupportEnricher
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    Trajectory,
)


def test_track_support_enricher_attaches_matched_observation_and_updates_provenance() -> None:
    scene = CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
        observations=(
            ObservationEvidence(
                observation_id="obs-close",
                source_agent_id="CAV_EGO",
                object_type="car",
                position=Point2D(10.2, 0.3),
                confidence=0.8,
                timestamp_index=0,
            ),
        ),
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
    )

    association_report = ObservationAssociator().associate(scene, max_distance=3.0)
    enriched = TrackSupportEnricher().enrich(scene, association_report)

    assert len(enriched.object_tracks[0].observations) == 1
    assert enriched.object_tracks[0].observations[0].observation_id == "obs-close"
    assert enriched.object_tracks[0].provenance.source_agent_ids == ("GT", "CAV_EGO")
    assert enriched.object_tracks[0].provenance.observation_ids == (
        "gt_track-1_0",
        "obs-close",
    )
