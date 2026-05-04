from __future__ import annotations

from kg_coop_drive.application.v2vgotqa_router import (
    AgentMotionPredictionHandler,
    ControlSettingsHandler,
    FutureTrajectoryHandler,
    InvisibleObjectsHandler,
    InvisibleSelectionPolicy,
    NotableObjectsHandler,
    NotableObjectLLMRankItem,
    NotableObjectLLMRankedItem,
    ObjectMotionPredictionHandler,
    OccludingObjectLLMRankItem,
    OccludingObjectLLMRankedItem,
    OccludingObjectsHandler,
    PlanningAwarenessHandler,
    V2VGoTQARouter,
)
from kg_coop_drive.application.planning_awareness import (
    PlanningAwarenessCandidate,
    PlanningAwarenessDecision,
    build_planning_awareness_orchestrator,
)
from kg_coop_drive.domain.benchmark import BenchmarkSample, BenchmarkTaskType
from kg_coop_drive.domain.scene import (
    AgentContext,
    CooperativeScene,
    ObjectTrack,
    Point2D,
    Pose2D,
    ProvenanceRecord,
    TrackStatus,
    Trajectory,
    Vector2D,
    VisibilityFact,
    VisibilityState,
)


def _scene() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-1",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="visible-car",
                object_type="car",
                position=Point2D(11.0, 0.5),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-1",),
                    latest_timestamp_index=0,
                ),
                velocity=Vector2D(2.0, 0.0),
            ),
            ObjectTrack(
                object_id="occluded-car",
                object_type="car",
                position=Point2D(12.0, -0.5),
                confidence=0.8,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-2",),
                    latest_timestamp_index=0,
                ),
                velocity=Vector2D(0.0, -1.0),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="visible-car",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="occluded-car",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def _sample(task_type: BenchmarkTaskType) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"sample-{task_type.value}",
        dataset_name="V2V-GoT-QA",
        split_name="val",
        file_name="demo.json",
        task_type=task_type,
        scene=_scene(),
        raw_record={},
    )


def _scene_with_competing_objects() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-2",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="good-visible",
                object_type="car",
                position=Point2D(11.0, 0.3),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO", "CAV_1"),
                    observation_ids=("obs-1", "obs-2"),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="far-visible",
                object_type="car",
                position=Point2D(11.0, 5.5),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-3",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="blocker-visible",
                object_type="car",
                position=Point2D(9.0, 0.0),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-4",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="hidden-target",
                object_type="car",
                position=Point2D(16.0, 0.1),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-5",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="good-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="far-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="blocker-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="hidden-target",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def _scene_with_late_path_visible_object() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-3",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_1",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(-75.7, 5.2))),
        ),
        future_trajectory=Trajectory(
            points=(
                Point2D(-68.0, 3.4),
                Point2D(-59.9, 1.4),
                Point2D(-51.8, 0.5),
                Point2D(-43.9, 0.6),
                Point2D(-35.5, 0.4),
                Point2D(-26.8, 0.3),
            )
        ),
        object_tracks=(
            ObjectTrack(
                object_id="late-path-visible",
                object_type="car",
                position=Point2D(-21.1, 1.5),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-6",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="far-off-visible",
                object_type="car",
                position=Point2D(-20.0, 10.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-7",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_1",
                object_id="late-path-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_1",
                object_id="far-off-visible",
                state=VisibilityState.VISIBLE,
            ),
        ),
    )


def _scene_with_supported_and_candidate_visible_objects() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-4",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="grounded-visible",
                object_type="car",
                position=Point2D(12.0, 0.6),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("GT", "CAV_EGO"),
                    observation_ids=("obs-8",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
            ObjectTrack(
                object_id="candidate-visible",
                object_type="car",
                position=Point2D(12.5, 0.5),
                confidence=0.99,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-9",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="grounded-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="candidate-visible",
                state=VisibilityState.VISIBLE,
            ),
        ),
    )


def _scene_with_far_only_objects() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-5",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="far-visible",
                object_type="car",
                position=Point2D(20.0, 10.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-10",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="far-occluded",
                object_type="car",
                position=Point2D(22.0, -8.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-11",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="far-visible",
                state=VisibilityState.VISIBLE,
            ),
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="far-occluded",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def _scene_with_three_competitive_blockers() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-6",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="blocker-a",
                object_type="car",
                position=Point2D(9.0, 0.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-12",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="blocker-b",
                object_type="car",
                position=Point2D(10.0, 0.2),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-13",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="blocker-c",
                object_type="car",
                position=Point2D(11.0, -0.2),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-14",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="hidden-a",
                object_type="car",
                position=Point2D(18.0, 0.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-15",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="hidden-b",
                object_type="car",
                position=Point2D(20.0, 0.3),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-16",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="blocker-a", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="blocker-b", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="blocker-c", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="hidden-a", state=VisibilityState.OCCLUDED),
            VisibilityFact(agent_id="CAV_EGO", object_id="hidden-b", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_weak_third_blocker() -> CooperativeScene:
    base_scene = _scene_with_three_competitive_blockers()
    object_tracks = tuple(
        ObjectTrack(
            object_id=object_track.object_id,
            object_type=object_track.object_type,
            position=Point2D(11.0, 9.0),
            confidence=object_track.confidence,
            provenance=object_track.provenance,
        )
        if object_track.object_id == "blocker-c"
        else object_track
        for object_track in base_scene.object_tracks
    )
    return CooperativeScene(
        scene_id="scene-7",
        local_timestamp_index=base_scene.local_timestamp_index,
        global_timestamp_index=base_scene.global_timestamp_index,
        asker_agent_id=base_scene.asker_agent_id,
        agents=base_scene.agents,
        future_trajectory=base_scene.future_trajectory,
        object_tracks=object_tracks,
        visibility_facts=base_scene.visibility_facts,
    )


def _scene_with_near_mid_third_blocker() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-8",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="blocker-a",
                object_type="car",
                position=Point2D(9.0, 0.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-17",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="blocker-b",
                object_type="car",
                position=Point2D(10.0, 0.2),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-18",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="near-mid-blocker",
                object_type="car",
                position=Point2D(36.0, 1.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-19",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
            ),
            ObjectTrack(
                object_id="hidden-a",
                object_type="car",
                position=Point2D(18.0, 0.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-20",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="hidden-b",
                object_type="car",
                position=Point2D(42.0, 1.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-21",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="blocker-a", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="blocker-b", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="near-mid-blocker", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="hidden-a", state=VisibilityState.OCCLUDED),
            VisibilityFact(agent_id="CAV_EGO", object_id="hidden-b", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_sparse_blocker_evidence_and_visible_fallback() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-9",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="paired-blocker",
                object_type="car",
                position=Point2D(9.0, 0.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-22",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="visible-fallback",
                object_type="car",
                position=Point2D(12.0, 4.0),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO",),
                    observation_ids=("obs-23",),
                    latest_timestamp_index=0,
                ),
            ),
            ObjectTrack(
                object_id="hidden-target",
                object_type="car",
                position=Point2D(18.0, 0.0),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-24",),
                    latest_timestamp_index=0,
                ),
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="paired-blocker", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="visible-fallback", state=VisibilityState.VISIBLE),
            VisibilityFact(agent_id="CAV_EGO", object_id="hidden-target", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_competing_invisible_objects() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-10",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="strong-hidden",
                object_type="car",
                position=Point2D(12.0, 0.2),
                confidence=0.95,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-25", "obs-26"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
            ObjectTrack(
                object_id="weak-hidden",
                object_type="car",
                position=Point2D(12.5, 2.8),
                confidence=0.45,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1",),
                    observation_ids=("obs-27",),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.CANDIDATE,
                conflict_score=0.8,
                uncertainty_score=0.7,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="strong-hidden", state=VisibilityState.OCCLUDED),
            VisibilityFact(agent_id="CAV_EGO", object_id="weak-hidden", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_near_asker_hidden_artifact() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-11",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="near-asker-hidden-artifact",
                object_type="car",
                position=Point2D(0.2, 0.1),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_EGO", "CAV_1"),
                    observation_ids=("obs-28", "obs-29"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
            ObjectTrack(
                object_id="plausible-hidden",
                object_type="car",
                position=Point2D(12.0, 0.1),
                confidence=0.9,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-30", "obs-31"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="near-asker-hidden-artifact",
                state=VisibilityState.OCCLUDED,
            ),
            VisibilityFact(agent_id="CAV_EGO", object_id="plausible-hidden", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_centerline_and_lateral_hidden_objects() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-12",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="far-centerline-hidden",
                object_type="car",
                position=Point2D(24.0, 0.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-32", "obs-33"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
            ObjectTrack(
                object_id="lateral-hidden",
                object_type="car",
                position=Point2D(24.0, 4.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-34", "obs-35"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="far-centerline-hidden", state=VisibilityState.OCCLUDED),
            VisibilityFact(agent_id="CAV_EGO", object_id="lateral-hidden", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_far_centerline_hidden_object() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-13",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="far-centerline-hidden",
                object_type="car",
                position=Point2D(24.0, 0.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-36", "obs-37"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="far-centerline-hidden", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_far_behind_lateral_hidden_object() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-14",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(-12.0, 3.0), Point2D(-18.0, 3.0), Point2D(-24.0, 3.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="far-behind-lateral-hidden",
                object_type="car",
                position=Point2D(-18.0, 3.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-38", "obs-39"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="far-behind-lateral-hidden", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_far_behind_centerline_hidden_object() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-15",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(-12.0, 0.0), Point2D(-18.0, 0.0), Point2D(-24.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="far-behind-centerline-hidden",
                object_type="car",
                position=Point2D(-18.0, 0.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-40", "obs-41"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="far-behind-centerline-hidden", state=VisibilityState.OCCLUDED),
        ),
    )


def _scene_with_behind_centerline_near_trajectory_hidden_object() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-16",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(-10.0, 0.0), Point2D(-18.0, 0.0), Point2D(-30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="behind-centerline-near-trajectory-hidden",
                object_type="car",
                position=Point2D(-18.0, 0.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-42", "obs-43"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="behind-centerline-near-trajectory-hidden",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def _scene_with_ahead_centerline_near_trajectory_hidden_object() -> CooperativeScene:
    return CooperativeScene(
        scene_id="scene-17",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="ahead-centerline-near-trajectory-hidden",
                object_type="car",
                position=Point2D(18.0, 0.2),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-44", "obs-45"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(
                agent_id="CAV_EGO",
                object_id="ahead-centerline-near-trajectory-hidden",
                state=VisibilityState.OCCLUDED,
            ),
        ),
    )


def _sample_with_scene(task_type: BenchmarkTaskType, scene: CooperativeScene) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"sample-{task_type.value}-custom",
        dataset_name="V2V-GoT-QA",
        split_name="val",
        file_name="demo.json",
        task_type=task_type,
        scene=scene,
        raw_record={},
    )


class _FakeOccludingLLMClient:
    def rerank_occluding_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[OccludingObjectLLMRankItem, ...],
    ) -> tuple[OccludingObjectLLMRankedItem, ...]:
        del asker_agent_id, raw_question
        return tuple(
            OccludingObjectLLMRankedItem(
                object_id=candidate.object_id,
                score=1.0 if candidate.object_id == "blocker-visible" else 0.1,
            )
            for candidate in candidates
        )


class _FakeNotableLLMClient:
    def rerank_notable_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[NotableObjectLLMRankItem, ...],
    ) -> tuple[NotableObjectLLMRankedItem, ...]:
        del asker_agent_id, raw_question
        return tuple(
            NotableObjectLLMRankedItem(
                object_id=candidate.object_id,
                score=1.0 if candidate.object_id == "good-visible" else 0.1,
            )
            for candidate in candidates
        )


def test_v2vgotqa_router_answers_notable_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.NOTABLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car",)
    assert "Notable visible objects" in answer.answer_text


def test_v2vgotqa_router_prioritizes_top_visible_relevant_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.NOTABLE_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ("good-visible", "blocker-visible")


def test_v2vgotqa_router_supports_energy_ranker_for_notable_objects() -> None:
    router = V2VGoTQARouter(
        handlers=(NotableObjectsHandler(ranker="energy"),)
    )

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.NOTABLE_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids[0] == "good-visible"


def test_v2vgotqa_router_supports_llm_ranker_for_notable_objects() -> None:
    router = V2VGoTQARouter(
        handlers=(NotableObjectsHandler(ranker="llm", llm_client=_FakeNotableLLMClient()),)
    )

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.NOTABLE_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids[0] == "good-visible"


def test_v2vgotqa_router_keeps_late_path_visible_object_for_notable_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.NOTABLE_OBJECTS, _scene_with_late_path_visible_object())
    )

    assert answer.supported is True
    assert answer.object_ids == ("late-path-visible",)


def test_v2vgotqa_router_prefers_grounded_visible_objects_over_candidates_for_notable_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.NOTABLE_OBJECTS,
            _scene_with_supported_and_candidate_visible_objects(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("grounded-visible",)


def test_v2vgotqa_router_answers_occluding_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.OCCLUDING_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car",)
    assert "occluding" in answer.answer_text.lower()


def test_v2vgotqa_router_prefers_visible_blockers_for_occluding_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.OCCLUDING_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids[0] == "blocker-visible"
    assert "blocker-visible" in answer.object_ids


def test_v2vgotqa_router_uses_llm_rerank_for_occluding_objects() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(llm_client=_FakeOccludingLLMClient()),)
    )

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.OCCLUDING_OBJECTS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ("blocker-visible", "good-visible")


def test_v2vgotqa_router_keeps_competitive_third_occluding_object() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(ranker="top3_open"),)
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.OCCLUDING_OBJECTS,
            _scene_with_three_competitive_blockers(),
        )
    )

    assert answer.supported is True
    assert len(answer.object_ids) == 3
    assert set(answer.object_ids) == {"blocker-a", "blocker-b", "blocker-c"}


def test_v2vgotqa_router_far_supported_ranker_filters_nearby_third_occluding_object() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(ranker="top3_far_supported"),)
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.OCCLUDING_OBJECTS,
            _scene_with_three_competitive_blockers(),
        )
    )

    assert answer.supported is True
    assert len(answer.object_ids) == 2
    assert "blocker-c" not in answer.object_ids


def test_v2vgotqa_router_hybrid_ranker_keeps_near_mid_third_occluding_object() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(ranker="top3_hybrid"),)
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.OCCLUDING_OBJECTS,
            _scene_with_near_mid_third_blocker(),
        )
    )

    assert answer.supported is True
    assert "near-mid-blocker" in answer.object_ids


def test_v2vgotqa_router_hybrid_ranker_filters_weak_third_occluding_object() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(ranker="top3_hybrid"),)
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.OCCLUDING_OBJECTS,
            _scene_with_weak_third_blocker(),
        )
    )

    assert answer.supported is True
    assert len(answer.object_ids) == 2
    assert "blocker-c" not in answer.object_ids


def test_v2vgotqa_router_risk_adaptive_ranker_uses_relative_occlusion_risk() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(ranker="risk_adaptive"),)
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.OCCLUDING_OBJECTS,
            _scene_with_three_competitive_blockers(),
        )
    )

    assert answer.supported is True
    assert len(answer.object_ids) >= 2
    assert set(answer.object_ids).issubset({"blocker-a", "blocker-b", "blocker-c"})


def test_v2vgotqa_router_risk_adaptive_backfills_sparse_blocker_evidence() -> None:
    router = V2VGoTQARouter(
        handlers=(OccludingObjectsHandler(ranker="risk_adaptive"),)
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.OCCLUDING_OBJECTS,
            _scene_with_sparse_blocker_evidence_and_visible_fallback(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("paired-blocker", "visible-fallback")


def test_v2vgotqa_router_filters_weak_third_occluding_object() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.OCCLUDING_OBJECTS,
            _scene_with_weak_third_blocker(),
        )
    )

    assert answer.supported is True
    assert len(answer.object_ids) == 2
    assert "blocker-c" not in answer.object_ids


def test_v2vgotqa_router_answers_invisible_objects() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.INVISIBLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car",)
    assert "invisible" in answer.answer_text.lower()


def test_invisible_objects_risk_adaptive_filters_weak_hidden_candidates() -> None:
    router = V2VGoTQARouter(
        handlers=(InvisibleObjectsHandler(ranker="risk_adaptive"),)
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_competing_invisible_objects(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("strong-hidden",)


def test_invisible_objects_legacy_ranker_keeps_broader_hidden_candidates() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="legacy",
                selection_policy=InvisibleSelectionPolicy(
                    min_distance_to_asker=0.0,
                    max_distance_to_trajectory=5.0,
                ),
            ),
        )
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_competing_invisible_objects(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("strong-hidden", "weak-hidden")


def test_invisible_objects_filters_near_asker_hidden_artifacts() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_near_asker_hidden_artifact(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("plausible-hidden",)


def test_invisible_objects_policy_can_tighten_absolute_risk_gate() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="risk_adaptive",
                selection_policy=InvisibleSelectionPolicy(min_risk=0.95),
            ),
        )
    )

    answer = router.answer(_sample(BenchmarkTaskType.INVISIBLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ()


def test_invisible_objects_road_region_prefers_lateral_hidden_hazard() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="road_region",
                selection_policy=InvisibleSelectionPolicy(
                    max_results=1,
                    max_distance_to_trajectory=8.0,
                ),
            ),
        )
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_centerline_and_lateral_hidden_objects(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("lateral-hidden",)


def test_invisible_objects_road_region_strict_suppresses_far_centerline_clutter() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="road_region_strict",
                selection_policy=InvisibleSelectionPolicy(
                    max_results=1,
                    max_distance_to_trajectory=8.0,
                ),
            ),
        )
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_far_centerline_hidden_object(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ()


def test_invisible_objects_temporal_guard_suppresses_far_behind_centerline_clutter() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="temporal_guard",
                selection_policy=InvisibleSelectionPolicy(
                    max_results=1,
                    max_distance_to_trajectory=6.0,
                ),
            ),
        )
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_far_behind_centerline_hidden_object(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ()


def test_invisible_objects_temporal_guard_keeps_far_behind_lateral_candidate() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="temporal_guard",
                selection_policy=InvisibleSelectionPolicy(
                    max_results=1,
                    max_distance_to_trajectory=6.0,
                ),
            ),
        )
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_far_behind_lateral_hidden_object(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("far-behind-lateral-hidden",)


def test_invisible_objects_backtrack_guard_suppresses_behind_centerline_trajectory_clutter() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="backtrack_guard",
                selection_policy=InvisibleSelectionPolicy(
                    max_results=1,
                    max_distance_to_trajectory=6.0,
                ),
            ),
        )
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_behind_centerline_near_trajectory_hidden_object(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ()


def test_invisible_objects_backtrack_guard_keeps_ahead_centerline_trajectory_candidate() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="backtrack_guard",
                selection_policy=InvisibleSelectionPolicy(
                    max_results=1,
                    max_distance_to_trajectory=6.0,
                ),
            ),
        )
    )

    answer = router.answer(
        _sample_with_scene(
            BenchmarkTaskType.INVISIBLE_OBJECTS,
            _scene_with_ahead_centerline_near_trajectory_hidden_object(),
        )
    )

    assert answer.supported is True
    assert answer.object_ids == ("ahead-centerline-near-trajectory-hidden",)


def test_invisible_objects_logreg_acceptor_uses_train_calibrated_model() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="logreg_acceptor",
                selection_policy=InvisibleSelectionPolicy(max_results=1),
                acceptor_model={
                    "feature_names": ["relative_x"],
                    "normalization": {"mean": [0.0], "std": [1.0]},
                    "bias": 0.0,
                    "weights": [1.0],
                    "threshold": 0.5,
                },
            ),
        )
    )

    answer = router.answer(_sample(BenchmarkTaskType.INVISIBLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car",)


def test_invisible_objects_logreg_acceptor_suppresses_low_probability_candidate() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="logreg_acceptor",
                selection_policy=InvisibleSelectionPolicy(max_results=1),
                acceptor_model={
                    "feature_names": ["relative_x"],
                    "normalization": {"mean": [0.0], "std": [1.0]},
                    "bias": 0.0,
                    "weights": [-1.0],
                    "threshold": 0.5,
                },
            ),
        )
    )

    answer = router.answer(_sample(BenchmarkTaskType.INVISIBLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ()


def test_invisible_objects_logreg_legacy_fallback_recovers_rejected_candidate() -> None:
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="logreg_legacy_fallback",
                selection_policy=InvisibleSelectionPolicy(max_results=1),
                acceptor_model={
                    "feature_names": ["relative_x"],
                    "normalization": {"mean": [0.0], "std": [1.0]},
                    "bias": 0.0,
                    "weights": [-1.0],
                    "threshold": 0.5,
                },
            ),
        )
    )

    answer = router.answer(_sample(BenchmarkTaskType.INVISIBLE_OBJECTS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car",)


def test_invisible_objects_logreg_lateral_rescue_recovers_train_mined_shape() -> None:
    scene = CooperativeScene(
        scene_id="scene-logreg-rescue",
        local_timestamp_index=0,
        global_timestamp_index=0,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(agent_id="CAV_1", pose=Pose2D(position=Point2D(5.0, 0.0))),
        ),
        future_trajectory=Trajectory(
            points=(Point2D(10.0, 0.0), Point2D(20.0, 0.0), Point2D(30.0, 0.0))
        ),
        object_tracks=(
            ObjectTrack(
                object_id="ahead-lateral-hidden",
                object_type="car",
                position=Point2D(18.0, 3.0),
                confidence=1.0,
                provenance=ProvenanceRecord(
                    source_agent_ids=("CAV_1", "CAV_2"),
                    observation_ids=("obs-46", "obs-47"),
                    latest_timestamp_index=0,
                ),
                status=TrackStatus.SUPPORTED,
            ),
        ),
        visibility_facts=(
            VisibilityFact(agent_id="CAV_EGO", object_id="ahead-lateral-hidden", state=VisibilityState.OCCLUDED),
        ),
    )
    router = V2VGoTQARouter(
        handlers=(
            InvisibleObjectsHandler(
                ranker="logreg_lateral_rescue",
                selection_policy=InvisibleSelectionPolicy(
                    max_results=1,
                    max_distance_to_trajectory=6.0,
                ),
                acceptor_model={
                    "feature_names": ["relative_x"],
                    "normalization": {"mean": [0.0], "std": [1.0]},
                    "bias": 0.0,
                    "weights": [-1.0],
                    "threshold": 0.5,
                },
            ),
        )
    )

    answer = router.answer(_sample_with_scene(BenchmarkTaskType.INVISIBLE_OBJECTS, scene))

    assert answer.supported is True
    assert answer.object_ids == ("ahead-lateral-hidden",)


def test_v2vgotqa_router_answers_object_motion_prediction() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.OBJECT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car", "occluded-car")
    assert answer.answer_text == (
        "Predicted object motion: visible-car=moving forward from (11.0, 0.5) to (13.0, 0.5); "
        "occluded-car=moving left from (12.0, -0.5) to (12.0, -1.5)."
    )


def test_v2vgotqa_router_answers_agent_motion_prediction() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.AGENT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert answer.answer_text == (
        "Predicted agent motion: CAV_1=hold position near (5.0, 0.0)."
    )


def test_v2vgotqa_router_answers_future_trajectory() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.FUTURE_TRAJECTORY))

    assert answer.supported is True
    assert answer.object_ids == ()
    assert answer.answer_text == (
        "Suggested future trajectory: [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]."
    )


def test_v2vgotqa_router_answers_control_settings() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.CONTROL_SETTINGS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car", "visible-car")
    assert answer.answer_text == (
        "Suggested control settings: speed=reduce speed sharply; steering=steer left; key objects: occluded-car, visible-car."
    )


def test_future_trajectory_handler_renders_points_directly() -> None:
    handler = FutureTrajectoryHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.FUTURE_TRAJECTORY))

    assert answer.supported is True
    assert answer.object_ids == ()
    assert answer.answer_text.endswith("[(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)].")


def test_control_settings_handler_renders_speed_and_steering() -> None:
    handler = ControlSettingsHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.CONTROL_SETTINGS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car", "visible-car")
    assert "speed=reduce speed sharply" in answer.answer_text
    assert "steering=steer left" in answer.answer_text


def test_object_motion_prediction_handler_projects_object_velocity() -> None:
    handler = ObjectMotionPredictionHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.OBJECT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car", "occluded-car")
    assert "visible-car=moving forward" in answer.answer_text
    assert "to (13.0, 0.5)" in answer.answer_text


def test_agent_motion_prediction_handler_renders_other_agent_positions() -> None:
    handler = AgentMotionPredictionHandler()

    answer = handler.answer(_sample(BenchmarkTaskType.AGENT_MOTION_PREDICTION))

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert "CAV_1=hold position near (5.0, 0.0)" in answer.answer_text


def test_agent_motion_prediction_handler_projects_agent_velocity() -> None:
    handler = AgentMotionPredictionHandler()
    scene = CooperativeScene(
        scene_id="scene-agent-motion",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(
                agent_id="CAV_1",
                pose=Pose2D(position=Point2D(5.0, 1.0)),
                velocity=Vector2D(0.0, 2.0),
            ),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.AGENT_MOTION_PREDICTION, scene)
    )

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert answer.answer_text == (
        "Predicted agent motion: CAV_1=move right from (5.0, 1.0) to (5.0, 3.0)."
    )


def test_agent_motion_prediction_handler_uses_planned_trajectory_when_velocity_is_static() -> None:
    handler = AgentMotionPredictionHandler()
    scene = CooperativeScene(
        scene_id="scene-agent-trajectory",
        local_timestamp_index=1,
        global_timestamp_index=1,
        asker_agent_id="CAV_EGO",
        agents=(
            AgentContext(agent_id="CAV_EGO", pose=Pose2D(position=Point2D(0.0, 0.0))),
            AgentContext(
                agent_id="CAV_1",
                pose=Pose2D(position=Point2D(5.0, 1.0)),
                velocity=Vector2D(0.0, 0.0),
                planned_trajectory=Trajectory(points=(Point2D(8.0, 0.2), Point2D(20.0, 0.5))),
            ),
        ),
        future_trajectory=Trajectory(points=(Point2D(10.0, 0.0),)),
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.AGENT_MOTION_PREDICTION, scene)
    )

    assert answer.supported is True
    assert answer.object_ids == ("CAV_1",)
    assert answer.answer_text == (
        "Predicted agent motion: CAV_1=move forward from (5.0, 1.0) to (25.0, 1.5)."
    )


def test_v2vgotqa_router_answers_planning_awareness() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.PLANNING_AWARENESS))

    assert answer.supported is True
    assert answer.object_ids == ("occluded-car", "visible-car")
    assert "aware" in answer.answer_text.lower()


def test_v2vgotqa_router_planning_awareness_limits_visible_results_to_notable_subset() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.PLANNING_AWARENESS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ("hidden-target", "good-visible")


def test_v2vgotqa_router_planning_awareness_returns_empty_when_only_far_objects_exist() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(
        _sample_with_scene(BenchmarkTaskType.PLANNING_AWARENESS, _scene_with_far_only_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ()
    assert answer.answer_text == "There is no notable object."


def test_planning_awareness_handler_can_use_injected_orchestrator_selection() -> None:
    class _FakeOrchestrator:
        def select(self, scene):
            object_track = scene.get_object("visible-car")
            assert object_track is not None
            candidate = PlanningAwarenessCandidate(
                object_track=object_track,
                visibility_state=VisibilityState.VISIBLE,
                distance_to_trajectory=0.0,
                score=1.0,
                rationale=("test-orchestrator",),
            )
            return PlanningAwarenessDecision(
                selected_candidates=(candidate,),
                considered_candidates=(candidate,),
            )

    handler = PlanningAwarenessHandler(
        orchestrator=_FakeOrchestrator(),
        selection_source="orchestrator",
    )

    answer = handler.answer(_sample(BenchmarkTaskType.PLANNING_AWARENESS))

    assert answer.supported is True
    assert answer.object_ids == ("visible-car",)


def test_planning_awareness_count_adaptive_policy_suppresses_near_duplicates() -> None:
    scene = _scene_with_competing_objects()
    orchestrator = build_planning_awareness_orchestrator(
        ranker="relational_importance",
        selection_policy="count_adaptive",
    )

    decision = orchestrator.select(scene)

    selected_ids = tuple(candidate.object_track.object_id for candidate in decision.selected_candidates)
    assert "good-visible" in selected_ids
    assert len(selected_ids) <= 3


def test_planning_awareness_logreg_acceptor_uses_frozen_model() -> None:
    feature_names = ("rank", "visibility_occluded", "visibility_visible")
    model = {
        "model_type": "logreg",
        "feature_names": list(feature_names),
        "normalization": {
            "mean": {name: 0.0 for name in feature_names},
            "std": {name: 1.0 for name in feature_names},
        },
        "weights": [0.0, 8.0, -8.0],
        "bias": 0.0,
        "threshold": 0.5,
    }
    orchestrator = build_planning_awareness_orchestrator(
        ranker="relational_importance",
        selection_policy="logreg_acceptor",
        acceptor_model=model,
    )
    handler = PlanningAwarenessHandler(
        orchestrator=orchestrator,
        selection_source="orchestrator",
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.PLANNING_AWARENESS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ("hidden-target",)


def test_planning_awareness_mlp_acceptor_uses_frozen_model() -> None:
    feature_names = ("rank", "visibility_occluded", "visibility_visible")
    model = {
        "model_type": "mlp",
        "feature_names": list(feature_names),
        "normalization": {
            "mean": {name: 0.0 for name in feature_names},
            "std": {name: 1.0 for name in feature_names},
        },
        "hidden": 1,
        "w1": [[0.0, 2.0, -2.0]],
        "b1": [0.0],
        "w2": [8.0],
        "b2": 0.0,
        "threshold": 0.5,
    }
    orchestrator = build_planning_awareness_orchestrator(
        ranker="relational_importance",
        selection_policy="mlp_acceptor",
        acceptor_model=model,
    )
    handler = PlanningAwarenessHandler(
        orchestrator=orchestrator,
        selection_source="orchestrator",
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.PLANNING_AWARENESS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ("hidden-target",)


def test_planning_awareness_trajectory_calibrated_acceptor_suppresses_far_extra() -> None:
    feature_names = ("rank", "distance_to_trajectory")
    model = {
        "model_type": "logreg",
        "feature_names": list(feature_names),
        "normalization": {
            "mean": {name: 0.0 for name in feature_names},
            "std": {name: 1.0 for name in feature_names},
        },
        "weights": [0.0, 0.0],
        "bias": 0.45,
        "threshold": 0.56,
        "near_duplicate_distance": 1.0,
        "trajectory_calibration": {
            "far_distance_to_trajectory": 4.0,
            "far_abs_y": 4.0,
            "far_moderate_max_probability": 0.65,
            "rescue_min_probability": 0.50,
            "rescue_max_rank": 6,
            "rescue_max_distance_to_trajectory": 4.0,
        },
    }
    orchestrator = build_planning_awareness_orchestrator(
        ranker="relational_importance",
        selection_policy="trajectory_calibrated_acceptor",
        acceptor_model=model,
    )
    handler = PlanningAwarenessHandler(
        orchestrator=orchestrator,
        selection_source="orchestrator",
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.PLANNING_AWARENESS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert "far-visible" not in answer.object_ids


def test_planning_awareness_count_gated_acceptor_can_predict_zero_objects() -> None:
    feature_names = ("rank", "visibility_occluded", "visibility_visible")
    count_feature_names = ("candidate_count",)
    model = {
        "model_type": "logreg",
        "feature_names": list(feature_names),
        "normalization": {
            "mean": {name: 0.0 for name in feature_names},
            "std": {name: 1.0 for name in feature_names},
        },
        "weights": [0.0, 8.0, -8.0],
        "bias": 8.0,
        "threshold": 0.5,
        "count_gate": {
            "model_type": "multinomial_logreg",
            "feature_names": list(count_feature_names),
            "normalization": {
                "mean": {name: 0.0 for name in count_feature_names},
                "std": {name: 1.0 for name in count_feature_names},
            },
            "weights": [[0.0], [0.0], [0.0], [0.0]],
            "biases": [8.0, 0.0, 0.0, 0.0],
            "label_values": [0, 1, 2, 3],
        },
    }
    orchestrator = build_planning_awareness_orchestrator(
        ranker="relational_importance",
        selection_policy="count_gated_acceptor",
        acceptor_model=model,
    )
    handler = PlanningAwarenessHandler(
        orchestrator=orchestrator,
        selection_source="orchestrator",
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.PLANNING_AWARENESS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert answer.object_ids == ()


def test_planning_awareness_soft_count_gated_acceptor_allows_one_extra_strong_candidate() -> None:
    feature_names = ("rank", "visibility_occluded", "visibility_visible")
    count_feature_names = ("candidate_count",)
    model = {
        "model_type": "logreg",
        "feature_names": list(feature_names),
        "normalization": {
            "mean": {name: 0.0 for name in feature_names},
            "std": {name: 1.0 for name in feature_names},
        },
        "weights": [0.0, 8.0, 8.0],
        "bias": 8.0,
        "threshold": 0.5,
        "count_gate": {
            "model_type": "multinomial_logreg",
            "feature_names": list(count_feature_names),
            "normalization": {
                "mean": {name: 0.0 for name in count_feature_names},
                "std": {name: 1.0 for name in count_feature_names},
            },
            "weights": [[0.0], [0.0], [0.0], [0.0]],
            "biases": [0.0, 8.0, 0.0, 0.0],
            "label_values": [0, 1, 2, 3],
            "soft_extra_min_probability": 0.5,
            "soft_extra_min_relative_to_k": 0.9,
        },
    }
    orchestrator = build_planning_awareness_orchestrator(
        ranker="relational_importance",
        selection_policy="soft_count_gated_acceptor",
        acceptor_model=model,
    )
    handler = PlanningAwarenessHandler(
        orchestrator=orchestrator,
        selection_source="orchestrator",
    )

    answer = handler.answer(
        _sample_with_scene(BenchmarkTaskType.PLANNING_AWARENESS, _scene_with_competing_objects())
    )

    assert answer.supported is True
    assert len(answer.object_ids) == 2


def test_v2vgotqa_router_marks_unsupported_tasks_explicitly() -> None:
    router = V2VGoTQARouter()

    answer = router.answer(_sample(BenchmarkTaskType.UNKNOWN))

    assert answer.supported is False
    assert answer.object_ids == ()
    assert "Unsupported Phase 5 task" in answer.answer_text
