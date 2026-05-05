#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.application.query_engine import SceneQueryEngine
from kg_coop_drive.application.candidate_track_creator import CandidateTrackCreator
from kg_coop_drive.application.candidate_track_resolver import CandidateTrackResolver
from kg_coop_drive.application.cross_agent_associator import CrossAgentAssociator
from kg_coop_drive.application.cross_agent_support_enricher import CrossAgentSupportEnricher
from kg_coop_drive.application.metrics_reporter import SceneMetricsReporter
from kg_coop_drive.application.observation_associator import ObservationAssociator
from kg_coop_drive.application.processed_scene_service import ProcessedSceneEnricher
from kg_coop_drive.application.relation_builder import RelationBuilder
from kg_coop_drive.application.scene_builder import QueryInterpreter, SceneBuilder
from kg_coop_drive.application.track_quality_assessor import TrackQualityAssessor
from kg_coop_drive.application.track_merger import TrackMerger
from kg_coop_drive.application.track_support_enricher import TrackSupportEnricher
from kg_coop_drive.application.visibility_reasoner import VisibilityReasoner
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
    adapter = V2VGoTSceneAdapter(str(repository_root))
    scene = adapter.load_first_scene()
    processed_loader = V2VGoTProcessedAssetLoader(str(repository_root))
    processed_enricher = ProcessedSceneEnricher()
    relation_builder = RelationBuilder()
    observation_associator = ObservationAssociator()
    track_support_enricher = TrackSupportEnricher()
    candidate_track_creator = CandidateTrackCreator()
    candidate_track_resolver = CandidateTrackResolver()
    cross_agent_associator = CrossAgentAssociator()
    cross_agent_support_enricher = CrossAgentSupportEnricher()
    track_merger = TrackMerger()
    track_quality_assessor = TrackQualityAssessor()
    visibility_reasoner = VisibilityReasoner()
    scene_metrics_reporter = SceneMetricsReporter()

    builder = SceneBuilder()
    query_engine = SceneQueryEngine()
    query_interpreter = QueryInterpreter()

    availability = processed_loader.inspect_availability(
        timestamp_index=scene.global_timestamp_index,
        split_name="val",
    )
    processed_data = processed_loader.load_frame_scene_data(
        timestamp_index=scene.global_timestamp_index,
        split_name="val",
    )
    if processed_data is not None:
        scene = processed_enricher.enrich(scene, processed_data)
    association_report = observation_associator.associate(scene, max_distance=3.0)
    scene = track_support_enricher.enrich(scene, association_report)
    scene = candidate_track_creator.promote(scene, association_report)
    scene, candidate_resolution_report = candidate_track_resolver.resolve(
        scene,
        min_candidate_confidence=0.25,
    )
    cross_agent_report = cross_agent_associator.associate(scene, max_distance=3.0)
    scene, cross_agent_support_report = cross_agent_support_enricher.enrich(
        scene,
        cross_agent_report,
    )
    scene, track_merge_report = track_merger.merge(scene, max_distance=1.0)
    scene = track_quality_assessor.assess(scene)
    scene, visibility_reasoning_report = visibility_reasoner.infer(
        scene,
        uncertain_distance=30.0,
        min_candidate_visible_confidence=0.5,
    )
    scene = relation_builder.build(scene)
    scene_metrics = scene_metrics_reporter.compute(scene, association_report, cross_agent_report)

    report = builder.build(scene)

    print_section("Scene Overview")
    print(report.interpretation.summary)

    print_section("What Was Parsed")
    print(f"Scene id: {scene.scene_id}")
    print(f"Local timestamp: {scene.local_timestamp_index}")
    print(f"Global timestamp: {scene.global_timestamp_index}")
    print(f"Asker agent: {scene.asker_agent_id}")
    print(f"Number of agents parsed: {len(scene.agents)}")
    print(f"Trajectory points parsed: {len(scene.future_trajectory.points)}")
    for agent in scene.agents:
        print(
            f"- {agent.agent_id}: position=({agent.pose.position.x:.2f}, {agent.pose.position.y:.2f}), "
            f"yaw={agent.pose.yaw_radians:.3f}"
        )

    print_section("Original QA")
    print(f"Question: {scene.raw_question}")
    print(f"Answer: {scene.raw_answer}")

    print_section("Interpretation")
    print("Build steps:")
    for step in report.build_steps:
        print(f"- {step}")
    if report.interpretation.assumptions:
        print("Assumptions:")
        for item in report.interpretation.assumptions:
            print(f"- {item}")
    print("Current capabilities:")
    for item in report.interpretation.observed_capabilities:
        print(f"- {item}")
    print("What is still missing:")
    for item in report.interpretation.missing_information:
        print(f"- {item}")

    print_section("Processed Asset Availability")
    print(f"npy root: {availability.npy_root}")
    print(f"timestamp: {availability.timestamp_index}")
    print(f"has_gt_boxes: {availability.has_gt_boxes}")
    print(f"has_gt_ids: {availability.has_gt_ids}")
    print(f"has_visibility_for_ego: {availability.has_visibility_for_ego}")
    print(f"has_visibility_for_cav1: {availability.has_visibility_for_cav1}")
    print(f"has_pred_for_ego: {availability.has_pred_for_ego}")
    print(f"has_pred_for_cav1: {availability.has_pred_for_cav1}")
    if processed_data is None:
        print(
            "Processed frame assets are not available in this environment for the selected timestamp, "
            "so the scene remains a metadata seed."
        )
    else:
        print(f"Loaded object tracks: {len(processed_data.object_tracks)}")
        print(f"Loaded observations: {len(processed_data.observations)}")
        print(f"Loaded visibility facts: {len(processed_data.visibility_facts)}")
        print("Processed sources:")
        for path in processed_data.source_paths:
            print(f"- {path}")

    print_section("Progress Metrics")
    print(f"total_tracks: {scene_metrics.total_tracks}")
    print(f"confirmed_tracks: {scene_metrics.confirmed_tracks}")
    print(f"supported_tracks: {scene_metrics.supported_tracks}")
    print(f"candidate_tracks: {scene_metrics.candidate_tracks}")
    print(f"total_observations: {scene_metrics.total_observations}")
    print(f"matched_observations: {scene_metrics.matched_observations}")
    print(f"unmatched_observations: {scene_metrics.unmatched_observations}")
    print(f"support_coverage: {scene_metrics.support_coverage:.2f}")
    print(f"average_track_confidence: {scene_metrics.average_track_confidence:.2f}")
    print(f"average_uncertainty_score: {scene_metrics.average_uncertainty_score:.2f}")
    print(f"average_conflict_score: {scene_metrics.average_conflict_score:.2f}")
    print(f"cross_agent_match_count: {scene_metrics.cross_agent_match_count}")
    print(f"relation_count: {scene_metrics.relation_count}")
    print(f"visibility_fact_count: {scene_metrics.visibility_fact_count}")

    print_section("Loaded Observations")
    if not scene.observations:
        print("No detector-backed observations are currently populated.")
    else:
        for observation in scene.observations[:12]:
            print(
                f"- source_agent={observation.source_agent_id}, position=({observation.position.x:.2f}, "
                f"{observation.position.y:.2f}), confidence={observation.confidence:.3f}"
            )

    print_section("Populated Objects")
    if not scene.object_tracks:
        print("No object tracks are currently populated.")
    else:
        for object_track in scene.object_tracks:
            print(
                f"- object_id={object_track.object_id}, type={object_track.object_type}, "
                f"position=({object_track.position.x:.2f}, {object_track.position.y:.2f}), "
                f"status={object_track.status.value}, "
                f"confidence={object_track.confidence:.2f}, "
                f"support_count={len(object_track.observations)}, "
                f"uncertainty={object_track.uncertainty_score:.2f}, "
                f"conflict={object_track.conflict_score:.2f}, "
                f"last_support_confidence={object_track.last_support_confidence:.2f}, "
                f"provenance_agents={list(object_track.provenance.source_agent_ids)}"
            )

    print_section("Derived Relations")
    if not scene.relations:
        print("No derived relation facts are currently populated.")
    else:
        for relation in scene.relations:
            print(
                f"- {relation.subject_id} {relation.relation_type.value} {relation.object_id} "
                f"(confidence={relation.confidence:.2f})"
            )

    print_section("Observation Association")
    association_explanation = query_interpreter.explain_association(
        report=association_report,
        max_distance=3.0,
    )
    print(f"{association_explanation.title}:")
    for step in association_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {association_explanation.outcome}")
    if association_report.matches:
        print("Matched pairs:")
        for match in association_report.matches:
            print(
                f"- track_id={match.track_id}, source_agent={match.source_agent_id}, "
                f"distance={match.distance_meters:.2f}m, confidence={match.observation_confidence:.3f}"
            )
    if association_report.unmatched_track_ids:
        print(
            "Unmatched tracks: "
            + ", ".join(association_report.unmatched_track_ids)
        )
    if association_report.unmatched_observation_ids:
        print(
            "Unmatched observations: "
            + ", ".join(association_report.unmatched_observation_ids)
        )

    print_section("Candidate Resolution")
    candidate_resolution_explanation = query_interpreter.explain_candidate_resolution(
        report=candidate_resolution_report,
        min_candidate_confidence=0.25,
    )
    print(f"{candidate_resolution_explanation.title}:")
    for step in candidate_resolution_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {candidate_resolution_explanation.outcome}")
    if candidate_resolution_report.kept_candidate_ids:
        print("Kept candidates: " + ", ".join(candidate_resolution_report.kept_candidate_ids))
    if candidate_resolution_report.pruned_candidate_ids:
        print("Pruned candidates: " + ", ".join(candidate_resolution_report.pruned_candidate_ids))

    print_section("Track Merge")
    track_merge_explanation = query_interpreter.explain_track_merge(
        report=track_merge_report,
        max_distance=1.0,
    )
    print(f"{track_merge_explanation.title}:")
    for step in track_merge_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {track_merge_explanation.outcome}")
    if track_merge_report.merges:
        print("Merge decisions:")
        for merge in track_merge_report.merges:
            print(
                f"- merged {merge.source_track_id} into {merge.target_track_id} "
                f"at distance {merge.distance_meters:.2f}m"
            )
    if track_merge_report.remaining_candidate_ids:
        print(
            "Remaining candidates: "
            + ", ".join(track_merge_report.remaining_candidate_ids)
        )

    print_section("Visibility Reasoning")
    visibility_explanation = query_interpreter.explain_visibility_reasoning(
        report=visibility_reasoning_report,
        uncertain_distance=30.0,
        min_candidate_visible_confidence=0.5,
    )
    print(f"{visibility_explanation.title}:")
    for step in visibility_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {visibility_explanation.outcome}")
    if visibility_reasoning_report.inferred_visible_pairs:
        print(
            "Inferred visible pairs: "
            + ", ".join(visibility_reasoning_report.inferred_visible_pairs)
        )
    if visibility_reasoning_report.inferred_uncertain_pairs:
        print(
            "Inferred uncertain pairs: "
            + ", ".join(visibility_reasoning_report.inferred_uncertain_pairs)
        )

    print_section("Cross-Agent Association")
    cross_agent_explanation = query_interpreter.explain_cross_agent_association(
        report=cross_agent_report,
        max_distance=3.0,
    )
    print(f"{cross_agent_explanation.title}:")
    for step in cross_agent_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {cross_agent_explanation.outcome}")
    if cross_agent_report.participating_agents:
        print("Participating agents: " + ", ".join(cross_agent_report.participating_agents))
    if cross_agent_report.matches:
        print("Cross-agent matches:")
        for match in cross_agent_report.matches:
            print(
                f"- {match.left_agent_id}:{match.left_observation_id} <-> "
                f"{match.right_agent_id}:{match.right_observation_id} "
                f"(distance={match.distance_meters:.2f}m, confidence={match.confidence:.2f})"
            )

    print_section("Cross-Agent Support Attachment")
    cross_agent_support_explanation = query_interpreter.explain_cross_agent_support_attachment(
        report=cross_agent_support_report,
    )
    print(f"{cross_agent_support_explanation.title}:")
    for step in cross_agent_support_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {cross_agent_support_explanation.outcome}")
    if cross_agent_support_report.enriched_track_ids:
        print(
            "Enriched tracks: "
            + ", ".join(cross_agent_support_report.enriched_track_ids)
        )

    print_section("Deterministic Query Walkthrough")
    selection = query_engine.select_objects(scene)
    selection_explanation = query_interpreter.explain_selection(selection)
    print(f"{selection_explanation.title}:")
    for step in selection_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {selection_explanation.outcome}")

    visible = query_engine.filter_by_visibility(
        selection,
        agent_id=scene.asker_agent_id,
        visibility=VisibilityState.VISIBLE,
    )
    visible_explanation = query_interpreter.explain_visibility_filter(
        agent_id=scene.asker_agent_id,
        visibility=VisibilityState.VISIBLE,
        result=visible,
    )
    print()
    print(f"{visible_explanation.title}:")
    for step in visible_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {visible_explanation.outcome}")

    near_trajectory = query_engine.filter_near_trajectory(visible, max_distance=3.0)
    trajectory_explanation = query_interpreter.explain_trajectory_filter(
        max_distance=3.0,
        result=near_trajectory,
    )
    print()
    print(f"{trajectory_explanation.title}:")
    for step in trajectory_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {trajectory_explanation.outcome}")

    front_of_ego = query_engine.filter_by_relation(
        selection,
        relation_type=RelationType.FRONT_OF,
        reference_id=scene.asker_agent_id,
    )
    front_explanation = query_interpreter.explain_relation_filter(
        relation_name=RelationType.FRONT_OF.value,
        reference_id=scene.asker_agent_id,
        result=front_of_ego,
    )
    print()
    print(f"{front_explanation.title}:")
    for step in front_explanation.steps:
        print(f"- {step}")
    print(f"Outcome: {front_explanation.outcome}")

    print_section("How To Read This Output")
    if processed_data is None:
        print(
            "If object selections stay empty, that is expected in environments where only the QA "
            "JSON is available. In that case, the script is still valuable because it exposes the "
            "scene context, parsed trajectory, and query flow transparently."
        )
    else:
        print(
            "If object selections are now non-empty, that means the scene is no longer just a seed: "
            "it has been enriched with timestamp-aligned GT object tracks and visibility facts from "
            "the processed V2V-GoT assets. Observation association then tells us which detector-backed "
            "evidences support those tracks before we move on to fusion."
        )


if __name__ == "__main__":
    main()
