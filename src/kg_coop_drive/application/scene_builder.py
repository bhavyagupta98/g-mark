from __future__ import annotations

from dataclasses import dataclass

from kg_coop_drive.domain.scene import (
    CandidateResolutionReport,
    CooperativeScene,
    CrossAgentAssociationReport,
    CrossAgentSupportAttachmentReport,
    ObservationAssociationReport,
    QueryResult,
    SceneBuildReport,
    SceneInterpretation,
    TrackMergeReport,
    TrackStatus,
    VisibilityReasoningReport,
    VisibilityState,
)


@dataclass(frozen=True)
class QueryExplanation:
    """Narrated explanation of one deterministic query execution."""

    title: str
    steps: tuple[str, ...]
    outcome: str


class SceneBuilder:
    """Builds an interpreted scene report from a canonical scene seed."""

    def build(self, scene: CooperativeScene) -> SceneBuildReport:
        """Create a narrated interpretation for one scene seed."""

        assumptions = []
        if not scene.object_tracks:
            assumptions.append(
                "No object tracks are populated yet; the current scene is a metadata-rich seed rather than a full fused graph."
            )
        if not scene.visibility_facts:
            assumptions.append(
                "Visibility labels are not yet derived from perception; visibility information currently lives only in the QA text."
            )
        if not scene.relations:
            assumptions.append(
                "No simple spatial relations were produced for this frame; either the geometry did not satisfy the current heuristics or richer predicates are still missing."
            )

        observed_capabilities = (
            "We can identify the querying agent, frame indices, and cooperating agent poses.",
            "We can parse and store the future trajectory that the question refers to.",
            "We can preserve the original QA prompt and answer for later grounding checks.",
        )
        if scene.observations:
            observed_capabilities += (
                f"We can load {len(scene.observations)} detector-backed observation evidences from processed assets.",
            )
        if scene.object_tracks:
            observed_capabilities += (
                f"We can build a final scene with {len(scene.object_tracks)} object tracks from processed GT assets plus detector-derived candidates.",
            )
        supported_track_count = sum(
            1 for object_track in scene.object_tracks if object_track.status == TrackStatus.SUPPORTED
        )
        prediction_only_track_count = sum(
            1 for object_track in scene.object_tracks if object_track.status == TrackStatus.CANDIDATE
        )
        if supported_track_count:
            observed_capabilities += (
                f"We can attach detector-backed support evidence onto {supported_track_count} confirmed object tracks.",
            )
        if prediction_only_track_count:
            observed_capabilities += (
                f"We can promote {prediction_only_track_count} unmatched observations into prediction-only candidate tracks.",
            )
        if scene.visibility_facts:
            observed_capabilities += (
                f"We can attach {len(scene.visibility_facts)} visibility facts from processed files or conservative scene reasoning.",
            )
        if scene.object_tracks:
            observed_capabilities += (
                "We can quantify per-track uncertainty and support conflict to help debug weak hypotheses.",
            )
        missing_information = (
            "Per-object detections or tracks are not yet mapped from processed perception files.",
            "Cross-agent object association and fusion are not yet applied.",
            "Visibility, occlusion, and notable-object labels are not yet instantiated as graph facts.",
        )
        if scene.object_tracks or scene.visibility_facts:
            missing_information = (
                "Cross-agent object association and fusion are not yet applied.",
                "Spatial relations are not yet derived from populated object tracks.",
                "The current populated graph uses GT object tracks as a bootstrap, not detector-driven fused tracks yet.",
            )
        if scene.observations and scene.object_tracks:
            missing_information = (
                "Cross-agent object association and fusion are not yet applied.",
                "Spatial relations are still heuristic and should be expanded with stronger geometric/semantic predicates.",
                "Prediction-only candidate tracks are now conservatively pruned, but richer merge and conflict-resolution policies are still missing.",
            )

        summary = (
            f"Scene {scene.scene_id} is a cooperative scene seed at local timestamp "
            f"{scene.local_timestamp_index} and global timestamp {scene.global_timestamp_index}. "
            f"The asker is {scene.asker_agent_id}, and the parsed trajectory contains "
            f"{len(scene.future_trajectory.points)} future points."
        )
        if scene.object_tracks:
            summary += f" The scene currently contains {len(scene.object_tracks)} populated object tracks."

        build_steps = (
            "Loaded one raw V2V-GoT QA record from the co_llm benchmark file.",
            "Parsed the scenario id, local/global timestamps, and asking agent id.",
            "Parsed cooperative agent poses for CAV_EGO and CAV_1.",
            "Parsed the future trajectory string into typed 2D points.",
            "Stored the original question and answer for interpretability and later validation.",
        )
        if scene.object_tracks or scene.visibility_facts:
            build_steps += (
                "Loaded processed timestamped GT and visibility arrays for the same frame.",
                "Converted GT boxes into bootstrap object tracks and attached per-agent visibility facts.",
            )
        if scene.observations:
            build_steps += (
                "Loaded detector-backed observation evidences from processed prediction assets.",
            )
        if supported_track_count:
            build_steps += (
                "Associated detector-backed observations to nearby tracks and attached them as support evidence.",
            )
        if prediction_only_track_count:
            build_steps += (
                "Promoted unmatched observations into prediction-only candidate object tracks.",
            )

        interpretation = SceneInterpretation(
            summary=summary,
            assumptions=tuple(assumptions),
            observed_capabilities=observed_capabilities,
            missing_information=missing_information,
        )
        return SceneBuildReport(
            scene=scene,
            interpretation=interpretation,
            build_steps=build_steps,
        )


class QueryInterpreter:
    """Produces human-readable explanations for deterministic query results."""

    def explain_selection(self, result: QueryResult) -> QueryExplanation:
        """Explain a simple selection result."""

        object_ids = [object_track.object_id for object_track in result.objects]
        outcome = (
            f"Selected {result.count()} objects: {object_ids}"
            if object_ids
            else "Selected 0 objects."
        )
        return QueryExplanation(
            title="Object Selection",
            steps=(
                "Started from the current cooperative scene snapshot.",
                "Returned all currently populated object tracks in the scene.",
            ),
            outcome=outcome,
        )

    def explain_visibility_filter(
        self,
        agent_id: str,
        visibility: VisibilityState,
        result: QueryResult,
    ) -> QueryExplanation:
        """Explain a visibility filter result."""

        object_ids = [object_track.object_id for object_track in result.objects]
        return QueryExplanation(
            title="Visibility Filter",
            steps=(
                f"Kept only objects with visibility state `{visibility.value}` for agent `{agent_id}`.",
                "Matched object ids against the scene's visibility facts.",
            ),
            outcome=(
                f"{len(object_ids)} objects remain after visibility filtering: {object_ids}"
                if object_ids
                else "No objects remain after visibility filtering."
            ),
        )

    def explain_trajectory_filter(
        self,
        max_distance: float,
        result: QueryResult,
    ) -> QueryExplanation:
        """Explain a near-trajectory filter result."""

        object_ids = [object_track.object_id for object_track in result.objects]
        return QueryExplanation(
            title="Near-Trajectory Filter",
            steps=(
                f"Compared each object position against each parsed trajectory point using a {max_distance}m threshold.",
                "Kept objects whose 2D distance to at least one future point stayed within the threshold.",
            ),
            outcome=(
                f"{len(object_ids)} objects are near the planned trajectory: {object_ids}"
                if object_ids
                else "No objects are currently near the planned trajectory under the chosen threshold."
            ),
        )

    def explain_relation_filter(
        self,
        relation_name: str,
        reference_id: str,
        result: QueryResult,
    ) -> QueryExplanation:
        """Explain a relation-based filter result."""

        object_ids = [object_track.object_id for object_track in result.objects]
        return QueryExplanation(
            title="Relation Filter",
            steps=(
                f"Kept only objects that satisfy relation `{relation_name}` relative to `{reference_id}`.",
                "Matched object ids against the derived relation facts in the scene graph.",
            ),
            outcome=(
                f"{len(object_ids)} objects satisfy the relation: {object_ids}"
                if object_ids
                else "No objects satisfy the requested relation."
            ),
        )

    def explain_association(
        self,
        report: ObservationAssociationReport,
        max_distance: float,
    ) -> QueryExplanation:
        """Explain observation-to-track association results."""

        if report.matches:
            outcome = (
                f"Matched {len(report.matches)} observation-track pairs, "
                f"left {len(report.unmatched_track_ids)} tracks unmatched, and "
                f"{len(report.unmatched_observation_ids)} observations unmatched."
            )
        else:
            outcome = (
                "No observation-track pairs satisfied the current matching threshold."
            )

        return QueryExplanation(
            title="Observation Association",
            steps=(
                f"Compared each detector-backed observation to each object track using a {max_distance}m nearest-neighbor threshold.",
                "Allowed one observation per track and one track per observation, then kept the closest compatible matches.",
            ),
            outcome=outcome,
        )

    def explain_candidate_resolution(
        self,
        report: CandidateResolutionReport,
        min_candidate_confidence: float,
    ) -> QueryExplanation:
        """Explain candidate keep/prune decisions."""

        outcome = (
            f"Kept {len(report.kept_candidate_ids)} candidates and pruned {len(report.pruned_candidate_ids)} "
            f"below the {min_candidate_confidence:.2f} confidence threshold."
        )
        return QueryExplanation(
            title="Candidate Resolution",
            steps=(
                "Reviewed prediction-only candidate tracks separately from confirmed/supported tracks.",
                f"Kept only candidates whose confidence met or exceeded {min_candidate_confidence:.2f}.",
            ),
            outcome=outcome,
        )

    def explain_track_merge(
        self,
        report: TrackMergeReport,
        max_distance: float,
    ) -> QueryExplanation:
        """Explain conservative candidate-to-track merge decisions."""

        outcome = (
            f"Merged {len(report.merges)} candidate tracks into stronger tracks and left "
            f"{len(report.remaining_candidate_ids)} candidates unmerged."
        )
        return QueryExplanation(
            title="Track Merge",
            steps=(
                "Compared surviving candidate tracks only against confirmed/supported tracks.",
                f"Merged only same-type tracks within a conservative {max_distance:.2f}m distance threshold.",
            ),
            outcome=outcome,
        )

    def explain_cross_agent_association(
        self,
        report: CrossAgentAssociationReport,
        max_distance: float,
    ) -> QueryExplanation:
        """Explain cross-agent observation matching results."""

        if len(report.participating_agents) < 2:
            outcome = "Fewer than two agents provided observations, so no cross-agent matching was attempted."
        else:
            outcome = (
                f"Matched {len(report.matches)} cross-agent observation pairs across "
                f"{len(report.participating_agents)} participating agents."
            )
        return QueryExplanation(
            title="Cross-Agent Association",
            steps=(
                "Grouped observations by source agent within the same frame.",
                f"Matched only same-type observations from different agents within a conservative {max_distance:.2f}m threshold.",
            ),
            outcome=outcome,
        )

    def explain_cross_agent_support_attachment(
        self,
        report: CrossAgentSupportAttachmentReport,
    ) -> QueryExplanation:
        """Explain cooperative provenance enrichment after cross-agent matching."""

        outcome = (
            f"Attached {report.attached_match_count} cross-agent matches onto "
            f"{len(report.enriched_track_ids)} existing tracks."
        )
        return QueryExplanation(
            title="Cross-Agent Support Attachment",
            steps=(
                "Looked for cross-agent matched observations where exactly one side was already attached to a track.",
                "Attached only the missing counterpart observation onto that existing track instead of creating a new fused object.",
            ),
            outcome=outcome,
        )

    def explain_visibility_reasoning(
        self,
        report: VisibilityReasoningReport,
        uncertain_distance: float,
        min_candidate_visible_confidence: float,
    ) -> QueryExplanation:
        """Explain conservative visibility fact inference."""

        outcome = (
            f"Preserved {report.preserved_fact_count} existing visibility facts, inferred "
            f"{len(report.inferred_visible_pairs)} visible pairs, and inferred "
            f"{len(report.inferred_uncertain_pairs)} uncertain pairs."
        )
        return QueryExplanation(
            title="Visibility Reasoning",
            steps=(
                "Kept processed GT visibility facts unchanged whenever they were already available.",
                "Marked an object as visible only when the same agent directly supported that track with an observation.",
                f"Required prediction-only candidate tracks to meet at least {min_candidate_visible_confidence:.2f} confidence before marking them visible.",
                f"Marked otherwise-unlabeled nearby pairs as uncertain when the agent-object distance stayed within {uncertain_distance:.2f}m.",
            ),
            outcome=outcome,
        )
