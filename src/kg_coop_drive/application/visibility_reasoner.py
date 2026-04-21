from __future__ import annotations

from dataclasses import replace
from math import dist

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    TrackStatus,
    VisibilityFact,
    VisibilityReasoningReport,
    VisibilityState,
)


class VisibilityReasoner:
    """Conservatively populates per-agent visibility facts when files are missing."""

    def infer(
        self,
        scene: CooperativeScene,
        uncertain_distance: float = 30.0,
        min_candidate_visible_confidence: float = 0.5,
    ) -> tuple[CooperativeScene, VisibilityReasoningReport]:
        """Preserve existing facts and fill missing pairs with cautious inference."""

        existing_facts = list(scene.visibility_facts)
        existing_pairs = {
            (fact.agent_id, fact.object_id): fact for fact in scene.visibility_facts
        }
        inferred_visible_pairs: list[str] = []
        inferred_uncertain_pairs: list[str] = []

        for object_track in scene.object_tracks:
            for agent in scene.agents:
                key = (agent.agent_id, object_track.object_id)
                if key in existing_pairs:
                    continue

                if object_track.observed_by(agent.agent_id) and (
                    object_track.status != TrackStatus.CANDIDATE
                    or object_track.confidence >= min_candidate_visible_confidence
                ):
                    existing_facts.append(
                        VisibilityFact(
                            agent_id=agent.agent_id,
                            object_id=object_track.object_id,
                            state=VisibilityState.VISIBLE,
                        )
                    )
                    inferred_visible_pairs.append(
                        f"{agent.agent_id}:{object_track.object_id}"
                    )
                    continue

                distance_meters = dist(
                    (agent.pose.position.x, agent.pose.position.y),
                    (object_track.position.x, object_track.position.y),
                )
                if distance_meters <= uncertain_distance:
                    existing_facts.append(
                        VisibilityFact(
                            agent_id=agent.agent_id,
                            object_id=object_track.object_id,
                            state=VisibilityState.UNCERTAIN,
                        )
                    )
                    inferred_uncertain_pairs.append(
                        f"{agent.agent_id}:{object_track.object_id}"
                    )

        return (
            replace(scene, visibility_facts=tuple(existing_facts)),
            VisibilityReasoningReport(
                preserved_fact_count=len(scene.visibility_facts),
                inferred_visible_pairs=tuple(inferred_visible_pairs),
                inferred_uncertain_pairs=tuple(inferred_uncertain_pairs),
            ),
        )
