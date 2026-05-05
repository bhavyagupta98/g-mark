from __future__ import annotations

from math import dist

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    CrossAgentAssociation,
    CrossAgentAssociationReport,
)


class CrossAgentAssociator:
    """Conservatively matches observations across different agents in one frame."""

    def associate(
        self,
        scene: CooperativeScene,
        max_distance: float = 3.0,
    ) -> CrossAgentAssociationReport:
        """Return plausible cross-agent observation matches."""

        agents = tuple(sorted({ob.source_agent_id for ob in scene.observations}))
        if len(agents) < 2:
            return CrossAgentAssociationReport(matches=tuple(), participating_agents=agents)

        candidates = []
        for left in scene.observations:
            for right in scene.observations:
                if left.source_agent_id >= right.source_agent_id:
                    continue
                if left.source_agent_id == right.source_agent_id:
                    continue
                if left.object_type != right.object_type:
                    continue
                distance_meters = dist(
                    (left.position.x, left.position.y),
                    (right.position.x, right.position.y),
                )
                if distance_meters <= max_distance:
                    confidence = max(0.0, min(1.0, 1.0 - (distance_meters / max_distance)))
                    candidates.append(
                        (
                            distance_meters,
                            -min(left.confidence, right.confidence),
                            left.observation_id,
                            left.source_agent_id,
                            right.observation_id,
                            right.source_agent_id,
                            confidence,
                        )
                    )

        candidates.sort()
        used_observations: set[str] = set()
        matches: list[CrossAgentAssociation] = []
        for (
            distance_meters,
            _negative_confidence,
            left_observation_id,
            left_agent_id,
            right_observation_id,
            right_agent_id,
            confidence,
        ) in candidates:
            if left_observation_id in used_observations or right_observation_id in used_observations:
                continue
            used_observations.add(left_observation_id)
            used_observations.add(right_observation_id)
            matches.append(
                CrossAgentAssociation(
                    left_observation_id=left_observation_id,
                    left_agent_id=left_agent_id,
                    right_observation_id=right_observation_id,
                    right_agent_id=right_agent_id,
                    distance_meters=distance_meters,
                    confidence=confidence,
                )
            )

        return CrossAgentAssociationReport(
            matches=tuple(matches),
            participating_agents=agents,
        )
