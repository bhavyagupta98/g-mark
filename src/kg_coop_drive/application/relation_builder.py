from __future__ import annotations

from dataclasses import replace
from math import dist

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    Point2D,
    RelationFact,
    RelationType,
)


class RelationBuilder:
    """Derives simple spatial relations from populated scene objects."""

    def build(
        self,
        scene: CooperativeScene,
        near_ego_distance: float = 10.0,
    ) -> CooperativeScene:
        """Return a new scene with derived relation facts populated."""

        relation_facts: list[RelationFact] = []
        asker_agent = self._find_asker_agent(scene)
        if asker_agent is None:
            return scene

        ego_position = asker_agent.pose.position
        for object_track in scene.object_tracks:
            if object_track.position.x > ego_position.x:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.FRONT_OF,
                        object_id=scene.asker_agent_id,
                        confidence=1.0,
                    )
                )
            if object_track.position.x < ego_position.x:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.BEHIND,
                        object_id=scene.asker_agent_id,
                        confidence=1.0,
                    )
                )
            if object_track.position.y > ego_position.y:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.LEFT_OF,
                        object_id=scene.asker_agent_id,
                        confidence=1.0,
                    )
                )
            if object_track.position.y < ego_position.y:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.RIGHT_OF,
                        object_id=scene.asker_agent_id,
                        confidence=1.0,
                    )
                )
            if self._is_near_position(ego_position, object_track.position, max_distance=near_ego_distance):
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.NEAR,
                        object_id=scene.asker_agent_id,
                        confidence=self._near_confidence(ego_position, object_track.position, near_ego_distance),
                    )
                )
            if self._is_near_trajectory(scene, object_track.position):
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.NEAR_TRAJECTORY,
                        object_id=scene.asker_agent_id,
                        confidence=1.0,
                    )
                )

        return replace(scene, relations=tuple(relation_facts))

    @staticmethod
    def _find_asker_agent(scene: CooperativeScene):
        for agent in scene.agents:
            if agent.agent_id == scene.asker_agent_id:
                return agent
        return None

    @staticmethod
    def _is_near_trajectory(scene: CooperativeScene, position: Point2D, max_distance: float = 3.0) -> bool:
        return any(
            dist((position.x, position.y), (point.x, point.y)) <= max_distance
            for point in scene.future_trajectory.points
        )

    @staticmethod
    def _is_near_position(reference: Point2D, position: Point2D, max_distance: float) -> bool:
        return dist((reference.x, reference.y), (position.x, position.y)) <= max_distance

    @staticmethod
    def _near_confidence(reference: Point2D, position: Point2D, max_distance: float) -> float:
        distance = dist((reference.x, reference.y), (position.x, position.y))
        return max(0.0, min(1.0, 1.0 - (distance / max_distance)))
