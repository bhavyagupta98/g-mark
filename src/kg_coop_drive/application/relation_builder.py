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
        near_first_waypoint_distance: float = 6.0,
        path_relevant_distance: float = 4.0,
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
            first_waypoint_confidence = self._waypoint_proximity_confidence(
                scene,
                object_track.position,
                max_distance=near_first_waypoint_distance,
            )
            if first_waypoint_confidence > 0.0:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.NEAR_FIRST_WAYPOINT,
                        object_id=scene.asker_agent_id,
                        confidence=first_waypoint_confidence,
                    )
                )
            path_relevant_confidence = self._path_relevance_confidence(
                scene,
                object_track.position,
                max_distance=path_relevant_distance,
            )
            if path_relevant_confidence > 0.0:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.PATH_RELEVANT,
                        object_id=scene.asker_agent_id,
                        confidence=path_relevant_confidence,
                    )
                )
            if len(object_track.provenance.source_agent_ids) >= 2:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.COOPERATIVELY_SUPPORTED,
                        object_id=scene.asker_agent_id,
                        confidence=min(1.0, 0.5 + 0.2 * len(object_track.provenance.source_agent_ids)),
                    )
                )
            if object_track.conflict_score <= 0.5:
                relation_facts.append(
                    RelationFact(
                        subject_id=object_track.object_id,
                        relation_type=RelationType.LOW_CONFLICT,
                        object_id=scene.asker_agent_id,
                        confidence=max(0.0, min(1.0, 1.0 - object_track.conflict_score)),
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

    @staticmethod
    def _waypoint_proximity_confidence(
        scene: CooperativeScene,
        position: Point2D,
        max_distance: float,
    ) -> float:
        if not scene.future_trajectory.points:
            return 0.0
        first_point = scene.future_trajectory.points[0]
        distance = dist((position.x, position.y), (first_point.x, first_point.y))
        if distance > max_distance:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (distance / max_distance)))

    @staticmethod
    def _path_relevance_confidence(
        scene: CooperativeScene,
        position: Point2D,
        max_distance: float,
    ) -> float:
        if not scene.future_trajectory.points:
            return 0.0
        best_distance = min(
            dist((position.x, position.y), (point.x, point.y))
            for point in scene.future_trajectory.points
        )
        if best_distance > max_distance:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (best_distance / max_distance)))
