from __future__ import annotations

from math import dist

from kg_coop_drive.domain.scene import (
    CooperativeScene,
    ProvenanceTrace,
    QueryAttributeValue,
    QueryComparison,
    QueryResult,
    RelationType,
    VisibilityState,
)


class SceneQueryEngine:
    """Deterministic KG query interface for the first prototype."""

    _COMPARABLE_ATTRIBUTES = {
        "confidence",
        "position_x",
        "position_y",
        "support_count",
        "uncertainty_score",
        "conflict_score",
    }

    def select_objects(self, scene: CooperativeScene) -> QueryResult:
        """Select all currently tracked objects."""

        return QueryResult(scene=scene, objects=scene.object_tracks)

    def filter_by_type(self, result: QueryResult, object_type: str) -> QueryResult:
        """Keep only objects of the requested semantic type."""

        filtered = tuple(
            object_track
            for object_track in result.objects
            if object_track.object_type == object_type
        )
        return QueryResult(scene=result.scene, objects=filtered)

    def filter_by_visibility(
        self,
        result: QueryResult,
        agent_id: str,
        visibility: VisibilityState = VisibilityState.VISIBLE,
    ) -> QueryResult:
        """Keep only objects with the requested visibility state for one agent."""

        allowed_ids = {
            fact.object_id
            for fact in result.scene.visibility_facts
            if fact.agent_id == agent_id and fact.state == visibility
        }
        filtered = tuple(
            object_track
            for object_track in result.objects
            if object_track.object_id in allowed_ids
        )
        return QueryResult(scene=result.scene, objects=filtered)

    def filter_by_source_agent(
        self,
        result: QueryResult,
        source_agent_id: str,
    ) -> QueryResult:
        """Keep only objects whose provenance includes the requested source agent."""

        filtered = tuple(
            object_track
            for object_track in result.objects
            if source_agent_id in object_track.provenance.source_agent_ids
        )
        return QueryResult(scene=result.scene, objects=filtered)

    def filter_by_relation(
        self,
        result: QueryResult,
        relation_type: RelationType,
        reference_id: str,
    ) -> QueryResult:
        """Keep objects linked to a reference entity through one relation type."""

        allowed_ids = {
            fact.subject_id
            for fact in result.scene.relations
            if fact.relation_type == relation_type and fact.object_id == reference_id
        }
        filtered = tuple(
            object_track
            for object_track in result.objects
            if object_track.object_id in allowed_ids
        )
        return QueryResult(scene=result.scene, objects=filtered)

    def filter_near_trajectory(
        self,
        result: QueryResult,
        max_distance: float,
    ) -> QueryResult:
        """Keep objects that lie within a distance threshold of any trajectory point."""

        points = result.scene.future_trajectory.points
        filtered = tuple(
            object_track
            for object_track in result.objects
            if any(
                dist(
                    (object_track.position.x, object_track.position.y),
                    (point.x, point.y),
                )
                <= max_distance
                for point in points
            )
        )
        return QueryResult(scene=result.scene, objects=filtered)

    @staticmethod
    def count(result: QueryResult) -> int:
        """Return the number of objects in a query result."""

        return result.count()

    @staticmethod
    def exists(result: QueryResult) -> bool:
        """Return whether a query result is non-empty."""

        return result.exists()

    def get_attribute(
        self,
        result: QueryResult,
        attribute_name: str,
    ) -> tuple[QueryAttributeValue, ...]:
        """Return structured attribute values for the selected objects."""

        return tuple(
            QueryAttributeValue(
                object_id=object_track.object_id,
                attribute_name=attribute_name,
                value=self._read_attribute(object_track, attribute_name),
            )
            for object_track in result.objects
        )

    def compare(
        self,
        result: QueryResult,
        attribute_name: str,
    ) -> tuple[QueryComparison, ...]:
        """Compare one attribute across all object pairs in a result."""

        comparisons: list[QueryComparison] = []
        for left_index, left_track in enumerate(result.objects):
            left_value = self._read_attribute(left_track, attribute_name)
            if left_value is None:
                continue
            for right_track in result.objects[left_index + 1 :]:
                right_value = self._read_attribute(right_track, attribute_name)
                if right_value is None:
                    continue
                relation = self._compare_values(attribute_name, left_value, right_value)
                comparisons.append(
                    QueryComparison(
                        attribute_name=attribute_name,
                        left_object_id=left_track.object_id,
                        right_object_id=right_track.object_id,
                        relation=relation,
                        left_value=left_value,
                        right_value=right_value,
                    )
                )
        return tuple(comparisons)

    @staticmethod
    def trace_provenance(result: QueryResult) -> tuple[ProvenanceTrace, ...]:
        """Return provenance traces for the selected objects."""

        return tuple(
            ProvenanceTrace(
                object_id=object_track.object_id,
                source_agent_ids=object_track.provenance.source_agent_ids,
                observation_ids=object_track.provenance.observation_ids,
                latest_timestamp_index=object_track.provenance.latest_timestamp_index,
            )
            for object_track in result.objects
        )

    @staticmethod
    def _read_attribute(object_track, attribute_name: str):
        if attribute_name == "confidence":
            return object_track.confidence
        if attribute_name == "object_type":
            return object_track.object_type
        if attribute_name == "status":
            return object_track.status.value
        if attribute_name == "position_x":
            return object_track.position.x
        if attribute_name == "position_y":
            return object_track.position.y
        if attribute_name == "support_count":
            return len(object_track.observations)
        if attribute_name == "uncertainty_score":
            return object_track.uncertainty_score
        if attribute_name == "conflict_score":
            return object_track.conflict_score
        return None

    def _compare_values(self, attribute_name: str, left_value: object, right_value: object) -> str:
        if attribute_name not in self._COMPARABLE_ATTRIBUTES:
            return "not_comparable"
        try:
            if left_value < right_value:
                return "less_than"
            if left_value > right_value:
                return "greater_than"
            return "equal"
        except TypeError:
            return "not_comparable"
