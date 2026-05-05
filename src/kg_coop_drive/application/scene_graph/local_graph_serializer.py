from __future__ import annotations

import json

from kg_coop_drive.domain.scene import CooperativeScene


class LocalGraphSerializer:
    """Serializes a local graph into a deterministic JSON string for inspection."""

    def to_json(self, scene: CooperativeScene) -> str:
        """Return a stable JSON rendering of a local graph scene."""

        payload = {
            "scene_id": scene.scene_id,
            "local_timestamp_index": scene.local_timestamp_index,
            "global_timestamp_index": scene.global_timestamp_index,
            "asker_agent_id": scene.asker_agent_id,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "position": {
                        "x": agent.pose.position.x,
                        "y": agent.pose.position.y,
                    },
                    "yaw_radians": agent.pose.yaw_radians,
                }
                for agent in scene.agents
            ],
            "object_tracks": [
                {
                    "object_id": track.object_id,
                    "object_type": track.object_type,
                    "position": {
                        "x": track.position.x,
                        "y": track.position.y,
                    },
                    "status": track.status.value,
                    "confidence": track.confidence,
                    "provenance_agents": list(track.provenance.source_agent_ids),
                    "support_count": len(track.observations),
                    "uncertainty_score": track.uncertainty_score,
                    "conflict_score": track.conflict_score,
                }
                for track in scene.object_tracks
            ],
            "relations": [
                {
                    "subject_id": relation.subject_id,
                    "relation_type": relation.relation_type.value,
                    "object_id": relation.object_id,
                    "confidence": relation.confidence,
                }
                for relation in scene.relations
            ],
            "visibility_facts": [
                {
                    "agent_id": fact.agent_id,
                    "object_id": fact.object_id,
                    "state": fact.state.value,
                }
                for fact in scene.visibility_facts
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True)
