from __future__ import annotations

from dataclasses import replace

from kg_coop_drive.domain.processed_scene import ProcessedFrameSceneData
from kg_coop_drive.domain.scene import CooperativeScene


class ProcessedSceneEnricher:
    """Attaches processed object and visibility data to an existing scene seed."""

    def enrich(
        self,
        scene: CooperativeScene,
        processed_data: ProcessedFrameSceneData,
    ) -> CooperativeScene:
        """Return a new scene with processed content populated."""

        return replace(
            scene,
            observations=processed_data.observations,
            object_tracks=processed_data.object_tracks,
            visibility_facts=processed_data.visibility_facts,
        )
