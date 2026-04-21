from __future__ import annotations

from pathlib import Path

import numpy as np

from kg_coop_drive.domain.processed_scene import (
    ProcessedFrameSceneData,
    ProcessedSceneAvailability,
)
from kg_coop_drive.domain.scene import (
    ObjectTrack,
    ObservationEvidence,
    Point2D,
    ProvenanceRecord,
    VisibilityFact,
    VisibilityState,
)


class V2VGoTProcessedAssetLoader:
    """Loads processed timestamped GT and visibility arrays from V2V-GoT assets."""

    def __init__(self, repository_root: str) -> None:
        self._repository_root = Path(repository_root).expanduser().resolve()

    def inspect_availability(
        self,
        timestamp_index: int,
        split_name: str = "val",
    ) -> ProcessedSceneAvailability:
        """Inspect whether the required processed files exist for one frame."""

        npy_root = self._resolve_npy_root(
            timestamp_index=timestamp_index,
            split_name=split_name,
        )
        return ProcessedSceneAvailability(
            npy_root=str(npy_root),
            timestamp_index=timestamp_index,
            has_gt_boxes=self._timestamp_path(npy_root, timestamp_index, "gt").exists(),
            has_gt_ids=self._timestamp_path(
                npy_root, timestamp_index, "gt_object_id"
            ).exists(),
            has_visibility_for_ego=self._visibility_path(
                npy_root, timestamp_index, "ego", visible=True
            ).exists(),
            has_visibility_for_cav1=self._visibility_path(
                npy_root, timestamp_index, "1", visible=True
            ).exists(),
            has_pred_for_ego=self._find_prediction_source(npy_root, timestamp_index, "ego") is not None,
            has_pred_for_cav1=self._find_prediction_source(npy_root, timestamp_index, "1") is not None,
        )

    def list_available_timestamps(
        self,
        split_name: str = "val",
    ) -> tuple[int, ...]:
        """Return sorted timestamps discovered under the resolved processed root."""

        npy_root = self._resolve_npy_root(timestamp_index=0, split_name=split_name)
        return self.list_available_timestamps_for_root(npy_root)

    def list_available_timestamps_for_root(
        self,
        npy_root: Path,
    ) -> tuple[int, ...]:
        """Return sorted timestamps discovered under one explicit processed root."""

        timestamp_indices: set[int] = set()

        for path in npy_root.glob("*.npy"):
            stem = path.stem
            prefix = stem.split("_", 1)[0]
            if prefix.isdigit():
                timestamp_indices.add(int(prefix))

        for cav_id in ("ego", "1"):
            cav_dir = npy_root / cav_id
            if not cav_dir.exists():
                continue
            for path in cav_dir.glob("*.npy"):
                stem = path.stem
                prefix = stem.split("_", 1)[0]
                if prefix.isdigit():
                    timestamp_indices.add(int(prefix))

        collm_root = npy_root / "co_llm"
        if collm_root.exists():
            for path in collm_root.glob("*/*.npy"):
                stem = path.stem
                prefix = stem.split("_", 1)[0]
                if prefix.isdigit():
                    timestamp_indices.add(int(prefix))

        return tuple(sorted(timestamp_indices))

    def list_candidate_npy_roots(
        self,
        split_name: str = "val",
    ) -> tuple[Path, ...]:
        """Return the candidate processed roots that currently exist on disk."""

        return tuple(
            candidate
            for candidate in self._candidate_npy_roots(split_name=split_name)
            if candidate.exists()
        )

    def load_frame_scene_data(
        self,
        timestamp_index: int,
        split_name: str = "val",
    ) -> ProcessedFrameSceneData | None:
        """Load GT object tracks and visibility facts for one timestamp if available."""

        availability = self.inspect_availability(
            timestamp_index=timestamp_index,
            split_name=split_name,
        )
        if not availability.has_gt_boxes or not availability.has_gt_ids:
            return None

        npy_root = Path(availability.npy_root)
        gt_corners = np.load(
            self._timestamp_path(npy_root, timestamp_index, "gt"),
            allow_pickle=True,
        )
        gt_object_ids = np.load(
            self._timestamp_path(npy_root, timestamp_index, "gt_object_id"),
            allow_pickle=True,
        )
        observations = self._load_observations(npy_root, timestamp_index)

        tracks = tuple(
            self._build_object_track(object_id=str(gt_object_ids[index]), corners=gt_corners[index], timestamp_index=timestamp_index)
            for index in range(len(gt_object_ids))
        )

        visibility_facts: list[VisibilityFact] = []
        for cav_id, agent_id in (("ego", "CAV_EGO"), ("1", "CAV_1")):
            visible_ids = self._load_visibility_ids(
                npy_root=npy_root,
                timestamp_index=timestamp_index,
                cav_id=cav_id,
                visible=True,
            )
            invisible_ids = self._load_visibility_ids(
                npy_root=npy_root,
                timestamp_index=timestamp_index,
                cav_id=cav_id,
                visible=False,
            )
            visibility_facts.extend(
                VisibilityFact(
                    agent_id=agent_id,
                    object_id=str(object_id),
                    state=VisibilityState.VISIBLE,
                )
                for object_id in visible_ids
            )
            visibility_facts.extend(
                VisibilityFact(
                    agent_id=agent_id,
                    object_id=str(object_id),
                    state=VisibilityState.OCCLUDED,
                )
                for object_id in invisible_ids
            )

        source_paths = [str(self._timestamp_path(npy_root, timestamp_index, "gt"))]
        source_paths.append(str(self._timestamp_path(npy_root, timestamp_index, "gt_object_id")))
        for observation in observations:
            source_paths.append(observation.observation_id.split("::", 1)[0])
        for cav_id in ("ego", "1"):
            for visible in (True, False):
                path = self._visibility_path(
                    npy_root=npy_root,
                    timestamp_index=timestamp_index,
                    cav_id=cav_id,
                    visible=visible,
                )
                if path.exists():
                    source_paths.append(str(path))
        return ProcessedFrameSceneData(
            timestamp_index=timestamp_index,
            observations=observations,
            object_tracks=tracks,
            visibility_facts=tuple(visibility_facts),
            source_paths=tuple(dict.fromkeys(source_paths)),
        )

    def _resolve_npy_root(self, timestamp_index: int, split_name: str) -> Path:
        candidates = self._candidate_npy_roots(split_name=split_name)

        preferred_suffixes = ("gt_object_id", "gt")
        for suffix in preferred_suffixes:
            for candidate in candidates:
                if self._timestamp_path(candidate, timestamp_index, suffix).exists():
                    return candidate

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def _candidate_npy_roots(self, split_name: str) -> tuple[Path, ...]:
        split_dir = (
            "no_fusion_keep_all"
            if split_name == "val"
            else "train_no_fusion_keep_all"
        )
        return (
            self._repository_root / "cobevt" / "npy",
            self._repository_root / "DMSTrack" / "V2V4Real" / "official_models" / split_dir / "npy",
            self._repository_root / "DMSTrack" / "V2V4Real" / "official_models" / "no_fusion_keep_all" / "npy",
            self._repository_root / "DMSTrack" / "V2V4Real" / "official_models" / "train_no_fusion_keep_all" / "npy",
        )

    def _load_observations(
        self,
        npy_root: Path,
        timestamp_index: int,
    ) -> tuple[ObservationEvidence, ...]:
        observations: list[ObservationEvidence] = []
        for cav_id, agent_id in (("ego", "CAV_EGO"), ("1", "CAV_1")):
            prediction_source = self._find_prediction_source(npy_root, timestamp_index, cav_id)
            if prediction_source is None:
                continue
            path, source_type = prediction_source
            if source_type == "co_llm_detection_box_score":
                box_scores = np.load(path, allow_pickle=True)
            else:
                pred_path = path
                score_path = self._prediction_score_path(npy_root, timestamp_index, cav_id)
                if score_path is None:
                    continue
                pred_corners = np.load(pred_path, allow_pickle=True)
                pred_scores = np.load(score_path, allow_pickle=True)
                box_scores = self._corners_and_scores_to_box_features(pred_corners, pred_scores)

            for index, feature in enumerate(box_scores):
                observations.append(
                    ObservationEvidence(
                        observation_id=f"{path}::obs_{cav_id}_{timestamp_index}_{index}",
                        source_agent_id=agent_id,
                        object_type="car",
                        position=Point2D(x=float(feature[3]), y=float(feature[5])),
                        confidence=float(feature[7]),
                        timestamp_index=timestamp_index,
                    )
                )
        return tuple(observations)

    @staticmethod
    def _timestamp_path(npy_root: Path, timestamp_index: int, suffix: str) -> Path:
        return npy_root / f"{timestamp_index:04d}_{suffix}.npy"

    @staticmethod
    def _visibility_path(
        npy_root: Path,
        timestamp_index: int,
        cav_id: str,
        visible: bool,
    ) -> Path:
        label = "visible" if visible else "invisible"
        return npy_root / f"{timestamp_index:04d}_gt_object_id_{label}_to_{cav_id}.npy"

    def _load_visibility_ids(
        self,
        npy_root: Path,
        timestamp_index: int,
        cav_id: str,
        visible: bool,
    ) -> tuple[str, ...]:
        path = self._visibility_path(
            npy_root=npy_root,
            timestamp_index=timestamp_index,
            cav_id=cav_id,
            visible=visible,
        )
        if not path.exists():
            return tuple()
        values = np.load(path, allow_pickle=True)
        return tuple(str(value) for value in values.tolist())

    def _find_prediction_source(
        self,
        npy_root: Path,
        timestamp_index: int,
        cav_id: str,
    ) -> tuple[Path, str] | None:
        per_agent_pred_path = npy_root / cav_id / f"{timestamp_index:04d}_pred.npy"
        if per_agent_pred_path.exists():
            return per_agent_pred_path, "per_agent_pred"

        collm_detection_path = (
            npy_root / "co_llm" / cav_id / f"{timestamp_index:04d}_detection_box_score.npy"
        )
        if collm_detection_path.exists():
            return collm_detection_path, "co_llm_detection_box_score"

        return None

    def prediction_source_exists_for_root(
        self,
        npy_root: Path,
        timestamp_index: int,
        cav_id: str,
    ) -> bool:
        """Return whether an observation source exists for one agent under one root."""

        return self._find_prediction_source(npy_root, timestamp_index, cav_id) is not None

    def _prediction_score_path(
        self,
        npy_root: Path,
        timestamp_index: int,
        cav_id: str,
    ) -> Path | None:
        score_path = npy_root / cav_id / f"{timestamp_index:04d}_pred_score.npy"
        if score_path.exists():
            return score_path
        return None

    @staticmethod
    def _corners_and_scores_to_box_features(
        pred_corners: np.ndarray,
        pred_scores: np.ndarray,
    ) -> np.ndarray:
        min_xyz = np.min(pred_corners, axis=1)
        max_xyz = np.max(pred_corners, axis=1)
        hwl = max_xyz - min_xyz
        center = np.mean(pred_corners, axis=1)
        yaw = np.zeros((pred_corners.shape[0], 1))
        box_features = np.concatenate(
            [
                hwl,
                center,
                yaw,
                np.expand_dims(pred_scores, axis=1),
            ],
            axis=1,
        )
        return box_features

    @staticmethod
    def _build_object_track(
        object_id: str,
        corners: np.ndarray,
        timestamp_index: int,
    ) -> ObjectTrack:
        center = np.mean(corners, axis=0)
        position = Point2D(x=float(center[0]), y=float(center[2]))
        provenance = ProvenanceRecord(
            source_agent_ids=("GT",),
            observation_ids=(f"gt_{object_id}_{timestamp_index}",),
            latest_timestamp_index=timestamp_index,
        )
        return ObjectTrack(
            object_id=object_id,
            object_type="car",
            position=position,
            confidence=1.0,
            provenance=provenance,
        )
