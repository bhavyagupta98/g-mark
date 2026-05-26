from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from kg_coop_drive.application.planning.object_motion_predictor import LearnedObjectMotionPredictor
from kg_coop_drive.application.v2vgotqa_evaluator import GraphAblationMode, V2VGoTQAPhase5AEvaluator
from kg_coop_drive.domain.benchmark import BenchmarkTaskType
from kg_coop_drive.domain.scene import VisibilityState
from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter

_Q5_OBJECT_RE = re.compile(
    r"There is a car at "
    r"\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)\s+"
    r"(?P<action>[^.]+?)\.\s*"
    r"The predicted future trajectory is\s*\[(?P<trajectory>[^\]]+)\]",
    re.IGNORECASE | re.DOTALL,
)
_POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")

_BASE_FEATURES = (
    "x",
    "y",
    "vx",
    "vy",
    "speed",
    "distance_to_trajectory",
    "distance_to_asker",
    "confidence",
    "support_count",
    "conflict_score",
    "uncertainty_score",
    "status_supported",
    "status_candidate",
    "visibility_visible",
    "visibility_occluded",
    "visibility_uncertain",
)

_KG_FEATURES = (
    "support_count",
    "conflict_score",
    "uncertainty_score",
    "status_supported",
    "status_candidate",
    "visibility_visible",
    "visibility_occluded",
    "visibility_uncertain",
)


@dataclass(frozen=True)
class ObjectMotionTrainingExample:
    sample_id: str
    qa_type_id: int
    feature_values: tuple[float, ...]
    target_dx: float
    target_dy: float


@dataclass(frozen=True)
class BuildSummary:
    requested_split: str
    total_samples: int
    matched_examples: int
    skipped_no_gt: int
    skipped_no_tracks: int
    skipped_no_match: int


class ObjectMotionDatasetBuilder:
    """Build Q5/Q7 object-motion training rows with strict split discipline.

    Important leakage rule:
    - Features are built from prepared scene/object state only.
    - Ground-truth answer text is used only to derive supervision targets (dx, dy),
      never as input features.
    """

    def __init__(
        self,
        *,
        v2vgot_root: str,
        file_name: str,
        baseline_mode: str,
        graph_ablation_mode: str,
        max_match_distance: float,
        include_kg_features: bool,
    ) -> None:
        self._v2vgot_root = str(Path(v2vgot_root).expanduser().resolve())
        self._file_name = file_name
        self._baseline_mode = baseline_mode
        self._graph_ablation_mode = graph_ablation_mode
        self._max_match_distance = float(max_match_distance)
        self._include_kg_features = bool(include_kg_features)
        self._adapter = V2VGoTQABenchmarkAdapter(self._v2vgot_root)
        self._evaluator = V2VGoTQAPhase5AEvaluator(
            self._v2vgot_root,
            graph_ablation=graph_ablation_mode,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self._include_kg_features:
            return _BASE_FEATURES
        return tuple(name for name in _BASE_FEATURES if name not in _KG_FEATURES)

    def build(
        self,
        *,
        split_name: str,
        qa_type_ids: tuple[int, ...],
        limit: int,
        progress_every: int,
    ) -> tuple[list[ObjectMotionTrainingExample], BuildSummary]:
        samples = tuple(
            sample
            for sample in self._adapter.load_samples(split_name=split_name, file_name=self._file_name)
            if sample.task_type == BenchmarkTaskType.OBJECT_MOTION_PREDICTION
            and int(sample.qa_type_id or -1) in qa_type_ids
        )
        if limit > 0:
            samples = samples[:limit]

        examples: list[ObjectMotionTrainingExample] = []
        skipped_no_gt = 0
        skipped_no_tracks = 0
        skipped_no_match = 0

        for idx, sample in enumerate(samples, start=1):
            if sample.split_name != split_name:
                raise ValueError(
                    f"Split leakage detected: expected split={split_name}, got sample split={sample.split_name}"
                )

            gt_rows = _parse_gt_answer(sample.raw_record)
            if not gt_rows:
                skipped_no_gt += 1
                continue

            scene = self._evaluator.prepare_sample(sample=sample, baseline_mode=self._baseline_mode)
            if not scene.object_tracks:
                skipped_no_tracks += 1
                continue

            visibility = {
                fact.object_id: fact.state
                for fact in scene.visibility_facts
                if fact.agent_id == scene.asker_agent_id
            }
            available_tracks = list(scene.object_tracks)

            for gt_x, gt_y, gt_future in gt_rows:
                if not available_tracks:
                    break
                best_track, best_distance = _nearest_track(gt_x=gt_x, gt_y=gt_y, tracks=available_tracks)
                if best_track is None or best_distance > self._max_match_distance:
                    skipped_no_match += 1
                    continue
                available_tracks.remove(best_track)
                vis_state = visibility.get(best_track.object_id)
                feature_map = LearnedObjectMotionPredictor._feature_map(  # noqa: SLF001
                    scene=scene,
                    object_track=best_track,
                    visibility_state=vis_state,
                )
                feat = tuple(float(feature_map.get(name, 0.0)) for name in self.feature_names)
                dx = float(gt_future[0][0] - best_track.position.x)
                dy = float(gt_future[0][1] - best_track.position.y)
                examples.append(
                    ObjectMotionTrainingExample(
                        sample_id=str(sample.sample_id),
                        qa_type_id=int(sample.qa_type_id or -1),
                        feature_values=feat,
                        target_dx=dx,
                        target_dy=dy,
                    )
                )

            if progress_every > 0 and (idx % progress_every == 0 or idx == len(samples)):
                print(f"dataset_build_progress split={split_name}: {idx}/{len(samples)} samples")

        summary = BuildSummary(
            requested_split=split_name,
            total_samples=len(samples),
            matched_examples=len(examples),
            skipped_no_gt=skipped_no_gt,
            skipped_no_tracks=skipped_no_tracks,
            skipped_no_match=skipped_no_match,
        )
        return examples, summary


def _parse_gt_answer(raw_record: dict[str, object]) -> list[tuple[float, float, list[tuple[float, float]]]]:
    conversations = raw_record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return []
    answer = conversations[1]
    if not isinstance(answer, dict):
        return []
    answer_text = str(answer.get("value", ""))
    rows: list[tuple[float, float, list[tuple[float, float]]]] = []
    for match in _Q5_OBJECT_RE.finditer(answer_text):
        points = [
            (float(raw_x), float(raw_y))
            for raw_x, raw_y in _POINT_RE.findall(str(match.group("trajectory")))
        ]
        if not points:
            continue
        rows.append((float(match.group("x")), float(match.group("y")), [points[0]]))
    return rows


def _nearest_track(*, gt_x: float, gt_y: float, tracks: Iterable[object]) -> tuple[object | None, float]:
    best_track = None
    best_distance = float("inf")
    for track in tracks:
        d = ((track.position.x - gt_x) ** 2 + (track.position.y - gt_y) ** 2) ** 0.5
        if d < best_distance:
            best_distance = d
            best_track = track
    return best_track, float(best_distance)
