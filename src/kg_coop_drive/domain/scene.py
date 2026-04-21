from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RelationType(str, Enum):
    """Supported relation vocabulary for the first KG prototype."""

    FRONT_OF = "front_of"
    BEHIND = "behind"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    NEAR = "near"
    OBSERVED_BY = "observed_by"
    NEAR_TRAJECTORY = "near_trajectory"


class VisibilityState(str, Enum):
    """Visibility labels attached per agent/object pair."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    UNCERTAIN = "uncertain"


class TrackStatus(str, Enum):
    """Lifecycle status for a scene object track."""

    CONFIRMED = "confirmed"
    SUPPORTED = "supported"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class Point2D:
    """2D position in a shared scene coordinate frame."""

    x: float
    y: float


@dataclass(frozen=True)
class Vector2D:
    """2D vector used for velocity-like quantities."""

    x: float
    y: float


@dataclass(frozen=True)
class Pose2D:
    """Agent pose in the shared coordinate frame."""

    position: Point2D
    yaw_radians: float = 0.0


@dataclass(frozen=True)
class Trajectory:
    """Ordered future-trajectory points for a querying agent."""

    points: tuple[Point2D, ...]


@dataclass(frozen=True)
class AgentContext:
    """Basic metadata for one agent participating in the scene."""

    agent_id: str
    pose: Pose2D


@dataclass(frozen=True)
class ObservationEvidence:
    """One local object-level report from a single agent at one time."""

    observation_id: str
    source_agent_id: str
    object_type: str
    position: Point2D
    confidence: float
    timestamp_index: int
    velocity: Vector2D | None = None


@dataclass(frozen=True)
class ProvenanceRecord:
    """Tracks where a fused object belief came from."""

    source_agent_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    latest_timestamp_index: int


@dataclass(frozen=True)
class VisibilityFact:
    """Visibility state of one object with respect to one agent."""

    agent_id: str
    object_id: str
    state: VisibilityState


@dataclass(frozen=True)
class ObjectTrack:
    """Persistent fused world entity in the cooperative scene."""

    object_id: str
    object_type: str
    position: Point2D
    confidence: float
    provenance: ProvenanceRecord
    status: TrackStatus = TrackStatus.CONFIRMED
    age_frames: int = 1
    miss_count: int = 0
    uncertainty_score: float = 0.0
    conflict_score: float = 0.0
    last_support_confidence: float = 0.0
    velocity: Vector2D | None = None
    observations: tuple[ObservationEvidence, ...] = field(default_factory=tuple)

    def observed_by(self, agent_id: str) -> bool:
        """Return whether the object has evidence from the given agent."""

        return agent_id in self.provenance.source_agent_ids


@dataclass(frozen=True)
class RelationFact:
    """Derived relation between two entities in the scene."""

    subject_id: str
    relation_type: RelationType
    object_id: str
    confidence: float


@dataclass(frozen=True)
class CooperativeScene:
    """Canonical Phase 2 scene container for querying and later fusion."""

    scene_id: str
    local_timestamp_index: int
    global_timestamp_index: int
    asker_agent_id: str
    agents: tuple[AgentContext, ...]
    future_trajectory: Trajectory
    observations: tuple[ObservationEvidence, ...] = field(default_factory=tuple)
    object_tracks: tuple[ObjectTrack, ...] = field(default_factory=tuple)
    relations: tuple[RelationFact, ...] = field(default_factory=tuple)
    visibility_facts: tuple[VisibilityFact, ...] = field(default_factory=tuple)
    raw_question: str = ""
    raw_answer: str = ""

    def get_object(self, object_id: str) -> ObjectTrack | None:
        """Return the object track with the requested identifier, if present."""

        for object_track in self.object_tracks:
            if object_track.object_id == object_id:
                return object_track
        return None


@dataclass(frozen=True)
class QueryResult:
    """Intermediate immutable query result over scene objects."""

    scene: CooperativeScene
    objects: tuple[ObjectTrack, ...]

    def count(self) -> int:
        """Return the number of selected objects."""

        return len(self.objects)

    def exists(self) -> bool:
        """Return whether the current selection is non-empty."""

        return bool(self.objects)


@dataclass(frozen=True)
class QueryAttributeValue:
    """Structured attribute lookup result for one selected object."""

    object_id: str
    attribute_name: str
    value: object


@dataclass(frozen=True)
class QueryComparison:
    """Structured comparison result across selected object attributes."""

    attribute_name: str
    left_object_id: str
    right_object_id: str
    relation: str
    left_value: object
    right_value: object


@dataclass(frozen=True)
class ProvenanceTrace:
    """Structured provenance trace for one selected object."""

    object_id: str
    source_agent_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    latest_timestamp_index: int


@dataclass(frozen=True)
class SceneInterpretation:
    """Human-readable interpretation of what a scene seed currently represents."""

    summary: str
    assumptions: tuple[str, ...]
    observed_capabilities: tuple[str, ...]
    missing_information: tuple[str, ...]


@dataclass(frozen=True)
class SceneBuildReport:
    """Structured result for building and interpreting a scene."""

    scene: CooperativeScene
    interpretation: SceneInterpretation
    build_steps: tuple[str, ...]


@dataclass(frozen=True)
class ObservationTrackAssociation:
    """One matched detector observation supporting one object track."""

    track_id: str
    observation_id: str
    source_agent_id: str
    distance_meters: float
    observation_confidence: float


@dataclass(frozen=True)
class ObservationAssociationReport:
    """Association result between current observations and object tracks."""

    matches: tuple[ObservationTrackAssociation, ...]
    unmatched_track_ids: tuple[str, ...]
    unmatched_observation_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateResolutionReport:
    """Resolution result for prediction-only candidate tracks."""

    kept_candidate_ids: tuple[str, ...]
    pruned_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrackMerge:
    """One conservative merge from a candidate track into a stronger track."""

    source_track_id: str
    target_track_id: str
    distance_meters: float


@dataclass(frozen=True)
class TrackMergeReport:
    """Merge decisions for candidate tracks after pruning."""

    merges: tuple[TrackMerge, ...]
    remaining_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class TemporalTrackUpdateReport:
    """Frame-to-frame persistence decisions for track identities."""

    persisted_track_ids: tuple[str, ...]
    new_track_ids: tuple[str, ...]
    retained_stale_track_ids: tuple[str, ...]
    pruned_stale_track_ids: tuple[str, ...]


@dataclass(frozen=True)
class CrossAgentAssociation:
    """One plausible same-object match between observations from different agents."""

    left_observation_id: str
    left_agent_id: str
    right_observation_id: str
    right_agent_id: str
    distance_meters: float
    confidence: float


@dataclass(frozen=True)
class CrossAgentAssociationReport:
    """Cross-agent observation matching results for one frame."""

    matches: tuple[CrossAgentAssociation, ...]
    participating_agents: tuple[str, ...]


@dataclass(frozen=True)
class CrossAgentSupportAttachmentReport:
    """Summary of cross-agent observations attached onto existing tracks."""

    attached_match_count: int
    enriched_track_ids: tuple[str, ...]


@dataclass(frozen=True)
class VisibilityReasoningReport:
    """Summary of preserved and conservatively inferred visibility facts."""

    preserved_fact_count: int
    inferred_visible_pairs: tuple[str, ...]
    inferred_uncertain_pairs: tuple[str, ...]
