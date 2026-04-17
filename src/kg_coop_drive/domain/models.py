from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssetReference:
    """Describes an external project or artifact we plan to reuse."""

    name: str
    path: str
    role: str
    strengths: tuple[str, ...] = field(default_factory=tuple)
    gaps: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PrototypeDefinition:
    """Captures the initial narrow prototype we are committing to build first."""

    name: str
    objective: str
    in_scope: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    success_criteria: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryModule:
    """Represents one logical module in the project layout."""

    name: str
    responsibility: str
