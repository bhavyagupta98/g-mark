from __future__ import annotations

from typing import Protocol


class AssetInventoryProvider(Protocol):
    """Supplies reusable project assets and their intended roles."""

    def load_assets(self) -> tuple[object, ...]:
        """Return the known reusable assets."""


class PrototypeDefinitionProvider(Protocol):
    """Supplies the current prototype scope for the project."""

    def load_definition(self) -> object:
        """Return the active prototype definition."""


class ProjectLayoutProvider(Protocol):
    """Supplies repository module layout information."""

    def load_modules(self) -> tuple[object, ...]:
        """Return the logical project modules."""


class DatasetInspector(Protocol):
    """Supplies a structured inspection report for a dataset or benchmark source."""

    def inspect(self) -> object:
        """Return a dataset inspection report."""


class SceneAdapter(Protocol):
    """Builds an internal scene representation from one external sample."""

    def build_scene(self, record: dict[str, object]) -> object:
        """Return the canonical scene object for one raw record."""


class SceneBuilderPort(Protocol):
    """Builds a narrated interpretation around one canonical scene."""

    def build(self, scene: object) -> object:
        """Return the interpreted scene report."""
