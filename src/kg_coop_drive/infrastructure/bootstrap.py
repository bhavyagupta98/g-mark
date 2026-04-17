from __future__ import annotations

from kg_coop_drive.domain.models import AssetReference, PrototypeDefinition, RepositoryModule


class StaticAssetInventory:
    """Filesystem-independent inventory for Phase 0 planning."""

    def load_assets(self) -> tuple[AssetReference, ...]:
        return (
            AssetReference(
                name="V2V-GoT",
                path="/Users/bhavya/Desktop/ms_projects/V2V-GoT",
                role="Dataset, benchmark, and evaluation reference",
                strengths=(
                    "Cooperative driving benchmark context",
                    "V2V4Real data plumbing",
                    "QA generation scripts",
                    "Grounding evaluation scripts",
                ),
                gaps=(
                    "No cooperative knowledge graph abstraction",
                    "No provenance-aware fusion layer",
                    "No typed graph reasoning tools",
                ),
            ),
            AssetReference(
                name="auto_drive_copy",
                path="/Users/bhavya/Desktop/ms_projects/auto_drive_copy",
                role="Tool-constrained reasoning design reference",
                strengths=(
                    "Typed tool execution loop",
                    "Grounded scene binding pattern",
                    "Deterministic relation and counting tools",
                ),
                gaps=(
                    "No multi-agent fusion",
                    "No cooperative visibility reasoning",
                    "No V2V object association",
                ),
            ),
        )


class StaticPrototypeDefinition:
    """Default Phase 0 prototype target."""

    def load_definition(self) -> PrototypeDefinition:
        return PrototypeDefinition(
            name="Two-agent single-frame cooperative KG prototype",
            objective=(
                "Build local scene graphs for two agents, fuse them into a cooperative "
                "graph, and answer a small deterministic QA subset."
            ),
            in_scope=(
                "Two agents",
                "Single-frame reasoning",
                "Existence queries",
                "Count queries",
                "Basic relative position queries",
                "Deterministic graph queries",
            ),
            out_of_scope=(
                "Planning and trajectory generation",
                "Long-horizon prediction",
                "Full LLM planner",
                "Networked V2V communication",
            ),
            success_criteria=(
                "Correct simple cross-agent object merges",
                "Provenance retained for fused objects",
                "Hand-picked QA subset answered deterministically",
            ),
        )


class StaticProjectLayout:
    """Logical repository modules for the initial implementation."""

    def load_modules(self) -> tuple[RepositoryModule, ...]:
        return (
            RepositoryModule(name="domain", responsibility="Core abstractions, models, and contracts"),
            RepositoryModule(name="application", responsibility="Use-case orchestration and workflow coordination"),
            RepositoryModule(name="infrastructure", responsibility="Filesystem and external project adapters"),
        )
