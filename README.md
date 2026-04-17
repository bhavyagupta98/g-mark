# KG Cooperative Drive

This repository contains the implementation plan and code scaffold for a cooperative driving system built around:

- grounded local scene facts per CAV
- structured V2V fact exchange
- provenance-aware cooperative knowledge graphs
- constrained reasoning over graph tools

## Current Status

Phase 0 is in progress.

The current repository includes:

- the implementation roadmap in [plan.md](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/plan.md:1)
- a Phase 0 asset inventory in [docs/phase0_asset_inventory.md](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/docs/phase0_asset_inventory.md:1)
- a Phase 0 prototype definition in [docs/phase0_prototype_definition.md](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/docs/phase0_prototype_definition.md:1)
- an initial modular Python package scaffold under [src/kg_coop_drive](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/src/kg_coop_drive)

## Design Principles

The codebase should remain:

- modular
- explicit
- testable
- inspectable
- easy to extend without large refactors

We will follow these engineering rules:

- Single Responsibility Principle for core classes
- Open/Closed Principle via interfaces and adapters
- Dependency Inversion for domain logic
- clear separation between domain, application, and infrastructure
- deterministic components before LLM-driven components

## Planned Package Layout

```text
src/kg_coop_drive/
  application/
  domain/
  infrastructure/
tests/
docs/
```

## Immediate Next Step

Phase 1 begins with:

1. loading one V2V-GoT sample
2. defining the canonical schema in code
3. mapping a narrow QA subset to structured graph queries
