# systemPatterns.md

System architecture, design patterns, and critical flows.
Initialized: 2025-08-14T00:00:00Z

## Architecture Overview
- Repository is primarily Python-based with analysis and simulation modules.

## Key Technical Decisions
- Memory Bank will be stored in memory-bank/ and treated as authoritative project context.

## Design Patterns in Use
- Modular analysis scripts grouped under `mtbs_fire_analysis/analysis` and `CA_Simulation/`.

## Component/Module Relationships
- Analysis scripts read data from `mtbs_fire_analysis/data/` and write outputs to `mtbs_fire_analysis/outputs/`.

## Critical Implementation Paths
- Data ingestion -> HLH fitting -> lookup generation -> burn probability scoring
