# Experiments Directory

This directory contains experimental implementations and approaches that are not yet part of the main codebase.

## v7 Tournament System

**Status**: Experimental (Not Production)

**Performance**: +3.8% improvement over v5 in fair benchmarks

**Core Innovation**:
- Elo-based tournament selection for skill evolution
- Head-to-head skill comparison on identical environment snapshots
- Solves the "fitness-environment coupling" problem

**Files**:
- `v7_tournament_system.py` - Main implementation
- `verify_v7_evolution.py` - Determinism and evolution tests
- `fair_benchmark.py` - Fair comparison vs v5

**Why Experimental?**:
- Increased complexity (2x code size vs v5)
- Performance gain (+3.8%) does not justify complexity in simple environments
- Architecture designed for future complex tasks

**Future Consideration**:
This system may be promoted to main codebase when applied to more complex environments where tournament-based selection shows significantly higher value.
