# Non-RSI Simulation Core

A single-file Python simulation demonstrating system-level performance acceleration without source code modification.

## Structure

The simulation implements two primary classes to model the "Fixed Agent x Collective System" interaction:

### Component B: Agent Class (Fixed Architecture)
Instances of the `Agent` class that process observations using static logic:
*   **World Model**: A hash-based state transition tracker (learns P(s'|s,a)).
*   **Planner**: A fixed-depth lookahead searcher using the world model.
*   **Skill Interpreter**: Executes data-driven tool sequences (DSL) stored in memory.
*   **Update Mechanism**: Updates internal weight parameters and state history; **no code modification**.

### Component C: Orchestrator Class (Collective System)
The `Orchestrator` class that manages the main loop and shared state:
*   **Shared Memory**: A dictionary-based store for artifacts and episodic logs.
*   **Project Graph**: A directed graph structure to track dependency chains.
*   **Scheduler**: Assigns `Agent` instances to `ProjectNode` items based on heuristic policy.
*   **Adaptation**: Adjusts global simulation parameters (e.g., risk tolerance) based on output variance.

## Simulation Logic
The script runs a loop where `Agent` objects interact with a synthetic `ResearchEnvironment`. The environment returns reward signals which are aggregated by the `Orchestrator` to update the global policy and shared memory context, demonstrating accumulation of performance without changing the Python source file.

## Usage
`python NON_RSI_AGI_CORE_v2.py --rounds 40 --agents 8`
