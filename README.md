# Non-RSI Intelligence Core

Implementation of the "Path B x Path C" architecture hypothesis for system-level self-improvement without source code modification.

## Core Concepts

### Path B: The Fixed-Architecture Node
Represents an individual agent with a static Python codebase. It consists of:
*   **World Model**: Learns transition probabilities and reward expectations.
*   **Planner**: Executes lookahead search over the world model.
*   **Skill Engine**: Interprets data-level tool sequences (DSL) that can be synthesized at runtime.
*   **Evolution**: Occurs via parameter updates and state accumulation, not source rewriting.

### Path C: The Collective System
Represents the multi-agent orchestration layer. It manages:
*   **Shared Knowledge Base**: Stores episodic memory, verified principles, and artifacts.
*   **Project Graph**: Tracks long-horizon goals and dependencies.
*   **Resource Allocation**: Dynamically assigns compute budgets and agent roles based on performance variance.
*   **Evolution**: Occurs via organizational policy adaptation and infrastructure growth.

## Mechanism
The system runs a simulation where 'B-type' agents generate tools and knowledge artifacts. The 'C-type' system integrates these outputs to optimize the collective policy, enabling the solution of progressively harder tasks through data-driven feedback loops rather than code modification.

## Usage
`python NON_RSI_AGI_CORE_v2.py --rounds 40 --agents 8`
