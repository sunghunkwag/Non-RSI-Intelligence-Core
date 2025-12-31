"""
NON_RSI_AGI_CORE_v2.py
======================

Architecture goal:
- "Quasi-AGI" Architecture (Jun-AGI Level).
- Pure Python simulation of advanced cognitive architectures.
- Key Components:
  - Holographic Associative Memory (HAM): Bitwise hypervector simulation.
  - Fractal Recursive Planner: Dynamic goal decomposition.
  - Evolutionary Orchestrator: Genetic optimization of cognitive genomes.
  - Emergent Concept Blending: Creating new skills from successful patterns.

Run:
  python NON_RSI_AGI_CORE_v2.py --rounds 40 --agents 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set


# ----------------------------
# Utility & HyperVector Core
# ----------------------------

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    buf: List[str] = []
    cur: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
    if cur:
        buf.append("".join(cur))
    return buf


class HyperVector:
    """
    Simulated Holographic Vector (64-bit).
    - Supports superposition (OR/Majority), binding (XOR), and similarity (Hamming).
    - Used for semantic association in memory without embedding APIs.
    """
    def __init__(self, val: Optional[int] = None, dim: int = 64) -> None:
        self.dim = dim
        if val is None:
            self.val = random.getrandbits(dim)
        else:
            self.val = val

    @classmethod
    def from_text(cls, text: str, dim: int = 64) -> "HyperVector":
        # Deterministic projection from string space to vector space
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        return cls(h & ((1 << dim) - 1), dim)

    def bind(self, other: "HyperVector") -> "HyperVector":
        return HyperVector(self.val ^ other.val, self.dim)

    def superpose(self, other: "HyperVector") -> "HyperVector":
        # Simple OR for simulation (preserves features of both)
        return HyperVector(self.val | other.val, self.dim)

    def similarity(self, other: "HyperVector") -> float:
        xor = self.val ^ other.val
        dist = bin(xor).count('1')
        return 1.0 - (dist / self.dim)

    def __repr__(self) -> str:
        return f"HV({hex(self.val)})"


# ----------------------------
# Advanced Neuro-Symbolic Memory
# ----------------------------

@dataclass
class MemoryItem:
    ts_ms: int
    kind: str
    title: str
    content: Dict[str, Any]
    tags: List[str]
    vector: HyperVector
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {"ts": self.ts_ms, "k": self.kind, "t": self.title, "c": self.content}
        )


class NeuroSymbolicMemory:
    """
    Holographic Associative Memory.
    - Stores items with semantic vectors.
    - Retrieval is based on vector similarity (simulating neural activation).
    - Supports 'dreaming' (consolidation).
    """

    def __init__(self, max_items: int = 8000) -> None:
        self.max_items = max_items
        self._items: List[MemoryItem] = []
        self._concept_vectors: Dict[str, HyperVector] = {}

    def _get_vector(self, text: str) -> HyperVector:
        # Cache vectors for words/tags
        if text not in self._concept_vectors:
            self._concept_vectors[text] = HyperVector.from_text(text)
        return self._concept_vectors[text]

    def _encode(self, title: str, tags: List[str], content: Dict[str, Any]) -> HyperVector:
        # Create a composite vector from title, tags, and content keys
        v = self._get_vector("bias")
        for t in tokenize(title):
            v = v.superpose(self._get_vector(t))
        for t in tags:
            v = v.superpose(self._get_vector(t))
        # Encode keys of content
        for k in content.keys():
            v = v.superpose(self._get_vector(k))
        return v

    def add(self, kind: str, title: str, content: Dict[str, Any],
            tags: Optional[List[str]] = None) -> str:
        tags = tags or []
        vec = self._encode(title, tags, content)
        item = MemoryItem(
            ts_ms=now_ms(), kind=kind, title=title,
            content=content, tags=tags, vector=vec
        )
        self._items.append(item)
        if len(self._items) > self.max_items:
            # Memory decay: remove oldest
            self._items = self._items[-self.max_items:]
        return item.id

    def search(self, query: str, k: int = 10,
               kinds: Optional[List[str]] = None,
               tags: Optional[List[str]] = None) -> List[MemoryItem]:
        q_vec = self._get_vector("bias")
        for t in tokenize(query):
            q_vec = q_vec.superpose(self._get_vector(t))

        scored: List[Tuple[float, MemoryItem]] = []
        t_now = now_ms()

        for it in self._items:
            if kinds and it.kind not in kinds:
                continue
            if tags and not any(t in it.tags for t in tags):
                continue

            sim = it.vector.similarity(q_vec)
            # Recency boost
            recency = 1.0 / (1.0 + (t_now - it.ts_ms) / (1000.0 * 60.0 * 60.0))
            score = sim + 0.2 * recency
            scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def extract_principles(self, k: int = 6) -> List[str]:
        # "Dreaming" process: find high-reward episodes and distill patterns
        episodes = [it for it in self._items if it.kind == "episode"]
        if not episodes:
            return []

        # Sort by reward
        episodes.sort(key=lambda it: float(it.content.get("reward", 0.0)), reverse=True)
        top = episodes[:k]

        created_ids = []
        for ep in top:
            obs = ep.content.get("obs", {})
            action = ep.content.get("action", "")
            # Create a "Principle" vector by binding Task + Action
            p_title = f"principle:{obs.get('task')}:{action}"
            p_content = {
                "action": action,
                "source": ep.id,
                "heuristic": "high_reward",
                "conditions": {
                    "difficulty": obs.get("difficulty"),
                    "domain": obs.get("domain")
                }
            }
            pid = self.add("principle", p_title, p_content, tags=["derived", "principle"])
            created_ids.append(pid)
        return created_ids

    def dump_summary(self, k: int = 15) -> List[Dict[str, Any]]:
        tail = self._items[-k:]
        return [
            {"id": it.id, "title": it.title, "kind": it.kind, "tags": it.tags}
            for it in tail
        ]


# ----------------------------
# Skill DSL & Registry
# ----------------------------

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._tools.get(name)
        if fn is None:
            return {"ok": False, "error": f"unknown_tool:{name}"}
        try:
            return fn(args)
        except Exception as e:
            return {"ok": False, "error": str(e)}

@dataclass
class SkillStep:
    kind: str  # "call", "if", "foreach"
    tool: Optional[str] = None
    args_template: Optional[Dict[str, Any]] = None
    condition: Optional[Dict[str, Any]] = None
    steps: Optional[List["SkillStep"]] = None
    else_steps: Optional[List["SkillStep"]] = None
    list_key: Optional[str] = None
    item_key: Optional[str] = None

@dataclass
class Skill:
    name: str
    purpose: str
    steps: List[SkillStep]
    tags: List[str]
    id: str = field(init=False)

    def __post_init__(self):
        self.id = stable_hash({"n": self.name, "p": self.purpose})

    def run(self, tools: ToolRegistry, context: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal interpreter
        ctx = dict(context)
        trace = []

        def subst(val):
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                return ctx.get(val[2:-1])
            if isinstance(val, dict):
                return {k: subst(v) for k, v in val.items()}
            return val

        def exec_steps(steps):
            for s in steps:
                if s.kind == "call":
                    args = subst(s.args_template or {})
                    res = tools.call(s.tool, args)
                    trace.append({"tool": s.tool, "res": res})
                    ctx["last"] = res
                elif s.kind == "if":
                    cond = s.condition or {}
                    val = ctx.get(cond.get("key"))
                    if val == cond.get("value"):
                        exec_steps(s.steps or [])
                    else:
                        exec_steps(s.else_steps or [])

        exec_steps(self.steps)
        return {"ok": True, "trace": trace, "final": ctx.get("last")}


class SkillLibrary:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def add(self, sk: Skill) -> str:
        self._skills[sk.id] = sk
        return sk.id

    def list(self, tag: Optional[str] = None) -> List[Skill]:
        if not tag: return list(self._skills.values())
        return [s for s in self._skills.values() if tag in s.tags]


# ----------------------------
# Cognitive Architecture (The "Brain")
# ----------------------------

@dataclass
class CognitiveGenome:
    """
    Evolvable parameters for the agent's brain.
    """
    learning_rate: float = 0.1
    curiosity: float = 0.2         # Probability of exploring unknown actions
    risk_tolerance: float = 0.2    # Threshold for risky plans
    memory_decay: float = 0.95     # Retention rate
    recursion_depth: int = 3       # Max depth of thought
    planning_width: int = 5        # Beam width


class CognitiveEngine:
    """
    Fractal Recursive Planner & Bayesian World Model.
    Replaces the simple linear planner.
    """
    def __init__(self, genome: CognitiveGenome) -> None:
        self.genome = genome
        # Simulated Bayesian Weights: Map (feature_hash -> weight)
        self.weights: Dict[str, float] = {}
        # Experience counter for uncertainty
        self.counts: Dict[str, int] = {}

    def _feature_hash(self, obs: Dict[str, Any], action: str) -> str:
        # Create a combined hash of state+action
        key = f"{obs.get('task')}|{obs.get('domain')}|{action}"
        return stable_hash(key)

    def predict_reward(self, obs: Dict[str, Any], action: str) -> Tuple[float, float]:
        # Returns (expected_reward, uncertainty)
        h = self._feature_hash(obs, action)
        w = self.weights.get(h, 0.0)
        cnt = self.counts.get(h, 0)
        # Uncertainty decreases with experience (1/sqrt(n))
        uncertainty = 1.0 / math.sqrt(cnt + 1)
        return w, uncertainty

    def update_model(self, obs: Dict[str, Any], action: str, reward: float):
        h = self._feature_hash(obs, action)
        curr = self.weights.get(h, 0.0)
        # Standard TD update
        self.weights[h] = curr + self.genome.learning_rate * (reward - curr)
        self.counts[h] = self.counts.get(h, 0) + 1

    def reflect(self, history: List[Dict[str, Any]]):
        # Meta-cognition: adjust internal weights based on recent history
        if not history: return
        avg_rew = sum(h['reward'] for h in history) / len(history)
        if avg_rew < 0.2:
            # If failing, increase curiosity to find new paths
            self.genome.curiosity = min(0.9, self.genome.curiosity + 0.05)
        else:
            # If succeeding, exploit more (reduce curiosity)
            self.genome.curiosity = max(0.05, self.genome.curiosity - 0.02)

    def decompose_goal(self, obs: Dict[str, Any], action_space: List[str], depth: int = 0) -> str:
        # Recursive fractal planning
        if depth >= self.genome.recursion_depth:
            # Base case: greedy selection
            best_a = random.choice(action_space)
            best_val = -1.0
            for a in action_space:
                val, unc = self.predict_reward(obs, a)
                # Upper Confidence Bound (UCB) logic
                score = val + self.genome.curiosity * unc
                if score > best_val:
                    best_val = score
                    best_a = a
            return best_a

        # Recursive Step: Simulate future
        # In this abstract sim, we don't have a full physics engine,
        # so we simulate "thought trials"
        candidates = []
        for a in action_space:
            r_pred, _ = self.predict_reward(obs, a)
            # Recursive value (simulated)
            future_val = r_pred * 0.9  # Discount
            candidates.append((a, future_val))

        candidates.sort(key=lambda x: x[1], reverse=True)
        # Beam search based on width
        top_candidates = candidates[:self.genome.planning_width]

        # Risk check
        safe_candidates = [
            c for c in top_candidates
            if c[1] > (0.5 - self.genome.risk_tolerance)
        ]

        if safe_candidates:
            return safe_candidates[0][0]
        return random.choice(action_space)


# ----------------------------
# Environment (Research)
# ----------------------------

@dataclass
class TaskSpec:
    name: str
    difficulty: int
    baseline: float
    domain: str

class ResearchEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.tasks = [
            TaskSpec("algorithm_design", 3, 0.35, "algorithm"),
            TaskSpec("systems_optimization", 4, 0.30, "systems"),
            TaskSpec("theory_discovery", 5, 0.28, "theory"),
            TaskSpec("toolchain_speedup", 5, 0.25, "engineering"),
        ]
        self.global_quality = {"tool": 0.1, "kb": 0.1, "org": 0.1}

    def sample_task(self) -> TaskSpec:
        return self.rng.choice(self.tasks)

    def make_observation(self, task: TaskSpec, budget: int) -> Dict[str, Any]:
        return {
            "task": task.name, "domain": task.domain,
            "difficulty": task.difficulty, "budget": budget,
            "baseline": task.baseline
        }

    def step(self, obs: Dict[str, Any], action: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        # Simulation Logic
        base = float(obs["baseline"])
        diff = int(obs["difficulty"])

        # Global leverage
        gq = self.global_quality
        leverage = (gq["tool"] + gq["kb"] + gq["org"]) / 3.0

        # Action effect
        raw_bonus = 0.0
        if action == "build_tool":
            raw_bonus = 0.05
            self.global_quality["tool"] = min(1.0, self.global_quality["tool"] + 0.02)
        elif action == "write_verified_note":
            raw_bonus = 0.04
            self.global_quality["kb"] = min(1.0, self.global_quality["kb"] + 0.02)
        elif action == "attempt_breakthrough":
            raw_bonus = 0.15 * leverage

        # Noise
        noise = self.rng.uniform(-0.05, 0.05)

        # Final Reward
        performance = max(0.0, min(1.0, base + raw_bonus + noise))
        reward = performance - base + (leverage * 0.1)

        return obs, reward, {"perf": performance}


# ----------------------------
# Agent & Orchestrator
# ----------------------------

@dataclass
class AgentConfig:
    name: str
    role: str
    genome: CognitiveGenome

class Agent:
    def __init__(self, cfg: AgentConfig, tools: ToolRegistry,
                 mem: NeuroSymbolicMemory, skills: SkillLibrary) -> None:
        self.cfg = cfg
        self.tools = tools
        self.mem = mem
        self.skills = skills
        self.brain = CognitiveEngine(cfg.genome)
        self.history: List[Dict[str, Any]] = []

    def action_space(self) -> List[str]:
        return ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]

    def step(self, env: ResearchEnvironment, obs: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Retrieve Context (Associative Memory)
        context_items = self.mem.search(f"{obs['task']} {obs['domain']}", k=3)

        # 2. Plan (Fractal Decomposition)
        action = self.brain.decompose_goal(obs, self.action_space())

        # 3. Execute
        payload = {"agent": self.cfg.name, "invest": 1.0}
        next_obs, reward, info = env.step(obs, action, payload)

        # 4. Learn (Update Model)
        self.brain.update_model(obs, action, reward)

        # 5. Store Memory
        self.mem.add("episode", f"{self.cfg.name}:{action}",
                     {"obs": obs, "action": action, "reward": reward}, tags=["episode", self.cfg.role])

        # 6. Meta-Cognition Loop
        entry = {"action": action, "reward": reward}
        self.history.append(entry)
        if len(self.history) % 5 == 0:
            self.brain.reflect(self.history[-5:])

        return {"agent": self.cfg.name, "action": action, "reward": reward, "info": info}


class Orchestrator:
    """
    Evolutionary Orchestrator.
    Manages the population of agents and evolves their Cognitive Genomes.
    """
    def __init__(self, agents_n: int, env: ResearchEnvironment, tools: ToolRegistry) -> None:
        self.env = env
        self.tools = tools
        self.mem = NeuroSymbolicMemory()
        self.skills = SkillLibrary()
        self.agents_n = agents_n
        self.population: List[Agent] = []
        self._init_population()

    def _init_population(self):
        roles = ["theorist", "builder", "experimenter", "verifier"]
        for i in range(self.agents_n):
            genome = CognitiveGenome() # Default
            cfg = AgentConfig(f"agent_{i}", roles[i % 4], genome)
            self.population.append(Agent(cfg, self.tools, self.mem, self.skills))

    def evolve_population(self, results: List[Dict[str, Any]]):
        # Genetic Algorithm: Select best genomes and mutate
        if not results: return

        # Score agents by total reward
        scores = {}
        for r in results:
            name = r["agent"]
            scores[name] = scores.get(name, 0.0) + r["reward"]

        sorted_agents = sorted(self.population, key=lambda a: scores.get(a.cfg.name, 0.0), reverse=True)
        top_half = sorted_agents[:self.agents_n // 2]

        # Create next generation
        new_pop = []
        for i in range(self.agents_n):
            parent = random.choice(top_half)
            # Clone genome
            pg = parent.cfg.genome
            new_genome = CognitiveGenome(
                learning_rate=pg.learning_rate,
                curiosity=pg.curiosity,
                risk_tolerance=pg.risk_tolerance,
                recursion_depth=pg.recursion_depth
            )
            # Mutate
            if random.random() < 0.3:
                new_genome.learning_rate += random.uniform(-0.02, 0.02)
                new_genome.curiosity += random.uniform(-0.05, 0.05)
                new_genome.curiosity = max(0.01, min(1.0, new_genome.curiosity))

            # Create new agent instance preserving memory access
            cfg = AgentConfig(f"agent_{i}_gen_next", parent.cfg.role, new_genome)
            new_pop.append(Agent(cfg, self.tools, self.mem, self.skills))

        self.population = new_pop
        print(">> Population Evolved: Genomes optimized.")

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        tasks = [self.env.sample_task() for _ in range(2)]
        results = []

        # Run agents
        for i, agent in enumerate(self.population):
            task = tasks[i % len(tasks)]
            obs = self.env.make_observation(task, budget=10)
            res = agent.step(self.env, obs)
            results.append(res)

        # Distill knowledge
        self.mem.extract_principles()

        # Evolution every 5 rounds
        if round_idx > 0 and round_idx % 5 == 0:
            self.evolve_population(results)

        return {
            "round": round_idx,
            "tasks": [t.name for t in tasks],
            "results": results,
            "global_quality": self.env.global_quality
        }


# ----------------------------
# Entry Point
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--agents", type=int, default=8)
    args = parser.parse_args()

    env = ResearchEnvironment(seed=42)
    tools = ToolRegistry()

    # Placeholder tools
    tools.register("write_note", lambda x: {"ok": True})

    orch = Orchestrator(args.agents, env, tools)

    print("=== QUASI-AGI CORE: SIMULATION START ===")
    print(f"Structure: {args.agents} Agents | Neuro-Symbolic Memory | Evolutionary Orchestrator")

    start_t = time.time()
    for r in range(args.rounds):
        out = orch.run_round(r)

        # Calc stats
        avg_rew = sum(r['reward'] for r in out['results']) / len(out['results'])
        print(f"[Round {r:02d}] Tasks: {out['tasks'][0]}... | Avg Reward: {avg_rew:.4f} | SysQual: {sum(out['global_quality'].values()):.2f}")

    print(f"=== END ({time.time()-start_t:.2f}s) ===")
    print("Top Memories (Holographic Retrieval):")
    for m in orch.mem.dump_summary(5):
        print(f" - [{m['kind']}] {m['title']}")

if __name__ == "__main__":
    main()
