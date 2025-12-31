"""
NON_RSI_AGI_CORE_v2.py
======================

Architecture goal:
- Fixed source code (no code-level RSI).
- AGI-oriented B×C structure:
  - B: world-model + planner + memory + skill-DSL interpreter (per agent)
  - C: multi-agent orchestrator + project/goal graph + evaluation/selection
- Self-improvement happens only via:
  - parameter updates (world model)
  - knowledge/memory accumulation
  - data-level skill programs
  - project graph + org policy adaptation
  NOT via modifying this file.

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
from typing import Any, Callable, Dict, List, Optional, Tuple


# ----------------------------
# Utility
# ----------------------------

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def now_ms() -> int:
    return int(time.time() * 1000)


# ----------------------------
# Shared Memory / Knowledge Base
# ----------------------------

@dataclass
class MemoryItem:
    ts_ms: int
    kind: str               # "episode" | "note" | "artifact" | "principle"
    title: str
    content: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {"ts": self.ts_ms, "k": self.kind, "t": self.title, "c": self.content, "tags": self.tags}
        )


class SharedMemory:
    """
    Shared KB across all agents and orchestrator.
    - Episodic: concrete runs, rewards, env stats
    - Semantic/Principles: distilled rules, patterns, strategies
    - Artifacts: tools, designs, verified modules
    """

    def __init__(self, max_items: int = 8000) -> None:
        self.max_items = max_items
        self._items: List[MemoryItem] = []

    def add(self, kind: str, title: str, content: Dict[str, Any],
            tags: Optional[List[str]] = None) -> str:
        tags = tags or []
        item = MemoryItem(ts_ms=now_ms(), kind=kind, title=title,
                          content=content, tags=tags)
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        return item.id

    # very small tokenizer
    def _tok(self, text: str) -> List[str]:
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

    def search(self, query: str, k: int = 10,
               kinds: Optional[List[str]] = None) -> List[MemoryItem]:
        qtok = set(self._tok(query))
        if not qtok:
            return self._items[-k:]

        t_now = now_ms()
        scored: List[Tuple[float, MemoryItem]] = []
        for it in self._items:
            if kinds is not None and it.kind not in kinds:
                continue
            txt = it.title + " " + json.dumps(it.content, ensure_ascii=False, default=str)
            itok = set(self._tok(txt))
            overlap = len(qtok.intersection(itok))
            if overlap == 0:
                continue
            recency = 1.0 / (1.0 + (t_now - it.ts_ms) / (1000.0 * 60.0 * 10.0))  # ~10min scale
            score = overlap + 0.3 * recency
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def dump_summary(self, k: int = 15) -> List[Dict[str, Any]]:
        tail = self._items[-k:]
        return [
            {
                "id": it.id,
                "ts_ms": it.ts_ms,
                "kind": it.kind,
                "title": it.title,
                "tags": it.tags,
            }
            for it in tail
        ]


# ----------------------------
# Tool interface (external world hook)
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
            return {"ok": False, "error": f"unknown_tool:{name}", "tool": name}
        try:
            out = fn(args)
            out = dict(out)
            out.setdefault("ok", True)
            out.setdefault("tool", name)
            return out
        except Exception as e:
            return {"ok": False, "error": repr(e), "tool": name}


# ----------------------------
# Skill DSL (data-level programs)
# ----------------------------

@dataclass
class SkillStep:
    tool: str
    args_template: Dict[str, Any]


@dataclass
class Skill:
    """
    Interpreted skill program:
    - sequence of tool calls
    - arguments can reference context via ${key}
    """
    name: str
    purpose: str
    steps: List[SkillStep]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {
                "name": self.name,
                "purpose": self.purpose,
                "steps": [(s.tool, s.args_template) for s in self.steps],
            }
        )

    def run(self, tools: ToolRegistry,
            context: Dict[str, Any]) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        ctx = dict(context)

        def subst(v: Any) -> Any:
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                key = v[2:-1]
                return ctx.get(key)
            if isinstance(v, dict):
                return {k: subst(x) for k, x in v.items()}
            if isinstance(v, list):
                return [subst(x) for x in v]
            return v

        for i, st in enumerate(self.steps):
            args = subst(st.args_template)
            if not isinstance(args, dict):
                args = {"value": args}
            res = tools.call(st.tool, args)
            trace.append({"i": i, "tool": st.tool, "args": args, "res": res})
            ctx["last"] = res
            ctx[f"step_{i}"] = res
            if not res.get("ok", False):
                break

        return {
            "ok": bool(trace and trace[-1]["res"].get("ok", False)),
            "trace": trace,
            "final": ctx.get("last"),
        }


class SkillLibrary:
    def __init__(self, max_skills: int = 3000) -> None:
        self.max_skills = max_skills
        self._skills: Dict[str, Skill] = {}

    def add(self, sk: Skill) -> str:
        self._skills[sk.id] = sk
        if len(self._skills) > self.max_skills:
            for sid in list(self._skills.keys())[: len(self._skills) - self.max_skills]:
                self._skills.pop(sid, None)
        return sk.id

    def list(self, tag: Optional[str] = None) -> List[Skill]:
        vals = list(self._skills.values())
        if tag is None:
            return vals
        return [s for s in vals if tag in s.tags]

    def get(self, sid: str) -> Optional[Skill]:
        return self._skills.get(sid)


# ----------------------------
# World Model (discrete predictive model)
# ----------------------------

@dataclass
class TransitionStat:
    n: int = 0
    reward_sum: float = 0.0
    next_counts: Dict[str, int] = field(default_factory=dict)


class WorldModel:
    """
    Simple discrete world-model:
    - states are hashed feature summaries
    - learns P(s' | s, a) and E[r | s, a]
    - shared across tasks/domains (domain-agnostic features)
    """

    def __init__(self) -> None:
        self._stats: Dict[Tuple[str, str], TransitionStat] = {}
        self._features: Dict[str, Dict[str, Any]] = {}

    def encode_state(self, obs: Dict[str, Any]) -> str:
        key = {
            "task": obs.get("task", ""),
            "phase": obs.get("phase", ""),
            "difficulty": int(obs.get("difficulty", 0)),
            "budget": int(obs.get("budget", 0)),
        }
        sid = stable_hash(key)
        self._features[sid] = key
        return sid

    def update(self, s: str, a: str, r: float, s_next: str) -> None:
        k = (s, a)
        st = self._stats.get(k)
        if st is None:
            st = TransitionStat()
            self._stats[k] = st
        st.n += 1
        st.reward_sum += r
        st.next_counts[s_next] = st.next_counts.get(s_next, 0) + 1

    def predict(self, s: str, a: str) -> Tuple[float, Dict[str, float]]:
        st = self._stats.get((s, a))
        if st is None or st.n == 0:
            return 0.0, {}
        er = st.reward_sum / max(1, st.n)
        tot = sum(st.next_counts.values())
        if tot <= 0:
            return er, {}
        dist = {sn: c / tot for sn, c in st.next_counts.items()}
        return er, dist


# ----------------------------
# Planner (lookahead over world model)
# ----------------------------

@dataclass
class PlanCandidate:
    actions: List[str]
    score: float


class Planner:
    def __init__(self, wm: WorldModel, depth: int = 3,
                 width: int = 6, gamma: float = 0.9) -> None:
        self.wm = wm
        self.depth = depth
        self.width = width
        self.gamma = gamma

    def propose(self, s: str, action_space: List[str]) -> List[PlanCandidate]:
        beam: List[PlanCandidate] = [PlanCandidate(actions=[], score=0.0)]

        for d in range(self.depth):
            new_beam: List[PlanCandidate] = []
            for cand in beam:
                for a in action_space:
                    er, nxt = self.wm.predict(s, a)
                    if nxt:
                        best_sn = max(nxt.items(), key=lambda kv: kv[1])[0]
                    else:
                        best_sn = s
                    sc = cand.score + (self.gamma ** d) * er
                    new_beam.append(
                        PlanCandidate(actions=cand.actions + [a], score=sc)
                    )
            new_beam.sort(key=lambda c: c.score, reverse=True)
            beam = new_beam[: self.width]

        return beam


# ----------------------------
# Project / Goal Graph (C-layer long-horizon structure)
# ----------------------------

@dataclass
class ProjectNode:
    id: str
    name: str
    task: str
    status: str = "open"      # "open" | "active" | "done"
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    value_estimate: float = 0.0
    history: List[str] = field(default_factory=list)  # memory ids


class ProjectGraph:
    """
    Long-horizon project DAG:
    - orchestrator attaches agent runs to nodes
    - nodes accumulate evidence and value estimates
    - closed-loop: future allocation depends on node/value structure
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ProjectNode] = {}

    def create_root(self, name: str, task: str) -> str:
        nid = stable_hash({"name": name, "task": task, "root": True})
        self._nodes[nid] = ProjectNode(id=nid, name=name, task=task, status="open")
        return nid

    def add_child(self, parent_id: str, name: str,
                  task: Optional[str] = None) -> str:
        parent = self._nodes[parent_id]
        nid = stable_hash({"name": name, "task": task or parent.task,
                           "parent": parent_id})
        node = ProjectNode(id=nid, name=name, task=task or parent.task,
                           status="open", parent_id=parent_id)
        self._nodes[nid] = node
        parent.children.append(nid)
        return nid

    def nodes_for_task(self, task: str) -> List[ProjectNode]:
        return [n for n in self._nodes.values() if n.task == task]

    def pick_node_for_round(self, task: str) -> ProjectNode:
        # prefer unfinished nodes; bias to higher value_estimate
        candidates = [n for n in self._nodes.values()
                      if n.task == task and n.status != "done"]
        if not candidates:
            # fallback: create a new root project for this task
            nid = self.create_root(name=f"{task}_root", task=task)
            return self._nodes[nid]
        candidates.sort(key=lambda n: n.value_estimate, reverse=True)
        return candidates[0]

    def update_node(self, nid: str, reward: float,
                    memory_id: Optional[str]) -> None:
        node = self._nodes[nid]
        # simple exponential update of value estimate
        alpha = 0.2
        node.value_estimate = (1 - alpha) * node.value_estimate + alpha * reward
        if memory_id:
            node.history.append(memory_id)
        if node.value_estimate > 0.15 and len(node.children) < 3:
            # spawn refinement subproject
            self.add_child(parent_id=nid,
                           name=f"{node.name}_refine_{len(node.children)}")


# ----------------------------
# Environment (research/engineering playground)
# ----------------------------

@dataclass
class TaskSpec:
    name: str
    difficulty: int
    baseline: float
    domain: str   # "algorithm" | "systems" | "theory" | "strategy" ...


class ResearchEnvironment:
    """
    Abstract multi-domain environment.
    - Each step is "run one agent on one project node for a given task/budget"
    - Reward ~ improvement over task baseline + infra gain
    - Global qualities (tool/kb/org) mediate acceleration
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.tasks: List[TaskSpec] = [
            TaskSpec("algorithm_design",      difficulty=3, baseline=0.35, domain="algorithm"),
            TaskSpec("systems_optimization", difficulty=4, baseline=0.30, domain="systems"),
            TaskSpec("verification_pipeline", difficulty=2, baseline=0.40, domain="verification"),
            TaskSpec("toolchain_speedup",    difficulty=5, baseline=0.25, domain="engineering"),
            TaskSpec("theory_discovery",     difficulty=5, baseline=0.28, domain="theory"),
            TaskSpec("strategy_optimization",difficulty=3, baseline=0.32, domain="strategy"),
        ]
        self.global_tool_quality = 0.10
        self.global_kb_quality = 0.10
        self.global_org_quality = 0.10

    def sample_task(self) -> TaskSpec:
        # favour tasks where we are still weak (using baselines as rough prior)
        # here we just random-choice with slight difficulty bias
        t = self.rng.choice(self.tasks)
        return t

    def make_observation(self, task: TaskSpec, budget: int,
                         phase: str = "research") -> Dict[str, Any]:
        return {
            "task": task.name,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "baseline": task.baseline,
            "budget": budget,
            "phase": phase,
        }

    def step(self, obs: Dict[str, Any], action: str,
             payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        task_name = str(obs["task"])
        diff = int(obs["difficulty"])
        base = float(obs["baseline"])
        budget = int(obs["budget"])

        tq = self.global_tool_quality
        kq = self.global_kb_quality
        oq = self.global_org_quality

        if action == "build_tool":
            invest = float(payload.get("invest", 1.0))
            gain = (0.02 + 0.10 * tq) * invest / (1.0 + 0.25 * diff)
            self.global_tool_quality = min(1.0, self.global_tool_quality + gain)
            raw = 0.02 * invest
        elif action == "write_verified_note":
            invest = float(payload.get("invest", 1.0))
            gain = (0.02 + 0.08 * kq) * invest / (1.0 + 0.25 * diff)
            self.global_kb_quality = min(1.0, self.global_kb_quality + gain)
            raw = 0.015 * invest
        elif action == "tune_orchestration":
            invest = float(payload.get("invest", 1.0))
            gain = (0.02 + 0.08 * oq) * invest / (1.0 + 0.25 * diff)
            self.global_org_quality = min(1.0, self.global_org_quality + gain)
            raw = 0.015 * invest
        elif action == "attempt_breakthrough":
            leverage = 0.30 * tq + 0.30 * kq + 0.30 * oq
            raw = (0.03 + 0.30 * leverage) * (1.0 + math.log(1 + budget) / 4.0) / (1.0 + 0.30 * diff)
        else:
            raw = 0.0

        noise = self.rng.uniform(-0.02, 0.02)
        performance = max(0.0, min(1.0, base + raw + noise))
        delta = performance - base
        infra_bonus = 0.02 * (tq + kq + oq) / 3.0
        reward = delta + infra_bonus

        next_obs = dict(obs)
        next_obs["phase"] = "integrate"
        info = {
            "task": task_name,
            "performance": performance,
            "delta": delta,
            "tq": self.global_tool_quality,
            "kq": self.global_kb_quality,
            "oq": self.global_org_quality,
        }
        return next_obs, reward, info


# ----------------------------
# Agent (B-type architecture)
# ----------------------------

@dataclass
class AgentConfig:
    name: str
    role: str = "general"     # "theorist" | "builder" | "experimenter" | "verifier" | "strategist"
    planner_depth: int = 3
    planner_width: int = 6
    risk: float = 0.2


class Agent:
    """
    B-type core:
    - WorldModel + Planner
    - SharedMemory + SkillLibrary + ToolRegistry
    - No self-modifying code; only state/memory/skills evolve.
    """

    def __init__(self, cfg: AgentConfig, tools: ToolRegistry,
                 shared_mem: SharedMemory, skills: SkillLibrary) -> None:
        self.cfg = cfg
        self.tools = tools
        self.mem = shared_mem
        self.skills = skills

        self.wm = WorldModel()
        self.planner = Planner(self.wm, depth=cfg.planner_depth,
                               width=cfg.planner_width)

    def action_space(self) -> List[str]:
        base = ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]
        r = self.cfg.role
        if r == "verifier":
            return ["write_verified_note", "build_tool", "tune_orchestration", "attempt_breakthrough"]
        if r == "builder":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "theorist":
            return ["attempt_breakthrough", "write_verified_note", "build_tool", "tune_orchestration"]
        if r == "experimenter":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "strategist":
            return ["tune_orchestration", "attempt_breakthrough", "build_tool", "write_verified_note"]
        return base

    def choose_action(self, obs: Dict[str, Any]) -> str:
        s = self.wm.encode_state(obs)
        candidates = self.planner.propose(s, self.action_space())
        if candidates and random.random() > self.cfg.risk:
            return candidates[0].actions[0]
        return random.choice(self.action_space())

    def maybe_synthesize_skill(self, obs: Dict[str, Any]) -> Optional[str]:
        task = obs.get("task", "")
        # simple heuristic triggers (data-level, not code-level)
        if task == "verification_pipeline" and random.random() < 0.25:
            sk = Skill(
                name=f"{self.cfg.name}_verify_pipeline",
                purpose="Evaluate candidate and write verified note.",
                steps=[
                    SkillStep("evaluate_candidate", {"task": "${task}", "candidate": "${candidate}"}),
                    SkillStep("write_note", {"title": "verified_result", "payload": "${step_0}"}),
                ],
                tags=["verification", "meta"],
            )
            return self.skills.add(sk)
        if task == "toolchain_speedup" and random.random() < 0.25:
            sk = Skill(
                name=f"{self.cfg.name}_toolchain_upgrade",
                purpose="Propose toolchain improvement artifact.",
                steps=[
                    SkillStep("tool_build_report", {"task": "${task}", "idea": "${idea}"}),
                    SkillStep("write_artifact", {"title": "tool_artifact", "payload": "${step_0}"}),
                ],
                tags=["toolchain", "artifact"],
            )
            return self.skills.add(sk)
        return None

    def act_on_project(self, env: ResearchEnvironment,
                       proj_node: ProjectNode,
                       obs: Dict[str, Any]) -> Dict[str, Any]:
        # Retrieve relevant principles and artifacts
        hints = self.mem.search(
            f"{obs.get('task','')} difficulty {obs.get('difficulty',0)}",
            k=6,
            kinds=["principle", "artifact", "note"],
        )

        context = {
            "task": obs.get("task"),
            "domain": obs.get("domain"),
            "difficulty": obs.get("difficulty"),
            "budget": obs.get("budget"),
            "project": {"id": proj_node.id, "name": proj_node.name},
            "candidate": {
                "type": "proposal",
                "from": self.cfg.name,
                "role": self.cfg.role,
                "hints": [h.title for h in hints],
            },
            "idea": {
                "from": self.cfg.name,
                "summary": "incremental improvement on project using accumulated tools/kb/org.",
            },
        }

        sid = self.maybe_synthesize_skill(obs)
        if sid:
            self.mem.add(
                "artifact",
                f"skill_added:{sid}",
                {"agent": self.cfg.name, "skill_id": sid},
                tags=["skill"],
            )

        action = self.choose_action(obs)
        invest = max(1.0, float(obs.get("budget", 1)) / 10.0)
        payload = {
            "invest": invest,
            "agent": self.cfg.name,
            "role": self.cfg.role,
            "task": obs.get("task"),
            "project_id": proj_node.id,
        }

        s = self.wm.encode_state(obs)
        next_obs, reward, info = env.step(obs, action, payload)
        s_next = self.wm.encode_state(next_obs)
        self.wm.update(s, action, reward, s_next)

        mem_id = self.mem.add(
            "episode",
            f"{self.cfg.name}:{action}:{obs.get('task')}:{proj_node.name}",
            {
                "obs": obs,
                "action": action,
                "payload": payload,
                "reward": reward,
                "info": info,
                "project_id": proj_node.id,
                "hints_used": [h.id for h in hints],
            },
            tags=["episode", self.cfg.role, obs.get("task", "task")],
        )

        # run suitable skills occasionally
        if random.random() < 0.35:
            tag = "verification" if action == "write_verified_note" else "toolchain"
            candidates = self.skills.list(tag=tag)
            if candidates:
                sk = random.choice(candidates)
                out = sk.run(self.tools, context)
                self.mem.add(
                    "note",
                    f"{self.cfg.name}:skill_run:{sk.name}",
                    {"skill_id": sk.id, "out": out},
                    tags=["skill_run", tag],
                )

        return {
            "agent": self.cfg.name,
            "role": self.cfg.role,
            "project_id": proj_node.id,
            "project_name": proj_node.name,
            "action": action,
            "reward": reward,
            "mem_id": mem_id,
            "info": info,
        }


# ----------------------------
# Orchestrator (C-layer: multi-agent + project graph)
# ----------------------------

@dataclass
class OrchestratorConfig:
    agents: int = 8
    base_budget: int = 20
    selection_top_k: int = 4
    budget_growth: float = 1.06


class Orchestrator:
    """
    C-layer:
    - maintains SharedMemory, SkillLibrary, ProjectGraph
    - runs multiple B-type agents per round
    - distills principles from best episodes
    - adapts org policy (role mix, risk) based on outcomes
    """

    def __init__(self, cfg: OrchestratorConfig,
                 env: ResearchEnvironment,
                 tools: ToolRegistry) -> None:
        self.cfg = cfg
        self.env = env
        self.tools = tools

        self.mem = SharedMemory()
        self.skills = SkillLibrary()
        self.projects = ProjectGraph()

        self._agents: List[Agent] = []
        self._org_policy: Dict[str, Any] = {
            "risk": 0.25,
            "role_mix": ["theorist", "builder", "experimenter", "verifier", "strategist"],
        }
        self._init_agents()

    def _init_agents(self) -> None:
        roles = self._org_policy["role_mix"]
        for i in range(self.cfg.agents):
            role = roles[i % len(roles)]
            cfg = AgentConfig(
                name=f"agent_{i:02d}",
                role=role,
                planner_depth=4 if role in ("theorist", "strategist") else 3,
                planner_width=7 if role == "strategist" else 6,
                risk=self._org_policy["risk"],
            )
            self._agents.append(Agent(cfg, self.tools, self.mem, self.skills))

    def _distill_principles(self, round_idx: int,
                            results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        results_sorted = sorted(results, key=lambda r: r["reward"],
                                reverse=True)
        top = results_sorted[: self.cfg.selection_top_k]
        bottom = results_sorted[-self.cfg.selection_top_k :]

        # store distillation note
        self.mem.add(
            "note",
            f"round_{round_idx}_distill",
            {
                "top": [
                    {
                        "agent": r["agent"],
                        "role": r["role"],
                        "task": r["info"]["task"],
                        "reward": r["reward"],
                        "action": r["action"],
                    }
                    for r in top
                ],
                "bottom": [
                    {
                        "agent": r["agent"],
                        "role": r["role"],
                        "task": r["info"]["task"],
                        "reward": r["reward"],
                        "action": r["action"],
                    }
                    for r in bottom
                ],
                "env": {
                    "tq": self.env.global_tool_quality,
                    "kq": self.env.global_kb_quality,
                    "oq": self.env.global_org_quality,
                },
                "policy": dict(self._org_policy),
            },
            tags=["distill", "round"],
        )

        # derive coarse "principle" items
        for r in top:
            self.mem.add(
                "principle",
                f"good_pattern:{r['info']['task']}:{r['action']}",
                {
                    "agent": r["agent"],
                    "role": r["role"],
                    "task": r["info"]["task"],
                    "action": r["action"],
                    "reward": r["reward"],
                    "env": {
                        "tq": r["info"]["tq"],
                        "kq": r["info"]["kq"],
                        "oq": r["info"]["oq"],
                    },
                },
                tags=["principle", "good"],
            )
        for r in bottom:
            self.mem.add(
                "principle",
                f"bad_pattern:{r['info']['task']}:{r['action']}",
                {
                    "agent": r["agent"],
                    "role": r["role"],
                    "task": r["info"]["task"],
                    "action": r["action"],
                    "reward": r["reward"],
                    "env": {
                        "tq": r["info"]["tq"],
                        "kq": r["info"]["kq"],
                        "oq": r["info"]["oq"],
                    },
                },
                tags=["principle", "bad"],
            )

        # org-level adaptation
        rewards = [r["reward"] for r in results]
        mean = sum(rewards) / max(1, len(rewards))
        var = sum((x - mean) ** 2 for x in rewards) / max(1, len(rewards))
        std = math.sqrt(var)

        # adapt role_mix based on env qualities
        tq = self.env.global_tool_quality
        kq = self.env.global_kb_quality
        oq = self.env.global_org_quality

        if tq < kq and tq < oq:
            self._org_policy["role_mix"] = [
                "builder", "builder", "experimenter",
                "verifier", "strategist"
            ]
        elif kq < tq and kq < oq:
            self._org_policy["role_mix"] = [
                "verifier", "verifier", "theorist",
                "builder", "strategist"
            ]
        elif oq < tq and oq < kq:
            self._org_policy["role_mix"] = [
                "strategist", "strategist", "builder",
                "experimenter", "verifier"
            ]
        else:
            self._org_policy["role_mix"] = [
                "theorist", "builder", "experimenter",
                "verifier", "strategist"
            ]

        # risk schedule: shrink exploration when variance large
        if std > 0.10:
            self._org_policy["risk"] = max(0.05, self._org_policy["risk"] - 0.02)
        else:
            self._org_policy["risk"] = min(0.35, self._org_policy["risk"] + 0.01)

        # propagate policy to agents
        roles = self._org_policy["role_mix"]
        for i, ag in enumerate(self._agents):
            ag.cfg.role = roles[i % len(roles)]
            ag.cfg.risk = self._org_policy["risk"]

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        # pick task
        task = self.env.sample_task()
        budget = int(self.cfg.base_budget * (self.cfg.budget_growth ** round_idx))
        # pick project node for this task
        proj_node = self.projects.pick_node_for_round(task.name)
        obs = self.env.make_observation(task, budget)

        results: List[Dict[str, Any]] = []
        for ag in self._agents:
            res = ag.act_on_project(self.env, proj_node, obs)
            results.append(res)
            # update project node with reward/evidence
            self.projects.update_node(proj_node.id, res["reward"], res["mem_id"])

        self._distill_principles(round_idx, results)

        return {
            "round": round_idx,
            "task": task.name,
            "project_id": proj_node.id,
            "project_name": proj_node.name,
            "obs": obs,
            "results": results,
            "env": {
                "tq": self.env.global_tool_quality,
                "kq": self.env.global_kb_quality,
                "oq": self.env.global_org_quality,
            },
            "policy": dict(self._org_policy),
        }


# ----------------------------
# Minimal tools (replace with real-world hooks)
# ----------------------------

def tool_write_note_factory(shared_mem: SharedMemory) -> ToolFn:
    def _fn(args: Dict[str, Any]) -> Dict[str, Any]:
        title = str(args.get("title", "note"))
        payload = args.get("payload", {})
        mid = shared_mem.add("note", title, {"payload": payload},
                             tags=["tool_note"])
        return {"ok": True, "memory_id": mid, "title": title}
    return _fn


def tool_write_artifact_factory(shared_mem: SharedMemory) -> ToolFn:
    def _fn(args: Dict[str, Any]) -> Dict[str, Any]:
        title = str(args.get("title", "artifact"))
        payload = args.get("payload", {})
        mid = shared_mem.add("artifact", title, {"payload": payload},
                             tags=["tool_artifact"])
        return {"ok": True, "memory_id": mid, "title": title}
    return _fn


def tool_evaluate_candidate(args: Dict[str, Any]) -> Dict[str, Any]:
    task = str(args.get("task", "unknown"))
    cand = args.get("candidate", {})
    # crude proxy for complexity/structure
    size = len(json.dumps(cand, default=str))
    score = (size % 97) / 100.0
    if "hints" in cand and isinstance(cand["hints"], list) and len(cand["hints"]) > 4:
        score *= 0.93  # mild overfitting penalty
    verdict = "pass" if score > 0.4 else "revise"
    return {"ok": True, "task": task, "score": score, "verdict": verdict}


def tool_tool_build_report(args: Dict[str, Any]) -> Dict[str, Any]:
    task = str(args.get("task", "unknown"))
    idea = args.get("idea", {})
    return {
        "ok": True,
        "task": task,
        "artifact": {
            "type": "tool_proposal",
            "idea": idea,
            "expected_effect": "increase evaluation throughput & reliability",
        },
    }


# ----------------------------
# Main entry
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = ResearchEnvironment(seed=args.seed)
    tools = ToolRegistry()

    # orchestrator owns shared memory/skills/projects
    orch_cfg = OrchestratorConfig(
        agents=args.agents,
        base_budget=20,
        selection_top_k=max(3, args.agents // 2),
    )
    orch = Orchestrator(orch_cfg, env, tools)

    # register tools (hook points for the “world”)
    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    print("=== NON-RSI AGI CORE v2: RUN START ===")
    for r in range(args.rounds):
        out = orch.run_round(r)
        top = sorted(out["results"], key=lambda x: x["reward"],
                     reverse=True)[:3]
        print(
            f"[Round {r:02d}] task={out['task']:<22} proj={out['project_name']:<20} "
            f"tq={out['env']['tq']:.3f} kq={out['env']['kq']:.3f} oq={out['env']['oq']:.3f} "
            f"risk={out['policy']['risk']:.2f} "
            f"top_rewards={[round(x['reward'],4) for x in top]}"
        )

    print("=== RUN END ===")
    print("Recent memory summary:")
    for it in orch.mem.dump_summary(k=15):
        print(it)


if __name__ == "__main__":
    main()
