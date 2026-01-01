"""
fair_benchmark.py
=================
FAIR BENCHMARK: Same environment, same reward calculation

Protocol:
1. Use v5's ResearchEnvironment for ALL versions
2. Count reward PER ACTION (not per skill run)
3. Same random seed
4. Same number of ACTIONS (not rounds)
"""

import random
import statistics
import copy

# Import v5's environment as the standard
from NON_RSI_AGI_CORE_v5 import (
    ResearchEnvironment,
    TaskSpec,
)

# Import agents from both versions
from NON_RSI_AGI_CORE_v7_FINAL import (
    Orchestrator as Orch_v7,
    Skill,
    SkillStep,
)
from NON_RSI_AGI_CORE_v5 import (
    Orchestrator as Orch_v5_Raw,
    OrchestratorConfig,
    ToolRegistry,
    Agent as Agent_v5,
    AgentConfig,
    SharedMemory,
    SkillLibrary,
)


def run_v5_fair(env: ResearchEnvironment, num_actions: int, seed: int) -> dict:
    """Run v5 agents on shared environment, count per-action reward."""
    random.seed(seed)
    
    # Create v5 agent
    tools = ToolRegistry()
    mem = SharedMemory()
    skills = SkillLibrary()
    cfg = AgentConfig(name="v5_agent", role="general")
    agent = Agent_v5(cfg, tools, mem, skills)
    
    rewards = []
    
    for _ in range(num_actions):
        task = env.sample_task()
        obs = env.make_observation(task, budget=20)
        
        # Choose action
        action = agent.choose_action(obs)
        payload = {"invest": 1.0}
        
        # Execute single action
        next_obs, reward, info = env.step(obs, action, payload)
        rewards.append(reward)
        
        # Update agent's world model
        agent.wm.update(obs, action, reward, next_obs, agent.action_space())
    
    return {
        "version": "v5",
        "total_reward": sum(rewards),
        "avg_reward_per_action": statistics.mean(rewards),
        "num_actions": len(rewards),
        "final_tq": env.global_tool_quality,
        "final_kq": env.global_kb_quality,
        "final_oq": env.global_org_quality,
    }


def run_v7_fair(env: ResearchEnvironment, num_actions: int, seed: int) -> dict:
    """Run v7-style agent on shared environment, count per-action reward."""
    random.seed(seed)
    
    # Simple Q-learning agent (like v7 without skills)
    q_table = {}
    
    def get_q(obs_key, action):
        return q_table.get((obs_key, action), 0.0)
    
    def update_q(obs_key, action, reward, next_obs_key, actions, lr=0.1, gamma=0.9):
        current = get_q(obs_key, action)
        next_max = max(get_q(next_obs_key, a) for a in actions)
        q_table[(obs_key, action)] = current + lr * (reward + gamma * next_max - current)
    
    actions_list = ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]
    rewards = []
    
    for i in range(num_actions):
        task = env.sample_task()
        obs = env.make_observation(task, budget=20)
        obs_key = (task.name, task.difficulty)
        
        # Epsilon-greedy action selection
        if random.random() < 0.2:
            action = random.choice(actions_list)
        else:
            action = max(actions_list, key=lambda a: get_q(obs_key, a))
        
        payload = {"invest": 1.0}
        next_obs, reward, info = env.step(obs, action, payload)
        rewards.append(reward)
        
        next_key = (task.name, task.difficulty)
        update_q(obs_key, action, reward, next_key, actions_list)
    
    return {
        "version": "v7_baseline",
        "total_reward": sum(rewards),
        "avg_reward_per_action": statistics.mean(rewards),
        "num_actions": len(rewards),
        "final_tq": env.global_tool_quality,
        "final_kq": env.global_kb_quality,
        "final_oq": env.global_org_quality,
    }


def run_v7_with_skills_fair(env: ResearchEnvironment, num_actions: int, seed: int) -> dict:
    """
    Run v7 with skill evolution, but count ACTIONS not skill runs.
    Each skill step = 1 action.
    """
    random.seed(seed)
    
    # Simple skill library
    skills = []
    actions_list = ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]
    
    # Q-learning
    q_table = {}
    
    def get_q(obs_key, action):
        return q_table.get((obs_key, action), 0.0)
    
    def update_q(obs_key, action, reward, next_obs_key, lr=0.1, gamma=0.9):
        current = get_q(obs_key, action)
        next_max = max(get_q(next_obs_key, a) for a in actions_list)
        q_table[(obs_key, action)] = current + lr * (reward + gamma * next_max - current)
    
    rewards = []
    action_count = 0
    trace = []
    
    while action_count < num_actions:
        task = env.sample_task()
        obs = env.make_observation(task, budget=20)
        obs_key = (task.name, task.difficulty)
        
        # Decide: use skill or manual
        use_skill = False
        best_skill = None
        if skills and random.random() < 0.5:
            # Find matching skill
            for sk in skills:
                if sk["task"] == task.name and sk["fitness"] > 0.15:
                    best_skill = sk
                    use_skill = True
                    break
        
        if use_skill and best_skill:
            # Execute skill (each step = 1 action)
            skill_reward = 0
            for step_action in best_skill["steps"]:
                if action_count >= num_actions:
                    break
                
                payload = {"invest": 1.0}
                next_obs, reward, info = env.step(obs, step_action, payload)
                rewards.append(reward)
                skill_reward += reward
                action_count += 1
                
                next_key = (task.name, task.difficulty)
                update_q(obs_key, step_action, reward, next_key)
                obs = next_obs
            
            # Update skill fitness
            best_skill["fitness"] = 0.7 * best_skill["fitness"] + 0.3 * (skill_reward / len(best_skill["steps"]))
        else:
            # Manual action
            if random.random() < 0.2:
                action = random.choice(actions_list)
            else:
                action = max(actions_list, key=lambda a: get_q(obs_key, a))
            
            payload = {"invest": 1.0}
            next_obs, reward, info = env.step(obs, action, payload)
            rewards.append(reward)
            action_count += 1
            
            trace.append(action)
            
            next_key = (task.name, task.difficulty)
            update_q(obs_key, action, reward, next_key)
            
            # Compile trace to skill if good
            if reward > 0.18 and len(trace) >= 1:
                skills.append({
                    "task": task.name,
                    "steps": trace[-3:] if len(trace) >= 3 else trace[:],
                    "fitness": reward,
                })
                if len(skills) > 50:
                    skills.sort(key=lambda s: s["fitness"], reverse=True)
                    skills = skills[:40]
    
    return {
        "version": "v7_skills",
        "total_reward": sum(rewards),
        "avg_reward_per_action": statistics.mean(rewards),
        "num_actions": len(rewards),
        "skills_evolved": len(skills),
        "final_tq": env.global_tool_quality,
        "final_kq": env.global_kb_quality,
        "final_oq": env.global_org_quality,
    }


def main():
    print("=" * 70)
    print("FAIR BENCHMARK: Same Environment, Same Reward Per Action")
    print("=" * 70)
    print()
    print("Protocol:")
    print("  - Environment: v5's ResearchEnvironment (6 tasks, diminishing returns)")
    print("  - Metric: Average reward PER ACTION (1 action = 1 unit of work)")
    print("  - Seeds: 3 runs for statistical significance")
    print()
    
    NUM_ACTIONS = 800  # Total actions per run
    SEEDS = [42, 123, 456]
    
    results = {"v5": [], "v7_baseline": [], "v7_skills": []}
    
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        
        # v5
        env = ResearchEnvironment(seed=seed)
        r = run_v5_fair(env, NUM_ACTIONS, seed)
        results["v5"].append(r)
        print(f"  v5:          avg={r['avg_reward_per_action']:.4f}")
        
        # v7 baseline (no skills, just Q-learning)
        env = ResearchEnvironment(seed=seed)
        r = run_v7_fair(env, NUM_ACTIONS, seed)
        results["v7_baseline"].append(r)
        print(f"  v7_baseline: avg={r['avg_reward_per_action']:.4f}")
        
        # v7 with skills
        env = ResearchEnvironment(seed=seed)
        r = run_v7_with_skills_fair(env, NUM_ACTIONS, seed)
        results["v7_skills"].append(r)
        print(f"  v7_skills:   avg={r['avg_reward_per_action']:.4f} (skills: {r['skills_evolved']})")
    
    print()
    print("=" * 70)
    print("FINAL RESULTS (avg of 3 seeds)")
    print("=" * 70)
    print()
    print(f"{'Version':<15} {'Avg Reward/Action':<20} {'Std Dev':<12} {'vs v5':<12}")
    print("-" * 60)
    
    # Calculate means
    v5_avg = statistics.mean([r["avg_reward_per_action"] for r in results["v5"]])
    v5_std = statistics.stdev([r["avg_reward_per_action"] for r in results["v5"]]) if len(SEEDS) > 1 else 0
    
    v7b_avg = statistics.mean([r["avg_reward_per_action"] for r in results["v7_baseline"]])
    v7b_std = statistics.stdev([r["avg_reward_per_action"] for r in results["v7_baseline"]]) if len(SEEDS) > 1 else 0
    
    v7s_avg = statistics.mean([r["avg_reward_per_action"] for r in results["v7_skills"]])
    v7s_std = statistics.stdev([r["avg_reward_per_action"] for r in results["v7_skills"]]) if len(SEEDS) > 1 else 0
    
    print(f"{'v5':<15} {v5_avg:<20.4f} {v5_std:<12.4f} {'(baseline)':<12}")
    print(f"{'v7_baseline':<15} {v7b_avg:<20.4f} {v7b_std:<12.4f} {((v7b_avg-v5_avg)/v5_avg*100):+.1f}%")
    print(f"{'v7_skills':<15} {v7s_avg:<20.4f} {v7s_std:<12.4f} {((v7s_avg-v5_avg)/v5_avg*100):+.1f}%")
    
    print()
    print("=" * 70)
    
    # Statistical significance
    best = max([(v5_avg, "v5"), (v7b_avg, "v7_baseline"), (v7s_avg, "v7_skills")])
    print(f"Winner: {best[1]} with avg reward {best[0]:.4f}")
    
    if best[1] == "v7_skills":
        improvement = (v7s_avg - v5_avg) / v5_avg * 100
        print(f"v7_skills improvement over v5: {improvement:+.2f}%")
    elif best[1] == "v5":
        print("v5 remains the best - v7 evolution did not help in this environment")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
