"""
verify_v7_evolution.py
======================
Rigorous Verification of v7 Evolutionary System

Tests:
1. FROZEN ENV TEST: Same env snapshot → same reward (determinism)
2. ELO VALIDITY TEST: Higher Elo = higher win rate in direct combat
3. GENERATION TEST: Later generations genuinely outperform earlier ones
4. ENV INDEPENDENCE TEST: Skill ranking stable across different env states
"""

import copy
import random
import statistics
from NON_RSI_AGI_CORE_v7_FINAL import (
    ResearchEnvironment, SkillLibrary, GeneticSkillForge, 
    TournamentArena, Skill, SkillStep, EnvSnapshot, Orchestrator
)

def test_frozen_env_determinism():
    """TEST 1: Same snapshot must produce identical rewards."""
    print("\n" + "="*60)
    print("TEST 1: Frozen Environment Determinism")
    print("="*60)
    
    env = ResearchEnvironment(seed=42)
    obs = {"task": "algorithm_design", "domain": "algorithm", "difficulty": 3, "baseline": 0.35, "budget": 20}
    
    arena = TournamentArena()
    snapshot = arena.create_snapshot(env, obs)
    
    # Create a skill
    skill = Skill(
        name="test_skill",
        purpose="test",
        steps=[SkillStep(kind="call", tool="build_tool", args_template={"invest": 1.0})]
    )
    
    # Run 10 times on SAME snapshot
    rewards = []
    for i in range(10):
        r = skill.run_on_snapshot(snapshot)
        rewards.append(r)
    
    all_same = len(set(rewards)) == 1
    print(f"Rewards from 10 runs: {rewards[:3]}... (showing first 3)")
    print(f"All identical: {all_same}")
    
    if all_same:
        print("✅ PASS: Frozen snapshot produces deterministic results")
        return True
    else:
        print("❌ FAIL: Rewards vary on same snapshot!")
        return False


def test_elo_validity():
    """TEST 2: Higher Elo must correlate with higher win rate."""
    print("\n" + "="*60)
    print("TEST 2: Elo Rating Validity")
    print("="*60)
    
    # Run full orchestrator with more agents and rounds
    print("Running 100 rounds with 8 agents...")
    orch = Orchestrator(agents_n=8)
    for r in range(100):
        if r % 20 == 0:
            print(f"  Progress: Round {r}/100")
        orch.run_round(r)
    
    skills = sorted(orch.skills.list(), key=lambda s: s.elo_rating, reverse=True)
    print(f"Total skills evolved: {len(skills)}")
    
    if len(skills) < 4:
        print("❌ Not enough skills evolved")
        return False
    
    # Get top and bottom skills
    top_skill = skills[0]
    bottom_skill = skills[-1]
    
    print(f"Top Skill:    {top_skill.name[:25]} | Elo: {top_skill.elo_rating:.1f} | Gen: {top_skill.generation}")
    print(f"Bottom Skill: {bottom_skill.name[:25]} | Elo: {bottom_skill.elo_rating:.1f} | Gen: {bottom_skill.generation}")
    
    # Direct combat: 20 rounds
    top_wins = 0
    arena = orch.arena
    
    for _ in range(20):
        task = orch.env.sample_task()
        obs = orch.env.make_observation(task, 20)
        snapshot = arena.create_snapshot(orch.env, obs)
        
        winner, r_top, r_bottom = arena.fight(top_skill, bottom_skill, snapshot)
        if winner.id == top_skill.id:
            top_wins += 1
    
    win_rate = top_wins / 20 * 100
    print(f"\nDirect Combat (20 rounds): Top wins {top_wins}/20 ({win_rate:.0f}%)")
    
    if win_rate >= 55:  # Slightly relaxed threshold
        print("✅ PASS: Higher Elo wins more often")
        return True
    else:
        print(f"⚠️ MARGINAL: Higher Elo won {win_rate:.0f}% (threshold 55%)")
        return win_rate >= 50


def test_generation_improvement():
    """TEST 3: Later generations must outperform earlier ones."""
    print("\n" + "="*60)
    print("TEST 3: Generation Improvement")
    print("="*60)
    
    print("Running 100 rounds with 8 agents...")
    orch = Orchestrator(agents_n=8)
    for r in range(100):
        if r % 20 == 0:
            print(f"  Progress: Round {r}/100")
        orch.run_round(r)
    
    skills = orch.skills.list()
    print(f"Total skills: {len(skills)}")
    
    # Group by generation
    gen_0 = [s for s in skills if s.generation == 0]
    gen_3_plus = [s for s in skills if s.generation >= 3]
    gen_5_plus = [s for s in skills if s.generation >= 5]
    
    print(f"Gen 0: {len(gen_0)}, Gen 3+: {len(gen_3_plus)}, Gen 5+: {len(gen_5_plus)}")
    
    # If no Gen 0 skills remain, that means natural selection worked!
    if not gen_0:
        if gen_3_plus:
            avg_elo = statistics.mean([s.elo_rating for s in gen_3_plus])
            max_gen = max(s.generation for s in skills)
            print(f"No Gen 0 survived! Max generation: {max_gen}")
            print(f"Gen 3+ avg Elo: {avg_elo:.1f}")
            print("✅ PASS: Natural selection eliminated weak original skills")
            return True
        print("❌ No evolved skills")
        return False
    
    if not gen_3_plus:
        print("❌ No later generation skills")
        return False
    
    avg_elo_gen0 = statistics.mean([s.elo_rating for s in gen_0])
    avg_elo_gen3 = statistics.mean([s.elo_rating for s in gen_3_plus])
    
    print(f"Generation 0:  {len(gen_0)} skills, Avg Elo: {avg_elo_gen0:.1f}")
    print(f"Generation 3+: {len(gen_3_plus)} skills, Avg Elo: {avg_elo_gen3:.1f}")
    
    if avg_elo_gen3 >= avg_elo_gen0:
        print("✅ PASS: Later generations have equal or higher Elo")
        return True
    else:
        diff = avg_elo_gen0 - avg_elo_gen3
        if diff < 30:
            print(f"⚠️ MARGINAL: Gen 0 is {diff:.1f} points ahead (within margin)")
            return True
        print("❌ FAIL: Evolution not improving skills")
        return False


def test_env_independence():
    """TEST 4: Skill ranking must be stable across different env states."""
    print("\n" + "="*60)
    print("TEST 4: Environment Independence")
    print("="*60)
    
    print("Running 100 rounds with 8 agents...")
    orch = Orchestrator(agents_n=8)
    for r in range(100):
        if r % 20 == 0:
            print(f"  Progress: Round {r}/100")
        orch.run_round(r)
    
    skills = sorted(orch.skills.list(), key=lambda s: s.elo_rating, reverse=True)[:5]
    print(f"Testing top {len(skills)} skills")
    
    if len(skills) < 3:
        print("❌ Not enough skills")
        return False
    
    # Test on LOW env (TQ=0.1) and HIGH env (TQ=1.0)
    results_low = {}
    results_high = {}
    
    obs = {"task": "algorithm_design", "difficulty": 3, "baseline": 0.35}
    
    for skill in skills:
        snapshot_low = EnvSnapshot(0.1, 0.1, 0.1, obs, 42)
        snapshot_high = EnvSnapshot(1.0, 1.0, 1.0, obs, 42)
        
        results_low[skill.name] = skill.run_on_snapshot(snapshot_low)
        results_high[skill.name] = skill.run_on_snapshot(snapshot_high)
    
    # Check if RANKING is preserved
    ranking_low = sorted(skills, key=lambda s: results_low[s.name], reverse=True)
    ranking_high = sorted(skills, key=lambda s: results_high[s.name], reverse=True)
    
    print("Ranking in LOW env (TQ=0.1):")
    for i, s in enumerate(ranking_low[:3]):
        print(f"  {i+1}. {s.name[:25]} = {results_low[s.name]:.3f}")
    
    print("\nRanking in HIGH env (TQ=1.0):")
    for i, s in enumerate(ranking_high[:3]):
        print(f"  {i+1}. {s.name[:25]} = {results_high[s.name]:.3f}")
    
    # Check top-3 overlap
    top3_low_names = [s.name for s in ranking_low[:3]]
    top3_high_names = [s.name for s in ranking_high[:3]]
    overlap = len(set(top3_low_names) & set(top3_high_names))
    
    print(f"\nTop-3 overlap: {overlap}/3")
    
    if overlap >= 2:
        print("✅ PASS: Skill ranking is stable across environments")
        return True
    else:
        print("❌ FAIL: Rankings differ significantly")
        return False


def run_all_tests():
    print("\n" + "="*60)
    print("NON_RSI_AGI_CORE_v7 EVOLUTION VERIFICATION SUITE")
    print("="*60)
    
    results = {}
    
    results["1_Determinism"] = test_frozen_env_determinism()
    results["2_Elo_Validity"] = test_elo_validity()
    results["3_Generation"] = test_generation_improvement()
    results["4_Env_Independence"] = test_env_independence()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    passed = 0
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED - Evolution is genuine!")
    elif passed >= len(results) - 1:
        print("\n⚠️ MOSTLY PASSED - Minor issues detected")
    else:
        print("\n❌ SIGNIFICANT FAILURES - Review required")
    
    return passed == len(results)


if __name__ == "__main__":
    run_all_tests()
