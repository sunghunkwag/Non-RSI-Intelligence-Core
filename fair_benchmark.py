"""
FAIR BENCHMARK: v2 vs v3
- IDENTICAL environments (same ResearchEnvironment)
- IDENTICAL seed
- ONLY difference: v3 has System 2 meta-cognitive reflection in Agent.choose_action()
"""

import subprocess
import sys
import re
import statistics

def run_simulation(script, rounds, seed):
    result = subprocess.run(
        [sys.executable, script, "--rounds", str(rounds), "--seed", str(seed), "--agents", "8"],
        capture_output=True, text=True, encoding='utf-8'
    )
    return result.stdout

def extract_rewards(output):
    """Extract top-3 average rewards from each round."""
    rewards = []
    for line in output.splitlines():
        match = re.search(r"top_rewards=\[(.*?)\]", line)
        if match:
            vals = [float(x) for x in match.group(1).split(",")]
            rewards.append(statistics.mean(vals))
    return rewards

def main():
    SEED = 42
    ROUNDS = 40
    
    print("=" * 60)
    print("  FAIR BENCHMARK: v2 (Baseline) vs v3 (System 2 Upgrade)")
    print("=" * 60)
    print(f"Controlled Seed: {SEED}")
    print(f"Rounds: {ROUNDS}")
    print(f"Agents: 8")
    print(f"Environment: IDENTICAL (v2's ResearchEnvironment)")
    print(f"Only Difference: v3 adds meta-cognitive reflection\n")
    
    print("Running v2 (Baseline)...")
    out_v2 = run_simulation("NON_RSI_AGI_CORE_v2.py", ROUNDS, SEED)
    rewards_v2 = extract_rewards(out_v2)
    
    print("Running v3 (System 2 Reflection)...")
    out_v3 = run_simulation("NON_RSI_AGI_CORE_v3_FAIR.py", ROUNDS, SEED)
    rewards_v3 = extract_rewards(out_v3)
    
    if not rewards_v2 or not rewards_v3:
        print("\nERROR: Could not parse results.")
        print(f"v2 rewards: {len(rewards_v2)}, v3 rewards: {len(rewards_v3)}")
        return
    
    # Statistics
    avg_v2 = statistics.mean(rewards_v2)
    avg_v3 = statistics.mean(rewards_v3)
    std_v2 = statistics.stdev(rewards_v2)
    std_v3 = statistics.stdev(rewards_v3)
    peak_v2 = max(rewards_v2)
    peak_v3 = max(rewards_v3)
    
    print("\n" + "=" * 60)
    print("                    RESULTS")
    print("=" * 60)
    
    print(f"\nv2 (Baseline):")
    print(f"  Average Reward:  {avg_v2:.4f}")
    print(f"  Peak Reward:     {peak_v2:.4f}")
    print(f"  Stability (std): {std_v2:.4f}")
    print(f"  Final 5 rounds:  {[round(r, 3) for r in rewards_v2[-5:]]}")
    
    print(f"\nv3 (System 2 Meta-Cognition):")
    print(f"  Average Reward:  {avg_v3:.4f}")
    print(f"  Peak Reward:     {peak_v3:.4f}")
    print(f"  Stability (std): {std_v3:.4f}")
    print(f"  Final 5 rounds:  {[round(r, 3) for r in rewards_v3[-5:]]}")
    
    # Comparison
    delta_avg = avg_v3 - avg_v2
    delta_pct = (delta_avg / avg_v2) * 100 if avg_v2 != 0 else 0
    delta_std = std_v3 - std_v2
    delta_peak = peak_v3 - peak_v2
    
    print("\n" + "=" * 60)
    print("                 HONEST COMPARISON")
    print("=" * 60)
    print(f"Average Reward Change: {delta_avg:+.4f} ({delta_pct:+.2f}%)")
    print(f"Peak Reward Change:    {delta_peak:+.4f}")
    print(f"Stability Change:      {delta_std:+.4f} (Negative is better)")
    
    if delta_pct > 5:
        print(f"\n✓ RESULT: v3 shows SIGNIFICANT improvement (+{delta_pct:.1f}%)")
    elif delta_pct > 0:
        print(f"\n~ RESULT: v3 shows marginal improvement (+{delta_pct:.1f}%)")
    else:
        print(f"\n✗ RESULT: v3 does NOT outperform v2 ({delta_pct:.1f}%)")
    
    # Save
    with open("fair_benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Fair Benchmark Results (Seed={SEED}, Rounds={ROUNDS})\\n")
        f.write(f"v2 avg: {avg_v2:.4f}, v3 avg: {avg_v3:.4f}\\n")
        f.write(f"Improvement: {delta_pct:+.2f}%\\n")
        f.write(f"v2 rewards: {rewards_v2}\\n")
        f.write(f"v3 rewards: {rewards_v3}\\n")
    
    print(f"\nResults saved to fair_benchmark_results.txt")

if __name__ == "__main__":
    main()
