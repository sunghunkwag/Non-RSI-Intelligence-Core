"""
COMPREHENSIVE BENCHMARK: v2 (Baseline) vs v4 (Enhanced AGI)
- v4 improvements:
  1. Balanced meta-cognition (success + failure analysis)
  2. Associative memory (knowledge graph)
  3. Probabilistic risk assessment (not binary)
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
    
    print("=" * 70)
    print("  HONEST BENCHMARK: v2 (Baseline) vs v4 (Enhanced Meta-Cognition)")
    print("=" * 70)
    print(f"Controlled Seed: {SEED}")
    print(f"Rounds: {ROUNDS}")
    print(f"Agents: 8")
    print(f"\nv4 Enhancements:")
    print("  • Balanced success/failure analysis (not just failures)")
    print("  • Associative knowledge graph (tags link related memories)")
    print("  • Probabilistic risk (60% switch vs 100% override)")
    print("  • Lower thresholds (0.25 success, 0.10 failure)\n")
    
    print("Running v2 (Baseline)...")
    out_v2 = run_simulation("NON_RSI_AGI_CORE_v2.py", ROUNDS, SEED)
    rewards_v2 = extract_rewards(out_v2)
    
    print("Running v4 (Enhanced Meta-Cognition)...")
    out_v4 = run_simulation("NON_RSI_AGI_CORE_v4.py", ROUNDS, SEED)
    rewards_v4 = extract_rewards(out_v4)
    
    if not rewards_v2 or not rewards_v4:
        print(f"\nERROR: v2={len(rewards_v2)} rounds, v4={len(rewards_v4)} rounds")
        return
    
    # Statistics
    avg_v2 = statistics.mean(rewards_v2)
    avg_v4 = statistics.mean(rewards_v4)
    std_v2 = statistics.stdev(rewards_v2)
    std_v4 = statistics.stdev(rewards_v4)
    peak_v2 = max(rewards_v2)
    peak_v4 = max(rewards_v4)
    
    # Learning progression (early vs late)
    early_v2 = statistics.mean(rewards_v2[:10])
    late_v2 = statistics.mean(rewards_v2[-10:])
    early_v4 = statistics.mean(rewards_v4[:10])
    late_v4 = statistics.mean(rewards_v4[-10:])
    
    print("\n" + "=" * 70)
    print("                         RESULTS")
    print("=" * 70)
    
    print(f"\nv2 (Baseline):")
    print(f"  Average:         {avg_v2:.4f}")
    print(f"  Peak:            {peak_v2:.4f}")
    print(f"  Stability (std): {std_v2:.4f}")
    print(f"  Early → Late:    {early_v2:.4f} → {late_v2:.4f} (Δ {late_v2 - early_v2:+.4f})")
    
    print(f"\nv4 (Enhanced Meta-Cognition):")
    print(f"  Average:         {avg_v4:.4f}")
    print(f"  Peak:            {peak_v4:.4f}")
    print(f"  Stability (std): {std_v4:.4f}")
    print(f"  Early → Late:    {early_v4:.4f} → {late_v4:.4f} (Δ {late_v4 - early_v4:+.4f})")
    
    # Comparison
    delta_avg = avg_v4 - avg_v2
    delta_pct = (delta_avg / avg_v2) * 100 if avg_v2 != 0 else 0
    delta_std = std_v4 - std_v2
    delta_peak = peak_v4 - peak_v2
    learning_improvement = (late_v4 - early_v4) - (late_v2 - early_v2)
    
    print("\n" + "=" * 70)
    print("                    HONEST VERDICT")
    print("=" * 70)
    print(f"Average Performance:  {delta_pct:+.2f}% change")
    print(f"Peak Performance:     {delta_peak:+.4f}")
    print(f"Stability:            {delta_std:+.4f} (Negative = Better)")
    print(f"Learning Speed:       {learning_improvement:+.4f} (How much more it learned)")
    
    if delta_pct > 3:
        print(f"\n✓ SUCCESS: v4 outperforms v2 by {delta_pct:.1f}%")
    elif delta_pct > 0:
        print(f"\n~ MARGINAL: v4 shows slight improvement (+{delta_pct:.1f}%)")
    else:
        print(f"\n✗ REGRESSION: v4 underperforms v2 ({delta_pct:.1f}%)")
    
    print(f"\nResults saved to v4_benchmark_results.txt")
    
    with open("v4_benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write(f"v4 Benchmark (Seed={SEED}, Rounds={ROUNDS})\\n")
        f.write(f"v2: avg={avg_v2:.4f}, peak={peak_v2:.4f}, std={std_v2:.4f}\\n")
        f.write(f"v4: avg={avg_v4:.4f}, peak={peak_v4:.4f}, std={std_v4:.4f}\\n")
        f.write(f"Change: {delta_pct:+.2f}%\\n")
        f.write(f"v2: {rewards_v2}\\n")
        f.write(f"v4: {rewards_v4}\\n")

if __name__ == "__main__":
    main()
