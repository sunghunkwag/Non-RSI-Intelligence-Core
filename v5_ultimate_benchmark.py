"""
ULTIMATE BENCHMARK: v2 vs v5
v5 combines ALL enhancements:
1. Enhanced WorldModel (non-linear features + experience replay)
2. Adaptive Planning (difficulty-based depth/width)
3. Improved Meta-Cognition (balanced success/failure)
4. Knowledge Graph (associative memory)
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
    
    print("=" * 75)
    print("  ULTIMATE BENCHMARK: v2 (Baseline) vs v5 (All Enhancements)")
    print("=" * 75)
    print(f"Seed: {SEED}, Rounds: {ROUNDS}, Agents: 8\n")
    print("v5 Enhancements:")
    print("  ✓ Non-linear feature combinations (diff×action, domain×diff)")
    print("  ✓ Experience replay buffer (batch learning from past)")
    print("  ✓ Adaptive planning (depth 2-5, width 4-8 based on difficulty)")
    print("  ✓ Balanced meta-cognition (success + failure analysis)")
    print("  ✓ Knowledge graph (associative tag linking)\n")
    
    print("Running v2 (Baseline)...")
    out_v2 = run_simulation("NON_RSI_AGI_CORE_v2.py", ROUNDS, SEED)
    rewards_v2 = extract_rewards(out_v2)
    
    print("Running v5 (Ultimate)...")
    out_v5 = run_simulation("NON_RSI_AGI_CORE_v5.py", ROUNDS, SEED)
    rewards_v5 = extract_rewards(out_v5)
    
    if not rewards_v2 or not rewards_v5:
        print(f"\nERROR: v2={len(rewards_v2)}, v5={len(rewards_v5)}")
        return
    
    # Statistics
    avg_v2 = statistics.mean(rewards_v2)
    avg_v5 = statistics.mean(rewards_v5)
    std_v2 = statistics.stdev(rewards_v2)
    std_v5 = statistics.stdev(rewards_v5)
    peak_v2 = max(rewards_v2)
    peak_v5 = max(rewards_v5)
    
    # Learning curve
    early_v2 = statistics.mean(rewards_v2[:10])
    late_v2 = statistics.mean(rewards_v2[-10:])
    early_v5 = statistics.mean(rewards_v5[:10])
    late_v5 = statistics.mean(rewards_v5[-10:])
    learning_v2 = late_v2 - early_v2
    learning_v5 = late_v5 - early_v5
    
    print("\n" + "=" * 75)
    print("                           RESULTS")
    print("=" * 75)
    
    print(f"\nv2 (Baseline):")
    print(f"  Average:         {avg_v2:.4f}")
    print(f"  Peak:            {peak_v2:.4f}")
    print(f"  Stability (std): {std_v2:.4f}")
    print(f"  Early → Late:    {early_v2:.4f} → {late_v2:.4f}  (Δ {learning_v2:+.4f})")
    
    print(f"\nv5 (Ultimate AGI Core):")
    print(f"  Average:         {avg_v5:.4f}")
    print(f"  Peak:            {peak_v5:.4f}")
    print(f"  Stability (std): {std_v5:.4f}")
    print(f"  Early → Late:    {early_v5:.4f} → {late_v5:.4f}  (Δ {learning_v5:+.4f})")
    
    # Comparison
    delta_avg = avg_v5 - avg_v2
    delta_pct = (delta_avg / avg_v2) * 100 if avg_v2 != 0 else 0
    delta_std = std_v5 - std_v2
    delta_peak = peak_v5 - peak_v2
    delta_learning = learning_v5 - learning_v2
    
    print("\n" + "=" * 75)
    print("                      FINAL VERDICT")
    print("=" * 75)
    print(f"Average Performance:  {delta_pct:+.2f}%")
    print(f"Peak Performance:     {delta_peak:+.4f}")
    print(f"Stability:            {delta_std:+.4f} (Negative = Better)")
    print(f"Learning Speed:       {delta_learning:+.4f} (Higher = Faster learning)")
    
    if delta_pct >= 10:
        print(f"\n🏆 BREAKTHROUGH: v5 achieves {delta_pct:.1f}% improvement!")
    elif delta_pct >= 5:
        print(f"\n✓ STRONG SUCCESS: v5 outperforms by {delta_pct:.1f}%")
    elif delta_pct > 0:
        print(f"\n~ MARGINAL: v5 shows +{delta_pct:.1f}% improvement")
    else:
        print(f"\n✗ REGRESSION: v5 underperforms ({delta_pct:.1f}%)")
    
    print(f"\nDetailed results saved to v5_ultimate_benchmark.txt")
    
    with open("v5_ultimate_benchmark.txt", "w", encoding="utf-8") as f:
        f.write(f"Ultimate Benchmark (Seed={SEED}, Rounds={ROUNDS})\\n")
        f.write(f"v2: avg={avg_v2:.4f}, peak={peak_v2:.4f}, std={std_v2:.4f}\\n")
        f.write(f"v5: avg={avg_v5:.4f}, peak={peak_v5:.4f}, std={std_v5:.4f}\\n")
        f.write(f"Improvement: {delta_pct:+.2f}%\\n")
        f.write(f"\\nv2 trajectory: {[round(r, 3) for r in rewards_v2]}\\n")
        f.write(f"v5 trajectory: {[round(r, 3) for r in rewards_v5]}\\n")

if __name__ == "__main__":
    main()
