import numpy as np
import json

results = json.load(open("evaluation_results.json"))

gaps_norm = [r["r_chosen_norm"] - r["r_rejected_norm"] for r in results]
print("mean gap:", np.mean(gaps_norm))
print("median gap:", np.median(gaps_norm))
print("% positive:", np.mean([g > 0 for g in gaps_norm]))

acc_raw  = sum(r["correct_raw"]  for r in results) / len(results)
acc_norm = sum(r["correct_norm"] for r in results) / len(results)

print(f"n pairs: {len(results)}")
print(f"raw accuracy: {acc_raw:.3f}")
print(f"length-normalized accuracy: {acc_norm:.3f}")

print(f"{'raw':>12} | {'normalized':>12}")
print(f"{acc_raw:>12.3f} | {acc_norm:>12.3f}")

