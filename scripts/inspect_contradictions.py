"""One-time diagnostic: see what contradictions the pipeline produced."""

import json

with open("results/contradiction_f1_raw.json", encoding="utf-8") as fh:
    d = json.load(fh)

print("=" * 60)
print("GROUND TRUTH (what we expected)")
print("=" * 60)
for i, gt in enumerate(d["ground_truth"]):
    print(f"\n[{i+1}] {gt['paper_a']} vs {gt['paper_b']}")
    print(f"    A: {gt['claim_a'][:200]}")
    print(f"    B: {gt['claim_b'][:200]}")

print("\n" + "=" * 60)
print("PIPELINE PRODUCED")
print("=" * 60)
for i, p in enumerate(d["predictions"]):
    print(f"\n[{i+1}] {p['paper_a']} vs {p['paper_b']}")
    print(f"    A: {p['claim_a'][:200]}")
    print(f"    B: {p['claim_b'][:200]}")
    print(f"    matched_ref: {p['matched_ref_index']}")