"""One-time diagnostic: inspect title and future_work BERTScore pairs."""

import json

with open("results/bertscore_raw.json", encoding="utf-8") as fh:
    raw = json.load(fh)

for r in raw:
    if r["field_name"] not in ("title", "future_work"):
        continue
    print(f"\n=== {r['paper_name']} | {r['field_name']} | F1={r['f1']:.3f} ===")
    print(f"PRED: {r['prediction'][:300]}")
    print(f"REF:  {r['reference'][:300]}")