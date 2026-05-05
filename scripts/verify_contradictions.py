"""One-time verification: check contradictions ground truth file parses correctly."""

import json

with open("evaluation/ground_truth/contradictions.json", encoding="utf-8") as fh:
    data = json.load(fh)

print(f"{len(data)} contradictions loaded")
for c in data:
    print(f"  - {c['paper_a']} vs {c['paper_b']}")