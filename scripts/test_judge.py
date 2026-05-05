"""Manually test what the match judge says about pipeline[1] vs ground_truth[1]."""

from openai import OpenAI

from litreviewrag import config
from litreviewrag.synthesis.contradiction import Contradiction
from evaluation.contradiction_f1 import _judge_match, _papers_match

client = OpenAI(api_key=config.OPENAI_API_KEY)

pred = Contradiction(
    paper_a="eval_paper_1.pdf",
    paper_b="eval_paper_2.pdf",
    claim_a="Our findings suggest that current models exhibit only moderate population-level alignment on average, with average simulation quality scores across all tests on SP-ABCBENCH ranging from 50 to 64 out of 100.",
    claim_b="The validity rates of the larger models are close to 100%, indicating a high fidelity of the simulated human behavior across various tests.",
    explanation="Disagreement on simulation fidelity",
)

ref = {
    "paper_a": "eval_paper_3.pdf",
    "paper_b": "eval_paper_1.pdf",
    "claim_a": "GPT-3 demonstrates high algorithmic fidelity and accurately reflects diverse human sub-populations' attitudes and behaviors",
    "claim_b": "Current large language models achieve only moderate alignment with real human security and privacy patterns, with simulation quality scores of just 50-64 out of 100",
    "explanation": "Disagreement on LLM simulation quality",
}

print(f"Papers match (share at least one)? {_papers_match(pred, ref)}")
print(f"Pred papers:  {pred.paper_a}, {pred.paper_b}")
print(f"Ref papers:   {ref['paper_a']}, {ref['paper_b']}")
print(f"Shared paper: {set([pred.paper_a, pred.paper_b]) & set([ref['paper_a'], ref['paper_b']])}")
print(f"\nJudge verdict: {_judge_match(client, pred, ref)}")