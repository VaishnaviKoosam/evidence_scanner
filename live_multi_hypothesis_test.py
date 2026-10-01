"""
live_multi_hypothesis_test.py
--------------------------------
Runs the REAL agent, with real network calls, across several different
hypotheses in a row -- and times each one, so you can actually see the
concurrency improvement for yourself, not just trust the test suite.

Place this in your project root (same level as agents/, tools/).
Run with: python live_multi_hypothesis_test.py

Requires: a working .env with GROQ_API_KEY, and real internet access.
"""
import time
import json
from agents.evidence_scanner_agent import build_evidence_scanner

HYPOTHESES = [
    "unstable angina",
    "type 2 diabetes",
    "migraine",
    "community-acquired pneumonia",
]

scanner = build_evidence_scanner()

total_start = time.perf_counter()

for h in HYPOTHESES:
    print(f"\n{'='*70}")
    print(f"HYPOTHESIS: {h}")
    print(f"{'='*70}")

    start = time.perf_counter()
    result = scanner.invoke({
        "request": {"hypotheses": [{"diagnosis": h}]},
        "top_n": 5,
    })
    elapsed = time.perf_counter() - start

    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Broadened (fallback retry triggered): {result['broadened']}")
    print(f"Evidence items returned: {len(result['evidence'])}")
    for item in result["evidence"][:3]:
        print(f"  [{item['tier_label']}] score={item['score']:.1f} -- {item['title'][:70]}")
        print(f"    Summary: {item['summary'][:100]}...")

total_elapsed = time.perf_counter() - total_start
print(f"\n{'='*70}")
print(f"TOTAL across {len(HYPOTHESES)} hypotheses: {total_elapsed:.2f} seconds")
print(f"Average per hypothesis: {total_elapsed / len(HYPOTHESES):.2f} seconds")
print(f"{'='*70}")