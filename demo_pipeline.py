"""
demo_pipeline.py
-------------------
A small end-to-end demo that ties together everything built so far
(layers 1-5), so you can SEE the whole pipeline work on one real
diagnostic hypothesis before we wrap it in CrewAI (layer 6).

Run this from the project root:
    python3 demo_pipeline.py

Requires:
    - Your .env file with a working GROQ_API_KEY
    - Real internet access (this won't work in Claude's sandbox --
      run it on your own machine)
"""

from tools.query_generator import generate_search_queries
from tools.pubmed_tool import pubmed_search
from tools.europepmc_tool import europepmc_search
from tools.semantic_scholar_tool import semantic_scholar_search
from tools.evidence_hierarchy import rank_evidence
from tools.summarizer import summarize_evidence_batch

HYPOTHESIS = "unstable angina"
RESULTS_PER_SOURCE = 3
TOP_N_TO_SHOW = 5


def run_pipeline(hypothesis: str):
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC HYPOTHESIS: {hypothesis}")
    print(f"{'='*70}\n")

    # --- Step 1: generate targeted search queries ---
    print("[1/4] Generating search queries...")
    queries = generate_search_queries(hypothesis, num_queries=3)
    for q in queries:
        print(f"      - {q}")

    # --- Step 2: fetch evidence from all 3 sources, for every query ---
    print("\n[2/4] Fetching evidence from PubMed, Europe PMC, Semantic Scholar...")
    all_evidence = []
    for q in queries:
        try:
            all_evidence.extend(pubmed_search(q, max_results=RESULTS_PER_SOURCE))
        except Exception as e:
            print(f"      ! PubMed failed for '{q}': {e}")
        try:
            all_evidence.extend(europepmc_search(q, max_results=RESULTS_PER_SOURCE))
        except Exception as e:
            print(f"      ! Europe PMC failed for '{q}': {e}")
        try:
            all_evidence.extend(semantic_scholar_search(q, max_results=RESULTS_PER_SOURCE))
        except Exception as e:
            print(f"      ! Semantic Scholar failed for '{q}': {e}")

    print(f"      Total evidence collected: {len(all_evidence)}")

    # --- Step 3: rank by evidence hierarchy ---
    print("\n[3/4] Ranking evidence by hierarchy (strongest first)...")
    ranked = rank_evidence(all_evidence)
    top_n = [r.evidence for r in ranked[:TOP_N_TO_SHOW]]

    # --- Step 4: summarize the top N ---
    print(f"\n[4/4] Summarizing top {len(top_n)} pieces of evidence...\n")
    summarize_evidence_batch(top_n, hypothesis)

    # --- Print final report ---
    print(f"{'='*70}")
    print("TOP EVIDENCE (strongest first)")
    print(f"{'='*70}")
    for scored, ev in zip(ranked[:TOP_N_TO_SHOW], top_n):
        print(f"\n[{scored.tier_label}]  (score: {scored.score:.1f})  source: {ev.source}")
        print(f"Title:   {ev.title}")
        print(f"Summary: {getattr(ev, 'summary', '(no summary)')}")
        if ev.url:
            print(f"URL:     {ev.url}")


if __name__ == "__main__":
    run_pipeline(HYPOTHESIS)