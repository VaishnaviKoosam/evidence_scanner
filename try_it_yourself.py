"""
try_it_yourself.py
---------------------
Interactive playground to test each tool one at a time with a REAL
diagnosis you type in, so you can see exactly what each layer
outputs. Requires internet + your .env GROQ_API_KEY to be set.

Run from the project root:
    python3 try_it_yourself.py
"""

from tools.query_generator import generate_search_queries
from tools.pubmed_tool import pubmed_search
from tools.europepmc_tool import europepmc_search
from tools.semantic_scholar_tool import semantic_scholar_search
from tools.evidence_hierarchy import rank_evidence
from tools.summarizer import summarize_evidence


def print_evidence_list(evidence_list, label):
    print(f"\n--- {label} ({len(evidence_list)} results) ---")
    if not evidence_list:
        print("  (no results)")
        return
    for i, ev in enumerate(evidence_list, 1):
        print(f"\n  [{i}] {ev.title}")
        print(f"      Source: {ev.source} | Year: {ev.pub_date} | Types: {ev.pub_types}")
        print(f"      Abstract: {ev.abstract[:150]}{'...' if len(ev.abstract) > 150 else ''}")
        if hasattr(ev, "citation_count"):
            print(f"      Citations: {ev.citation_count}")


def menu():
    print("\n" + "=" * 60)
    print("Which tool do you want to test?")
    print("  1. query_generator   -- see the 3 generated search queries")
    print("  2. pubmed_tool       -- see real PubMed results")
    print("  3. europepmc_tool    -- see real Europe PMC results")
    print("  4. semantic_scholar_tool -- see real Semantic Scholar results")
    print("  5. evidence_hierarchy -- see ranking of combined results")
    print("  6. summarizer        -- see LLM summary of one result")
    print("  7. Run ALL of the above, in order")
    print("  0. Quit")
    return input("Enter choice: ").strip()


def main():
    diagnosis = input("\nEnter a diagnostic hypothesis (e.g. 'unstable angina'): ").strip()
    if not diagnosis:
        diagnosis = "unstable angina"
        print(f"(using default: '{diagnosis}')")

    while True:
        choice = menu()

        if choice == "0":
            print("Bye!")
            break

        elif choice in ("1", "7"):
            print("\n>>> Calling query_generator.generate_search_queries()...")
            queries = generate_search_queries(diagnosis, num_queries=3)
            for q in queries:
                print(f"  - {q}")

        elif choice in ("2", "7"):
            print("\n>>> Calling pubmed_tool.pubmed_search()...")
            results = pubmed_search(diagnosis, max_results=3)
            print_evidence_list(results, "PubMed results")

        elif choice in ("3", "7"):
            print("\n>>> Calling europepmc_tool.europepmc_search()...")
            results = europepmc_search(diagnosis, max_results=3)
            print_evidence_list(results, "Europe PMC results")

        elif choice in ("4", "7"):
            print("\n>>> Calling semantic_scholar_tool.semantic_scholar_search()...")
            results = semantic_scholar_search(diagnosis, max_results=3)
            print_evidence_list(results, "Semantic Scholar results")

        elif choice in ("5", "7"):
            print("\n>>> Combining all 3 sources and calling evidence_hierarchy.rank_evidence()...")
            combined = (
                pubmed_search(diagnosis, max_results=3)
                + europepmc_search(diagnosis, max_results=3)
                + semantic_scholar_search(diagnosis, max_results=3)
            )
            ranked = rank_evidence(combined)
            print(f"\n--- Ranked evidence, strongest first ({len(ranked)} total) ---")
            for scored in ranked:
                print(f"  [{scored.tier_label}] score={scored.score:.1f} -- {scored.evidence.title[:80]}")

        elif choice in ("6", "7"):
            print("\n>>> Calling summarizer.summarize_evidence() on a real PubMed result...")
            results = pubmed_search(diagnosis, max_results=1)
            if results:
                ev = results[0]
                print(f"\n  Original abstract:\n  {ev.abstract[:300]}...")
                summary = summarize_evidence(ev, diagnosis)
                print(f"\n  LLM summary:\n  {summary}")
            else:
                print("  (no PubMed results to summarize)")

        else:
            print("Not a valid choice, try again.")


if __name__ == "__main__":
    main()