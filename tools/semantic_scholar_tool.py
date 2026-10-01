"""
semantic_scholar_tool.py
--------------------------
Bottom-layer, framework-agnostic tool for querying the Semantic
Scholar Graph API. No key required for light/free use (shared public
rate limit -- ~100 requests per 5 minutes; request a free key from
Semantic Scholar if you need more headroom and set SEMANTIC_SCHOLAR_API_KEY).

Why this source matters beyond PubMed/Europe PMC: it gives us
citationCount, which we'll use in the hierarchy classifier as a
secondary signal -- e.g. a highly-cited cohort study can reasonably
outrank a barely-cited "systematic review" of dubious quality.

API docs: https://api.semanticscholar.org/api-docs/graph
"""

import os
import requests
from tools.pubmed_tool import Evidence  # reuse the same shared shape

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")  # optional

# Fields we actually need -- keeping this narrow keeps responses small
# and fast, and avoids pulling fields we'd have to ignore anyway.
FIELDS = "title,abstract,authors,venue,year,publicationTypes,externalIds,citationCount,url"


def semantic_scholar_search(query: str, max_results: int = 8) -> list[Evidence]:
    """
    Search Semantic Scholar for a hypothesis-specific query and return
    normalized Evidence objects, each carrying a citation_count for
    use as a secondary evidence-strength signal downstream.

    Parameters
    ----------
    query : str
        Free-text query, e.g. "chest pain angina diagnosis"
    max_results : int
        Number of results to fetch.

    Returns
    -------
    list[Evidence]
    """
    params = {
        "query": query,
        "limit": max_results,
        "fields": FIELDS,
    }
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

    resp = requests.get(S2_BASE, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    papers = data.get("data", [])
    return [_to_evidence(p) for p in papers]


def _to_evidence(p: dict) -> Evidence:
    authors = [a.get("name", "") for a in (p.get("authors") or [])]
    pub_types = p.get("publicationTypes") or []
    external_ids = p.get("externalIds") or {}

    ev = Evidence(
        source="SemanticScholar",
        title=(p.get("title") or "").strip(),
        abstract=(p.get("abstract") or "").strip(),
        authors=authors,
        journal=p.get("venue"),
        pub_date=str(p.get("year")) if p.get("year") else None,
        pub_types=pub_types,
        doi=external_ids.get("DOI"),
        external_id=external_ids.get("PubMed") or p.get("paperId"),
        url=p.get("url"),
    )
    # citation_count isn't part of the shared Evidence shape (PubMed/
    # Europe PMC don't have it), so we attach it as an extra attribute
    # rather than bloating the dataclass for every source.
    ev.citation_count = p.get("citationCount", 0)
    return ev


if __name__ == "__main__":
    # Manual smoke test -- requires network access to
    # api.semanticscholar.org, NOT available in this sandbox.
    # Run locally / in your own project environment to verify.
    results = semantic_scholar_search("chest pain angina diagnosis", max_results=3)
    for r in results:
        print(f"[{r.source}] {r.title} ({r.pub_date}) -- citations={getattr(r, 'citation_count', 0)}")