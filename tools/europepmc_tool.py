"""
europepmc_tool.py
------------------
Bottom-layer, framework-agnostic tool for querying the Europe PMC REST
API. This is our free substitute for Google Scholar: Europe PMC
indexes PubMed + preprints + a lot of guideline content, with no key
required and a permissive ToS (unlike scraping Scholar directly).

API docs: https://europepmc.org/RestfulWebService
No API key required.
"""

import requests
from tools.pubmed_tool import Evidence  # reuse the same shared shape

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def europepmc_search(query: str, max_results: int = 8, filters: dict | None = None) -> list[Evidence]:
    """
    Search Europe PMC for a hypothesis-specific query and return
    normalized Evidence objects.

    Parameters
    ----------
    query : str
        Free-text query, e.g. "chest pain angina diagnosis"
    max_results : int
        Number of results to fetch.
    filters : dict, optional
        e.g. {"publication_type": "guideline"} to bias toward guideline
        content, which is otherwise hard to get for free.

    Returns
    -------
    list[Evidence]
    """
    params = {
        "query": _build_query_string(query, filters),
        "format": "json",
        "pageSize": max_results,
        "resultType": "core",  # gives full abstract + pub types, not just IDs
    }

    resp = requests.get(EPMC_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("resultList", {}).get("result", [])
    return [_to_evidence(r) for r in results]


def _build_query_string(query: str, filters: dict | None) -> str:
    q = query
    if filters and "publication_type" in filters:
        # Europe PMC supports PUB_TYPE filtering, e.g. PUB_TYPE:"guideline"
        q += f' AND PUB_TYPE:"{filters["publication_type"]}"'
    return q


def _to_evidence(r: dict) -> Evidence:
    authors = []
    if r.get("authorString"):
        authors = [a.strip() for a in r["authorString"].split(",")]

    pub_types = r.get("pubTypeList", {}).get("pubType", [])
    if isinstance(pub_types, str):
        pub_types = [pub_types]

    pmid = r.get("pmid")
    url = f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id', '')}"

    return Evidence(
        source="EuropePMC",
        title=r.get("title", "").strip(),
        abstract=r.get("abstractText", "").strip(),
        authors=authors,
        journal=r.get("journalInfo", {}).get("journal", {}).get("title"),
        pub_date=r.get("pubYear"),
        pub_types=pub_types,
        doi=r.get("doi"),
        external_id=pmid or r.get("id"),
        url=url,
    )


if __name__ == "__main__":
    # Manual smoke test -- requires network access to
    # www.ebi.ac.uk, NOT available in this sandbox.
    # Run locally / in your own project environment to verify.
    results = europepmc_search("chest pain angina diagnosis", max_results=3)
    for r in results:
        print(f"[{r.source}] {r.title} ({r.pub_date}) -- types={r.pub_types}")