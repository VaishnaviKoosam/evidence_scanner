"""
pubmed_tool.py
----------------
Bottom-layer, framework-agnostic tool for querying PubMed via NCBI's
E-utilities API. No CrewAI / LangGraph dependency here on purpose --
this is a plain function so it can be reused by either framework later.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
No API key required, but rate limits are 3 req/sec without a key,
10 req/sec with a free NCBI API key (set NCBI_API_KEY env var).
"""

import os
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")  # optional, raises rate limit


@dataclass
class Evidence:
    """A single normalized piece of literature evidence.
    Every source-specific tool (PubMed, Europe PMC, Semantic Scholar)
    should return a list of these, so downstream code never has to
    care which API the evidence came from.
    """
    source: str                 # e.g. "PubMed"
    title: str
    abstract: str
    authors: list = field(default_factory=list)
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    pub_types: list = field(default_factory=list)   # e.g. ["Meta-Analysis"]
    doi: Optional[str] = None
    external_id: Optional[str] = None                # e.g. PMID
    url: Optional[str] = None


def _rate_limit_delay():
    # Stay comfortably under NCBI's limits.
    time.sleep(0.34 if not NCBI_API_KEY else 0.11)


def pubmed_search(query: str, max_results: int = 8, filters: Optional[dict] = None) -> list[Evidence]:
    """
    Search PubMed for a hypothesis-specific query and return normalized
    Evidence objects (title + abstract + metadata), ready for the
    evidence-hierarchy classifier in the next layer.

    Parameters
    ----------
    query : str
        Free-text query, e.g. "chest pain angina diagnosis"
    max_results : int
        Number of PMIDs to fetch details for.
    filters : dict, optional
        e.g. {"publication_type": "Meta-Analysis"} or {"date_range": "5"}
        (last 5 years). Kept simple on purpose -- extend as needed.

    Returns
    -------
    list[Evidence]
    """
    params = {
        "db": "pubmed",
        "term": _build_query_string(query, filters),
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=15)
    resp.raise_for_status()
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])

    if not id_list:
        return []

    _rate_limit_delay()
    return _fetch_details(id_list)


def _build_query_string(query: str, filters: Optional[dict]) -> str:
    q = query
    if filters:
        if "publication_type" in filters:
            q += f' AND "{filters["publication_type"]}"[Publication Type]'
        if "date_range" in filters:
            q += f' AND ("last {filters["date_range"]} years"[PDat])'
    return q


def _fetch_details(pmids: list[str]) -> list[Evidence]:
    """Given a list of PMIDs, fetch title/abstract/metadata via efetch."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=20)
    resp.raise_for_status()
    return _parse_efetch_xml(resp.text)


def _parse_efetch_xml(xml_text: str) -> list[Evidence]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    results = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join("".join(p.itertext()) for p in abstract_parts) if abstract_parts else ""

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        pub_date_el = article.find(".//PubDate/Year")
        pub_date = pub_date_el.text if pub_date_el is not None else None

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None:
                name = last.text
                if fore is not None:
                    name = f"{fore.text} {name}"
                authors.append(name)

        pub_types = [pt.text for pt in article.findall(".//PublicationTypeList/PublicationType") if pt.text]

        doi = None
        for eid in article.findall(".//ArticleIdList/ArticleId"):
            if eid.attrib.get("IdType") == "doi":
                doi = eid.text

        results.append(Evidence(
            source="PubMed",
            title=title.strip(),
            abstract=abstract.strip(),
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            pub_types=pub_types,
            doi=doi,
            external_id=pmid,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
        ))

    return results


if __name__ == "__main__":
    # Quick manual smoke test -- requires outbound network access to
    # eutils.ncbi.nlm.nih.gov, which is NOT available in this sandbox.
    # Run this locally / in your project environment to verify.
    results = pubmed_search("chest pain angina diagnosis", max_results=3)
    for r in results:
        print(f"[{r.source}] {r.title} ({r.pub_date}) -- types={r.pub_types}")