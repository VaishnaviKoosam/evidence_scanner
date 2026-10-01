"""Collect literature evidence, retrying empty searches with the raw hypothesis."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

# Load before importing connectors, which read API keys at import time.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from tools.pubmed_tool import Evidence, pubmed_search
from tools.europepmc_tool import europepmc_search
from tools.semantic_scholar_tool import semantic_scholar_search
from tools.query_generator import generate_search_queries

logger = logging.getLogger(__name__)


def _search_queries(queries: list[str], max_results: int) -> list[Evidence]:
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(search, query, max_results=max_results): (index, search)
            for index, (query, search) in enumerate(
                (query, search)
                for query in queries
                for search in (pubmed_search, europepmc_search, semantic_scholar_search)
            )
        }
        results = [[] for _ in futures]
        for future in as_completed(futures):
            index, search = futures[future]
            try:
                results[index].extend(future.result())
            except Exception as exc:
                # A failed source must not discard results or skip other sources.
                # Never log raw exceptions: request URLs can contain API keys.
                detail = type(exc).__name__
                response = getattr(exc, "response", None)
                if response is not None:
                    detail += f" (HTTP {response.status_code})"
                cause = exc.__cause__ or exc.__context__
                if cause is not None:
                    detail += f"; underlying {type(cause).__name__}"
                logger.warning(
                    "Evidence source %s failed: %s; continuing",
                    getattr(search, "__name__", type(search).__name__), detail,
                )
    return [item for batch in results for item in batch]


def gather_evidence(hypothesis: str, num_queries: int = 3,
                    max_results: int = 3) -> dict:
    """Return evidence, queries_used and broadened for a plain hypothesis string.

    Search all sources for each generated query. If the combined result is
    empty, retry all sources once with the original, unmodified hypothesis.
    max_results limits each source per query. Ranking and summarization are
    left to the caller. Query generation supplies its own deterministic fallback.
    """
    queries = generate_search_queries(hypothesis, num_queries=num_queries)
    evidence = _search_queries(queries, max_results)
    broadened = not evidence
    if broadened:
        queries = [hypothesis]
        evidence = _search_queries(queries, max_results)
    return {
        "evidence": evidence,
        "broadened": broadened,
        "queries_used": queries,
    }

