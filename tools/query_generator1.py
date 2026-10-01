"""
query_generator.py
---------------------
Given a diagnostic hypothesis (from the Symptom RAG Analyzer, per the
base paper), generate targeted search-query strings to fire at the
PubMed / Europe PMC / Semantic Scholar connectors -- rather than just
searching the raw hypothesis text, which tends to return noisy,
low-relevance results.

Design choice: the LLM call itself lives in one small, isolated
function (_call_llm). Everything else -- prompt building, response
parsing, and a deterministic fallback -- is plain Python and fully
testable without any network access. If _call_llm fails or returns
garbage, generate_search_queries() still returns usable queries via
the fallback, so a flaky LLM call never breaks the pipeline.

Uses Groq's free tier (OpenAI-compatible chat completions endpoint).
Swap providers by rewriting _call_llm only -- the prompt/parsing/
fallback logic is provider-agnostic.
"""

import os
import json
import re
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = """You are helping a clinical decision support system search medical literature.

Given the diagnostic hypothesis below, generate {num_queries} distinct search queries \
to find supporting or refuting evidence. Vary the queries so they are likely to surface \
different types of evidence (e.g. one aimed at systematic reviews/meta-analyses, one at \
clinical guidelines, one at randomized controlled trials or primary studies).

Diagnostic hypothesis: "{hypothesis}"

Respond with ONLY a JSON array of strings, nothing else. Example format:
["query one", "query two", "query three"]
"""


def generate_search_queries(hypothesis: str, num_queries: int = 3) -> list[str]:
    """
    Generate `num_queries` targeted search strings for a diagnostic
    hypothesis. Falls back to deterministic heuristic queries if the
    LLM call fails or returns unparseable output.
    """
    prompt = PROMPT_TEMPLATE.format(hypothesis=hypothesis, num_queries=num_queries)

    try:
        raw_response = _call_llm(prompt)
        queries = _parse_query_response(raw_response)
        if queries:
            return queries[:num_queries]
    except Exception:
        # Network hiccup, bad API key, malformed JSON -- any failure
        # here should degrade gracefully, not crash the agent.
        pass

    return _fallback_queries(hypothesis, num_queries)


def _call_llm(prompt: str) -> str:
    """Isolated LLM call -- the only function that touches the network."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    resp = requests.post(
        GROQ_BASE,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_query_response(raw_response: str) -> list[str]:
    """
    Parse the LLM's response into a clean list of query strings.
    Handles the common case of the model wrapping JSON in markdown
    code fences despite being told not to.
    """
    cleaned = re.sub(r"```json|```", "", raw_response).strip()
    parsed = json.loads(cleaned)

    if not isinstance(parsed, list):
        return []

    return [q.strip() for q in parsed if isinstance(q, str) and q.strip()]


def _fallback_queries(hypothesis: str, num_queries: int) -> list[str]:
    """
    Deterministic backup queries, used when the LLM is unavailable.
    Not as sharp as an LLM-tuned query, but guarantees the pipeline
    always has something reasonable to search with.
    """
    templates = [
        f"{hypothesis} systematic review meta-analysis",
        f"{hypothesis} clinical guideline",
        f"{hypothesis} randomized controlled trial",
        f"{hypothesis} diagnosis management",
    ]
    return templates[:num_queries]


if __name__ == "__main__":
    # Manual smoke test -- requires network access to api.groq.com and
    # a GROQ_API_KEY set, NOT available in this sandbox.
    # Run locally / in your own project environment to verify.
    queries = generate_search_queries("unstable angina", num_queries=3)
    for q in queries:
        print(f"- {q}")