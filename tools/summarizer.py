"""
summarizer.py
---------------
Condenses each Evidence object's (often dense, jargon-heavy) abstract
into a short, hypothesis-relevant summary line -- this is what
actually gets shown to the Clinical Data Fusion Agent downstream,
rather than raw abstract text.

Same pattern as query_generator.py: the LLM call is isolated in
_call_llm so it's the only network-touching function, and there's a
deterministic fallback (truncate to first sentence / N words) so a
flaky LLM call degrades gracefully instead of breaking the pipeline.
"""

import os
import re
import requests
from dotenv import load_dotenv
from tools.pubmed_tool import Evidence

load_dotenv()  # reads .env in the project root and loads it into os.environ

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = """You are summarizing a piece of medical literature for a clinical \
decision support system evaluating the diagnostic hypothesis: "{hypothesis}"

Abstract:
\"\"\"{abstract}\"\"\"

Write ONE concise sentence (max 35 words) summarizing what this evidence says that is \
relevant to the hypothesis above. State only what the abstract supports -- do not add \
outside information. If the abstract is not clearly relevant, say so briefly.

Respond with ONLY the sentence, no preamble.
"""

FALLBACK_MAX_WORDS = 40


def summarize_evidence(evidence: Evidence, hypothesis: str) -> str:
    """
    Return a short, hypothesis-relevant summary string for one piece
    of evidence. Falls back to a truncated version of the abstract if
    the LLM call fails or the abstract is empty.
    """
    if not evidence.abstract:
        return f"No abstract available for: {evidence.title}"

    prompt = PROMPT_TEMPLATE.format(hypothesis=hypothesis, abstract=evidence.abstract)

    try:
        summary = _call_llm(prompt).strip()
        if summary:
            return summary
    except Exception:
        pass

    return _fallback_summary(evidence.abstract)


def summarize_evidence_batch(evidence_list: list[Evidence], hypothesis: str) -> list[Evidence]:
    """
    Summarize a list of Evidence objects in place, attaching a
    `.summary` attribute to each (kept as an extra attribute rather
    than a dataclass field, same reasoning as citation_count -- not
    every piece of evidence in the pipeline necessarily needs one).
    """
    for ev in evidence_list:
        ev.summary = summarize_evidence(ev, hypothesis)
    return evidence_list


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
            "temperature": 0.2,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _fallback_summary(abstract: str) -> str:
    """
    Deterministic backup summary: first sentence if it's reasonably
    short, otherwise the first FALLBACK_MAX_WORDS words. Not as sharp
    as an LLM summary, but guarantees something usable downstream.
    """
    first_sentence_match = re.match(r"^(.{0,250}?[.!?])\s", abstract + " ")
    if first_sentence_match:
        return first_sentence_match.group(1).strip()

    words = abstract.split()
    truncated = " ".join(words[:FALLBACK_MAX_WORDS])
    suffix = "..." if len(words) > FALLBACK_MAX_WORDS else ""
    return truncated + suffix


if __name__ == "__main__":
    # Manual smoke test -- requires network access to api.groq.com and
    # a GROQ_API_KEY set, NOT available in this sandbox.
    # Run locally / in your own project environment to verify.
    sample = Evidence(
        source="PubMed",
        title="Diagnostic accuracy of chest pain protocols",
        abstract="We conducted a systematic review of 42 studies evaluating chest "
                 "pain triage protocols in emergency departments. Pooled sensitivity "
                 "was 94% and specificity was 88% for identifying acute coronary syndrome.",
    )
    print(summarize_evidence(sample, "unstable angina"))