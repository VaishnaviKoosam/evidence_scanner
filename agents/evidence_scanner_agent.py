"""LangGraph Evidence-Based Scanner Agent.

Role: validate or challenge a diagnostic hypothesis against external literature.
Goal: hand ranked, summarized evidence to the downstream Clinical Data Fusion Agent.
Context: the upstream Text Clarifier supplies a diagnosis, optionally in JSON.
The graph orchestrates existing tools; evidence strength is not diagnostic confidence.

Run from the project root: python -m agents.evidence_scanner_agent
"""
import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

# Also support: python agents/evidence_scanner_agent.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from tools.evidence_aggregator import gather_evidence
from tools.evidence_hierarchy import ScoredEvidence, rank_evidence
from tools.pubmed_tool import Evidence
from tools.summarizer import summarize_evidence_batch


class EvidenceResult(TypedDict):
    tier_label: str
    score: float
    title: str
    summary: str
    url: str | None
    source: str


class ScannerResult(TypedDict):
    hypothesis: str
    broadened: bool
    evidence: list[EvidenceResult]


class ScannerInput(TypedDict):
    request: str | dict
    top_n: int


class ScannerState(TypedDict, total=False):
    request: str | dict
    top_n: int
    hypothesis: str
    broadened: bool
    gathered: list[Evidence]
    ranked: list[ScoredEvidence]
    evidence: list[EvidenceResult]


def extract_hypothesis(request: str | dict) -> str:
    """Extract exactly one diagnosis at the agent boundary, preserving its text.

    Accept plain text, a Text Clarifier dictionary, or its JSON serialization.
    Multiple hypotheses require separate invocations to avoid silent selection.
    Confidence is upstream metadata, not an evidence-ranking input.
    """
    if isinstance(request, str):
        stripped = request.strip()
        if not stripped:
            raise ValueError("The diagnostic hypothesis must not be empty")
        if stripped.startswith(("{", "[")):
            try:
                request = json.loads(request)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid Text Clarifier JSON") from exc
        else:
            return request
    if not isinstance(request, dict):
        raise ValueError("Expected a diagnosis string or Text Clarifier object")
    hypotheses = request.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 1:
        raise ValueError("Provide exactly one hypothesis per scanner invocation")
    entry = hypotheses[0]
    diagnosis = entry.get("diagnosis") if isinstance(entry, dict) else None
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        raise ValueError("Each hypothesis requires a nonempty diagnosis string")
    return diagnosis


def _extract(state: ScannerState) -> dict:
    top_n = state.get("top_n", 5)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    return {"hypothesis": extract_hypothesis(state["request"]), "top_n": top_n}


def _gather(state: ScannerState) -> dict:
    result = gather_evidence(state["hypothesis"])
    return {"gathered": result["evidence"], "broadened": result["broadened"]}


def _rank(state: ScannerState) -> dict:
    return {"ranked": rank_evidence(state["gathered"])[:state["top_n"]]}


def _summarize(state: ScannerState) -> dict:
    ranked = state["ranked"]
    if ranked:
        # The existing batch function attaches summaries in place and provides
        # deterministic fallbacks for failed LLM calls or missing abstracts.
        summarize_evidence_batch([item.evidence for item in ranked], state["hypothesis"])
    return {"evidence": [
        {"tier_label": item.tier_label, "score": item.score,
         "title": item.evidence.title, "summary": item.evidence.summary,
         "url": item.evidence.url, "source": item.evidence.source}
        for item in ranked
    ]}


def build_evidence_scanner():
    """Compile a reusable graph with a JSON-serializable downstream output."""
    builder = StateGraph(ScannerState, input_schema=ScannerInput, output_schema=ScannerResult)
    builder.add_node("extract_hypothesis", _extract)
    builder.add_node("gather_evidence", _gather)
    builder.add_node("rank_evidence", _rank)
    builder.add_node("summarize_evidence", _summarize)
    builder.add_edge(START, "extract_hypothesis")
    builder.add_edge("extract_hypothesis", "gather_evidence")
    builder.add_edge("gather_evidence", "rank_evidence")
    builder.add_edge("rank_evidence", "summarize_evidence")
    builder.add_edge("summarize_evidence", END)
    return builder.compile()


def scan_evidence(hypothesis: str, top_n: int = 5) -> ScannerResult:
    """Plain-string entry point; structured handoffs use the graph's request key."""
    if not isinstance(hypothesis, str):
        raise ValueError("scan_evidence expects a diagnosis string")
    return build_evidence_scanner().invoke({"request": hypothesis, "top_n": top_n})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hypothesis", nargs="?", default="unstable angina")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    # Demonstrate an actual Text Clarifier handoff and compiled graph invocation.
    handoff = {"hypotheses": [{"diagnosis": args.hypothesis, "confidence": 0.8}]}
    result = build_evidence_scanner().invoke({"request": handoff, "top_n": args.top_n})
    print(json.dumps(result, indent=2, ensure_ascii=False))
