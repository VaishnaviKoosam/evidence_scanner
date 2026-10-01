"""Offline integration tests using the real compiled LangGraph, no pytest."""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agents.evidence_scanner_agent as agent
import tools.evidence_aggregator as aggregator
import tools.summarizer as summarizer
from tools.pubmed_tool import Evidence


def papers():
    types = ["Case Reports", "review", "guideline", "cohort", "RCT", "Meta-Analysis", ""]
    return [Evidence(source="PubMed", title=f"Paper {i}", abstract=f"Abstract {i}.",
                     pub_types=[kind], url=f"https://example.org/{i}")
            for i, kind in enumerate(types)]


def test_ranks_before_summarizing_default_top_five():
    evidence = papers()
    summary_calls = []

    def fake_batch(selected, hypothesis):
        summary_calls.append(([ev.title for ev in selected], hypothesis))
        for ev in selected:
            ev.summary = "Offline summary"
        return selected

    with patch.object(agent, "gather_evidence", return_value={"evidence": evidence, "broadened": False}) as gather:
        with patch.object(agent, "summarize_evidence_batch", side_effect=fake_batch):
            result = agent.scan_evidence("unstable angina")
    gather.assert_called_once_with("unstable angina")
    assert summary_calls == [(["Paper 5", "Paper 4", "Paper 2", "Paper 3", "Paper 1"], "unstable angina")]
    assert set(result) == {"hypothesis", "broadened", "evidence"}
    assert result["hypothesis"] == "unstable angina"
    assert result["broadened"] is False
    assert len(result["evidence"]) == 5
    assert result["evidence"][0]["tier_label"] == "Systematic Review / Meta-Analysis"
    assert result["evidence"][0]["score"] == 100.0
    for row in result["evidence"]:
        assert set(row) == {"tier_label", "score", "title", "summary", "url", "source"}
        assert row["summary"] == "Offline summary"
    assert not hasattr(evidence[0], "summary")
    assert not hasattr(evidence[6], "summary")
    json.dumps(result)
    print("test_ranks_before_summarizing_default_top_five PASSED")


def test_structured_handoffs_and_llm_fallback():
    handoff = {"hypotheses": [{"diagnosis": "  Unstable Angina  ", "confidence": 0.8}]}
    graph = agent.build_evidence_scanner()
    for request in (handoff, json.dumps(handoff)):
        with patch.object(agent, "gather_evidence", return_value={"evidence": papers(), "broadened": True}) as gather:
            with patch.object(summarizer, "_call_llm", side_effect=RuntimeError("offline")) as llm:
                result = graph.invoke({"request": request, "top_n": 2})
        gather.assert_called_once_with("  Unstable Angina  ")
        assert result["hypothesis"] == "  Unstable Angina  "
        assert result["broadened"] is True
        assert len(result["evidence"]) == 2
        assert llm.call_count == 2
        assert result["evidence"][0]["summary"] == "Abstract 5."
    print("test_structured_handoffs_and_llm_fallback PASSED")


def test_empty_evidence_skips_summarization():
    with patch.object(agent, "gather_evidence", return_value={"evidence": [], "broadened": True}):
        with patch.object(agent, "summarize_evidence_batch") as summarize:
            result = agent.build_evidence_scanner().invoke({"request": "unknown condition"})
    assert result == {"hypothesis": "unknown condition", "broadened": True, "evidence": []}
    summarize.assert_not_called()
    print("test_empty_evidence_skips_summarization PASSED")


def test_invalid_inputs_fail_before_search():
    bad_requests = ["", "  ", None, [], "{broken", "[]", {}, {"hypotheses": []},
                    {"hypotheses": [{"diagnosis": "a"}, {"diagnosis": "b"}]},
                    {"hypotheses": [{"diagnosis": 42}]}, {"hypotheses": [{}]}]
    graph = agent.build_evidence_scanner()
    inputs = [{"request": value} for value in bad_requests]
    inputs += [{"request": "angina", "top_n": value} for value in (0, -1, True, 1.5, "5", None)]
    with patch.object(agent, "gather_evidence") as gather:
        for state in inputs:
            try:
                graph.invoke(state)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Expected input rejection: {state!r}")
        gather.assert_not_called()
    print("test_invalid_inputs_fail_before_search PASSED")


def test_real_pipeline_partial_failure_and_broadened_retry():
    item = papers()[5]
    calls = []

    def pubmed(query, max_results):
        calls.append(query)
        return [item] if query == "angina" else []

    with patch.multiple(aggregator,
                        generate_search_queries=lambda *args, **kwargs: ["specific query"],
                        pubmed_search=pubmed,
                        europepmc_search=lambda *args, **kwargs: []):
        with patch.object(aggregator, "semantic_scholar_search", side_effect=RuntimeError("429")):
            with patch.object(aggregator.logger, "warning"):
                with patch.object(summarizer, "_call_llm", side_effect=RuntimeError("offline")):
                    result = agent.scan_evidence("angina")
    assert calls == ["specific query", "angina"]
    assert result["broadened"] is True
    assert result["evidence"][0]["summary"] == "Abstract 5."
    assert result["evidence"][0]["title"] == "Paper 5"
    print("test_real_pipeline_partial_failure_and_broadened_retry PASSED")


if __name__ == "__main__":
    # Block HTTP as well as sockets: swallowed connector errors cannot hide calls.
    with patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")) as http:
        with patch("socket.socket", side_effect=AssertionError("Network forbidden")) as network:
            test_ranks_before_summarizing_default_top_five()
            test_structured_handoffs_and_llm_fallback()
            test_empty_evidence_skips_summarization()
            test_invalid_inputs_fail_before_search()
            test_real_pipeline_partial_failure_and_broadened_retry()
            assert http.call_count == 0
            assert network.call_count == 0
    print("\nAll LangGraph scanner tests passed.")
