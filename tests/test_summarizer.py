"""
Offline tests for summarizer.py. LLM call is monkeypatched, so these
run with no network access and no API key needed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.summarizer as summarizer
from tools.pubmed_tool import Evidence


def test_summarize_uses_llm_when_available():
    def mock_llm(prompt):
        return "This meta-analysis supports troponin testing for angina diagnosis."

    original = summarizer._call_llm
    summarizer._call_llm = mock_llm
    try:
        ev = Evidence(source="PubMed", title="t", abstract="Some dense abstract text here.")
        result = summarizer.summarize_evidence(ev, "unstable angina")
        assert result == "This meta-analysis supports troponin testing for angina diagnosis."
    finally:
        summarizer._call_llm = original
    print("test_summarize_uses_llm_when_available PASSED")


def test_summarize_falls_back_when_llm_fails():
    def broken_llm(prompt):
        raise RuntimeError("simulated failure")

    original = summarizer._call_llm
    summarizer._call_llm = broken_llm
    try:
        ev = Evidence(
            source="PubMed", title="t",
            abstract="Troponin testing shows high sensitivity. Further details follow in the full text."
        )
        result = summarizer.summarize_evidence(ev, "unstable angina")
        assert result == "Troponin testing shows high sensitivity."
    finally:
        summarizer._call_llm = original
    print("test_summarize_falls_back_when_llm_fails PASSED")


def test_summarize_handles_empty_abstract():
    ev = Evidence(source="PubMed", title="Mystery paper", abstract="")
    result = summarizer.summarize_evidence(ev, "unstable angina")
    assert "Mystery paper" in result
    assert "No abstract" in result
    print("test_summarize_handles_empty_abstract PASSED")


def test_fallback_summary_truncates_long_abstract_without_sentence_end():
    long_text = " ".join(["word"] * 100)  # no punctuation at all
    result = summarizer._fallback_summary(long_text)
    assert result.endswith("...")
    assert len(result.split()) <= summarizer.FALLBACK_MAX_WORDS + 1  # +1 for "..."
    print("test_fallback_summary_truncates_long_abstract_without_sentence_end PASSED")


def test_summarize_evidence_batch_attaches_summary_to_each():
    def mock_llm(prompt):
        return "Mocked summary."

    original = summarizer._call_llm
    summarizer._call_llm = mock_llm
    try:
        evs = [
            Evidence(source="PubMed", title="a", abstract="abstract a"),
            Evidence(source="EuropePMC", title="b", abstract="abstract b"),
        ]
        result = summarizer.summarize_evidence_batch(evs, "sepsis")
        assert all(hasattr(e, "summary") for e in result)
        assert all(e.summary == "Mocked summary." for e in result)
    finally:
        summarizer._call_llm = original
    print("test_summarize_evidence_batch_attaches_summary_to_each PASSED")


if __name__ == "__main__":
    test_summarize_uses_llm_when_available()
    test_summarize_falls_back_when_llm_fails()
    test_summarize_handles_empty_abstract()
    test_fallback_summary_truncates_long_abstract_without_sentence_end()
    test_summarize_evidence_batch_attaches_summary_to_each()
    print("\nAll tests passed.")