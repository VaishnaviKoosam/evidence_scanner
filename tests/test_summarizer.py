"""
Offline tests for summarizer.py. LLM call is monkeypatched, so these
run with no network access and no API key needed.
"""
import sys
import json
import requests
import os
import time
from threading import Barrier, Lock
from unittest.mock import call, patch
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



def test_batch_concurrent_completion_and_independent_fallback():
    evs = [
        Evidence(source="PubMed", title=str(i),
                 abstract=f"Abstract {i}. Further details.")
        for i in range(6)
    ]
    barrier = Barrier(3)
    completed = []
    lock = Lock()
    active = peak = 0

    def mock_llm(prompt):
        nonlocal active, peak
        index = next(i for i in range(6) if f"Abstract {i}." in prompt)
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if index < 3:
                barrier.wait(timeout=3)
            time.sleep(0.10 if index == 0 else 0.01)
            if index == 1:
                raise RuntimeError("one LLM call fails")
            return f"Summary {index}."
        finally:
            with lock:
                active -= 1
                completed.append(index)

    with patch.object(summarizer, "_call_llm", side_effect=mock_llm) as llm:
        result = summarizer.summarize_evidence_batch(evs, "sepsis")
    assert result is evs
    assert llm.call_count == len(evs)
    assert all(ev.summary for ev in evs)
    assert [ev.summary for ev in evs] == [
        "Summary 0.", "Abstract 1.", "Summary 2.",
        "Summary 3.", "Summary 4.", "Summary 5."
    ]
    assert completed[0] != 0
    assert peak == 3
    print("test_batch_concurrent_completion_and_independent_fallback PASSED "
          "(3 workers; correct per-item summaries and fallback)")



def groq_response(status, content=""):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode()
    return response


def test_call_llm_retries_transient_http_errors():
    for status in (429, 500, 502, 503, 599):
        with patch.object(summarizer, "GROQ_API_KEY", "offline-key"),              patch.object(summarizer.requests, "post", side_effect=[
                 groq_response(status), groq_response(200, "Recovered")
             ]) as post, patch.object(summarizer.time, "sleep") as sleep:
            assert summarizer._call_llm("offline prompt") == "Recovered"
        assert post.call_count == 2
        sleep.assert_called_once_with(1)
        assert post.call_args_list[0] == post.call_args_list[1]
    print("test_call_llm_retries_transient_http_errors PASSED (429 and 5xx)")


def test_call_llm_persistent_429_raises_after_two_retries():
    with patch.object(summarizer, "GROQ_API_KEY", "offline-key"),          patch.object(summarizer.requests, "post", return_value=groq_response(429)) as post,          patch.object(summarizer.time, "sleep") as sleep:
        try:
            summarizer._call_llm("offline prompt")
        except requests.HTTPError as exc:
            assert exc.response.status_code == 429
        else:
            raise AssertionError("Persistent 429 must propagate to the caller")
    assert post.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    print("test_call_llm_persistent_429_raises_after_two_retries PASSED")


def test_call_llm_does_not_retry_other_errors():
    for status in (400, 401, 403, 404):
        with patch.object(summarizer, "GROQ_API_KEY", "offline-key"),              patch.object(summarizer.requests, "post", return_value=groq_response(status)) as post,              patch.object(summarizer.time, "sleep") as sleep:
            try:
                summarizer._call_llm("offline prompt")
            except requests.HTTPError as exc:
                assert exc.response.status_code == status
            else:
                raise AssertionError("Non-retryable HTTP error must propagate")
        post.assert_called_once()
        sleep.assert_not_called()
    with patch.object(summarizer, "GROQ_API_KEY", "offline-key"),          patch.object(summarizer.requests, "post", side_effect=requests.Timeout("offline")) as post,          patch.object(summarizer.time, "sleep") as sleep:
        try:
            summarizer._call_llm("offline prompt")
        except requests.Timeout:
            pass
        else:
            raise AssertionError("Non-HTTP errors must propagate")
    post.assert_called_once()
    sleep.assert_not_called()
    print("test_call_llm_does_not_retry_other_errors PASSED")


def test_persistent_rate_limit_keeps_existing_fallback():
    ev = Evidence(source="PubMed", title="Offline",
                  abstract="First supported sentence. Additional details.")
    with patch.object(summarizer, "GROQ_API_KEY", "offline-key"),          patch.object(summarizer.requests, "post", return_value=groq_response(429)) as post,          patch.object(summarizer.time, "sleep") as sleep:
        result = summarizer.summarize_evidence_batch([ev], "sepsis")
    assert result == [ev]
    assert ev.summary == "First supported sentence."
    assert post.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    print("test_persistent_rate_limit_keeps_existing_fallback PASSED")


if __name__ == "__main__":
    test_call_llm_retries_transient_http_errors()
    test_call_llm_persistent_429_raises_after_two_retries()
    test_call_llm_does_not_retry_other_errors()
    test_persistent_rate_limit_keeps_existing_fallback()
    test_batch_concurrent_completion_and_independent_fallback()
    test_summarize_uses_llm_when_available()
    test_summarize_falls_back_when_llm_fails()
    test_summarize_handles_empty_abstract()
    test_fallback_summary_truncates_long_abstract_without_sentence_end()
    test_summarize_evidence_batch_attaches_summary_to_each()
    print("\nAll tests passed.")