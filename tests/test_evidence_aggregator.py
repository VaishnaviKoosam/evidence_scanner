"""Offline aggregator tests. Run: python tests/test_evidence_aggregator.py.

All four external dependencies are patched at the aggregator import site.
Patches restore automatically, including when an assertion fails.
"""
import os
import sys
import time
from threading import Lock
from contextlib import contextmanager
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.evidence_aggregator as aggregator
from tools.pubmed_tool import Evidence

SOURCE_NAMES = ("pubmed_search", "europepmc_search", "semantic_scholar_search")
QUERIES = ["unstable angina systematic review", "unstable angina guideline"]


@contextmanager
def mocked_pipeline(hypothesis, responses=None, failed_source=None):
    responses = responses or {}
    calls = []
    generation_calls = []

    def fake_generate(value, num_queries=3):
        generation_calls.append(value)
        assert value == hypothesis
        return list(QUERIES)

    def make_search(name):
        def fake_search(query, max_results=8):
            calls.append((name, query))
            if name == failed_source:
                raise RuntimeError("simulated 429 rate limit")
            return responses.get((name, query), [])
        return fake_search

    replacements = {name: make_search(name) for name in SOURCE_NAMES}
    replacements["generate_search_queries"] = fake_generate
    with patch.multiple(aggregator, **replacements):
        yield calls, generation_calls


def expected_calls(queries):
    return [(source, query) for query in queries for source in SOURCE_NAMES]


def sample(source, title):
    return Evidence(source=source, title=title, abstract="Offline fixture abstract.")


def test_normal_results():
    hypothesis = "unstable angina"
    evidence = [sample(source, f"Paper {i}") for i, source in enumerate(SOURCE_NAMES)]
    responses = {
        (SOURCE_NAMES[0], QUERIES[0]): [evidence[0]],
        (SOURCE_NAMES[1], QUERIES[0]): [evidence[1]],
        (SOURCE_NAMES[2], QUERIES[1]): [evidence[2]],
    }
    with mocked_pipeline(hypothesis, responses) as (calls, generated):
        result = aggregator.gather_evidence(hypothesis)
        assert result["evidence"] == evidence
        assert all(actual is original for actual, original in zip(result["evidence"], evidence))
        assert result["broadened"] is False
        assert result["queries_used"] == QUERIES
        assert sorted(calls) == sorted(expected_calls(QUERIES))
        assert generated == [hypothesis]
    print("test_normal_results PASSED")


def test_zero_initial_results_retry_raw_hypothesis():
    # Whitespace and case make accidental normalization detectable.
    hypothesis = "  Unstable Angina  "
    evidence = sample("PubMed", "Evidence found by broadened query")
    responses = {(SOURCE_NAMES[0], hypothesis): [evidence]}
    with mocked_pipeline(hypothesis, responses) as (calls, generated):
        result = aggregator.gather_evidence(hypothesis)
        assert result["evidence"] == [evidence]
        assert result["broadened"] is True
        assert result["queries_used"] == [hypothesis]
        assert sorted(calls) == sorted(expected_calls(QUERIES + [hypothesis]))
        assert generated == [hypothesis]
    print("test_zero_initial_results_retry_raw_hypothesis PASSED")


def test_zero_results_after_retry_stop():
    hypothesis = "unstable angina"
    with mocked_pipeline(hypothesis) as (calls, generated):
        result = aggregator.gather_evidence(hypothesis)
        assert result["evidence"] == []
        assert result["broadened"] is True
        assert result["queries_used"] == [hypothesis]
        assert sorted(calls) == sorted(expected_calls(QUERIES + [hypothesis]))
        assert generated == [hypothesis]
    print("test_zero_results_after_retry_stop PASSED")


def test_partial_source_failure():
    hypothesis = "unstable angina"
    # Exercise each source failing, both during initial searches and the retry.
    for failed_source in SOURCE_NAMES:
        for retry in (False, True):
            responses = {}
            evidence = []
            for query in ([hypothesis] if retry else QUERIES):
                for source in SOURCE_NAMES:
                    if source != failed_source:
                        item = sample(source, f"{source}: {query}")
                        responses[(source, query)] = [item]
                        evidence.append(item)
            with mocked_pipeline(hypothesis, responses, failed_source) as (calls, generated):
                with patch.object(aggregator.logger, "warning") as warning:
                    result = aggregator.gather_evidence(hypothesis)
                assert warning.call_count == (3 if retry else 2)
                assert result["evidence"] == evidence
                assert result["broadened"] is retry
                assert result["queries_used"] == ([hypothesis] if retry else QUERIES)
                assert sorted(calls) == sorted(expected_calls(QUERIES + ([hypothesis] if retry else [])))
                assert generated == [hypothesis]
    print("test_partial_source_failure PASSED (all 3 sources, initial and retry)")


def test_failure_diagnostics_do_not_expose_keys():
    import requests
    hypothesis = "unstable angina"
    response = requests.Response()
    response.status_code = 429
    failure = requests.HTTPError("https://example.org/?api_key=SECRET", response=response)
    with mocked_pipeline(hypothesis):
        with patch.object(aggregator, "pubmed_search", side_effect=failure):
            with patch.object(aggregator.logger, "warning") as warning:
                result = aggregator.gather_evidence(hypothesis)
    assert result["evidence"] == []
    assert warning.call_count == len(QUERIES) + 1
    for call in warning.call_args_list:
        message = call.args[0] % call.args[1:]
        assert "HTTPError (HTTP 429)" in message
        assert "SECRET" not in message
        assert "api_key" not in message
    print("test_failure_diagnostics_do_not_expose_keys PASSED")



def test_search_fanout_runs_concurrently_with_independent_failure():
    queries = ["q0", "q1", "q2"]
    delays = [0.40, 0.10, 0.20, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    items = [sample(SOURCE_NAMES[i % 3], f"Paper {i}") for i in range(9)]
    completed = []
    active = peak = 0
    lock = Lock()

    def make_search(source_index):
        def search(query, max_results):
            nonlocal active, peak
            assert max_results == 7
            index = queries.index(query) * 3 + source_index
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(delays[index])
                if index == 4:
                    raise RuntimeError("secret URL must never be logged")
                return [items[index]]
            finally:
                with lock:
                    active -= 1
                    completed.append(index)
        search.__name__ = SOURCE_NAMES[source_index]
        return search

    with patch.multiple(aggregator, **{
        name: make_search(i) for i, name in enumerate(SOURCE_NAMES)
    }), patch.object(aggregator.logger, "warning") as warning:
        start = time.perf_counter()
        result = aggregator._search_queries(queries, max_results=7)
        elapsed = time.perf_counter() - start
    expected = [item for i, item in enumerate(items) if i != 4]
    assert result == expected
    assert all(actual is original for actual, original in zip(result, expected))
    assert sorted(completed) == list(range(9))
    assert completed[0] != 0
    assert 1 < peak <= 5
    assert elapsed < sum(delays) * 0.7, (elapsed, sum(delays))
    warning.assert_called_once_with(
        "Evidence source %s failed: %s; continuing", "europepmc_search", "RuntimeError"
    )
    print(f"test_search_fanout_runs_concurrently_with_independent_failure PASSED "
          f"({elapsed:.2f}s vs {sum(delays):.2f}s sequential; peak {peak} workers)")


if __name__ == "__main__":
    # Fail closed if a future edit accidentally bypasses the mocked functions.
    with patch("socket.socket", side_effect=AssertionError("Network forbidden in offline tests")) as network:
        test_search_fanout_runs_concurrently_with_independent_failure()
        test_normal_results()
        test_zero_initial_results_retry_raw_hypothesis()
        test_zero_results_after_retry_stop()
        test_partial_source_failure()
        test_failure_diagnostics_do_not_expose_keys()
        assert network.call_count == 0, "An unexpected network call was attempted"
    print("\nAll aggregator tests passed.")
