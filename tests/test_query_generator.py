"""
Offline tests for query_generator.py. The LLM call (_call_llm) is
monkeypatched so these tests need no network access and no API key --
they verify the parsing logic and the fallback path, which are the
parts we actually control.
"""
import sys
import json
import requests
import os
from unittest.mock import call, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.query_generator as qg


def test_parse_clean_json_array():
    raw = '["chest pain systematic review", "chest pain guideline", "chest pain RCT"]'
    result = qg._parse_query_response(raw)
    assert result == ["chest pain systematic review", "chest pain guideline", "chest pain RCT"]
    print("test_parse_clean_json_array PASSED")


def test_parse_json_wrapped_in_markdown_fences():
    raw = '```json\n["query one", "query two"]\n```'
    result = qg._parse_query_response(raw)
    assert result == ["query one", "query two"]
    print("test_parse_json_wrapped_in_markdown_fences PASSED")


def test_parse_garbage_returns_empty_and_triggers_fallback():
    raw = "sorry, I can't help with that"
    try:
        result = qg._parse_query_response(raw)
        assert False, "should have raised on invalid JSON"
    except json.JSONDecodeError:
        print("test_parse_garbage_returns_empty_and_triggers_fallback PASSED (raised as expected)")


def test_generate_search_queries_uses_fallback_when_llm_fails():
    def broken_llm(prompt):
        raise RuntimeError("simulated network failure")

    original = qg._call_llm
    qg._call_llm = broken_llm
    try:
        result = qg.generate_search_queries("unstable angina", num_queries=3)
        assert len(result) == 3
        assert all("unstable angina" in q for q in result)
        # result[0] is ALWAYS the deterministic diagnostic-validation query,
        # guaranteed in code regardless of LLM success/failure (see
        # _diagnostic_validation_query) -- the evidence-quality-type
        # queries (systematic review, guideline, etc.) fill the remaining
        # slots, starting at result[1].
        assert "diagnostic criteria" in result[0]
        assert "systematic review" in result[1]
    finally:
        qg._call_llm = original
    print("test_generate_search_queries_uses_fallback_when_llm_fails PASSED")


def test_generate_search_queries_uses_llm_when_available():
    def mock_llm(prompt):
        return '["mocked query 1", "mocked query 2", "mocked query 3"]'

    original = qg._call_llm
    qg._call_llm = mock_llm
    try:
        result = qg.generate_search_queries("sepsis", num_queries=3)
        # The guaranteed diagnostic query always occupies slot 0, so only
        # num_queries - 1 of the LLM's returned queries are actually used
        # (the LLM's 3rd suggestion is discarded here since num_queries=3).
        assert result[0] == "sepsis diagnostic criteria differential diagnosis clinical presentation"
        assert result[1:] == ["mocked query 1", "mocked query 2"]
        assert len(result) == 3
    finally:
        qg._call_llm = original
    print("test_generate_search_queries_uses_llm_when_available PASSED")


def test_fallback_respects_num_queries():
    result = qg._fallback_queries("sepsis", num_queries=2)
    assert len(result) == 2
    print("test_fallback_respects_num_queries PASSED")



def groq_response(status, content=""):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode()
    return response


def test_call_llm_retries_transient_http_errors():
    for status in (429, 500, 502, 503, 599):
        with patch.object(qg, "GROQ_API_KEY", "offline-key"),              patch.object(qg.requests, "post", side_effect=[
                 groq_response(status), groq_response(200, "Recovered")
             ]) as post, patch.object(qg.time, "sleep") as sleep:
            assert qg._call_llm("offline prompt") == "Recovered"
        assert post.call_count == 2
        sleep.assert_called_once_with(1)
        assert post.call_args_list[0] == post.call_args_list[1]
    print("test_call_llm_retries_transient_http_errors PASSED (429 and 5xx)")


def test_call_llm_persistent_429_raises_after_two_retries():
    with patch.object(qg, "GROQ_API_KEY", "offline-key"),          patch.object(qg.requests, "post", return_value=groq_response(429)) as post,          patch.object(qg.time, "sleep") as sleep:
        try:
            qg._call_llm("offline prompt")
        except requests.HTTPError as exc:
            assert exc.response.status_code == 429
        else:
            raise AssertionError("Persistent 429 must propagate to the caller")
    assert post.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    print("test_call_llm_persistent_429_raises_after_two_retries PASSED")


def test_call_llm_does_not_retry_other_errors():
    for status in (400, 401, 403, 404):
        with patch.object(qg, "GROQ_API_KEY", "offline-key"),              patch.object(qg.requests, "post", return_value=groq_response(status)) as post,              patch.object(qg.time, "sleep") as sleep:
            try:
                qg._call_llm("offline prompt")
            except requests.HTTPError as exc:
                assert exc.response.status_code == status
            else:
                raise AssertionError("Non-retryable HTTP error must propagate")
        post.assert_called_once()
        sleep.assert_not_called()
    with patch.object(qg, "GROQ_API_KEY", "offline-key"),          patch.object(qg.requests, "post", side_effect=requests.Timeout("offline")) as post,          patch.object(qg.time, "sleep") as sleep:
        try:
            qg._call_llm("offline prompt")
        except requests.Timeout:
            pass
        else:
            raise AssertionError("Non-HTTP errors must propagate")
    post.assert_called_once()
    sleep.assert_not_called()
    print("test_call_llm_does_not_retry_other_errors PASSED")


def test_persistent_rate_limit_keeps_existing_fallback():
    with patch.object(qg, "GROQ_API_KEY", "offline-key"),          patch.object(qg.requests, "post", return_value=groq_response(429)) as post,          patch.object(qg.time, "sleep") as sleep:
        result = qg.generate_search_queries("sepsis", num_queries=3)
    assert result == [
        qg._diagnostic_validation_query("sepsis"),
        "sepsis systematic review meta-analysis", "sepsis clinical guideline"
    ]
    assert post.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    print("test_persistent_rate_limit_keeps_existing_fallback PASSED")


if __name__ == "__main__":
    test_call_llm_retries_transient_http_errors()
    test_call_llm_persistent_429_raises_after_two_retries()
    test_call_llm_does_not_retry_other_errors()
    test_persistent_rate_limit_keeps_existing_fallback()
    import json
    test_parse_clean_json_array()
    test_parse_json_wrapped_in_markdown_fences()
    test_parse_garbage_returns_empty_and_triggers_fallback()
    test_generate_search_queries_uses_fallback_when_llm_fails()
    test_generate_search_queries_uses_llm_when_available()
    test_fallback_respects_num_queries()
    print("\nAll tests passed.")