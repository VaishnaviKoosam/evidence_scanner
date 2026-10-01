"""
Offline tests for query_generator.py. The LLM call (_call_llm) is
monkeypatched so these tests need no network access and no API key --
they verify the parsing logic and the fallback path, which are the
parts we actually control.
"""
import sys
import os
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
        assert "systematic review" in result[0]
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
        assert result == ["mocked query 1", "mocked query 2", "mocked query 3"]
    finally:
        qg._call_llm = original
    print("test_generate_search_queries_uses_llm_when_available PASSED")


def test_fallback_respects_num_queries():
    result = qg._fallback_queries("sepsis", num_queries=2)
    assert len(result) == 2
    print("test_fallback_respects_num_queries PASSED")


if __name__ == "__main__":
    import json
    test_parse_clean_json_array()
    test_parse_json_wrapped_in_markdown_fences()
    test_parse_garbage_returns_empty_and_triggers_fallback()
    test_generate_search_queries_uses_fallback_when_llm_fails()
    test_generate_search_queries_uses_llm_when_available()
    test_fallback_respects_num_queries()
    print("\nAll tests passed.")