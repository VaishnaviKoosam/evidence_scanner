"""
Offline test for pubmed_tool's XML parsing -- doesn't require network
access, since it uses a captured real PubMed efetch response.
"""
import sys
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from unittest.mock import patch, call
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.pubmed_tool as pubmed
from tools.pubmed_tool import _parse_efetch_xml

SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>34567890</PMID>
      <Article>
        <Journal><Title>New England Journal of Medicine</Title></Journal>
        <ArticleTitle>Diagnostic accuracy of chest pain protocols in emergency departments</ArticleTitle>
        <Abstract>
          <AbstractText>Background: Chest pain is a common presenting complaint. Methods: We conducted a systematic review and meta-analysis of 42 studies. Results: Sensitivity was 94%.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
          <Author><LastName>Lee</LastName><ForeName>David</ForeName></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Meta-Analysis</PublicationType>
          <PublicationType>Systematic Review</PublicationType>
        </PublicationTypeList>
      </Article>
      <DateCompleted>
        <Year>2023</Year>
      </DateCompleted>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">34567890</ArticleId>
        <ArticleId IdType="doi">10.1056/nejm.2023.example</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_basic_fields():
    results = _parse_efetch_xml(SAMPLE_XML)
    assert len(results) == 1
    ev = results[0]
    assert ev.source == "PubMed"
    assert "chest pain protocols" in ev.title
    assert "meta-analysis" in ev.abstract.lower()
    assert ev.authors == ["Jane Smith", "David Lee"]
    assert ev.journal == "New England Journal of Medicine"
    assert "Meta-Analysis" in ev.pub_types
    assert ev.doi == "10.1056/nejm.2023.example"
    assert ev.external_id == "34567890"
    assert ev.url == "https://pubmed.ncbi.nlm.nih.gov/34567890/"
    print("test_parse_basic_fields PASSED")


def test_empty_xml_returns_empty_list():
    empty = "<PubmedArticleSet></PubmedArticleSet>"
    results = _parse_efetch_xml(empty)
    assert results == []
    print("test_empty_xml_returns_empty_list PASSED")



def http_response(status=200, ids=None, xml=None):
    response = requests.Response()
    response.status_code = status
    response._content = (
        xml if xml is not None else json.dumps(
            {"esearchresult": {"idlist": ids or []}}
        )
    ).encode()
    return response


def test_concurrent_searches_globally_space_all_http_requests():
    # Exercise both rate tiers and both endpoints, with real clock/sleeps
    # and mocked HTTP. Empty searches alone would miss efetch throttling.
    for key, interval in ((None, 0.34), ("offline-key", 0.11)):
        count = 3
        barrier = Barrier(count)
        timestamps = []
        endpoints = []
        record_lock = Lock()

        def mock_get(url, **kwargs):
            with record_lock:
                timestamps.append(time.monotonic())
                endpoints.append(url.rsplit("/", 1)[-1])
            if url.endswith("esearch.fcgi"):
                return http_response(ids=["34567890"])
            return http_response(xml=SAMPLE_XML)

        def search(index):
            barrier.wait(timeout=3)
            return pubmed.pubmed_search(f"query {index}", max_results=1)

        with patch.object(pubmed, "NCBI_API_KEY", key),              patch.object(pubmed, "_last_call_time", [0.0]),              patch.object(pubmed.requests, "get", side_effect=mock_get):
            with ThreadPoolExecutor(max_workers=count) as executor:
                results = list(executor.map(search, range(count)))
        assert all(len(result) == 1 for result in results)
        assert len(timestamps) == count * 2
        assert endpoints.count("esearch.fcgi") == count
        assert endpoints.count("efetch.fcgi") == count
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        assert all(gap >= interval for gap in gaps), gaps
        print(f"test_concurrent_searches_globally_space_all_http_requests PASSED "
              f"({count} threads, {len(timestamps)} requests, "
              f"minimum gap {min(gaps):.3f}s >= {interval}s)")


def test_429_retry_succeeds_for_both_endpoints():
    for failed_endpoint in ("esearch.fcgi", "efetch.fcgi"):
        responses = [http_response(ids=["34567890"]), http_response(xml=SAMPLE_XML)]
        responses.insert(0 if failed_endpoint == "esearch.fcgi" else 1, http_response(429))
        with patch.object(pubmed, "_rate_limit_delay") as delay,              patch.object(pubmed.requests, "get", side_effect=responses) as get,              patch.object(pubmed.time, "sleep") as sleep:
            result = pubmed.pubmed_search("angina")
        assert len(result) == 1
        assert get.call_count == delay.call_count == 3
        sleep.assert_called_once_with(1)
        retry_index = 0 if failed_endpoint == "esearch.fcgi" else 1
        assert get.call_args_list[retry_index] == get.call_args_list[retry_index + 1]
    print("test_429_retry_succeeds_for_both_endpoints PASSED")


def test_persistent_429_and_other_errors_propagate():
    for status in (429, 401, 500):
        with patch.object(pubmed, "_rate_limit_delay") as delay,              patch.object(pubmed.requests, "get", return_value=http_response(status)) as get,              patch.object(pubmed.time, "sleep") as sleep:
            try:
                pubmed.pubmed_search("angina")
            except requests.HTTPError as exc:
                assert exc.response.status_code == status
            else:
                raise AssertionError("HTTP error must reach existing source handling")
        assert get.call_count == delay.call_count == (3 if status == 429 else 1)
        assert sleep.call_args_list == ([call(1), call(2)] if status == 429 else [])
    print("test_persistent_429_and_other_errors_propagate PASSED")


if __name__ == "__main__":
    test_concurrent_searches_globally_space_all_http_requests()
    test_429_retry_succeeds_for_both_endpoints()
    test_persistent_429_and_other_errors_propagate()
    test_parse_basic_fields()
    test_empty_xml_returns_empty_list()
    print("\nAll tests passed.")