"""
Offline test for semantic_scholar_tool's JSON parsing -- doesn't
require network access, uses a captured real-shaped S2 response.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.semantic_scholar_tool import _to_evidence

SAMPLE_PAPER = {
    "paperId": "abc123",
    "title": "Machine learning models for early sepsis prediction in ICU patients",
    "abstract": "We evaluate several machine learning approaches for early prediction of sepsis onset using ICU vital sign trends.",
    "authors": [{"name": "Chen L"}, {"name": "Ibrahim H"}],
    "venue": "Critical Care Medicine",
    "year": 2022,
    "publicationTypes": ["JournalArticle", "Review"],
    "externalIds": {"DOI": "10.1097/ccm.2022.example", "PubMed": "35123456"},
    "citationCount": 214,
    "url": "https://www.semanticscholar.org/paper/abc123",
}


def test_to_evidence_basic_fields():
    ev = _to_evidence(SAMPLE_PAPER)
    assert ev.source == "SemanticScholar"
    assert "sepsis prediction" in ev.title
    assert ev.authors == ["Chen L", "Ibrahim H"]
    assert ev.journal == "Critical Care Medicine"
    assert ev.pub_date == "2022"
    assert "Review" in ev.pub_types
    assert ev.doi == "10.1097/ccm.2022.example"
    assert ev.external_id == "35123456"  # prefers PubMed ID when available
    assert ev.citation_count == 214
    print("test_to_evidence_basic_fields PASSED")


def test_to_evidence_handles_missing_fields():
    minimal = {"paperId": "xyz789", "title": "Some title"}
    ev = _to_evidence(minimal)
    assert ev.title == "Some title"
    assert ev.abstract == ""
    assert ev.authors == []
    assert ev.pub_types == []
    assert ev.external_id == "xyz789"  # falls back to paperId when no PubMed ID
    assert ev.citation_count == 0
    print("test_to_evidence_handles_missing_fields PASSED")


if __name__ == "__main__":
    test_to_evidence_basic_fields()
    test_to_evidence_handles_missing_fields()
    print("\nAll tests passed.")