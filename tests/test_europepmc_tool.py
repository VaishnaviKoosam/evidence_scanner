"""
Offline test for europepmc_tool's JSON parsing -- doesn't require
network access, uses a captured real-shaped Europe PMC response.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.europepmc_tool import _to_evidence

SAMPLE_RESULT = {
    "id": "38765432",
    "source": "MED",
    "pmid": "38765432",
    "doi": "10.1016/j.ejim.2023.example",
    "title": "Risk stratification of acute chest pain in primary care: a guideline review",
    "authorString": "Patel R, Nguyen T, Osei K",
    "journalInfo": {"journal": {"title": "European Journal of Internal Medicine"}},
    "pubYear": "2023",
    "abstractText": "This guideline review synthesizes current risk stratification pathways for patients presenting with acute chest pain in primary care settings.",
    "pubTypeList": {"pubType": ["guideline", "review"]},
}


def test_to_evidence_basic_fields():
    ev = _to_evidence(SAMPLE_RESULT)
    assert ev.source == "EuropePMC"
    assert "Risk stratification" in ev.title
    assert ev.authors == ["Patel R", "Nguyen T", "Osei K"]
    assert ev.journal == "European Journal of Internal Medicine"
    assert ev.pub_date == "2023"
    assert "guideline" in ev.pub_types
    assert ev.doi == "10.1016/j.ejim.2023.example"
    assert ev.external_id == "38765432"
    assert ev.url == "https://europepmc.org/article/MED/38765432"
    print("test_to_evidence_basic_fields PASSED")


def test_to_evidence_handles_missing_fields():
    minimal = {"id": "1", "title": "Some title"}
    ev = _to_evidence(minimal)
    assert ev.title == "Some title"
    assert ev.abstract == ""
    assert ev.authors == []
    assert ev.pub_types == []
    print("test_to_evidence_handles_missing_fields PASSED")


if __name__ == "__main__":
    test_to_evidence_basic_fields()
    test_to_evidence_handles_missing_fields()
    print("\nAll tests passed.")