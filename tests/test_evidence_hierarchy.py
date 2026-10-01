"""
Offline tests for evidence_hierarchy.py -- pure logic, no network
or API needed at all, since this whole layer is deterministic.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.pubmed_tool import Evidence
from tools.evidence_hierarchy import classify_evidence, score_evidence, rank_evidence


def test_classify_meta_analysis_as_tier1():
    ev = Evidence(source="PubMed", title="t", abstract="", pub_types=["Meta-Analysis"])
    tier, label = classify_evidence(ev)
    assert tier == 1
    assert "Systematic Review" in label
    print("test_classify_meta_analysis_as_tier1 PASSED")


def test_classify_guideline_lowercase_europepmc_style():
    ev = Evidence(source="EuropePMC", title="t", abstract="", pub_types=["guideline"])
    tier, label = classify_evidence(ev)
    assert tier == 3
    assert label == "Clinical Practice Guideline"
    print("test_classify_guideline_lowercase_europepmc_style PASSED")


def test_classify_case_report_as_weak_evidence():
    ev = Evidence(source="PubMed", title="t", abstract="", pub_types=["Case Reports"])
    tier, label = classify_evidence(ev)
    assert tier == 5
    print("test_classify_case_report_as_weak_evidence PASSED")


def test_classify_missing_pub_types_as_unclassified():
    ev = Evidence(source="PubMed", title="t", abstract="", pub_types=[])
    tier, label = classify_evidence(ev)
    assert tier == 6
    assert label == "Unclassified"
    print("test_classify_missing_pub_types_as_unclassified PASSED")


def test_citation_count_never_beats_a_higher_tier():
    # A cohort study with a huge citation count still shouldn't beat
    # a meta-analysis with zero citations -- tier gaps are designed
    # to be bigger than the max citation bonus.
    huge_citations_cohort = Evidence(
        source="SemanticScholar", title="t", abstract="", pub_types=["cohort"]
    )
    huge_citations_cohort.citation_count = 50000
    zero_citation_meta = Evidence(
        source="PubMed", title="t", abstract="", pub_types=["Meta-Analysis"]
    )
    scored_cohort = score_evidence(huge_citations_cohort)
    scored_meta = score_evidence(zero_citation_meta)
    assert scored_meta.score > scored_cohort.score
    print("test_citation_count_never_beats_a_higher_tier PASSED")


def test_citation_count_breaks_ties_within_same_tier():
    low_cite = Evidence(source="SemanticScholar", title="low", abstract="", pub_types=["cohort"])
    low_cite.citation_count = 2
    high_cite = Evidence(source="SemanticScholar", title="high", abstract="", pub_types=["cohort"])
    high_cite.citation_count = 500
    scored_low = score_evidence(low_cite)
    scored_high = score_evidence(high_cite)
    assert scored_high.score > scored_low.score
    print("test_citation_count_breaks_ties_within_same_tier PASSED")


def test_rank_evidence_sorts_strongest_first():
    sample = [
        Evidence(source="PubMed", title="case report", abstract="", pub_types=["Case Reports"]),
        Evidence(source="EuropePMC", title="guideline", abstract="", pub_types=["guideline"]),
        Evidence(source="PubMed", title="meta-analysis", abstract="", pub_types=["Meta-Analysis"]),
    ]
    ranked = rank_evidence(sample)
    titles_in_order = [r.evidence.title for r in ranked]
    assert titles_in_order == ["meta-analysis", "guideline", "case report"]
    print("test_rank_evidence_sorts_strongest_first PASSED")


if __name__ == "__main__":
    test_classify_meta_analysis_as_tier1()
    test_classify_guideline_lowercase_europepmc_style()
    test_classify_case_report_as_weak_evidence()
    test_classify_missing_pub_types_as_unclassified()
    test_citation_count_never_beats_a_higher_tier()
    test_citation_count_breaks_ties_within_same_tier()
    test_rank_evidence_sorts_strongest_first()
    print("\nAll tests passed.")