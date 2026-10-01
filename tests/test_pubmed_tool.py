"""
Offline test for pubmed_tool's XML parsing -- doesn't require network
access, since it uses a captured real PubMed efetch response.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


if __name__ == "__main__":
    test_parse_basic_fields()
    test_empty_xml_returns_empty_list()
    print("\nAll tests passed.")