# Evidence Scanner integration contract

Verified against the source and project .venv on 2026-09-29.
Python in this environment: 3.13.2. The pinned direct dependencies require Python >=3.10.

## Dependencies

requirements.txt pins every directly imported external distribution:

```
requests==2.34.2
python-dotenv==1.2.3
langgraph==1.2.11
```

The import name dotenv belongs to python-dotenv. Standard-library modules and
local agents/tools packages are excluded. Transitive dependencies are installed
by pip; this is not a full environment lock. PDF-export packages installed in the
local environment are not imported by this application and are not required.

## Main entry point: plain diagnosis

Module: agents.evidence_scanner_agent
Exact existing signature:

```python
def scan_evidence(hypothesis: str, top_n: int = 5) -> ScannerResult:
```

- hypothesis: a nonblank diagnosis string. A serialized Text Clarifier JSON object
  string also reaches the shared parser, but a Python dictionary is rejected by
  this convenience function; use the graph interface for dictionaries.
- top_n: positive integer (bool is rejected), default 5.
- Builds a fresh graph and invokes it synchronously. Network requests block.
- Returns the ScannerResult dictionary described below.
- Invalid inputs raise ValueError. Source errors are logged and collection
  continues; unavailable LLM calls use deterministic fallbacks.

```python
from agents.evidence_scanner_agent import scan_evidence
result = scan_evidence("unstable angina", top_n=5)
```

## Main entry point: reusable LangGraph

Exact existing signature (currently no return annotation):

```python
def build_evidence_scanner():
```

Takes no arguments and returns a CompiledStateGraph, not an evidence result.
Building it makes no external API calls. Reuse the returned graph for invocations.

```python
from agents.evidence_scanner_agent import build_evidence_scanner
scanner = build_evidence_scanner()
result = scanner.invoke({
    "request": {
        "hypotheses": [{"diagnosis": "unstable angina", "confidence": 0.8}]
    },
    "top_n": 5,
})
```

request accepts a nonblank plain string, a Text Clarifier dictionary, or its
JSON object serialization. Exactly one hypotheses entry is supported. diagnosis
must be a nonblank string. confidence is optional and ignored. Original diagnosis
text, including surrounding whitespace, is preserved. Multiple diagnoses require
separate invocations. top_n defaults to 5 at runtime if omitted, although the
ScannerInput TypedDict currently declares it as required.

Parent graphs must map their state to request/top_n and place the result in the
appropriate downstream state field. This subgraph does not automatically read
arbitrary parent-state fields or invoke Clinical Data Fusion.

## Return contract for scan_evidence / scanner.invoke

```python
{
    "hypothesis": str,
    "broadened": bool,
    "evidence": [
        {
            "tier_label": str,
            "score": float,
            "title": str,
            "summary": str,
            "url": str | None,
            "source": str,
        }
    ],
}
```

The evidence list has zero to top_n entries, ordered by descending score.
Current sources are PubMed, EuropePMC, and SemanticScholar. Scores are study-type
ranking scores, not probabilities of diagnostic correctness. broadened indicates
that empty initial collection caused a retry with the unmodified hypothesis.
An empty list can mean no results or failed sources; it does not disprove the
hypothesis. Summary fallback usage and per-source errors are not returned in the
schema. Internal graph state and queries_used are excluded from the final output.

## Other public helpers (not full-agent entry points)

```python
def extract_hypothesis(request: str | dict) -> str:
```

In agents.evidence_scanner_agent. Validates/extracts one diagnosis, returning its
original string. No API calls; invalid inputs raise ValueError. Importing its
module still loads the wrapper dependencies and environment configuration.

```python
def gather_evidence(hypothesis: str, num_queries: int = 3,
                    max_results: int = 3) -> dict:
```

In tools.evidence_aggregator. Retrieval only; returns:

```python
{
    "evidence": list[Evidence],
    "broadened": bool,
    "queries_used": list[str],
}
```

Evidence is the shared dataclass in tools.pubmed_tool, with source, title,
abstract, authors, journal, pub_date, pub_types, doi, external_id, and url.
Semantic Scholar adds citation_count dynamically. These raw objects are not
JSON-ready dictionaries. This helper does not rank or summarize. max_results
limits each source per query, not the final combined count. It needs the same
literature services and query-generation configuration as the full agent, but
makes no summary calls.

## Services and configuration

No database, vector store, local LLM server, model file, or running web service
of your own is required.

| Service | Endpoint used by the code | Configuration |
| --- | --- | --- |
| Groq | https://api.groq.com/openai/v1/chat/completions | GROQ_API_KEY for model-backed queries/summaries; deterministic fallback without it or on failure |
| PubMed | https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi and efetch.fcgi | NCBI_API_KEY optional |
| Europe PMC | https://www.ebi.ac.uk/europepmc/webservices/rest/search | No API key in the connector |
| Semantic Scholar | https://api.semanticscholar.org/graph/v1/paper/search | SEMANTIC_SCHOLAR_API_KEY optional; source failures, including rate limits, are logged and skipped |

The Groq model string is hardcoded as llama-3.3-70b-versatile in the query generator
and summarizer; provider availability is external and was not tested for this
handoff. At least one literature service must return results to produce evidence.

## Portability / shared-pipeline considerations

- The aggregator loads evidence_scanner/.env using a path derived from its own
  file. The location is portable if the directory layout is retained; teammates
  must supply their own configuration. Exported environment variables also work.
- API-key values are cached in module globals when connectors/LLM modules are
  imported. Configure environment variables before importing the agent; changing
  the environment afterward does not refresh those globals.
- No D: drive or user-specific absolute path is hardcoded in the runtime agent.
- The agents.* and tools.* absolute imports assume the evidence_scanner directory
  is on sys.path. Run from that directory or add it to PYTHONPATH. Simply copying
  the directory under a different shared package namespace does not fix imports.
- Generic top-level agents/tools names can collide with other teammates' modules.
  Resolve the shared package layout when integrating; no import refactor was made.
- The local .venv is machine-specific. Recreate it from requirements.txt on each
  machine rather than copying it.
- Functions use synchronous requests. For an async parent pipeline, account for
  blocking calls when choosing the integration wrapper.
- No patient history or symptom list is accepted. Upstream confidence is ignored;
  the public contract evaluates literature for one diagnosis string.

Verification: all external import roots were enumerated with Python AST, exact
versions read with importlib.metadata, callable signatures inspected, and pip
check reported no broken requirements. No live API calls were made for this
integration inventory.
