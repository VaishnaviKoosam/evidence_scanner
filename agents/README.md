# LangGraph Evidence-Based Scanner

Install dependencies from this directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Run the real literature search (requires internet; Groq uses `GROQ_API_KEY`
from the local `.env`, with deterministic fallbacks when unavailable):

```powershell
.venv\Scripts\python -m agents.evidence_scanner_agent "unstable angina" --top-n 5
```

The graph extracts the diagnosis, gathers evidence, ranks it, and summarizes
only the top N papers. Output contains `hypothesis`, `broadened`, and `evidence`;
each evidence entry contains `tier_label`, `score`, `title`, `summary`, `url`,
and `source`. Empty searches return an empty evidence list. Existing source
failure handling, raw-hypothesis retry, and LLM fallbacks remain in the tools.

Use the compiled graph as a node/subgraph in a larger LangGraph workflow:

```python
from agents.evidence_scanner_agent import build_evidence_scanner, scan_evidence

result = scan_evidence("unstable angina")
graph = build_evidence_scanner()
result = graph.invoke({
    "request": {"hypotheses": [{"diagnosis": "unstable angina", "confidence": 0.8}]},
    "top_n": 5,
})
```

`request` accepts plain diagnosis text, a dictionary, or a JSON object string.
Supply exactly one diagnosis per invocation; invoke separately for multiple
hypotheses. Upstream confidence does not affect literature ranking. The role
is to supply supporting or challenging literature to Clinical Data Fusion;
the graph does not itself confirm a diagnosis. Framework imports live only
in `agents/`; existing query generation and summarization provide the LLM calls.

Run the offline tests without API keys:

```powershell
.venv\Scripts\python -B tests/test_evidence_aggregator.py
.venv\Scripts\python -B tests/test_evidence_scanner_agent.py
```

Implementation follows the official LangGraph StateGraph API:
https://reference.langchain.com/python/langgraph/graph/state/StateGraph
