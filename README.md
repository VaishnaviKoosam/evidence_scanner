# Evidence Scanner

Search PubMed, Europe PMC, and Semantic Scholar for a diagnostic hypothesis,
rank the literature by evidence strength, and summarize the top results.

## Install

Use Python 3.10 or newer. Run from the `evidence_scanner` repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`
instead. Literature searches require internet access.

## Environment variables

Copy `.env.example` to `.env` and set your keys before running or importing the
scanner. The scanner loads the root `.env`; exported environment variables
also work.

| Variable | Required? | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Required for LLM-backed query generation and summarization | Without it, queries use deterministic templates and summaries use abstract excerpts. |
| `NCBI_API_KEY` | Optional | Raises PubMed's rate limit from 3 to 10 requests/second. PubMed requests share a thread-safe limiter. |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Raises Semantic Scholar's rate limit for authenticated requests. |

## Run from the CLI

```powershell
python -m agents.evidence_scanner_agent "unstable angina" --top-n 5
```

## Run from Python

```python
import json
from agents.evidence_scanner_agent import build_evidence_scanner

request = {"hypotheses": [{"diagnosis": "unstable angina"}]}
scanner = build_evidence_scanner()
result = scanner.invoke({"request": request, "top_n": 5})
print(json.dumps(result, indent=2))
```

Supply one diagnosis per invocation. The result contains the hypothesis,
whether an empty search was broadened, and up to five ranked evidence entries
for this example. Scores describe evidence strength, not diagnostic confidence.

## Captured live example

Input:

```json
{"hypotheses": [{"diagnosis": "unstable angina"}]}
```

Output from the captured live run, reproduced verbatim below. New runs may
return different literature or summaries.

```json
{
  "hypothesis": "unstable angina",
  "broadened": false,
  "evidence": [
    {
      "tier_label": "Systematic Review / Meta-Analysis",
      "score": 100.0,
      "title": "Course of the effects of LDL-cholesterol reduction on cardiovascular risk over time: A meta-analysis of 60 randomized controlled trials.",
      "summary": "Individuals with or at high risk of cardiovascular disease (CVD) often receive long-term treatment with low-density lipoprotein cholesterol(LDL-C) lowering therapies, but whether the effects of LDL-C reduction remain stable over time is uncertain.",
      "url": "https://pubmed.ncbi.nlm.nih.gov/39126771/",
      "source": "PubMed"
    },
    {
      "tier_label": "Systematic Review / Meta-Analysis",
      "score": 100.0,
      "title": "The efficacy and safety of Xueshuantong (lyophilized) for injection in the treatment of unstable angina pectoris: A systematic review and meta-analysis.",
      "summary": "Objective: Xueshuantong (lyophilized) for injection (XST) isan effective botanical drug for treating unstable angina pectoris (UAP).",
      "url": "https://pubmed.ncbi.nlm.nih.gov/37089918/",
      "source": "PubMed"
    },
    {
      "tier_label": "Systematic Review / Meta-Analysis",
      "score": 100.0,
      "title": "Comparative effectiveness and safety of statins as a class andof specific statins for primary prevention of cardiovascular disease: A systematic review, meta-analysis, and network meta-analysis of randomized trials with94,283 participants.",
      "summary": "The current guidelines of statins for primary cardiovasculardisease (CVD) prevention were based on results from systematic reviews and meta-analyses that suffer from limitations.",
      "url": "https://pubmed.ncbi.nlm.nih.gov/30716508/",
      "source": "PubMed"
    },
    {
      "tier_label": "Systematic Review / Meta-Analysis",
      "score": 100.0,
      "title": "Effectiveness and safety of Breviscapine in the treatment of coronary heart disease: a meta-analysis of randomized controlled trials",
      "summary": "No abstract available for: Effectiveness and safety of Breviscapine in the treatment of coronary heart disease: a meta-analysis of randomized controlled trials",
      "url": "https://europepmc.org/article/PMC/PMC13612275",
      "source": "EuropePMC"
    },
    {
      "tier_label": "Systematic Review / Meta-Analysis",
      "score": 100.0,
      "title": "Cardiovascular disease and amyloidosis in phase III multiple myeloma trials: a systematic review of eligibility, screening and event reporting.",
      "summary": "<h4>Background</h4>Cardiovascular disease (CVD) and light chain (AL) amyloidosis are important causes of morbidity and mortality in patients with multiple myeloma (MM).",
      "url": "https://europepmc.org/article/MED/42680232",
      "source": "EuropePMC"
    }
  ]
}
```

See [INTEGRATION.md](INTEGRATION.md) for the full technical contract.
