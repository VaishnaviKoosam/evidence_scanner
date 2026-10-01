"""
evidence_hierarchy.py
------------------------
Deterministic, rule-based classifier that scores each Evidence object
by where it sits on the clinical evidence hierarchy. No LLM, no API --
pure logic, which matters for a clinical system: the scoring has to be
auditable and reproducible, not a black box.

Hierarchy (strongest to weakest), per standard evidence-based-medicine
pyramids:
  1. Systematic Review / Meta-Analysis
  2. Randomized Controlled Trial (RCT)
  3. Clinical Practice Guideline
  4. Cohort / Observational Study
  5. Case Report / Case Series
  6. Unclassified (couldn't determine study type from metadata)

Each source labels pub_types differently:
  - PubMed:        "Meta-Analysis", "Randomized Controlled Trial", ...
  - Europe PMC:     "guideline", "review", "research-article", ...
  - Semantic Scholar: "JournalArticle", "Review", "ClinicalTrial", ...
So classification works by case-insensitive keyword matching against
whatever strings are in pub_types, rather than expecting an exact enum.

Citation count (Semantic Scholar only) is used as a *secondary*,
tie-breaking signal on top of the tier -- it never lets a case report
outrank a systematic review, but it can distinguish a highly-cited
cohort study from an obscure one within the same tier.
"""

import math
from dataclasses import dataclass
from tools.pubmed_tool import Evidence

# Ordered strongest -> weakest. Each tier has a base_score used for
# ranking and a set of keywords matched case-insensitively against
# the joined pub_types string.
TIERS = [
    {
        "tier": 1,
        "label": "Systematic Review / Meta-Analysis",
        "base_score": 100,
        "keywords": ["meta-analysis", "meta analysis", "systematic review"],
    },
    {
        "tier": 2,
        "label": "Randomized Controlled Trial",
        "base_score": 80,
        "keywords": ["randomized controlled trial", "clinicaltrial", "rct"],
    },
    {
        "tier": 3,
        "label": "Clinical Practice Guideline",
        "base_score": 65,
        "keywords": ["guideline", "practice guideline", "consensus"],
    },
    {
        "tier": 4,
        "label": "Cohort / Observational Study",
        "base_score": 45,
        "keywords": ["cohort", "observational study", "comparative study", "multicenter study"],
    },
    {
        "tier": 5,
        "label": "Case Report / Case Series",
        "base_score": 20,
        "keywords": ["case reports", "case report", "case series"],
    },
    # Generic review (not "systematic") is real but weaker evidence
    # than a primary study -- it's a summary of other work, so it
    # sits below cohort studies rather than being unclassified.
    {
        "tier": 4.5,
        "label": "Narrative Review",
        "base_score": 35,
        "keywords": ["review", "journalarticle"],
    },
]

UNCLASSIFIED = {
    "tier": 6,
    "label": "Unclassified",
    "base_score": 10,
}


@dataclass
class ScoredEvidence:
    evidence: Evidence
    tier: float
    tier_label: str
    score: float          # final score, tier base + citation bonus


def classify_evidence(evidence: Evidence) -> tuple[float, str]:
    """
    Determine which evidence tier an Evidence object belongs to.

    Priority order (highest confidence first):
      1. Specific pub_types metadata (e.g. "Case Reports", "Meta-Analysis")
         -- this is official, structured data, so it always wins if present.
      2. Specific title keywords -- ONLY consulted if pub_types had no
         specific signal at all (empty, or only generic tags like
         "Review"/"JournalArticle"). This exists because some sources
         (Semantic Scholar) have no "Guideline" category in their
         taxonomy, so a real guideline can only be recognized via title.
      3. Generic pub_types fallback ("review"/"journalarticle" ->
         Narrative Review) -- lowest confidence, used last.

    Without this strict ordering, a case report whose title happens to
    mention "guideline-recommended therapy" in passing would wrongly
    get promoted to Clinical Practice Guideline -- explicit metadata
    must never be overridden by a stray word in a title.

    Returns (tier_number, tier_label). Lower tier number = stronger evidence.
    """
    pub_types_text = " ".join(evidence.pub_types).lower() if evidence.pub_types else ""
    title_text = (evidence.title or "").lower()
    GENERIC_KEYWORDS = ("review", "journalarticle")

    # Pass 1: specific pub_types metadata -- authoritative, wins outright.
    for tier_def in TIERS:
        specific_keywords = [kw for kw in tier_def["keywords"] if kw not in GENERIC_KEYWORDS]
        if specific_keywords and any(kw in pub_types_text for kw in specific_keywords):
            return tier_def["tier"], tier_def["label"]

    # Pass 2: pub_types had no specific signal -- fall back to title,
    # which catches guidelines from sources with no "Guideline" tag.
    for tier_def in TIERS:
        specific_keywords = [kw for kw in tier_def["keywords"] if kw not in GENERIC_KEYWORDS]
        if specific_keywords and any(kw in title_text for kw in specific_keywords):
            return tier_def["tier"], tier_def["label"]

    # Pass 3: generic pub_types tags only ("Review"/"JournalArticle").
    if any(kw in pub_types_text for kw in GENERIC_KEYWORDS):
        for tier_def in TIERS:
            if any(kw in tier_def["keywords"] for kw in GENERIC_KEYWORDS):
                return tier_def["tier"], tier_def["label"]

    return UNCLASSIFIED["tier"], UNCLASSIFIED["label"]


def _citation_bonus(evidence: Evidence) -> float:
    """
    Secondary signal: citation count, log-scaled so a paper with 2000
    citations doesn't completely dwarf the tier-based score, it just
    nudges ranking within a tier. Only Semantic Scholar results carry
    this attribute; everything else contributes 0.
    """
    citation_count = getattr(evidence, "citation_count", 0) or 0
    if citation_count <= 0:
        return 0.0
    # log1p keeps this bounded and gentle: 10 citations -> ~2.4,
    # 1000 citations -> ~6.9. Capped so it can never cross a tier gap
    # (smallest gap between tiers above is 10 points).
    return min(math.log1p(citation_count), 9.0)


def score_evidence(evidence: Evidence) -> ScoredEvidence:
    """Classify + score a single Evidence object."""
    tier_def = next((t for t in TIERS if classify_evidence(evidence)[0] == t["tier"]), None)
    tier, label = classify_evidence(evidence)
    base_score = tier_def["base_score"] if tier_def else UNCLASSIFIED["base_score"]

    final_score = base_score + _citation_bonus(evidence)
    return ScoredEvidence(evidence=evidence, tier=tier, tier_label=label, score=final_score)


def rank_evidence(evidence_list: list[Evidence]) -> list[ScoredEvidence]:
    """
    Score and sort a mixed list of Evidence (from any/all sources),
    strongest evidence first. This is the function the Evidence-Based
    Scanner Agent will actually call once all sources are merged.
    """
    scored = [score_evidence(e) for e in evidence_list]
    return sorted(scored, key=lambda s: s.score, reverse=True)


if __name__ == "__main__":
    # Small manual demo with synthetic evidence -- no network needed.
    sample = [
        Evidence(source="PubMed", title="Case report of rare angina presentation",
                 abstract="", pub_types=["Case Reports"]),
        Evidence(source="EuropePMC", title="ESC guideline on chest pain triage",
                 abstract="", pub_types=["guideline"]),
        Evidence(source="PubMed", title="Meta-analysis of troponin sensitivity",
                 abstract="", pub_types=["Meta-Analysis"]),
    ]
    for scored in rank_evidence(sample):
        print(f"[{scored.tier_label}] score={scored.score:.1f} -- {scored.evidence.title}")