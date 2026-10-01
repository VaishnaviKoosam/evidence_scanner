"""
show_raw_output.py -- prints the EXACT, full, untruncated
ScannerResult dict for several hypotheses in a row. This is exactly
what gets handed to the Fusion Agent each time -- nothing truncated,
nothing prettified.

Run with: python show_raw_output_multi.py
"""
import json
from agents.evidence_scanner_agent import build_evidence_scanner

HYPOTHESES = ["unstable angina", "type 2 diabetes", "migraine", "community-acquired pneumonia"]

scanner = build_evidence_scanner()

for h in HYPOTHESES:
    result = scanner.invoke({"request": {"hypotheses": [{"diagnosis": h}]}, "top_n": 5})
    print(f"\n{'='*70}\nRAW OUTPUT FOR: {h}\n{'='*70}")
    print(json.dumps(result, indent=2, ensure_ascii=False))