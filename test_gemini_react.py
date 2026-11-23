from dotenv import load_dotenv
import os, json, time
load_dotenv()
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
prompt = """You are an autonomous Tier-3 SOC cybersecurity ReAct triage agent investigating an incident.
Incident ID: BEN-000
Incident Alerts:
- Alert BEN-000-A00 (2026-03-14T10:00:00Z): Scheduled backup completed successfully on WKSTN-001, 1200 files, no egress
- Alert BEN-000-A01 (2026-03-14T10:05:00Z): Antivirus definitions auto-update finished on WKSTN-001 from internal mirror 10.0.0.5; signatures up-to-date.

History of actions taken so far:
No actions taken yet.

Available Tools:
1. "extract_iocs": {"text": "<alert text>"} -> extracts IP, domain, hash, process
2. "map_attack_technique": {"text": "<alert text>"} -> maps text to ATT&CK technique ID
3. "attribute_actor": {"technique_ids": ["T..."]} -> attributes threat actor
4. "search_cves": {"text": "<alert text>"} -> searches for CVEs and CVSS score
5. "build_attack_graph": {} -> builds provenance graph across alerts
6. "score_risk": {"cvss": 0.8, "actor": 0.6, "stage": 0.7, "graph": 0.4, "conf": 0.8} -> computes calibrated risk score
7. "recommend_mitigations": {"technique_ids": ["T..."]} -> retrieves ranked mitigations
8. "finish_triage": {"is_malicious": bool, "identified_techniques": ["T..."], "recommended_mitigations": ["M..."], "risk_level": "Low/Medium/High/Critical", "rationale": "..."} -> concludes triage

Select the next action as strict JSON:
{"thought": "<reasoning>", "action": "<tool_name>", "action_input": {...}}"""

t0 = time.perf_counter()
r = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt,
    config={"response_mime_type": "application/json", "temperature": 0.0}
)
lat = (time.perf_counter() - t0) * 1000.0
print(f"Latency ms: {lat:.2f}")
print("Response text:\n", r.text)
obj = json.loads(r.text)
print("Parsed action:", obj.get("action"))
print("Parsed input:", obj.get("action_input"))
if hasattr(r, "usage_metadata"):
    print("Tokens:", r.usage_metadata.prompt_token_count, r.usage_metadata.candidates_token_count)
