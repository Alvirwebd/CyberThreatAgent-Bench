from dotenv import load_dotenv
import os, json, time, requests
load_dotenv()

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
8. "finish_triage": {"is_malicious": bool, "identified_techniques": ["T..."], "recommended_mitigations": ["M..."], "risk_level": "Low|Medium|High|Critical", "rationale": "..."} -> concludes triage

Select the next action as strict JSON:
{"thought": "<reasoning>", "action": "<tool_name>", "action_input": {...}}"""

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}", "Content-Type": "application/json"}
payload = {
    "model": "qwen/qwen3.8-27b",
    "messages": [
        {"role": "system", "content": "You are a cybersecurity SOC triage assistant. Respond strictly via a valid JSON object."},
        {"role": "user", "content": prompt}
    ],
    "response_format": {"type": "json_object"},
    "temperature": 0.0
}

t0 = time.perf_counter()
res = requests.post(url, headers=headers, json=payload, timeout=15)
lat = (time.perf_counter() - t0) * 1000.0
print("Status:", res.status_code, "Latency ms:", lat)
if res.status_code == 200:
    data = res.json()
    content = data["choices"][0]["message"]["content"]
    print("Content:\n", content)
    obj = json.loads(content)
    print("Parsed Action:", obj.get("action"))
    print("Usage:", data.get("usage"))
else:
    print("Error:", res.text)
