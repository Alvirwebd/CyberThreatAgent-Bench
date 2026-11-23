import json, time
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell

shell = InteractiveShell.instance()

with open('Project2-CyberThreatAgent_Bench.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Load notebook environment up to stage 9
code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
for i, c in enumerate(code_cells[:19]):
    src = ''.join(c.get('source', []))
    if 'class Config:' in src:
        src = src.replace('fast: bool = False', 'fast: bool = True')
    clean = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('%matplotlib'))
    res = shell.run_cell(clean)

from ctab_bench.agentic_react_live import call_remote_llm, REACT_CACHE_FILE

print("Testing 1 benign scenario and 1 malicious scenario with live ReAct agent...")

hybrid = shell.user_ns['MAPPERS']['hybrid']
extract_iocs = shell.user_ns['extract_iocs']
ATTR = shell.user_ns['ATTR']
detect_cves = shell.user_ns['detect_cves']
search_related_cves = shell.user_ns['search_related_cves']
build_attack_graph = shell.user_ns['build_attack_graph']
compute_risk_score = shell.user_ns['compute_risk_score']
CALIBRATED_WEIGHTS = shell.user_ns['CALIBRATED_WEIGHTS']
df_mitigations = shell.user_ns['df_mitigations']
mitigations_for_techniques = shell.user_ns['mitigations_for_techniques']

def execute_tool(action, action_input):
    if action == "extract_iocs":
        text = action_input.get("text", "")
        return extract_iocs(text)
    elif action == "map_attack_technique":
        text = action_input.get("text", "")
        top = hybrid.ranked(text, top_k=1, allow_abstain=True)
        return top[0] if top else {"attack_id": None, "name": "Abstain"}
    elif action == "attribute_threat_actor":
        tids = action_input.get("technique_ids", [])
        return ATTR.rank(tids, top_k=1)
    elif action == "retrieve_cves":
        text = action_input.get("text", "")
        cves = detect_cves(text)
        return search_related_cves(text, cves, max_results=1)
    elif action == "score_incident_risk":
        return compute_risk_score(action_input, weights=CALIBRATED_WEIGHTS)
    elif action == "recommend_mitigations":
        tids = action_input.get("technique_ids", [])
        return mitigations_for_techniques(tids, df_mitigations, top_k=3)
    return {"error": f"Unknown tool {action}"}

print("Tool execution engine ready.")
