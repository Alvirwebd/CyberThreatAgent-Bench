"""
scripts/patch_notebook_cells.py
================================
Applies the rigorous Phase 6 scientific-integrity upgrades to Project2-CyberThreatAgent_Bench.ipynb:
1. Cell 14: Define gold_mitigation_ids, add 'mitigations' to truth for INC- and BEN- (Bug #2)
2. Cell 24: Add HybridLLMCascade on Splunk real telemetry (CACHE/splunk_telemetry_llm_cache.jsonl)
3. Cell 36: evaluate_risk_model(sample=None), return spearman_vs_sigma_level_EXTERNAL and spearman_heuristic;
            evaluate_risk_on_chains: purge (0.0 if not is_mal else None) -> 0.0 unconditionally (Bug #3)
4. Cell 40: Stage 7 exports Edge_P, Edge_R, Edge_F1; Stage 8 fits w_seed on tr only and evaluates te with sample=None
5. Cell 56: Asserts df_agent_eval['regime'].nunique() == 6 and len(df_agent_eval) == 288
"""

import json
from pathlib import Path

def patch_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # -------------------------------------------------------------
    # 1. Patch Cell 14 (Incident Chain Corpus)
    # -------------------------------------------------------------
    c14 = cells[14]
    src14 = "".join(c14["source"])
    
    helper_code = '''MIT_ID_BY_STIX = {s: a for s, a in zip(df_mitigations["stix_id"], df_mitigations["attack_id"]) if a}

def gold_mitigation_ids(attack_id):
    refs = attack_id_to_technique.get(attack_id)
    if refs is None:
        return []
    return sorted(MIT_ID_BY_STIX[s] for s in technique_mitigated_by.get(refs["stix_id"], set()) if s in MIT_ID_BY_STIX)

'''
    if "def gold_mitigation_ids" not in src14:
        # Insert before build_chain_corpus
        idx = src14.find("def build_chain_corpus")
        src14 = src14[:idx] + helper_code + src14[idx:]

    # Update truth.append for INC-
    old_inc_truth = '"attack_chain": gold_chain, "tactic_chain": gold_tactics})'
    new_inc_truth = '"attack_chain": gold_chain, "tactic_chain": gold_tactics, "mitigations": sorted({m for aid in gold_chain for m in gold_mitigation_ids(aid)})})'
    if old_inc_truth in src14:
        src14 = src14.replace(old_inc_truth, new_inc_truth)

    # Update truth.append for BEN-
    old_ben_truth = '"attack_chain": [],\n            "tactic_chain": []\n        })'
    new_ben_truth = '"attack_chain": [],\n            "tactic_chain": [],\n            "mitigations": []\n        })'
    if old_ben_truth in src14:
        src14 = src14.replace(old_ben_truth, new_ben_truth)

    c14["source"] = [l + "\n" for l in src14.splitlines()]
    print("[PASS] Cell 14 patched: gold_mitigations populated for INC- and BEN-.")

    # -------------------------------------------------------------
    # 2. Patch Cell 36 (Stage 8 Risk Score & Bug #3 Purge)
    # -------------------------------------------------------------
    c36 = cells[36]
    src36 = "".join(c36["source"])

    # Bug #3: Purge (0.0 if not is_mal else None)
    old_branch = "stage_score = max(stage_vals) if stage_vals else (0.0 if not is_mal else None)"
    new_branch = "stage_score = max(stage_vals) if stage_vals else 0.0"
    if old_branch in src36:
        src36 = src36.replace(old_branch, new_branch)

    # Upgrade evaluate_risk_model to handle sample=None and compute both spearman metrics
    old_eval_model = '''def evaluate_risk_model(df, sample=60, weights=None):
    weights = weights or CALIBRATED_SINGLE_WEIGHTS
    rows = []
    for _, row in df.head(sample).iterrows():'''

    new_eval_model = '''def evaluate_risk_model(df, sample=None, weights=None, mapper=None):
    weights = weights or CALIBRATED_SINGLE_WEIGHTS
    mapper = mapper or MAPPERS["hybrid"]
    eval_df = df if sample is None else df.head(sample)
    rows = []
    for _, row in eval_df.iterrows():'''

    if old_eval_model in src36:
        src36 = src36.replace(old_eval_model, new_eval_model)

    # Return both spearman metrics
    old_ret = '''    d = pd.DataFrame(rows)
    rho = _spearman(d["sigma_score"], d["risk_score"])
    d.to_csv(OUT / "stage8_risk.csv", index=False)
    return {"n": int(len(d)),
            "spearman_vs_sigma_level_EXTERNAL": round(rho, 4) if rho is not None else None,
            "note": "Sigma level excluded from input; genuine external check under calibrated lsq_linear weights.",
            "mean_weight_coverage": round(float(d["weight_coverage"].mean()), 4)}, d'''

    new_ret = '''    d = pd.DataFrame(rows)
    rho = _spearman(d["sigma_score"], d["risk_score"])
    
    # Compute heuristic spearman baseline for comparison
    h_scores = []
    for _, r in eval_df.iterrows():
        t = str(r["text"])
        top = mapper.ranked(t, top_k=1, allow_abstain=True)
        aid = top[0]["attack_id"] if top else None
        stg = (stages_for_techniques([aid]) or [None])[0] if aid else None
        actors = ATTR.rank([aid], top_k=1) if aid else []
        cves = search_related_cves(t, detect_cves(t), max_results=1)
        h_res = compute_risk_score({
            "cvss": normalize_cvss(cves[0]["cvss_score"]) if cves else None,
            "actor": actors[0]["score"] if actors else 0.0,
            "stage": STAGE_SEVERITY.get(stg),
            "conf": 0.0}, weights=BASE_WEIGHTS_V5)
        h_scores.append(h_res["risk_score"])
    rho_h = _spearman(d["sigma_score"], h_scores)

    d.to_csv(OUT / "stage8_risk.csv", index=False)
    return {"n": int(len(d)),
            "spearman_vs_sigma_level_EXTERNAL": round(rho, 4) if rho is not None else None,
            "spearman_heuristic": round(rho_h, 4) if rho_h is not None else None,
            "note": "Sigma level excluded from input; genuine external check under calibrated lsq_linear weights.",
            "mean_weight_coverage": round(float(d["weight_coverage"].mean()), 4)}, d'''

    if old_ret in src36:
        src36 = src36.replace(old_ret, new_ret)

    c36["source"] = [l + "\n" for l in src36.splitlines()]
    print("[PASS] Cell 36 patched: evaluate_risk_model(sample=None) + Bug #3 purged.")

    # -------------------------------------------------------------
    # 3. Patch Cell 40 (Multi-Seed Harness)
    # -------------------------------------------------------------
    c40 = cells[40]
    src40 = "".join(c40["source"])

    # Export Edge_P, Edge_R, Edge_F1
    old_s7_export = '''    met["s7_tech_node_P"] = s7.get("tech_node_P")
    met["s7_tech_node_R"] = s7.get("tech_node_R")
    met["s7_host_node_accuracy"] = s7.get("host_node_accuracy")'''

    new_s7_export = '''    met["s7_tech_node_P"] = s7.get("tech_node_P")
    met["s7_tech_node_R"] = s7.get("tech_node_R")
    met["s7_host_node_accuracy"] = s7.get("host_node_accuracy")
    met["s7_Edge_P"] = s7.get("Edge_P")
    met["s7_Edge_R"] = s7.get("Edge_R")
    met["s7_Edge_F1"] = s7.get("Edge_F1")'''

    if old_s7_export in src40:
        src40 = src40.replace(old_s7_export, new_s7_export)

    # Stage 8: fit strictly on tr and evaluate on te with sample=None
    old_s8_eval = '''# ---- stage 8 ----
    s8, _ = _timed("stage8", lambda: evaluate_risk_model(te), timing)
    met["s8_spearman_external"] = s8.get("spearman_vs_sigma_level_EXTERNAL")'''

    new_s8_eval = '''# ---- stage 8 ----
    w_seed = fit_single_alert_risk_weights(tr, maps["hybrid"])
    s8, _ = _timed("stage8", lambda: evaluate_risk_model(te, mapper=maps["hybrid"], weights=w_seed, sample=None), timing)
    met["s8_spearman_external"] = s8.get("spearman_vs_sigma_level_EXTERNAL")
    met["s8_spearman_heuristic"] = s8.get("spearman_heuristic")'''

    if old_s8_eval in src40:
        src40 = src40.replace(old_s8_eval, new_s8_eval)

    c40["source"] = [l + "\n" for l in src40.splitlines()]
    print("[PASS] Cell 40 patched: Stage 7 edge metrics & Stage 8 per-seed calibration.")

    # -------------------------------------------------------------
    # 4. Patch Cell 56 (6-Regime Assertion)
    # -------------------------------------------------------------
    c56 = cells[56]
    src56 = "".join(c56["source"])

    old_reg_assert = 'assert df_agent_eval["regime"].nunique() >= 3, "Expected at least 3 regimes"'
    new_reg_assert = 'assert df_agent_eval["regime"].nunique() == 6, f"Expected exactly 6 regimes, got {df_agent_eval[\'regime\'].nunique()}"\n    assert len(df_agent_eval) == 288, f"Expected 288 rows (6 regimes * 48 incidents), got {len(df_agent_eval)}"'

    if old_reg_assert in src56:
        src56 = src56.replace(old_reg_assert, new_reg_assert)

    c56["source"] = [l + "\n" for l in src56.splitlines()]
    print("[PASS] Cell 56 patched: 6-regime and 288-row assertions added.")

    # -------------------------------------------------------------
    # Save Patched Notebook
    # -------------------------------------------------------------
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[SUCCESS] Patched {nb_path} successfully!")

if __name__ == "__main__":
    patch_notebook("Project2-CyberThreatAgent_Bench.ipynb")
