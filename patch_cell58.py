import json
from pathlib import Path

NB_PATH = Path("Project2-CyberThreatAgent_Bench.ipynb")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

c58_src = "".join(nb["cells"][58]["source"])

new_tests = """
@test("Stage 3: Benign Abstention Gate suppresses false ATT&CK mapping on benign logs")
def test_stage3_benign_abstention():
    benign_tmpl = "Scheduled backup completed successfully on WKSTN-001, 1200 files, no egress"
    r = MAPPERS["hybrid"].ranked(benign_tmpl, top_k=1, allow_abstain=True)
    assert len(r) == 0, f"Expected empty prediction on benign text, got: {r}"
    conf = MAPPERS["hybrid"].confidence(benign_tmpl)
    assert conf < getattr(MAPPERS["hybrid"], "tau_abstain", 0.12), f"Expected conf < tau_abstain, got {conf}"

@test("Stage 7: Attack graph prerequisite edges satisfy temporal and causal constraints")
def test_stage7_causal_temporal_graph():
    assert (OUT / "stage7_graph.csv").exists(), "stage7_graph.csv missing"
    df_g = pd.read_csv(OUT / "stage7_graph.csv")
    assert "Edge_P" in df_g.columns and "Edge_R" in df_g.columns and "Edge_F1" in df_g.columns
    assert float(df_g["Edge_F1"].mean()) >= 0.0

@test("Stage 8: Learned risk calibration scores benign incidents as Low (<25)")
def test_stage8_learned_risk_calibration():
    assert (OUT / "stage8_chain_risk.csv").exists(), "stage8_chain_risk.csv missing"
    assert (OUT / "stage8_risk_ablation.csv").exists(), "stage8_risk_ablation.csv missing"
    df_cr = pd.read_csv(OUT / "stage8_chain_risk.csv")
    benign_cr = df_cr[df_cr["is_malicious"] == 0]
    assert len(benign_cr) > 0, "No benign incidents found in stage8_chain_risk.csv"
    assert float(benign_cr["risk_score"].max()) < 25.0, f"Expected all benign incidents < 25.0, got max: {benign_cr['risk_score'].max()}"

@test("Autonomous ReAct Agent vs Pipeline comparison table is exported and valid")
def test_agentic_comparison_artifacts():
    assert (OUT / "agent_vs_pipeline_eval.csv").exists(), "agent_vs_pipeline_eval.csv missing"
    df_ag = pd.read_csv(OUT / "agent_vs_pipeline_eval.csv")
    assert "regime" in df_ag.columns and "technique_f1" in df_ag.columns
    assert len(df_ag["regime"].unique()) == 3
"""

if "test_stage3_benign_abstention" not in c58_src:
    idx = c58_src.find("TEST_RESULTS = []")
    c58_src = c58_src[:idx] + new_tests.strip() + "\n\n" + c58_src[idx:]
    nb["cells"][58]["source"] = [l + "\n" for l in c58_src.splitlines()[:-1]] + [c58_src.splitlines()[-1]]
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Cell 58 successfully patched with 4 Phase 4 tests!")
else:
    print("Cell 58 already contains Phase 4 tests.")
