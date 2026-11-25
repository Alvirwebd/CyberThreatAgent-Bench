"""
tests/test_benchmark_contracts.py
=================================
Comprehensive Scientific Contract Test Suite for CyberThreatAgent-Bench.
Verifies determinism, zero-leakage, dataset validity, routing contracts, and agent regimes.
"""

import os
import json
import re
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import networkx as nx

import ctab
from ctab.normalizer import ingest_alerts, normalize_alert
from ctab.splitting import deterministic_split
from ctab.cascade import MarginGatedRouter
from ctab.graph import build_attack_graph, compute_graph_metrics
from ctab.risk import compute_risk_score, fit_chain_risk_weights

ROOT = Path(__file__).resolve().parent.parent

def test_package_metadata():
    assert ctab.__version__ == "1.0.0"

def test_alert_normalizer_contracts():
    raw_alerts = [
        {"alert_id": "A1", "timestamp": "2026-03-14T10:00:00Z", "raw_text": "Phishing attachment opened from hxxp://evil[.]com"},
        {"alert_id": "A2", "timestamp": "invalid_date", "raw_text": "Normal user login"},
        {"alert_id": "A3", "timestamp": "2026-03-14T10:05:00Z", "raw_text": ""},
        {"alert_id": "A1", "timestamp": "2026-03-14T10:00:00Z", "raw_text": "Phishing attachment opened from hxxp://evil[.]com"},
    ]
    accepted, summary = ingest_alerts(raw_alerts)
    assert summary["rows_in"] == 4
    assert summary["rows_accepted"] == 1
    assert summary["duplicates_removed"] == 1
    assert summary["rows_rejected"] == 2
    assert accepted[0]["raw_text_clean"] == "Phishing attachment opened from http://evil.com"

def test_deterministic_split_zero_leakage():
    data = [{"sample_id": f"S{i}", "text": f"Telemetry alert event description {i}"} for i in range(100)]
    df = pd.DataFrame(data)
    train_df, test_df = deterministic_split(df, test_size=0.30, seed=42)

    assert len(train_df) + len(test_df) == 100
    assert len(test_df) > 20
    # Zero overlap
    train_texts = set(train_df["text"])
    test_texts = set(test_df["text"])
    assert len(train_texts & test_texts) == 0

def test_margin_gated_router_contracts():
    router = MarginGatedRouter(margin_threshold=0.10, tau_abstain=0.12)

    # 1. Low confidence -> abstain
    res_abstain = router.route([{"attack_id": "T1059", "score": 0.05}], confidence_score=0.08)
    assert res_abstain["decision"] == "abstain"

    # 2. Ambiguous margin -> escalate to cloud LLM
    res_esc = router.route([
        {"attack_id": "T1059.001", "score": 0.52},
        {"attack_id": "T1059.003", "score": 0.48}
    ], confidence_score=0.75)
    assert res_esc["decision"] == "escalate"
    assert res_esc["candidate"] == "T1059.001"

    # 3. Decisive margin -> accept local
    res_local = router.route([
        {"attack_id": "T1059.001", "score": 0.85},
        {"attack_id": "T1059.003", "score": 0.30}
    ], confidence_score=0.85)
    assert res_local["decision"] == "accept_local"
    assert res_local["candidate"] == "T1059.001"

def test_stage8_risk_bounded_weights():
    np.random.seed(42)
    X = np.random.uniform(0.1, 0.9, size=(50, 5))
    y = np.random.uniform(0.1, 0.9, size=50)

    weights = fit_chain_risk_weights(X, y, bounds=(0.05, 0.45))
    assert len(weights) == 5
    assert abs(sum(weights.values()) - 1.0) < 1e-3
    for k, v in weights.items():
        assert 0.03 <= v <= 0.50, f"Weight {k}={v} out of bounds"

    # Test risk score monotonicity
    low_res = compute_risk_score({"cvss": 0.1, "actor": 0.1, "stage": 0.1, "graph": 0.1, "conf": 0.1}, weights=weights)
    high_res = compute_risk_score({"cvss": 0.9, "actor": 0.9, "stage": 0.9, "graph": 0.9, "conf": 0.9}, weights=weights)
    assert high_res["risk_score"] > low_res["risk_score"]

def test_attack_graph_construction_and_metrics():
    alerts = [
        {"pred_attack_id": "T1566.001", "gt_host": "WKSTN-101", "timestamp": "2026-03-14T10:00:00Z"},
        {"pred_attack_id": "T1059.001", "gt_host": "WKSTN-101", "timestamp": "2026-03-14T10:03:00Z"},
        {"pred_attack_id": "T1003.001", "gt_host": "WKSTN-101", "timestamp": "2026-03-14T10:06:00Z"},
    ]
    G = build_attack_graph(alerts, incident_id="INC-001")
    assert isinstance(G, nx.DiGraph)
    assert "WKSTN-101" in G.nodes
    assert len(G.nodes) == 4 # 1 host + 3 techniques

    metrics = compute_graph_metrics(G, gold_techniques=["T1566.001", "T1059.001", "T1003.001"], gold_host="WKSTN-101")
    assert metrics["tech_node_P"] == 1.0
    assert metrics["tech_node_R"] == 1.0
    assert metrics["host_node_accuracy"] == 1.0
    assert metrics["Edge_P"] == 1.0
    assert metrics["Edge_R"] == 1.0

def test_splunk_telemetry_dataset_integrity():
    csv_path = ROOT / "ctab_bench" / "snapshots" / "telemetry" / "splunk_attack_data.csv"
    assert csv_path.exists(), f"Missing telemetry file: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) >= 300, f"Expected >= 300 telemetry events, got {len(df)}"
    assert df["attack_id"].nunique() >= 25, f"Expected >= 25 techniques, got {df['attack_id'].nunique()}"

    # Verify zero short rows (< 80 chars)
    short_rows = (df["raw_text"].astype(str).str.len() < 80).sum()
    assert short_rows == 0, f"Found {short_rows} telemetry rows with length < 80 chars"

    # Verify zero leaked T1xxx tags in raw text
    tag_pattern = re.compile(r"\bT1\d{3}(?:\.\d{3})?\b")
    leaked_count = sum(1 for t in df["raw_text"].astype(str) if tag_pattern.search(t))
    assert leaked_count == 0, f"Found {leaked_count} rows with leaked ATT&CK technique IDs in raw telemetry"

def test_tram_bootstrap_corpus_integrity():
    csv_path = ROOT / "ctab_bench" / "snapshots" / "tram_bootstrap_corpus.csv"
    assert csv_path.exists(), f"Missing TRAM bootstrap corpus: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) >= 10000, f"Expected >= 10,000 sentences, got {len(df)}"
    assert df["attack_id"].nunique() >= 40, f"Expected >= 40 techniques, got {df['attack_id'].nunique()}"

def test_agentic_eval_and_manifest_contracts():
    csv_path = ROOT / "artifacts" / "agent_vs_pipeline_eval.csv"
    assert csv_path.exists(), f"Missing agent evaluation CSV: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) in (288, 384), f"Expected 288 or 384 rows in agent evaluation, got {len(df)}"
    assert df["incident_id"].nunique() == 48, f"Expected 48 incidents, got {df['incident_id'].nunique()}"
    assert df["regime"].nunique() in (6, 8), f"Expected 6 or 8 regimes, got {df['regime'].nunique()}"
    assert (df["latency_ms"] == 2500.0).sum() == 0, "Found synthetic fallback latencies (2500.0 ms)!"

    # Verify model purity per regime
    for reg, grp in df.groupby("regime"):
        assert grp["model"].nunique() == 1, f"Regime {reg} has mixed models: {grp['model'].unique()}"

    # Check manifest publication contracts
    manifest_path = ROOT / "artifacts" / "manifest_v5.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        cfg = manifest.get("config", {})
        dstats = manifest.get("dataset_statistics", {})
        if not cfg.get("fast", True):
            samples = dstats.get("samples") or cfg.get("samples")
            assert samples == 2530, f"Expected samples=2530 in manifest, got {samples}"

def test_scorecard_aggregated_no_nans():
    csv_path = ROOT / "artifacts" / "scorecard_aggregated.csv"
    assert csv_path.exists(), f"Missing scorecard_aggregated.csv: {csv_path}"
    df = pd.read_csv(csv_path)
    nan_count = df.isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaNs in scorecard_aggregated.csv!"

def test_latex_mitre_attack_escaped():
    latex_dir = ROOT / "artifacts" / "latex"
    if latex_dir.exists():
        for tex_file in latex_dir.glob("*.tex"):
            content = tex_file.read_text(encoding="utf-8")
            # Raw ATT&CK not preceded by a backslash
            unescaped = re.findall(r"(?<!\\)ATT&CK", content)
            assert len(unescaped) == 0, f"Found unescaped ATT&CK in {tex_file.name}: {len(unescaped)} occurrences"

def test_no_q1_labels_in_documentation():
    for doc_name in ["README.md", "docs/RESEARCH_CLAIMS_AND_VALIDITY.md"]:
        doc_path = ROOT / doc_name
        if doc_path.exists():
            content = doc_path.read_text(encoding="utf-8")
            q1_matches = re.findall(r"\b[Q]1[- ](?:grade|Scientific Integrity|grade integrity)?\b", content, re.IGNORECASE)
            assert len(q1_matches) == 0, f"Found legacy tier strings in {doc_name}: {q1_matches}"

def test_hierarchical_mapper_distinct_predictions_and_seed7_reproduction():
    """
    Verifies that HierarchicalMapper:
    1. Returns distinct predictions on distinct alert texts (zero hardcoded stubs).
    2. Reproduces the exact seed-7 publication benchmark score (Top1_Exact == 0.4128).
    """
    from ctab import load_attack, get_canonical_benchmark, group_split, HierarchicalMapper, evaluate_mapping

    df_techniques, _, _, _ = load_attack()
    bench_df = get_canonical_benchmark()

    # 1. Distinct predictions check (zero hardcoded stubs)
    sample_train = bench_df.iloc[:200]
    sample_mapper = HierarchicalMapper(df_techniques, sample_train, alpha=0.65, C=8.0, auto_tune_alpha=False)

    p1 = sample_mapper.ranked("PowerShell EncodedCommand downloading payload from external C2", top_k=1)[0]["attack_id"]
    p2 = sample_mapper.ranked("Dump lsass process memory with procdump", top_k=1)[0]["attack_id"]
    p3 = sample_mapper.ranked("Scheduled task creation via schtasks.exe daily persistence", top_k=1)[0]["attack_id"]

    distinct_preds = {p1, p2, p3}
    assert len(distinct_preds) >= 2, f"HierarchicalMapper returned constant stub predictions: {distinct_preds}"

    # 2. Seed-7 exact publication score reproduction
    tr, te, _ = group_split(bench_df, seeds=(7, 21, 42, 84, 123), seed=7)
    mapper_s7 = HierarchicalMapper(df_techniques, tr, alpha=0.65, C=8.0, auto_tune_alpha=True)
    fn = lambda text, top_k=5: mapper_s7.ranked(text, top_k=top_k)
    met, _ = evaluate_mapping(te, fn)

    assert met["Top1_Exact"] == pytest.approx(0.4128, abs=0.01), \
        f"Expected seed-7 Top1_Exact ~0.4128, got {met['Top1_Exact']}"




# =============================================================================
# Bug-5 Hardening Contracts (Audit-Required)
# =============================================================================
import hashlib as _hashlib

def test_table_dataset_has_correct_n():
    """table_dataset.tex must report 2,530 not the stale 400."""
    tex_path = ROOT / "artifacts" / "latex" / "table_dataset.tex"
    if not tex_path.exists():
        pytest.skip("table_dataset.tex not yet generated")
    tex = tex_path.read_text(encoding="utf-8")
    assert "2,530" in tex or "2530" in tex, "table_dataset.tex must contain 2,530"
    assert " 400" not in tex, "table_dataset.tex must not contain stale N=400"


def test_table_baselines_and_readme_have_correct_scores():
    """table_baselines.tex and README must report ~0.389 (not stale 0.255)."""
    sc = pd.read_csv(ROOT / "artifacts" / "scorecard_aggregated.csv").set_index("metric")
    expected = round(float(sc.loc["s3_hybrid_Top1_Exact", "mean"]), 3)
    assert abs(expected - 0.389) < 0.005, f"scorecard s3_hybrid_Top1_Exact={expected}, expected ~0.389"
    for f_rel in ["artifacts/latex/table_baselines.tex", "README.md"]:
        f_path = ROOT / f_rel
        if not f_path.exists():
            continue
        content = f_path.read_text(encoding="utf-8")
        fmt_val = f"{expected:.3f}"
        assert fmt_val in content or "0.389" in content, \
            f"{f_rel} must contain {fmt_val} (s3_hybrid_Top1_Exact), not stale 0.255"


def test_no_unescaped_underscores_in_texttt():
    """No unescaped underscores inside texttt blocks in any .tex file."""
    latex_dir = ROOT / "artifacts" / "latex"
    if not latex_dir.exists():
        pytest.skip("artifacts/latex not yet generated")
    for tex in latex_dir.glob("*.tex"):
        content = tex.read_text(encoding="utf-8")
        matches = re.findall(r"\\texttt\{[^}]*(?<!\\)_[^}]*\}", content)
        assert len(matches) == 0, \
            f"Unescaped underscore in texttt in {tex.name}: {matches}"


def test_grid_summary_latency_not_stale():
    """agent_architecture_grid_summary.csv must have 8 rows and no stale 27367.1 / Benign_FER=1.0 cascade row."""
    sum_path = ROOT / "artifacts" / "agent_architecture_grid_summary.csv"
    if not sum_path.exists():
        pytest.skip("agent_architecture_grid_summary.csv not yet generated")
    df = pd.read_csv(sum_path)
    assert len(df) == 8, f"Expected 8 regime rows, got {len(df)}"
    assert (df["Latency_ms"] < 25000).all(), \
        f"Stale pre-revision latency found: {df[df['Latency_ms'] >= 25000][['Architecture','Latency_ms']].to_dict()}"
    gem_casc = df[df["Regime"] == "Confidence-Cascaded Neuro-Symbolic (gemini-3.5-flash-lite)"]
    assert not gem_casc.empty and float(gem_casc["Benign_FER"].iloc[0]) == 0.0, \
        f"Stale Benign_FER on Confidence-Cascaded (gemini-3.5-flash-lite): {gem_casc.to_dict()}"


def test_per_stage_mitigation_non_degenerate():
    """per_stage_vs_global_escalation.csv per_stage_malicious_mit_ndcg3 must be < 0.99 (non-flat)."""
    csv_path = ROOT / "artifacts" / "per_stage_vs_global_escalation.csv"
    if not csv_path.exists():
        pytest.skip("per_stage_vs_global_escalation.csv not yet generated")
    df = pd.read_csv(csv_path)
    assert "per_stage_malicious_mit_ndcg3" in df.columns
    assert (df["per_stage_malicious_mit_ndcg3"] < 0.99).all(), \
        f"Per-Stage mitigation nDCG@3 is flat 1.0: {df['per_stage_malicious_mit_ndcg3'].tolist()}"


def test_valid_mitigations_all_m1xxx():
    """get_mitigation_engine().valid_mitigations must contain only M1xxx IDs."""
    from ctab.mitigation import get_mitigation_engine
    eng = get_mitigation_engine()
    non_m = [m for m in eng.valid_mitigations if not m.startswith("M")]
    assert len(non_m) == 0, f"Non-M1xxx IDs in valid_mitigations ({len(non_m)}): {non_m[:5]}"
    assert len(eng.valid_mitigations) >= 40, \
        f"Expected >=40 M1xxx mitigations, got {len(eng.valid_mitigations)}"


def test_manifest_sha256_matches_files():
    """manifest_v5.json artifact_sha256 must match actual on-disk hashes (LF-normalised)."""
    manifest_path = ROOT / "artifacts" / "manifest_v5.json"
    if not manifest_path.exists():
        pytest.skip("artifacts/manifest_v5.json not yet generated")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    hashes = manifest.get("artifact_sha256", {})
    mismatches = []
    for rel, expected_hash in hashes.items():
        fp = ROOT / "artifacts" / rel
        if fp.exists():
            data = fp.read_bytes()
            if fp.suffix.lower() in {".csv", ".tex", ".json", ".md", ".txt"}:
                data = data.replace(b"\r\n", b"\n")
            actual = _hashlib.sha256(data).hexdigest()
            if actual != expected_hash:
                mismatches.append(f"{rel}: expected {expected_hash[:8]}, got {actual[:8]}")
    assert len(mismatches) == 0, "SHA256 mismatches in manifest_v5.json:\n" + "\n".join(mismatches)


def test_no_q1_pattern_anywhere():
    """Word-boundary Q1 must not appear in README or docs."""
    for path_str in ["README.md", "docs/RESEARCH_CLAIMS_AND_VALIDITY.md"]:
        fp = ROOT / path_str
        if not fp.exists():
            continue
        content = fp.read_text(encoding="utf-8")
        # [Q]1 so this file itself does not cause grep to match
        matches = re.findall(r"\b[Q]1\b", content)
        assert len(matches) == 0, \
            f"Found {len(matches)} 'Q1' in {path_str}: {matches[:5]}"


def test_no_legacy_root_scratch_files():
    """Root directory must be free of TASK_STATUS.md and legacy scratch scripts."""
    patterns = [
        "patch_*.py", "test_*.py", "TASK_STATUS.md", "agentic_benchmark.py",
        "build_*.py", "check_*.py", "extract_*.py", "generate_*.py",
        "print_*.py", "update_*.py"
    ]
    found = []
    for pat in patterns:
        found.extend([p.name for p in ROOT.glob(pat)])
    assert found == [], f"Legacy root scratch files still present: {found}"


def test_no_stale_reviewer_strings():
    """Zero occurrences of stale/overclaimed terms across docs, README, LaTeX, and scripts."""
    # Character-class / split patterns so this test file never self-matches
    bad_patterns = [
        (r"Qwen 2\.5", "Qwen 2.5"),
        (r"super-additive", "super-additive"),
        (r"n=999", "n=999"),
        (r"10,370", "10,370"),
        (r"24 automated contract", "24 automated contract"),
        (r"CyberThreatAgent-Bench-Release", "CyberThreatAgent-Bench-Release"),
    ]
    targets = [ROOT / "README.md", ROOT / "docs" / "RESEARCH_CLAIMS_AND_VALIDITY.md"]
    if (ROOT / "artifacts" / "latex").exists():
        targets.extend((ROOT / "artifacts" / "latex").glob("*.tex"))
    hits = []
    for fp in targets:
        if not fp.exists():
            continue
        txt = fp.read_text(encoding="utf-8")
        for pat, label in bad_patterns:
            if re.search(pat, txt, re.IGNORECASE):
                hits.append(f"{fp.name}: {label}")
    assert hits == [], f"Stale reviewer strings found: {hits}"


def test_readme_and_tex_match_grid_summary():
    """README.md and table_agentic_comparison.tex must contain all 8 regimes matching CSV to 3dp."""
    sum_path = ROOT / "artifacts" / "agent_architecture_grid_summary.csv"
    if not sum_path.exists():
        pytest.skip("agent_architecture_grid_summary.csv not yet generated")
    gdf = pd.read_csv(sum_path)
    assert len(gdf) == 8
    readme_txt = (ROOT / "README.md").read_text(encoding="utf-8")
    tex_txt = (ROOT / "artifacts" / "latex" / "table_agentic_comparison.tex").read_text(encoding="utf-8")
    for _, r in gdf.iterrows():
        f1_s = f"{r['Tech_F1']:.3f}"
        ndcg_s = f"{r['Mit_nDCG3']:.3f}"
        assert f1_s in readme_txt, f"README missing Tech_F1={f1_s} for {r['Regime']}"
        assert f1_s in tex_txt, f"table_agentic_comparison.tex missing Tech_F1={f1_s} for {r['Regime']}"
        assert ndcg_s in readme_txt, f"README missing Mit_nDCG3={ndcg_s} for {r['Regime']}"
        assert ndcg_s in tex_txt, f"table_agentic_comparison.tex missing Mit_nDCG3={ndcg_s} for {r['Regime']}"


def test_benign_calibration_includes_agents():
    """benign_calibration_120.csv must include autonomous agent rows alongside pipeline baselines."""
    cal_path = ROOT / "artifacts" / "benign_calibration_120.csv"
    if not cal_path.exists():
        pytest.skip("benign_calibration_120.csv not yet generated")
    df = pd.read_csv(cal_path)
    assert len(df) >= 4, f"Expected >= 4 rows (pipeline + autonomous agents), got {len(df)}"
    arches = " ".join(df["Architecture"].astype(str).tolist())
    assert "Plan-and-Execute" in arches or "ReAct" in arches, \
        f"benign_calibration_120.csv missing autonomous agent rows: {df['Architecture'].tolist()}"
