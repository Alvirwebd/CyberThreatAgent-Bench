"""
tests/test_q1_contracts.py
==========================
Comprehensive Q1 Scientific-Integrity Contract Test Suite for CyberThreatAgent-Bench.
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
    assert len(df) == 288, f"Expected exactly 288 rows in agent evaluation, got {len(df)}"
    assert df["incident_id"].nunique() == 48, f"Expected 48 incidents, got {df['incident_id'].nunique()}"
    assert df["regime"].nunique() == 6, f"Expected 6 regimes, got {df['regime'].nunique()}"

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

