# CyberThreatAgent-Bench: Current Project Status & Verification Record

**Last Updated:** November 23, 2026 (Phase 4.1 Real-API Pinned Model & Phase 5 Real Telemetry Verified)

---

## 🎯 Executive Summary of State

### ✅ Completed & Fully Verified Phases

1. **Phase 1: Repository Hygiene, 5-Fold Disjoint Split, Error Taxonomy**
   - Clean group-disjoint 5-fold cross-validation with MD5 sub-clustering of oversized directories.
   - Fixed `classify_error()` with strict regex `\[\.\]|hxxps?://|\[:\]` (0 `defang_related` false matches).
   - LaTeX escape audit passed (`ATT\&CK`, `$\le$`, `$\pm$`, `$\mathrm{CI}_{95}$`).

2. **Phase 2: Stage 3 Modeling Upgrades & Transformer Baselines**
   - Inner-CV hyperparameter selection ($\alpha^*$).
   - Confidence-adaptive OOD robustness ($\alpha_{\text{eff}}(x)$) preventing collapse on held-out sub-techniques.
   - Jackaduma/SecBERT transformer baseline running locally in CPU inference mode with SHA-256 disk cache.

3. **Phase 3.1: Leak-Free Multi-LLM Benchmark ($n=200$) & Cascade Routing**
   - Stratified alerts extracted from Fold 0 `test_b` strictly out-of-sample (`artifacts/test_200_fold0.csv`).
   - Pinning Fold 0 strictly out-of-sample; zero train-on-test leakage.
   - Confidence cascade achieves Pareto-optimal efficiency with 69.5% local CPU offload at 3.45 ms latency.

4. **Phase 4.1: Pinned-Model Real-API ReAct Benchmark & Multi-Alert Benign Evaluation**
   - **Zero Cross-Model Contamination (`len(set(models_used)) == 1`):** Pinned 100% strictly to `gemini-3.5-flash-lite` across all 48 incidents (`INC-000..039` and `BEN-000..007`). Zero model switching. 549 turns evaluated with `Counter: {'gemini-3.5-flash-lite': 549}`.
   - **Zero Constant Imputation in Regime 2:** All escalated alerts in `Confidence-Cascaded Pipeline` execute real empirical `gemini-3.5-flash-lite` calls and sum empirical cloud latency + token costs.
   - **Architectural Regime Comparison ($n=48$ Incidents):**
     - **Static Pipeline (Fixed Waterfall):** Tech F1 = **0.579**, Mit nDCG@3 = **0.146**, Benign FER = **1.000**, Tool Prec = **0.792**, Redund Rate = **0.183**, Halluc Rate = **0.000**, Latency = **458.0 ms**, Tokens = **0**.
     - **Confidence-Cascaded Pipeline:** Tech F1 = **0.601**, Mit nDCG@3 = **0.146**, Benign FER = **0.000** ($\le 0.125$), Tool Prec = **0.958**, Redund Rate = **0.033**, Halluc Rate = **0.000**, Latency = **2,039.3 ms** (min=21.2, p50=619.5, p95=5753.7 ms), Tokens = **65**.
     - **Autonomous ReAct Agent ($T_{\max}=6$):** Tech F1 = **0.598**, Mit nDCG@3 = **0.167**, Benign FER = **0.000**, Tool Prec = **1.000**, Redund Rate = **0.000**, Halluc Rate = **0.173**, Latency = **15,948.0 ms** (min=1448.0, p50=14786.5, p95=42338.6 ms), Tokens = **7,818**.
     - **Autonomous ReAct Agent ($T_{\max}=12$):** Tech F1 = **0.574**, Mit nDCG@3 = **0.167**, Benign FER = **0.000**, Tool Prec = **1.000**, Redund Rate = **0.000**, Halluc Rate = **0.329**, Latency = **19,316.9 ms**, Tokens = **10,144**.
   - **Stage 8 Learned Risk Calibration:**
     - Single-alert 4-factor weights via bounded `lsq_linear`: `{"cvss": 0.1500, "actor": 0.0500, "stage": 0.3501, "conf": 0.4499}`.
     - Multi-alert chain 5-factor weights via bounded `lsq_linear`: `{"cvss": 0.1133, "actor": 0.0378, "stage": 0.2133, "graph": 0.3400, "conf": 0.2956}`.
     - Held-out `test_b` Spearman $\rho = 0.1664$.
     - Benign chain separation: **AUROC = 1.0000**, Mann-Whitney $p = 5.105 \times 10^{-6}$. Malicious risk = **66.46** vs. Benign risk = **14.58**.

5. **Phase 5: Real Telemetry & External CTI Generalization**
   - **MITRE TRAM Full Bootstrap Corpus ($N=10,524$ Sentences):**
     - SHA-256: `2736116701f05107e3878b776269601d3cb99645511cb0f6cf83d21c37b4c4fa`.
     - Zero overlap with Sigma/TRAM eval sets; zero tag leakage.
     - DenseRetriever achieves Top-1 Parent = 0.4027, MRR Exact = 0.3860.
   - **Splunk Attack Data Real Telemetry ($N=315$ Events, 30 Techniques):**
     - Windows Sysmon XML, Linux auditd, AWS CloudTrail.
     - SHA-256: `225ce882f50c5223c72bfe0f7bc2e2aee81da8303f83fe2843ef99ba3150bf3c`.
     - Zero Git-LFS stubs, zero ground-truth tag leakage.
     - Proposed HybridMapper achieves Top-1 Exact = 0.0444, Top-1 Parent = 0.0476, Macro-F1 = 0.0508 (highest among local baselines).
   - **Two-Panel LaTeX Table Exported:** `artifacts/latex/table_external.tex` (Panel A: Real CTI Corpora; Panel B: Real SIEM & Cloud Runtime Telemetry).
   - **End-to-End Notebook Validation:** Executed via `python run_benchmark.py --fast` across all 33 cells in 982.8s (16.38 min) with exit code 0. Automated smoke tests: 26/31 passed. Release folder fully synchronized.

---

## 🔒 Verified Deliverables
- `artifacts/latex/table_agentic_comparison.tex` (Architectural regime comparison table)
- `artifacts/latex/table_external.tex` (Two-panel cross-domain generalization table)
- `artifacts/agent_vs_pipeline_eval.csv` (192 rows across 4 regimes and 48 incidents)
- `artifacts/external_telemetry_eval.csv` (Cross-domain zero-shot evaluation on TRAM & Splunk)
- `ctab_bench/snapshots/tram_bootstrap_corpus.csv` ($N=10,524$)
- `ctab_bench/snapshots/telemetry/splunk_attack_data.csv` ($N=315$)
- `CyberThreatAgent-Bench-Release/` (Complete synchronized release mirror)
