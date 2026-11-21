# CyberThreatAgent-Bench: Autonomous Cybersecurity Agent Benchmark

[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/reproducibility-verified%205--seed-green.svg)](artifacts/manifest_v5.json)

Official reproducibility repository for **CyberThreatAgent-Bench**, an open-source, multi-stage benchmark for autonomous cybersecurity threat detection and investigation agents.

---

## 🏛 Architecture Overview

CyberThreatAgent-Bench evaluates agentic security pipelines across 9 modular, production-aligned operational stages:

```
[Raw Alerts] ──> Stage 1: Ingestion & Deduplication
                     │
                     ▼
                 Stage 2: IOC Extraction (Defang/Refang, IPv4/v6, Domains, Hashes)
                     │
                     ▼
                 Stage 3: Hierarchical ATT&CK Technique Mapping (Hybrid / Dense / HybridDenseRouter)
                     │
                     ▼
                 Stage 4: Threat Actor Attribution (Noisy Observational Profiles)
                     │
                     ▼
                 Stage 5: Kill-Chain Stage Identification
                     │
                     ▼
                 Stage 6: Controlled CVE Retrieval & Normalization
                     │
                     ▼
                 Stage 7: Autonomous Attack Graph Reconstruction (Hosts & Techniques)
                     │
                     ▼
                 Stage 8: Multi-Factor Incident Risk Scoring
                     │
                     ▼
                 Stage 9: Course-of-Action / Mitigation Recommendation (nDCG@3)
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone this repository and install the verified dependencies:

```bash
git clone https://github.com/Alvirwebd/CyberThreatAgent-Bench.git
cd CyberThreatAgent-Bench
pip install -r requirements.txt
```

*Verified on Python 3.10, 3.11, and 3.12.*

### 2. Benchmark Execution

Run the fast verification sanity check (2 seeds `[7, 21]`, alert sub-sampling to 400 items, and smoke test checks):
```bash
python run_benchmark.py --fast
```

Execute the full 5-seed publication benchmark (matches all manuscript tables and figures across seeds `[7, 21, 42, 84, 123]`):
```bash
python run_benchmark.py --full
```

Or open and run interactively:
```bash
jupyter notebook Project2-CyberThreatAgent_Bench.ipynb
```

---

## 📊 Pre-Generated Publication Artifacts

All benchmark outputs, pre-rendered publication figures, and LaTeX tables generated across the 5 evaluation seeds are stored in `artifacts/`:

- `artifacts/benchmark_summary.csv`: Aggregated 5-seed mean $\pm$ Student's $t$ $\text{CI}_{95}$ across all 9 pipeline stages.
- `artifacts/scorecard_raw_by_seed.csv`: Complete per-seed trial logs across all 5 seeds.
- `artifacts/scorecard_aggregated.csv`: Complete per-metric multi-seed statistics.
- `artifacts/agentic_routing.csv`: Uncertainty margin threshold sweep and escalation rates for `HybridDenseRouter`.
- `artifacts/hybrid_llm_routing.csv`: Sensitivity curve and Pareto frontier for `HybridLLMCascade`.
- `artifacts/external_validation.csv`: Real-world analyst-annotated CTI text evaluation ($n=163$).
- `artifacts/manifest_v5.json`: Cryptographic SHA-256 manifest of frozen inputs and outputs.
- `artifacts/datasheet.json`: Standardized machine-readable dataset documentation.
- `artifacts/outputs/latex/`: Formatted publication tables ready for direct `\input{}`:
  - `table_scorecard.tex`: Multi-stage performance scorecard.
  - `table_baselines.tex`: 5-seed baseline comparison table with statistical significance markers ($p < 0.01$).
  - `table_dataset.tex`: Dataset and corpus statistics.
  - `table_external.tex`: Real-world out-of-distribution evaluation on MITRE TRAM.
- `artifacts/outputs/figures/`: High-resolution (300 DPI) `.png` and vector `.pdf` figures (Pareto frontiers, leakage probes, noise decay curves, attack graphs).

---

## 📑 Baseline Performance Summary (Stage 3 ATT&CK Mapping)

Evaluated across $n=5$ independent seeds on group-disjoint splits:

| Method | Top-1 Exact | Top-1 Parent | Top-3 Exact | MRR Exact | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Guessing Floor | 0.001 $\pm$ 0.001 | 0.009 $\pm$ 0.006 | 0.003 $\pm$ 0.001 | 0.002 $\pm$ 0.001 | 0.000 $\pm$ 0.001 |
| Majority Class | 0.028 $\pm$ 0.020 | 0.037 $\pm$ 0.020 | 0.096 $\pm$ 0.024 | 0.067 $\pm$ 0.012 | 0.000 $\pm$ 0.000 |
| Rule-Based Keyword | 0.115 $\pm$ 0.046 | 0.159 $\pm$ 0.069 | 0.196 $\pm$ 0.050 | 0.159 $\pm$ 0.046 | 0.069 $\pm$ 0.034 |
| Okapi BM25 | 0.165 $\pm$ 0.051 | 0.232 $\pm$ 0.073 | 0.253 $\pm$ 0.039 | 0.213 $\pm$ 0.046 | 0.108 $\pm$ 0.042 |
| Zero-Shot TF-IDF | 0.226 $\pm$ 0.053 | 0.290 $\pm$ 0.080 | 0.321 $\pm$ 0.049 | 0.281 $\pm$ 0.046 | 0.137 $\pm$ 0.044 |
| Dense Retriever (`bge-small`) | 0.204 $\pm$ 0.036 | 0.311 $\pm$ 0.020 | 0.320 $\pm$ 0.019 | 0.267 $\pm$ 0.021 | 0.139 $\pm$ 0.039 |
| Supervised Dense (`bge-small`) | 0.199 $\pm$ 0.035 | **0.403 $\pm$ 0.030** | **0.444 $\pm$ 0.027** | 0.325 $\pm$ 0.028 | 0.165 $\pm$ 0.026 |
| **Proposed Hybrid Mapper** | **0.294 $\pm$ 0.042**$^{\dagger}$ | 0.376 $\pm$ 0.029 | 0.436 $\pm$ 0.029 | **0.366 $\pm$ 0.035**$^{\dagger}$ | **0.216 $\pm$ 0.033**$^{\dagger}$ |
| HybridDenseRouter | 0.276 $\pm$ 0.058 | 0.386 $\pm$ 0.056 | 0.435 $\pm$ 0.046 | 0.356 $\pm$ 0.053 | 0.199 $\pm$ 0.026 |
| LLM (`gemini-2.0-flash-lite`)* | 0.683 | 0.717 | -- | 0.683 | -- |

$^{\dagger}$ Statistically significant improvement over next-best baseline ($p < 0.01$, paired bootstrap & Wilcoxon signed-rank tests on $n=999$ unique held-out test items from the primary split).  
$^{*}$ Evaluated on $n=60$ test alerts with strict JSON schema at temperature 0 (unparseable predictions counted as incorrect).

---

### 🔀 Hybrid-to-LLM Cascade (Accuracy-Latency-Cost Pareto Trade-Off)

Evaluated on $n=60$ held-out test alerts with `HybridLLMCascade`: the fast local `HierarchicalMapper` predicts first, and only escalates to the frontier LLM when the top-2 prediction margin falls below uncertainty threshold $\theta$:

| Margin Threshold ($\theta$) | Escalation Rate | Top-1 Exact Accuracy (95% CI) | $p_{50}$ Latency | Mean Latency | Cost Savings vs. Full LLM | Operating Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0.05 | 20.00% | 0.3667 [0.2500, 0.4833] | 65.74 ms | 199.30 ms | 80.00% | Ultra-low latency edge filter |
| 0.10 | 33.33% | 0.4167 [0.2833, 0.5333] | 65.96 ms | 289.09 ms | 66.67% | Cost-conscious tier (66.7% cost saved) |
| 0.15 | 45.00% | 0.4500 [0.3167, 0.5667] | 67.33 ms | 369.73 ms | 55.00% | Balanced local-first tier |
| **0.20** | **63.33%** | **0.5333 [0.4000, 0.6667]** | **612.43 ms** | **481.16 ms** | **36.67%** | **Validation-Selected Operating Threshold ($\theta^*$)** |
| 0.30 | 81.67% | 0.6333 [0.5000, 0.7500] | 650.34 ms | 615.08 ms | 18.33% | High-accuracy tier |
| 0.40 | 96.67% | 0.6833 [0.5667, 0.8000] | 711.36 ms | 713.66 ms | 3.33% | Full cloud parity ($p=1.0$ vs standalone LLM) |

> **Key Pareto Observations:**
> - **Strict Out-of-Sample Threshold Selection:** The default operating point $\theta^* = 0.20$ was selected strictly on internal cross-validation folds of the training split (never tuned on the 60 test alerts), achieving $0.5333$ exact accuracy ($95\%\text{ CI}: [0.4000, 0.6667]$) with $36.67\%$ API query savings.
> - **Empirical Pareto Frontier:** Escalating only uncertain alerts (margin $< 0.40$) matches 100% standalone LLM accuracy ($0.6833$, paired McNemar/bootstrap $p = 1.0$) while offloading clear-cut procedural alerts to the deterministic local mapper.

---

## 🔬 Verified Artifacts & Methodology

- **Cryptographic Provenance:** [`artifacts/manifest_v5.json`](artifacts/manifest_v5.json) records SHA-256 digests for all inputs (MITRE ATT&CK 16.1, pinned Sigma rules) and outputs.
- **5-Seed Scorecard & Raw Trials:** [`artifacts/scorecard_raw_by_seed.csv`](artifacts/scorecard_raw_by_seed.csv) preserves trial-by-trial logs across seeds `[7, 21, 42, 84, 123]` with Student's $t$ $\text{CI}_{95}$ in [`artifacts/scorecard_aggregated.csv`](artifacts/scorecard_aggregated.csv).
- **Real-World CTI Evaluation:** [`artifacts/external_validation.csv`](artifacts/external_validation.csv) documents out-of-distribution evaluation on real-world analyst-annotated CTI text ($n=163$) from the MITRE TRAM corpus with zero rule-family overlap.
- **Statistical Significance Testing:** Paired bootstrap resampling ($B=2,000$) and Wilcoxon signed-rank tests confirm statistical significance ($p < 0.01$) on unique held-out test alerts from the primary split ($n=999$).
- **Regression Contract Suite:** 24 automated contract smoke tests (target-leakage prevention, schema normalization, score bounds $[0, 100]$, graph acyclicity, and zero-overlap TRAM isolation) pass deterministically.
- **Environment Verification:** Verified and tested across Python 3.10, 3.11, and 3.12 (see [`requirements.txt`](requirements.txt)).

---

## 📖 Citation

If you use CyberThreatAgent-Bench in your research or wish to reference this work, please cite:

```bibtex
@misc{shahriar2026cyberthreatagentbench,
  author       = {Shahriar, Alvir},
  title        = {CyberThreatAgent-Bench: A Multi-Stage Benchmark for Autonomous Threat Detection and ATT&CK Mapping Agents},
  year         = {2026},
  howpublished = {\url{https://github.com/Alvirwebd/CyberThreatAgent-Bench}},
  note         = {Open-source research benchmark and reproducibility repository}
}
```

---

## 📄 License
This repository is released under the [Apache 2.0 License](LICENSE).
