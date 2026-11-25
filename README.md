# CyberThreatAgent-Bench: Autonomous Cybersecurity Agent Benchmark

[![Python 3.10 – 3.14](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue.svg)](https://www.python.org/downloads/)
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
Clone this repository and install the verified package and dependencies:

```bash
git clone https://github.com/Alvirwebd/CyberThreatAgent-Bench.git
cd CyberThreatAgent-Bench
pip install -e .
pip install -r requirements.txt
```

*Verified on Python 3.10 – 3.14 (NumPy 2.4, Pandas 3.0, Scikit-Learn 1.9).*

### 2. Contract Test Suite (Reproducibility & Scientific Integrity)
Verify all 26 architectural, data integrity, and reproducibility contracts:
```bash
pytest tests/test_benchmark_contracts.py -v
```

### 3. Benchmark Execution

Run the fast verification sanity check (2 seeds `[7, 21]`, alert sub-sampling to 400 items, and smoke test checks):
```bash
python run_benchmark.py --fast
```

Execute the full 5-seed publication benchmark (matches all manuscript tables and figures across seeds `[7, 21, 42, 84, 123]`):
```bash
python run_benchmark.py --full
```

Run the 8-cell architecture x backbone evaluation across 48 incidents:
```bash
python scripts/run_agentic_8cell_grid.py
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
- `artifacts/latex/`: Formatted publication tables ready for direct `\input{}`:
  - `table_scorecard.tex`: Multi-stage performance scorecard.
  - `table_baselines.tex`: 5-seed baseline comparison table with statistical significance markers ($p < 0.01$).
  - `table_dataset.tex`: Dataset and corpus statistics.
  - `table_external.tex`: Real-world out-of-distribution evaluation on MITRE TRAM.
- `artifacts/figures/`: High-resolution (300 DPI) `.png` and vector `.pdf` figures (Pareto frontiers, leakage probes, noise decay curves, attack graphs).

---

## 📑 Baseline Performance Summary (Stage 3 ATT&CK Mapping)

Evaluated across $K=5$ independent seeds on group-disjoint splits with both strict exact-match and multi-label (Top-1 Any Valid Tag) scoring across dual hardware regimes:

| Method | Top-1 Exact | Top-1 Any Valid Tag | Top-3 Exact | MRR Exact | Macro-F1 | Hardware Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Local CPU Inference ($N=2,530$)** | | | | | | *Zero Cloud Dependency* |
| Random Guessing Floor | 0.002 $\pm$ 0.003 | 0.003 $\pm$ 0.002 | 0.004 $\pm$ 0.003 | 0.003 $\pm$ 0.002 | 0.001 $\pm$ 0.001 | Local CPU (0.02 ms) |
| Majority Class | 0.033 $\pm$ 0.009 | 0.042 $\pm$ 0.009 | 0.125 $\pm$ 0.023 | 0.080 $\pm$ 0.017 | 0.000 $\pm$ 0.000 | Local CPU (0.01 ms) |
| Rule-Based Keyword | 0.128 $\pm$ 0.032 | 0.163 $\pm$ 0.043 | 0.214 $\pm$ 0.040 | 0.174 $\pm$ 0.034 | 0.081 $\pm$ 0.030 | Local CPU (0.35 ms) |
| Okapi BM25 | 0.186 $\pm$ 0.048 | 0.220 $\pm$ 0.046 | 0.270 $\pm$ 0.052 | 0.233 $\pm$ 0.048 | 0.124 $\pm$ 0.053 | Local CPU (1.85 ms) |
| Zero-Shot TF-IDF | 0.246 $\pm$ 0.047 | 0.297 $\pm$ 0.057 | 0.338 $\pm$ 0.049 | 0.299 $\pm$ 0.045 | 0.159 $\pm$ 0.054 | Local CPU (1.25 ms) |
| Dense Retriever (`bge-small`) | 0.225 $\pm$ 0.030 | 0.251 $\pm$ 0.030 | 0.332 $\pm$ 0.044 | 0.283 $\pm$ 0.031 | 0.160 $\pm$ 0.046 | Local CPU (12.80 ms) |
| Supervised Dense (`bge-small`) | 0.236 $\pm$ 0.052 | 0.258 $\pm$ 0.056 | 0.513 $\pm$ 0.042 | 0.379 $\pm$ 0.043 | 0.190 $\pm$ 0.016 | Local CPU (13.50 ms) |
| SecBERT Bi-Encoder (`SecBERT`) | 0.046 $\pm$ 0.021 | 0.057 $\pm$ 0.018 | 0.075 $\pm$ 0.032 | 0.062 $\pm$ 0.026 | 0.049 $\pm$ 0.027 | Local CPU (11.50 ms) |
| SecBERT Hybrid Mapper | 0.162 $\pm$ 0.026 | 0.182 $\pm$ 0.029 | 0.327 $\pm$ 0.045 | 0.252 $\pm$ 0.034 | 0.123 $\pm$ 0.021 | Local CPU (13.20 ms) |
| **Proposed Hybrid Mapper** | **0.389 $\pm$ 0.071** | **0.425 $\pm$ 0.066** | **0.507 $\pm$ 0.083** | **0.451 $\pm$ 0.074** | **0.242 $\pm$ 0.064** | **Local CPU (3.45 ms)** |
| **Agentic Router** | **0.382 $\pm$ 0.062**$^{\dagger}$ | **0.416 $\pm$ 0.059** | **0.532 $\pm$ 0.056** | **0.459 $\pm$ 0.059**$^{\dagger}$ | **0.254 $\pm$ 0.051** | **Local CPU (4.10 ms)** |
| **B. Multi-LLM Scaling ($n=200$, Held-Out Fold 0)** | | | | | | *Cloud API / Groq LPU* |
| Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | **0.405** | **0.460** | **0.555** | **0.492** | -- | Google Cloud API (804.7 ms) |
| GPT-OSS 120B (`openai/gpt-oss-120b`) | 0.325 | 0.350 | 0.510 | 0.414 | -- | Groq LPU (1039.0 ms) |
| Qwen 3.8 27B (`qwen/qwen3.8-27b`) | 0.240 | 0.265 | 0.370 | 0.302 | -- | Groq LPU (342.4 ms) |
| GPT-OSS 20B (`openai/gpt-oss-20b`) | 0.140 | 0.145 | 0.180 | 0.163 | -- | Groq LPU (1110.3 ms) |

$^{\dagger}$ Statistically significant improvement over next-best local baseline ($p < 0.01$, paired bootstrap $B=2000$, Wilcoxon signed-rank, and McNemar tests on non-overlapping out-of-fold test predictions across 5 folds).  
$^{*}$ Multi-LLM scaling evaluated on the identical held-out test alerts ($n=200$, Fold 0 strictly out-of-sample) with ranked Top-5 schema and strict JSON parsing at temperature 0 (reasoning tokens stripped; unparseable predictions penalized as incorrect). Latency reported as median ($p_{50}$) per-alert inference time. Note: For 1-to-1 comparison on this identical $n=200$ test subset, the Proposed Hybrid Mapper achieves **0.425** Top-1 Exact and **0.492** MRR@5 with 3.45 ms local CPU latency.

#### Stage 3 Modeling Upgrades:
- **Inner-CV Hyperparameter Selection ($\alpha^*$):** Precomputed score matrices allow fast 3-fold inner cross-validation strictly on training splits to select the optimal hybrid blend weight $\alpha^* \in [0.2, 0.7]$ without data leakage.
- **Confidence-Adaptive Open-Set OOD Robustness:** Dynamic query weighting $\alpha_{\text{eff}}(x) = \alpha^* \cdot \text{clip}(c_{\text{sup}}(x) / \tau, 0.05, 1.0)$ shifts weight to zero-shot semantic matching whenever supervised margin or query-to-train similarity drops. Under predicted parents, child zero-shot affinities are preserved so unseen sub-techniques are never suppressed by training-only sub-classifiers.
- **Cybersecurity Transformer Baselines (SecBERT):** Domain-specific pre-trained embeddings (jackaduma/SecBERT) with SHA-256 disk caching, batch size 64, and PyTorch inference mode enable CPU execution in seconds.

---

## 🤖 8-Cell Architecture $\times$ Backbone Grid Evaluation (Table \ref{tab:agentic_comparison})

CyberThreatAgent-Bench comprehensively evaluates 8 architectural regime $\times$ backbone cells across $n=48$ standardized, multi-turn incident scenarios (40 multi-stage malicious intrusion campaigns `INC-000..039` and 8 benign enterprise activity sequences `BEN-000..007`). All metrics derive from empirical local CPU execution, deterministic confidence-thresholded escalation, and live HTTPS tool execution on pinned models with zero model-switching, zero synthetic tokens, and zero hardcoded latency values:

| Architecture Regime | Backbone Model | Tech F1 | Mit nDCG@3 | Benign FER (n=8) | Tool Prec | Redund Rate | Halluc Rate | Latency (ms) | Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Static 9-Stage Waterfall (Local CPU)** | `local_cpu` | 0.384 | 0.767 | 1.000 | 0.842 | 0.158 | 0.000 | 696.3 | 0 |
| **2. Confidence-Cascaded Neuro-Symbolic (gemini-3.5-flash-lite)** | `gemini-3.5-flash-lite` | 0.496 | 0.818 | 0.000 | 0.860 | 0.012 | 0.000 | 21,198.5 | 400 |
| **3. Confidence-Cascaded Neuro-Symbolic (qwen/qwen3.8-27b)** | `qwen/qwen3.8-27b` | 0.448 | 0.818 | 0.000 | 0.872 | 0.000 | 0.005 | 10,311.9 | 405 |
| **4. Plan-and-Execute Agent (gemini-3.5-flash-lite)** | `gemini-3.5-flash-lite` | 0.526 | 0.722 | 0.000 | 0.955 | 0.045 | 0.104 | 5,916.1 | 2,721 |
| **5. Plan-and-Execute Agent (qwen/qwen3.8-27b)** | `qwen/qwen3.8-27b` | 0.520 | 0.152 | 0.000 | 1.000 | 0.000 | 0.142 | 2,382.0 | 1,659 |
| **6. Autonomous ReAct Agent (gemini-3.5-flash-lite, T_max=6)** | `gemini-3.5-flash-lite` | 0.530 | 0.478 | 0.000 | 1.000 | 0.000 | 0.000 | 13,712.2 | 7,924 |
| **7. Autonomous ReAct Agent (gemini-3.5-flash-lite, T_max=12)** | `gemini-3.5-flash-lite` | 0.519 | 0.711 | 0.000 | 1.000 | 0.000 | 0.008 | 16,734.7 | 10,232 |
| **8. Autonomous ReAct Agent (qwen/qwen3.8-27b, T_max=6)** | `qwen/qwen3.8-27b` | 0.375 | 0.400 | 0.000 | 1.000 | 0.000 | 0.000 | 2,921.8 | 4,329 |

> **Key Takeaways & Scientific Findings:**
> - **Benign False Escalation:** Static waterfall pipelines over-escalate on benign enterprise noise (`Benign FER = 1.000`), whereas autonomous agents and confidence-gated cascades reach `0/8` (`Benign FER (n=8) = 0.000`, 95% Wilson CI `[0.00, 0.32]`; see `artifacts/benign_calibration_120.csv` for tight `N=120` bounds `[0.000, 0.031]`).
> - **Technique Identification:** `Autonomous ReAct Agent (gemini-3.5-flash-lite, T_max=6)` achieves the highest Technique F1 (**0.530**).
> - **Strict STIX-Only Mitigation:** Under authentic MITRE ATT&CK STIX `M1xxx` graded `nDCG@3` (zero playbook padding), `Confidence-Cascaded Neuro-Symbolic (gemini-3.5-flash-lite)` achieves **0.818**.

---

### 🔀 Hybrid-to-LLM Cascade (Accuracy-Latency-Cost Pareto Trade-Off)

Evaluated on $n=200$ held-out test alerts with `HybridLLMCascade` paired with champion LLM (`gemini-3.5-flash-lite`): the fast local `HierarchicalMapper` (Fold 0 out-of-sample) predicts first, and only escalates to the champion LLM when the top-2 prediction margin falls below uncertainty threshold $\theta$:

| Margin Threshold ($\theta$) | Escalation Rate | Top-1 Exact Accuracy (95% CI) | $p_{50}$ Latency | Mean Latency | Cost Savings vs. Full LLM | Operating Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0.05 | 18.50% | 0.4300 [0.3650, 0.5000] | 3.45 ms | 182.30 ms | 81.50% | Ultra-low latency edge filter |
| **0.10** | **30.50%** | **0.4350 [0.3650, 0.5100]** | **3.45 ms** | **288.44 ms** | **69.50%** | **Validation-Selected Operating Threshold ($\theta^*$)** |
| 0.15 | 40.50% | 0.4350 [0.3700, 0.5050] | 3.45 ms | 375.30 ms | 59.50% | Balanced local-first tier |
| 0.20 | 51.00% | 0.4150 [0.3500, 0.4850] | 624.54 ms | 461.04 ms | 49.00% | Intermediate escalation tier |
| 0.30 | 68.50% | 0.4100 [0.3450, 0.4850] | 747.38 ms | 610.40 ms | 31.50% | High escalation tier |
| 0.40 | 79.50% | 0.4000 [0.3350, 0.4700] | 773.43 ms | 707.65 ms | 20.50% | Cloud-heavy tier |

> **Key Pareto Observations:**
> - **Mathematical Oracle Union Ceiling:** Local Hybrid standalone achieves $0.4250$ ($85/200$) and Champion LLM standalone achieves $0.4050$ ($81/200$), establishing a theoretical oracle union ceiling of $0.5600$ ($112/200$). All cascade operating points strictly satisfy $\text{Acc}(\theta) \le \text{Oracle}_{\text{union}} = 0.5600$ with zero train-on-test leakage.
> - **Cost-Latency Efficiency at Non-Inferior Accuracy:** At validation-selected $\theta^* = 0.10$, `HybridLLMCascade` matches standalone accuracy ($0.4350$ vs. $0.4250$ local hybrid / $0.4050$ cloud LLM; McNemar $p = 0.43$, non-inferior) while **eliminating $69.5\%$ of cloud LLM calls and cutting median latency $10.6\times$ ($804.7\text{ ms} \to 75.9\text{ ms}$)**.

---

## 🔬 Verified Artifacts & Methodology

- **Cryptographic Provenance:** [`artifacts/manifest_v5.json`](artifacts/manifest_v5.json) records SHA-256 digests for all inputs (MITRE ATT&CK 16.1, pinned Sigma rules) and outputs.
- **5-Seed Scorecard & Raw Trials:** [`artifacts/scorecard_raw_by_seed.csv`](artifacts/scorecard_raw_by_seed.csv) preserves trial-by-trial logs across seeds `[7, 21, 42, 84, 123]` with Student's $t$ $\text{CI}_{95}$ in [`artifacts/scorecard_aggregated.csv`](artifacts/scorecard_aggregated.csv).
- **Real-World CTI Evaluation:** [`artifacts/external_validation.csv`](artifacts/external_validation.csv) documents out-of-distribution evaluation on real-world analyst-annotated CTI text ($n=163$) from the MITRE TRAM corpus with zero rule-family overlap.
- **Statistical Significance Testing:** Paired bootstrap resampling ($B=2,000$) and Wilcoxon signed-rank tests confirm statistical significance ($p < 0.01$) on unique held-out test alerts across 5 group-disjoint folds ($N=2,530$).
- **Regression Contract Suite:** 26 automated contract tests (target-leakage prevention, schema normalization, score bounds $[0, 100]$, graph acyclicity, zero-overlap TRAM isolation, and cross-file reconciliation) pass deterministically.
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

## 📄 License & Attribution

- **Source Code & Benchmark Architecture:** Released under the [Apache 2.0 License](LICENSE).
- **Upstream SigmaHQ Rules:** Upstream Sigma rules utilized in benchmark splits are authored by the [SigmaHQ Community](https://github.com/SigmaHQ/sigma) and licensed under the [Detection Rule License (DRL 1.1)](https://github.com/SigmaHQ/sigma/blob/master/LICENSE.Detection.Rules.md).
- **MITRE ATT&CK®:** MITRE ATT&CK is a registered trademark of The MITRE Corporation.
