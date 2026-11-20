# CyberThreatAgent-Bench: Autonomous Cybersecurity Agent Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
                 Stage 3: Hierarchical ATT&CK Technique Mapping (Hybrid / Dense / Agentic Router)
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
git clone https://github.com/your-org/CyberThreatAgent-Bench-Release.git
cd CyberThreatAgent-Bench-Release
pip install -r requirements.txt
```

### 2. One-Command Benchmark Execution

Run the fast verification sanity check (2 seeds, subset alerts):
```bash
python run_benchmark.py --fast
```

Execute the full 5-seed publication benchmark (matches all manuscript numbers):
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

- `artifacts/benchmark_summary.csv`: Aggregated 5-seed mean $\pm$ Student's $t$ $	ext{CI}_{95}$ across all 9 pipeline stages.
- `artifacts/scorecard_aggregated.csv`: Complete per-metric multi-seed statistics.
- `artifacts/agentic_routing.csv`: Uncertainty margin threshold sweep and escalation rates.
- `artifacts/external_validation.csv`: Zero-overlap real-world incident test results.
- `artifacts/manifest_v5.json`: Cryptographic SHA-256 manifest of frozen inputs and outputs.
- `artifacts/datasheet.json`: Standardized machine-readable dataset documentation.
- `artifacts/outputs/latex/`: Formatted publication tables ready for direct `\input{}`:
  - `table_scorecard.tex`: Multi-stage performance scorecard.
  - `table_baselines.tex`: 5-seed baseline comparison table with statistical significance markers ($p < 0.01$).
  - `table_dataset.tex`: Dataset and corpus statistics.
  - `table_external.tex`: Real-world out-of-distribution evaluation.
- `artifacts/outputs/figures/`: High-resolution (300 DPI) `.png` and vector `.pdf` figures (Pareto frontiers, leakage probes, noise decay curves, attack graphs).

---

## 📜 Baseline Performance Summary (Stage 3 ATT&CK Mapping)

Evaluated across $n=5$ independent seeds on group-disjoint splits:

| Method | Top-1 Exact | Top-3 Exact | MRR Exact | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| Random Guessing Floor | 0.003 $\pm$ 0.001 | 0.008 $\pm$ 0.001 | 0.005 $\pm$ 0.001 | 0.001 $\pm$ 0.001 |
| Majority Class | 0.042 $\pm$ 0.001 | 0.081 $\pm$ 0.001 | 0.057 $\pm$ 0.001 | 0.004 $\pm$ 0.001 |
| Rule-Based Keyword | 0.158 $\pm$ 0.008 | 0.245 $\pm$ 0.008 | 0.198 $\pm$ 0.008 | 0.082 $\pm$ 0.008 |
| Okapi BM25 | 0.194 $\pm$ 0.088 | 0.284 $\pm$ 0.121 | 0.248 $\pm$ 0.098 | 0.118 $\pm$ 0.056 |
| Zero-Shot TF-IDF | 0.213 $\pm$ 0.107 | 0.302 $\pm$ 0.052 | 0.260 $\pm$ 0.086 | 0.131 $\pm$ 0.060 |
| Dense Retriever (`bge-small`) | 0.206 $\pm$ 0.057 | 0.288 $\pm$ 0.060 | 0.251 $\pm$ 0.048 | 0.122 $\pm$ 0.030 |
| Supervised Dense (`bge-small`) | 0.035 $\pm$ 0.018 | 0.105 $\pm$ 0.033 | 0.077 $\pm$ 0.018 | 0.032 $\pm$ 0.015 |
| **Proposed Hybrid Mapper** | **0.265 $\pm$ 0.063**$^{\dagger}$ | **0.450 $\pm$ 0.085**$^{\dagger}$ | **0.366 $\pm$ 0.072**$^{\dagger}$ | **0.213 $\pm$ 0.031**$^{\dagger}$ |
| Agentic Router | 0.194 $\pm$ 0.012 | 0.312 $\pm$ 0.012 | 0.299 $\pm$ 0.012 | 0.145 $\pm$ 0.012 |
| LLM (`gemini-2.0-flash-lite`)* | 0.683 | -- | 0.683 | -- |

$^{\dagger}$ Statistically significant improvement over next-best baseline ($p < 0.01$, paired bootstrap & Wilcoxon signed-rank tests across 5 seeds).  
$^{*}$ Evaluated on $n=60$ test alerts with strict JSON schema at temperature 0 (unparseable predictions counted as incorrect).

---

## 📖 Citation

If you use CyberThreatAgent-Bench in your research, please cite:

```bibtex
@inproceedings{alvir2026cyberthreatagentbench,
  title     = {CyberThreatAgent-Bench: A Multi-Stage Benchmark for Autonomous Threat Detection and Investigation Agents},
  author    = {Alvir, M. and DeepMind Team},
  booktitle = {Proceedings of the IEEE Symposium on Security and Privacy (S&P)},
  year      = {2026}
}
```

---

## 📄 License
This repository is released under the [Apache 2.0 License](LICENSE).
