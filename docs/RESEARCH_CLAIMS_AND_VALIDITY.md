# CyberThreatAgent-Bench: Research Claims, Hypotheses, and Construct Validity

This document defines the formal research claims (C1–C5), statistical test criteria, explicit falsification boundaries, experimental designs, and construct validity threats for **CyberThreatAgent-Bench**.

---

## 🔬 Core Research Claims & Falsification Criteria

### Claim 1 (C1): Cross-Domain Generalization & Granular Chance-Level Performance
> **Claim Formulation:**  
> ATT&CK technique mapping models trained exclusively on structured detection rules (Sigma) exhibit monotonic degradation across domain shifts:
> $$\text{Sigma Rules} > \text{TRAM Prose} > \text{Splunk Content-Bearing} > \text{Splunk Raw Stream}$$
> Raw Top-1 accuracy cannot be directly compared across different datasets without evaluating both the true open-vocabulary candidate label space ($K_{\text{open}} = 248$, $p_{0,\text{open}} = 1/248 = 0.40\%$, corresponding to the full operational space scored by the mapper) and the closed-support subset floor ($p_{0,\text{closed}} = 1/|K_{\text{closed}}|$, e.g., $1/50 = 2.00\%$ on TRAM, $1/29 = 3.45\%$ on Splunk). Equivalence to chance is declared only if the 90% CI lies entirely below $1.5\times$ chance (Two One-Sided Tests, TOST); otherwise, the model lift must be reported alongside its confidence interval.

* **Statistical Test:**  
  For each corpus $D \in \{\text{Sigma}, \text{TRAM}, \text{Splunk-Content}, \text{Splunk-Raw}\}$, compute empirical Top-1 accuracy $\hat{p}_D$, candidate space sizes ($|K_{\text{open}}| = 248$ and $|K_{\text{closed}}|$), and exact random chance floors ($p_{0, \text{open}} = 1/248$ and $p_{0, \text{closed}} = 1/|K_D|$). Compute lift over chance as both ratio $\text{Lift}_{\text{ratio}} = \hat{p}_D / p_{0}$ and absolute margin $\text{Lift}_{\text{abs}} = \hat{p}_D - p_{0}$ with Wilson score confidence intervals. Test for strict monotonic degradation ($\hat{p}_{\text{Sigma}} > \hat{p}_{\text{TRAM}} > \hat{p}_{\text{Splunk-Content}} > \hat{p}_{\text{Splunk-Raw}}$). Equivalence to random chance is evaluated via TOST against an equivalence margin of $\delta = 1.5 \times p_0$.
* **Corpus Profiles & Baseline Performance:**

| Corpus / Domain | Samples ($N$) | Closed Vocab ($|K|$) | Open Chance ($1/248$) | Closed Chance ($1/|K|$) | Model Top-1 ($\hat{p}$) | Lift vs. Closed Floor | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sigma Test Set** | 2,530 | 248 | 0.40% | 0.40% | 38.9% $\pm$ 8.8% | **97.3x** | In-Distribution Benchmark ($p < 0.001$) |
| **MITRE TRAM Analyst Text** | 1,500 | 50 | 0.40% | 2.00% | 19.53% (Hybrid) / 30.33% (Dense) | **9.8x / 15.2x** | Strong CTI Narrative Transfer ($p < 0.001$) |
| **Splunk Content-Bearing Telemetry** | 329 | 29 | 0.40% | 3.45% | 4.56% (Hybrid) / 7.29% (Cascade) | **1.3x / 2.1x** | Significant Runtime Signal ($p < 0.01$) |
| **Splunk Raw Unfiltered Stream** | 450 | 29 | 0.40% | 3.45% | 4.44% | **1.3x** | Evaluated via TOST (Eq. bound 1.5x = 5.17%) |

* **Falsification Condition:**  
  Equivalence to chance is declared only if the 90% CI lies entirely below $1.5\times$ chance (TOST); otherwise report empirical lift with its CI. If Splunk Content-Bearing telemetry fails to significantly exceed chance ($p \ge 0.05$ via one-sample binomial test) or if cross-domain ordering violates monotonicity, Claim 1 is falsified.

---

### Claim 2 (C2): Error Propagation, Conditional Failure Risk, and Stage Elasticity
> **Claim Formulation:**  
> Errors introduced in upstream security analysis stages propagate causally to downstream synthesis stages. We distinguish **causal error propagation** from **compounding error amplification**:
> 1. *Causal Propagation* holds if counterfactual oracle interventions demonstrate strictly positive downstream recovery ($\Delta_{\text{oracle}}(S_2 \to S_3) > 0$ and $\Delta_{\text{oracle}}(S_2 \to S_9) > 0$), and upstream extraction failure significantly increases downstream classification failure risk ($P(\text{S3 error} \mid \text{S2 error}) > P(\text{S3 error} \mid \text{S2 clean})$).
> 2. *Compounding vs. Buffering* is governed by stage elasticity $\eta$. Downstream stages either amplify ($\eta > 1$) or buffer ($\eta \le 1$) upstream error relative to remaining performance retention.

* **Statistical Test:**  
  1. **Intervention Analysis:** Counterfactually replace Stage 2 output with gold-standard IOC/CLI extraction while holding all other inputs fixed. Compute recovery $\Delta_{\text{oracle}}(S_2 \to S_3)$ on Technique $F_1$ and $\Delta_{\text{oracle}}(S_2 \to S_9)$ on Mitigation $\text{nDCG@3}$, with cluster-bootstrap 95% confidence intervals ($B=2,000$ iterations clustered by `incident_id`).
  2. **Conditional Risk Ratio:** Measure failure risk ratio $\text{RR}_{2 \to 3} = \frac{P(\text{S3 error} \mid \text{S2 error})}{P(\text{S3 error} \mid \text{S2 clean})}$.
  3. **Stage Elasticity:** Define relative performance retention $R_{3 \to 7} = \frac{\text{EdgeF1}(\text{Pred } S_3)}{\text{EdgeF1}(\text{Oracle } S_3)}$ and $R_{3 \to 9} = \frac{\text{MitNDCG3}(\text{Pred } S_3)}{\text{MitNDCG3}(\text{Oracle } S_3)}$. Compute stage elasticity:
     $$\eta_{3 \to 7} = \frac{1 - R_{3 \to 7}}{1 - \text{TechF1}_{S_3}}, \quad \eta_{3 \to 9} = \frac{1 - R_{3 \to 9}}{1 - \text{TechF1}_{S_3}}$$
     An elasticity $\eta > 1$ confirms compounding amplification; $\eta \le 1$ demonstrates downstream structural buffering.
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  Causal error propagation is rejected if $\Delta_{\text{oracle}}(S_2 \to S_9) \le 0$ or if the bootstrap 95% CI includes zero. Compounding amplification is rejected if $\eta_{3 \to 7} \le 1$ or $\eta_{3 \to 9} \le 1$, in which case downstream dynamics must be reported as buffered/attenuated propagation.

---

### Claim 3 (C3): Confidence-Gated Hybrid Routing & Per-Stage Escalation Advantage
> **Claim Formulation (Bet):**  
> 1. Gating single-alert triage by local prediction uncertainty $\theta_3$—resolving decisive predictions locally on CPU and escalating ambiguous cases to a cloud LLM—achieves Pareto superiority over random routing:
>    $$\text{Acc}_{\text{cascade}}(c) > (1 - c)\text{Acc}_{\text{local}} + c\text{Acc}_{\text{LLM}}$$
>    across operating escalation fractions $c \in [0.10, 0.50]$ on held-out test data (`test_200_fold0.csv`).
> 2. In multi-stage end-to-end incident investigation, **per-stage escalation** (combining Stage 3 mapping confidence $\theta_3$ with Stage 8 graph risk/coherence gating $\theta_8$) achieves strictly higher downstream Mitigation $\text{nDCG@3}$ than a single global confidence threshold ($\theta_{\text{global}}$) at matched total LLM call budgets ($10\% - 50\%$).

* **Statistical Test:**  
  Fit `HierarchicalMapper` strictly on Fold-0 training data (`train_b`), assert zero rule ID or label leakage against held-out `test_200_fold0.csv`, and measure genuine CPU latencies and top-2 prediction margins $\Delta_{\text{margin}} = p_{(1)} - p_{(2)}$.
  1. Compare cascade accuracy against random routing convex combination at matched budget $c$. Test significance via paired McNemar test ($p < 0.05$).
  2. In multi-stage incident evaluation across matched LLM call budgets ($10\%, 20\%, 30\%, 40\%, 50\%$), compare per-stage escalation $(\theta_3, \theta_8)$ against single global threshold $\theta_{\text{global}}$ on Mitigation $\text{nDCG@3}$.
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  If the cascade fails to outperform random routing $\text{Acc}_{\text{cascade}}(c) \le (1 - c)\text{Acc}_{\text{local}} + c\text{Acc}_{\text{LLM}}$ at any tested budget $c \in [0.10, 0.50]$, or if per-stage escalation yields lower $\text{Mitigation nDCG@3}$ than global thresholding at matched LLM budgets, Claim 3 is falsified.

---

### Claim 4 (C4): Autonomous Tool-Augmented Agents Mitigate Benign Alert Fatigue (Pre-Registered Bet)
> **Claim Formulation (Bet):**  
> Static waterfall pipelines and single-threshold cascaded pipelines suffer from near-total false escalation on benign enterprise noise ($\text{Benign FER} \approx 1.0$), whereas autonomous multi-turn agents with active telemetry query tools and triage termination verdicts reduce benign false escalations by $> 80\%$ ($\text{Benign FER} \le 0.20$) without sacrificing malicious incident recall ($F_1 \ge 0.35$). This behavior is evaluated across a complete **Architecture $\times$ Model Backbone** grid:
> $$\{\text{Static Waterfall, Per-Stage Cascade, ReAct } (T=6), \text{ReAct } (T=12), \text{Plan-and-Execute}\} \times \{\text{gemini-3.5-flash-lite}, \text{qwen/qwen3.8-27b}\}$$

* **Statistical Test:**  
  The formal falsification test is evaluated on the matched **$N = 120$ real benign event sequences** (`artifacts/external/benign_sequences_real_120.csv`), matching event counts (3–6 events), Windows/Sysmon administrative actions, scheduled tasks, and service updates, with zero synthetic "benign" label leakage. Calculate Benign False Escalation Rate ($\text{FER} = \frac{\text{FP}_{\text{benign}}}{N_{\text{benign}}}$) with exact Wilson score 95% confidence intervals and malicious Technique $F_1$.
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  If autonomous agents (ReAct or Plan-and-Execute) exhibit $\text{Benign FER} > 0.20$ on the matched $N = 120$ real benign enterprise activity set, OR if benign suppression is achieved via degenerate abstention that causes malicious Technique $F_1$ to drop below $0.35$, Claim 4 is falsified.

---

### Claim 5 (C5): Reasoning Horizon Expansion Enhances Course-of-Action Mitigation Quality (Pre-Registered Bet)
> **Claim Formulation (Bet):**  
> Expanding the agent interaction reasoning budget from $T_{\max}=6$ to $T_{\max}=12$ turns enables iterative MITRE technique disambiguation and multi-step course-of-action retrieval, yielding a statistically significant improvement in mitigation ranking quality ($\Delta \text{Mit nDCG@3} > 0.10$) while maintaining tool precision $> 0.95$ and redundancy rate $< 0.05$ across both frontier and open-weight backbones (`gemini-3.5-flash-lite`, `qwen/qwen3.8-27b`).

* **Statistical Test:**  
  Compare paired trajectories on identical incidents under $T_{\max}=6$ vs. $T_{\max}=12$ using identical pinned backbones. Compute paired difference in Mitigation $\text{nDCG@3}$, tool call precision ($\frac{\text{Valid Tools}}{\text{Total Calls}}$), redundant query rate, and hallucination rate ($\frac{\text{Invalid IDs}}{\text{Total Preds}}$).
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  If extending reasoning budget from $T_{\max}=6$ to $T_{\max}=12$ produces $\Delta \text{Mit nDCG@3} \le 0$, OR if extended turns cause tool call redundancy to exceed $10\%$ or hallucination rate to exceed $5\%$, the hypothesis of scalable reasoning benefits is rejected (demonstrating diminishing/negative returns to autonomous agent search depth).

---

## 📐 Family-Wise Error Rate Control (Analysis Plan)

To control false discovery across the five primary claims (C1–C5), statistical hypothesis testing enforces the **Holm–Bonferroni method**:
1. Rank unadjusted $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(5)}$ across the five pre-registered tests.
2. Evaluate sequentially at thresholds $\alpha / (5 - k + 1)$ for $k \in \{1, \dots, 5\}$ at family-wise significance level $\alpha = 0.05$.
3. If $p_{(k)} > \alpha / (5 - k + 1)$, testing terminates and all subsequent hypotheses are declared non-significant.

---

## 🏗 Procedural Synthetic Multi-Alert Generator & Threats to Construct Validity

### Role of Real Corpora vs. Synthetic Incident Chains
CyberThreatAgent-Bench enforces strict separation of evaluation roles between real corpora and synthetic chains:
1. **Primary Evaluation on Real Corpora:**
   - **Detection Rule Generalization ($N=2,530$):** Canonical in-distribution evaluation on expert-authored Sigma rules spanning 248 MITRE ATT&CK techniques.
   - **Real Operational CTI ($N=1,500$):** Unstructured prose extracted from real-world threat reports annotated in the MITRE TRAM corpus across 50 techniques.
   - **Real Runtime SIEM Telemetry ($N=329$):** Authentic Windows event log telemetry from Splunk Attack Data across 29 techniques.
   - **Real Enterprise Benign Baseline ($N=120$):** Multi-event administrative, scheduled task, service deployment, and maintenance telemetry sequences extracted from production Windows environments without synthetic markers.
2. **Restricted Role of Synthetic Multi-Alert Generator:**
   - Multi-stage incident chains (`INC-001` to `INC-040`) are synthetically composed using 20 intrusion campaign templates spanning Windows, Linux, and Cloud environments.
   - **Explicit Limitation:** Synthetic chains are used *strictly* for controlled causal/architectural ablations (e.g., measuring Stage 2 $\to$ Stage 3 $\to$ Stage 7 $\to$ Stage 9 error propagation under counterfactual oracles, comparing reasoning horizons $T=6$ vs. $T=12$, and testing multi-agent escalation protocols).
   - Synthetic chains do *not* substitute for real operational corpora when reporting primary classification generalization or baseline enterprise false-positive rates.

---

## ⚠️ Construct Validity & Threats to Validity

### 1. Threats to Internal Validity
* **Synthetic Multi-Stage Chaining:** While alert descriptions derive from validated Sigma detection rules, multi-alert temporal chaining is synthetically assembled. Real-world adversary dwell time, interleaved unrelated enterprise alerts, and network telemetry jitter are approximated.
* **Deterministic Replay & Cache Integrity:** Local models run deterministically with fixed random seeds. For cloud LLM backbones, responses are logged to append-only JSONL caches with model IDs, timestamps, and request hashes to ensure auditability against cloud API drift.

### 2. Threats to External Validity
* **Candidate Label Space Cardinality Shift:** Real-world SOCs encounter varied distributions of ATT&CK techniques. Cross-dataset comparisons must always report lift over respective chance baselines ($1/248$ open-vocabulary vs. $1/50$ and $1/29$ closed-support subsets).
* **Textual Density vs. Raw Telemetry:** Sigma detection rules possess rich descriptive metadata, whereas raw SIEM syslog feeds are condensed and noisy. Models trained solely on rule text exhibit substantial domain shift when applied to raw streams.

### 3. Threats to Construct Validity
* **Mitigation Utility vs. Technique Accuracy:** Stage 9 mitigation nDCG@3 evaluates MITRE D3FEND/ATT&CK recommendations queried using the model's *predicted* technique. While this reflects end-to-end SOC analyst decision utility, it penalizes mitigation score whenever upstream technique mapping fails.
* **Risk Score Calibration Target:** Risk scores are calibrated against human-assigned Sigma severity levels without exposing the severity label to the feature extractor. Correlation reflects ordinal rank consistency rather than actuarial loss probability.
