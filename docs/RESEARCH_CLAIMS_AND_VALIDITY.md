# CyberThreatAgent-Bench: Research Claims, Hypotheses, and Construct Validity

This document defines the formal research claims (C1–C5), statistical test criteria, explicit falsification boundaries, experimental designs, and construct validity threats for **CyberThreatAgent-Bench**.

---

## 🔬 Core Research Claims & Falsification Criteria

### Claim 1 (C1): Cross-Domain Generalization & Granular Chance-Level Performance
> **Claim Formulation:**  
> ATT&CK technique mapping models trained exclusively on structured detection rules (Sigma) do not transfer uniformly across operational contexts. Raw Top-1 accuracy cannot be directly compared across different datasets without adjusting for the underlying candidate label space cardinality and empirical chance rate. Equivalence to chance is declared only if the 90% CI lies entirely below 1.5x chance (Two One-Sided Tests, TOST); otherwise, the model lift must be reported alongside its confidence interval.

* **Statistical Test:**  
  For each corpus $D \in \{\text{Sigma}, \text{TRAM}, \text{Splunk-Content}, \text{Splunk-Raw}\}$, compute empirical Top-1 accuracy $\hat{p}_D$, candidate space size $|K_D|$, and exact random chance floor $p_{0, D} = \frac{1}{|K_D|}$. Compute lift over chance as both ratio $\text{Lift}_{\text{ratio}} = \hat{p}_D / p_{0, D}$ and absolute margin $\text{Lift}_{\text{abs}} = \hat{p}_D - p_{0, D}$ with Wilson score confidence intervals. Equivalence to random chance is evaluated via TOST against an equivalence margin of $\delta = 1.5 \times p_{0, D}$.
* **Corpus Profiles & Baseline Performance:**

| Corpus / Domain | Samples ($N$) | Candidate Techniques ($|K|$) | Chance Floor ($p_0$) | Model Top-1 ($\hat{p}$) | Lift over Chance ($\hat{p} / p_0$) | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sigma Test Set** | 2,530 | 248 | 0.40% | 38.9% $\pm$ 8.8% | **97x** | High In-Distribution Fidelity ($p < 0.001$) |
| **MITRE TRAM Analyst Text** | 1,500 | 50 | 2.00% | 32.7% $\pm$ 2.4% | **16x** | Strong CTI Transfer ($p < 0.001$) |
| **Splunk Content-Bearing Telemetry** | 329 | 29 | 3.45% | 12.5% $\pm$ 3.1% | **3.6x** | Significant Runtime Signal ($p < 0.001$) |
| **Splunk Raw Unstructured Telemetry** | 329 | 29 | 3.45% | ~4.4% $\pm$ 1.8% | **1.3x** | Evaluated via TOST (Eq. bound 1.5x = 5.17%) |

* **Falsification Condition:**  
  Equivalence to chance is declared only if the 90% CI lies entirely below $1.5\times$ chance (TOST); otherwise report the empirical lift with its CI. If Splunk Content-Bearing telemetry fails to significantly exceed chance ($p \ge 0.05$ via one-sample binomial test), Claim 1 is falsified.

---

### Claim 2 (C2): Error Propagation vs. Compounding Amplification in Multi-Stage Pipelines
> **Claim Formulation:**  
> Errors introduced in upstream security analysis stages propagate to downstream synthesis stages. However, "compounding" is strictly an amplification claim: downstream recovery under an upstream oracle must exceed upstream recovery when normalized by remaining headroom. If downstream normalized recovery does not strictly exceed upstream normalized recovery, the system exhibits *additive error propagation*, not compounding error amplification.

* **Statistical Test:**  
  Intervention: replace Stage 2 output with gold-standard IOC/CLI extraction, everything else held fixed. Measure its recovery at Stage 3 and at Stage 9, each normalised by that stage's remaining headroom:
  $$r_3 = \frac{\Delta(S_2 \to S_3)}{\text{gap}(S_3)}, \quad r_9 = \frac{\Delta(S_2 \to S_9)}{\text{gap}(S_9)}$$
  where $\text{gap} = \text{oracle-with-gold-}S_2 - \text{observed}$. Compute incident-cluster bootstrap 95% confidence intervals (resampling by `incident_id` across $B=2,000$ iterations) for the difference $r_9 - r_3$. (Note: The $S_3 \to S_7/S_9$ oracle is a separate sensitivity analysis figure and is not part of this statistic.)
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  Compounding is supported if $r_9 > r_3$ with a cluster-bootstrap 95% CI excluding zero. If $r_9 \le r_3$, the hypothesis of compounding error amplification is **rejected**, and the phenomenon must be reported as **additive propagation**.

---

### Claim 3 (C3): Confidence-Gated Hybrid Cascade Pareto Superiority (Pre-Registered Bet)
> **Claim Formulation (Bet):**  
> Gating alert triage by local prediction uncertainty $\theta$—resolving decisive predictions locally on CPU and escalating ambiguous cases to a cloud LLM—achieves Pareto dominance over standalone local models and standalone cloud LLMs, yielding super-additive accuracy at substantially reduced compute cost.

* **Statistical Test:**  
  Fit `HierarchicalMapper` on Fold-0 training data (`train_b`), assert zero rule ID or label leakage against held-out `test_200_fold0.csv`, and measure genuine CPU latencies and top-2 prediction margins $\Delta_{\text{margin}} = p_{(1)} - p_{(2)}$. Evaluate cascade across thresholds $\theta \in [0.05, 0.40]$ against champion LLM (`gemini-3.5-flash-lite`, standalone Top-1 = 0.4050). Super-additivity is claimed only if the paired McNemar test is statistically significant ($p < 0.05$) AND the cascade beats the better standalone model by $\ge 5$ alerts of 200 ($\Delta \ge +0.025$); a smaller difference is reported as no significant difference.
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  If local standalone Top-1 $\le 0.4050$ on $n=200$, the premise of local superiority on confident subsets is compromised (`CASCADE PREMISE BROKEN`). Furthermore, if for all operating thresholds $\theta \in [0.05, 0.40]$, the cascade accuracy $\text{Acc}(\theta) \le \max(\text{Acc}_{\text{local}}, \text{Acc}_{\text{LLM}})$ or exceeds the theoretical oracle union ceiling $\text{Acc}_{\text{oracle}} = \text{Acc}(\text{local} \cup \text{LLM})$, the claim of Pareto cascade superiority is rejected.

---

### Claim 4 (C4): Autonomous Tool-Augmented Agents Mitigate Benign Alert Fatigue (Pre-Registered Bet)
> **Claim Formulation (Bet):**  
> Static waterfall pipelines and thresholded cascaded pipelines suffer from near-total false escalation on benign enterprise noise ($\text{Benign FER} \approx 1.0$), whereas autonomous multi-turn agents with active telemetry query tools and triage termination verdicts reduce benign false escalations by $> 80\%$ ($\text{Benign FER} \le 0.20$) without sacrificing malicious incident recall ($F_1 \ge 0.35$).

* **Statistical Test:**  
  The formal falsification test is evaluated on the matched $N \ge 120$ benign set (matching event counts 3–6, OS/host mix, and security-relevant benign noise: failed logons, blocked connections, admin scripts, and service restarts). The initial 48-incident set (8 benign) is designated as a preliminary exploratory pilot and supports no formal claim. Calculate Benign False Escalation Rate ($\text{FER} = \frac{\text{FP}_{\text{benign}}}{N_{\text{benign}}}$) with Wilson score 95% confidence intervals and malicious Technique $F_1$.
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  If autonomous agents (ReAct or Plan-and-Execute) exhibit $\text{Benign FER} > 0.20$ on the matched $N \ge 120$ benign enterprise activity set, OR if benign suppression is achieved via degenerate abstention that causes malicious Technique $F_1$ to drop below $0.35$, Claim 4 is falsified.

---

### Claim 5 (C5): Reasoning Horizon Expansion Enhances Course-of-Action Mitigation Quality (Pre-Registered Bet)
> **Claim Formulation (Bet):**  
> Expanding the agent interaction reasoning budget from $T_{\max}=6$ to $T_{\max}=12$ turns enables iterative MITRE technique disambiguation and multi-step course-of-action retrieval, yielding a statistically significant improvement in mitigation ranking quality ($\Delta \text{Mit nDCG@3} > 0.10$) while maintaining tool precision $> 0.95$ and redundancy rate $< 0.05$.

* **Statistical Test:**  
  Compare paired trajectories on identical incidents under $T_{\max}=6$ vs. $T_{\max}=12$ using the identical pinned LLM backbone (`gemini-3.5-flash-lite`). Compute paired difference in Mitigation $\text{nDCG@3}$, tool call precision ($\frac{\text{Valid Tools}}{\text{Total Calls}}$), redundant query rate, and hallucination rate ($\frac{\text{Invalid IDs}}{\text{Total Preds}}$).
* **Falsification Condition (Pre-Stated Null Hypothesis):**  
  If extending reasoning budget from $T_{\max}=6$ to $T_{\max}=12$ produces $\Delta \text{Mit nDCG@3} \le 0$, OR if extended turns cause tool call redundancy to exceed $10\%$ or hallucination rate to exceed $5\%$, the hypothesis of scalable reasoning benefits is rejected (demonstrating diminishing/negative returns to autonomous agent search depth).

---

## 📐 Family-Wise Error Rate Control (Analysis Plan)

To control false discovery across the five primary claims (C1–C5), statistical hypothesis testing enforces the **Holm–Bonferroni method**:
1. Rank unadjusted $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(5)}$ across the five pre-registered tests.
2. Evaluate sequentially at thresholds $\alpha / (5 - k + 1)$ for $k \in \{1, \dots, 5\}$ at family-wise significance level $\alpha = 0.05$.
3. If $p_{(k)} > \alpha / (5 - k + 1)$, testing terminates and all subsequent hypotheses are declared non-significant.

---

## 🏗 Synthetic Multi-Alert Incident Chain Generator Documentation

To evaluate multi-stage temporal reasoning where public labeled operational incidents are scarce, CyberThreatAgent-Bench incorporates a synthetic multi-alert incident generator:
1. **Template Grounding:** 20 distinct intrusion campaign templates spanning Windows enterprise networks, Linux server environments, and cloud infrastructure (AWS/Azure).
2. **Kill-Chain Progression:** Alerts follow standardized cyber kill-chain progression: Initial Access $\to$ Execution $\to$ Persistence / Privilege Escalation $\to$ Defense Evasion $\to$ Credential Access $\to$ Discovery $\to$ Lateral Movement $\to$ Collection / Exfiltration / Impact.
3. **Entity & Host Binding:** Realistic hostnames (`WKSTN-101`, `SRV-DC01`), IP subnets, user accounts, and process trees are synthetically assigned to preserve realistic graph connectivity.
4. **Controlled Benign Counterparts:** Matched benign enterprise incident sequences (`BEN-000..007` in pilot; $N \ge 120$ in benchmark) represent noisy background operations: automated vulnerability scanning, administrator remote desktop sessions, software patch deployments, and legitimate scheduled scripts.

---

## ⚠️ Construct Validity & Threats to Validity

### 1. Threats to Internal Validity
* **Synthetic Incident Sequences:** While individual alert descriptions derive from human-authored Sigma detection rules and real CTI reports, multi-alert temporal chaining is synthetically composed. Real-world adversary dwell time, interleaved benign alerts, and network telemetry jitter are approximated.
* **Deterministic Replay & Seed Sensitivity:** Baseline models are evaluated across 5 random seeds to measure split variance, but LLM agent behaviors exhibit minor temperature-0 platform drift across cloud API infrastructure updates.

### 2. Threats to External Validity
* **Disjoint Technique Candidate Cardinalities:** Evaluating transfer across Sigma (248 techniques), MITRE TRAM (50 techniques), and Splunk telemetry (29 techniques) creates varying chance baselines. All reported metrics must be interpreted relative to their corpus chance floor.
* **Alert Format Homogeneity:** Sigma YAML rules contain richer textual descriptions than raw SIEM syslog feeds. Models trained on Sigma text experience domain shift when processing condensed raw telemetry.

### 3. Threats to Construct Validity
* **Mitigation Evaluation Metric:** Stage 9 mitigation nDCG@3 is evaluated by querying candidate MITRE playbooks using the model's *predicted* technique rather than gold ground-truth technique. This captures real-world decision utility, but can penalize mitigation quality purely due to upstream technique misclassification.
* **Risk Score Calibration Target:** Risk scores are validated externally against human-assigned Sigma rule severity levels (`low`, `medium`, `high`, `critical`) without including the label in the input feature space. Spearman rank correlation reflects ordinal alignment, not absolute dollar-loss risk.
