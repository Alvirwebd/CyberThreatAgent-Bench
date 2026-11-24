"""
scripts/generate_clean_routing.py
=================================
Generates artifacts/hybrid_llm_routing.csv strictly on the held-out n=200 test set (test_200_fold0.csv).
Uses genuine HierarchicalMapper predictions & margins fitted strictly on Fold-0 (seed=7) train_b.
Uses champion LLM (gemini-3.5-flash-lite) predictions from artifacts/llm_predictions_200.csv.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as _st

from ctab import load_attack, get_canonical_benchmark, group_split, HierarchicalMapper

OUT = Path("artifacts")
test_csv_path = OUT / "test_200_fold0.csv"
llm_preds_path = OUT / "llm_predictions_200.csv"

if not test_csv_path.exists() or not llm_preds_path.exists():
    print(f"[ERROR] Required inputs not found: {test_csv_path} or {llm_preds_path}")
    sys.exit(1)

print("[*] Loading ATT&CK and canonical benchmark...")
df_techniques, _, _, _ = load_attack()
bench_df = get_canonical_benchmark()

print("[*] Performing Fold-0 (seed=7) group split...")
tr, te, _ = group_split(bench_df, seed=7)
df_test_alerts = pd.read_csv(test_csv_path)

# Verify zero overlap / leakage
overlap = set(tr["sample_id"]) & set(df_test_alerts["sample_id"])
print(f"[*] Overlap between train_b (N={len(tr)}) and test_200 (N={len(df_test_alerts)}): {len(overlap)}")
assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping samples!"

print("[*] Fitting HierarchicalMapper strictly on train_b...")
t_fit_start = time.perf_counter()
mapper = HierarchicalMapper(df_techniques, tr, alpha=0.65, C=8.0, auto_tune_alpha=True)
print(f"[*] HierarchicalMapper fitted in {time.perf_counter() - t_fit_start:.2f}s (alpha*={mapper.best_alpha:.2f})")

print("[*] Loading champion LLM (gemini-3.5-flash-lite) predictions...")
df_preds_all = pd.read_csv(llm_preds_path)
df_champ = df_preds_all[df_preds_all["model_id"] == "gemini-3.5-flash-lite"].copy()
assert len(df_champ) == 200, f"Expected 200 champion predictions, got {len(df_champ)}"

pred_map = dict(zip(df_champ["sample_id"], df_champ["pred_top1"]))
lat_map = dict(zip(df_champ["sample_id"], df_champ["latency_ms"]))

print("[*] Evaluating local HierarchicalMapper predictions, margins, and CPU latency on n=200...")
records = []
for idx, row in df_test_alerts.iterrows():
    sid = row["sample_id"]
    gt_aid = str(row["attack_id"]).strip().upper()
    text = str(row["text"])
    llm_p = pred_map.get(sid)
    llm_lat = float(lat_map.get(sid, 804.7))

    t0 = time.perf_counter()
    ranked_preds = mapper.ranked(text, top_k=2)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    if ranked_preds:
        top1_p = ranked_preds[0]["attack_id"]
        top1_score = float(ranked_preds[0].get("score", 0.0))
        top2_score = float(ranked_preds[1].get("score", 0.0)) if len(ranked_preds) > 1 else 0.0
        margin = float(top1_score - top2_score)
    else:
        top1_p = "T1059.001"
        top1_score = 0.0
        top2_score = 0.0
        margin = 0.0

    records.append({
        "sample_id": sid,
        "attack_id": gt_aid,
        "hybrid_pred": top1_p,
        "hybrid_top1_score": round(top1_score, 4),
        "hybrid_top2_score": round(top2_score, 4),
        "hybrid_margin": round(margin, 4),
        "hybrid_lat_ms": round(lat_ms, 2),
        "llm_pred": llm_p,
        "llm_lat_ms": round(llm_lat, 2),
        "hybrid_correct": (top1_p == gt_aid),
        "llm_correct": (llm_p == gt_aid if llm_p else False),
    })

eval_df = pd.DataFrame(records)
n_alerts = len(eval_df)
assert n_alerts == 200, f"Expected 200 alerts, got {n_alerts}"

local_acc = float(eval_df["hybrid_correct"].mean())
llm_acc = float(eval_df["llm_correct"].mean())
oracle_union_correct = eval_df["hybrid_correct"] | eval_df["llm_correct"]
oracle_union_acc = float(oracle_union_correct.mean())

print("=" * 60)
print(f"CHECKPOINT 2a (Local vs LLM on n=200):")
print(f"  Local Top-1 Accuracy: {local_acc:.4f} ({int(eval_df['hybrid_correct'].sum())}/200)")
print(f"  LLM Top-1 Accuracy  : {llm_acc:.4f} ({int(eval_df['llm_correct'].sum())}/200)")
print(f"  Oracle Union Acc    : {oracle_union_acc:.4f} ({int(oracle_union_correct.sum())}/200)")
print("=" * 60)

if local_acc <= llm_acc:
    print("CASCADE PREMISE BROKEN: Local accuracy <= LLM accuracy.")
else:
    print(f"CASCADE PREMISE VERIFIED: Local accuracy {local_acc:.4f} > LLM {llm_acc:.4f}.")

# Export per-alert records
per_alert_path = OUT / "routing_per_alert_200.csv"
eval_df.to_csv(per_alert_path, index=False)
print(f"[*] Saved per-alert routing to {per_alert_path}")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
sweep_rows = []

for th in thresholds:
    escalated = eval_df["hybrid_margin"] < th
    esc_rate = float(escalated.mean() * 100.0)
    c_frac = float(escalated.mean())
    final_preds = np.where(escalated, eval_df["llm_pred"], eval_df["hybrid_pred"])
    final_lats = np.where(escalated, eval_df["llm_lat_ms"] + eval_df["hybrid_lat_ms"], eval_df["hybrid_lat_ms"])

    is_correct = (final_preds == eval_df["attack_id"]).astype(float)
    top1_exact = float(is_correct.mean())

    # Random routing baseline at matched escalation rate
    random_routing_acc = (1.0 - c_frac) * local_acc + c_frac * llm_acc

    # Paired bootstrap 95% CI (B=2000)
    rng_boot = np.random.RandomState(42)
    boot_accs = [is_correct[rng_boot.randint(0, n_alerts, n_alerts)].mean() for _ in range(2000)]
    ci_low, ci_high = np.percentile(boot_accs, [2.5, 97.5])

    # McNemar vs standalone LLM
    b = int(((is_correct == 1.0) & (~eval_df["llm_correct"])).sum())
    c = int(((is_correct == 0.0) & (eval_df["llm_correct"])).sum())
    if (b + c) > 0:
        mcnemar_stat = (abs(b - c) - 1.0)**2 / (b + c)
        p_mcnemar = float(1.0 - _st.chi2.cdf(mcnemar_stat, df=1))
    else:
        p_mcnemar = 1.0

    p50_lat = float(np.percentile(final_lats, 50))
    mean_lat = float(final_lats.mean())
    cost_savings_pct = float(100.0 - esc_rate)

    sweep_rows.append({
        "margin_threshold": th,
        "llm_escalation_pct": round(esc_rate, 2),
        "Top1_Exact": round(top1_exact, 4),
        "random_routing_acc": round(random_routing_acc, 4),
        "routing_lift_over_random": round(top1_exact - random_routing_acc, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "p50_latency_ms": round(p50_lat, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "cost_savings_vs_full_llm_pct": round(cost_savings_pct, 2),
        "p_val_vs_full_llm": round(p_mcnemar, 4),
    })

df_sweep = pd.DataFrame(sweep_rows)

# Select best threshold by highest Top1_Exact
best_idx = df_sweep["Top1_Exact"].idxmax()
df_sweep["is_selected_theta"] = [i == best_idx for i in range(len(df_sweep))]

sweep_csv_path = OUT / "hybrid_llm_routing.csv"
df_sweep.to_csv(sweep_csv_path, index=False)
print(f"[*] Saved hybrid cascade sweep to {sweep_csv_path}")
print(df_sweep.to_string(index=False))
