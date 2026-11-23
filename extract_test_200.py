import sys, json, hashlib, os
from pathlib import Path
import pandas as pd
import numpy as np

os.chdir("c:/Research/Aiworks/CyberThreatAgent_Bench")
import nbformat
nb = nbformat.read("Project2-CyberThreatAgent_Bench.ipynb", as_version=4)

ctx = {}
for i, cell in enumerate(nb.cells[:15]):
    if cell.cell_type == "code":
        src = cell.source
        exec(src, ctx)

bench = ctx["bench"]
CFG = ctx["CFG"]
group_split = ctx["group_split"]

train_b, test_b, manifest = group_split(bench, CFG, seed=CFG.seeds[0])

print(f"Total benchmark alerts: {len(bench)}")
print(f"Fold 0 train_b alerts: {len(train_b)}")
print(f"Fold 0 test_b alerts: {len(test_b)}")
assert set(test_b["sample_id"]).isdisjoint(set(train_b["sample_id"]))
print("Disjoint check: PASSED!")

# Stratified sampling of exactly n=200 alerts from Fold 0 test_b
proportions = test_b["source_family"].value_counts(normalize=True)
sample_indices = []
for fam, count in (proportions * 200).round().astype(int).items():
    sub_fam = test_b[test_b["source_family"] == fam]
    take = min(len(sub_fam), max(1, count))
    sample_indices.extend(sub_fam.sample(n=take, random_state=42).index.tolist())

rng = np.random.RandomState(42)
if len(sample_indices) < 200:
    remain = [i for i in test_b.index if i not in sample_indices]
    sample_indices.extend(rng.choice(remain, size=200 - len(sample_indices), replace=False).tolist())
elif len(sample_indices) > 200:
    sample_indices = sample_indices[:200]

test_200 = test_b.iloc[sample_indices].reset_index(drop=True)
print(f"Sampled test_200: {len(test_200)} items")
print(f"Unique source families in test_200: {test_200['source_family'].nunique()}")
print(f"Unique ATT&CK labels in test_200: {test_200['attack_id'].nunique()}")

# Hard assertion: test_200 strictly disjoint from train_b
assert set(test_200["sample_id"]).isdisjoint(set(train_b["sample_id"]))
print("Assertion PASSED: test_200 is strictly disjoint from Fold 0 train_b!")

# Save to artifacts/test_200_fold0.csv
OUT = Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)
test_200.to_csv(OUT / "test_200_fold0.csv", index=False)
print(f"[SAVED] {OUT / 'test_200_fold0.csv'}")

# Also fit Fold 0 HierarchicalMapper and evaluate baseline on test_200!
df_techniques = ctx["df_techniques"]
fit_mappers = None
for c in nb.cells:
    if c.cell_type == "code" and "def fit_mappers" in c.source:
        exec(c.source, ctx)
        fit_mappers = ctx["fit_mappers"]
        break

print("Fitting Fold 0 HierarchicalMapper on train_b (seed 7)...")
maps_fold0 = fit_mappers(df_techniques, train_b, CFG, seed=CFG.seeds[0])
hybrid_fold0 = maps_fold0["hybrid"]

# Evaluate on test_200
correct_1 = []
mrr_list = []
for _, r in test_200.iterrows():
    ranked = hybrid_fold0.ranked(r["text"], top_k=5)
    preds = [x["attack_id"] for x in ranked]
    gt = r["attack_id"]
    correct_1.append(preds[0] == gt if preds else False)
    # MRR@5
    rr = 0.0
    for rank, p in enumerate(preds[:5], 1):
        if p == gt:
            rr = 1.0 / rank
            break
    mrr_list.append(rr)

print("\n--- OUT-OF-SAMPLE LOCAL HYBRID BASELINE ON n=200 TEST ALERTS ---")
print(f"Top-1 Exact: {np.mean(correct_1):.4f} ({sum(correct_1)}/{len(test_200)})")
print(f"MRR@5 Exact: {np.mean(mrr_list):.4f}")
