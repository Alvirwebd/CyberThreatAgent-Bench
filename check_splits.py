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

Config = ctx["Config"]
build_benchmark_df = ctx["build_benchmark_df"]
group_split = ctx["group_split"]

cfg_full = Config(fast=False)
bench_full = build_benchmark_df(cfg_full)
split_full = group_split(bench_full, cfg_full, seed=cfg_full.seeds[0])
print(f"FULL mode: bench_df = {len(bench_full)}, Fold 0 test_b = {len(split_full['test_idx'])}")

# Save Fold 0 test_b alerts to disk
test_b_full = bench_full.iloc[split_full["test_idx"]].reset_index(drop=True)
train_b_full = bench_full.iloc[split_full["train_idx"]].reset_index(drop=True)
print(f"Disjoint check: {set(test_b_full['sample_id']).isdisjoint(set(train_b_full['sample_id']))}")

# Sample deterministic stratified n=200 subset
# Stratify by tactic or attack_id prefix / source_family
rng = np.random.RandomState(42)
if len(test_b_full) > 200:
    # Stratified sampling across source_family
    sample_indices = []
    # group by source_family
    by_fam = test_b_full.groupby("source_family", group_keys=False)
    # take proportional samples
    proportions = test_b_full["source_family"].value_counts(normalize=True)
    for fam, count in (proportions * 200).round().astype(int).items():
        sub_fam = test_b_full[test_b_full["source_family"] == fam]
        take = min(len(sub_fam), max(1, count))
        sample_indices.extend(sub_fam.sample(n=take, random_state=42).index.tolist())
    
    # Adjust to exactly 200
    if len(sample_indices) < 200:
        remain = [i for i in test_b_full.index if i not in sample_indices]
        sample_indices.extend(rng.choice(remain, size=200 - len(sample_indices), replace=False).tolist())
    elif len(sample_indices) > 200:
        sample_indices = sample_indices[:200]
    
    test_200 = test_b_full.iloc[sample_indices].reset_index(drop=True)
else:
    test_200 = test_b_full.copy()

print(f"Sampled test_200: {len(test_200)} items")
print(f"Unique source families in test_200: {test_200['source_family'].nunique()}")
print(f"Unique ATT&CK labels in test_200: {test_200['attack_id'].nunique()}")
assert set(test_200['sample_id']).isdisjoint(set(train_b_full['sample_id']))
print("Strict out-of-sample disjointness assertion: PASSED!")
