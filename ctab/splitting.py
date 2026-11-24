"""
ctab.splitting
==============
Deterministic group-disjoint and hash-partitioned dataset splitting.
Reproduces Cell 12 of Project2-CyberThreatAgent_Bench.ipynb.
"""

import hashlib
import numpy as np
import pandas as pd


def deterministic_hash(text, seed=42):
    """Generates a stable 64-bit integer hash for data splitting."""
    h = hashlib.sha256(f"{seed}::{text}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def deterministic_split(df, test_size=0.30, seed=42, text_col="text"):
    """
    Splits dataframe deterministically using content hashing so no train/test leakage occurs.
    """
    hashes = np.array([deterministic_hash(str(t), seed=seed) for t in df[text_col]])
    threshold = int(test_size * (2**64 - 1))
    is_test = hashes < threshold

    test_df = df[is_test].copy().reset_index(drop=True)
    train_df = df[~is_test].copy().reset_index(drop=True)
    return train_df, test_df


def group_split(bench_df, seeds=(7, 21, 42, 84, 123), seed=7):
    """
    Deterministic group-disjoint partitioner across n_folds = len(seeds).

    Sub-partitions oversized directories (>30% of corpus, e.g. rules/windows/process_creation)
    via rule_id hash buckets into balanced clusters. Greedily balances group clusters into
    disjoint folds 0..n_folds-1. Maps each benchmark seed bijectively via
    seeds.index(seed) to test_fold so all test folds are strictly non-overlapping
    with unique test_sha256 fingerprints.
    """
    n_folds = max(2, len(seeds))
    fam = bench_df["source_family"].astype(str).values
    fams = pd.Series(fam).value_counts()
    total_n = len(bench_df)

    # Sub-partition oversized families (> 30% of rules)
    group_keys = []
    for _, row in bench_df.iterrows():
        f = str(row["source_family"])
        if fams[f] / max(1, total_n) > 0.30:
            h = int(hashlib.md5(str(row["rule_id"]).encode()).hexdigest(), 16) % n_folds
            group_keys.append(f"{f}__sub{h}")
        else:
            group_keys.append(f)

    group_series = pd.Series(group_keys)
    g_counts = group_series.value_counts()

    # Deterministic greedy balancing
    load = [0] * n_folds
    fold_of = {}
    for g in g_counts.index:
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += int(g_counts[g])

    folds = np.array([fold_of[g] for g in group_keys])

    # Bijective mapping from seed index to test_fold
    seed_idx = seeds.index(seed) if seed in seeds else (seed % n_folds)
    test_fold = seed_idx % n_folds

    te_mask = (folds == test_fold)
    tr = bench_df[~te_mask].reset_index(drop=True)
    te = bench_df[te_mask].reset_index(drop=True)
    # Closed-set filter by construction
    te = te[te["attack_id"].isin(set(tr["attack_id"]))].reset_index(drop=True)

    manifest = {
        "seed": seed,
        "train_n": int(len(tr)),
        "test_n": int(len(te)),
        "n_folds": int(n_folds),
        "test_fold": int(test_fold),
        "test_ids": sorted(te["sample_id"].tolist()),
        "train_sha256": hashlib.sha256(",".join(sorted(tr["sample_id"])).encode()).hexdigest()[:16],
        "test_sha256": hashlib.sha256(",".join(sorted(te["sample_id"])).encode()).hexdigest()[:16]
    }
    return tr, te, manifest
