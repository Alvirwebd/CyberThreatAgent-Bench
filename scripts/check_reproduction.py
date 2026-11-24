"""
scripts/check_reproduction.py
=============================
Standalone verification script importing ONLY from ctab.
Verifies HierarchicalMapper reproduces:
  - seed=7 s3_hybrid_Top1_Exact = 0.4128
  - 5-seed (7, 21, 42, 84, 123) mean s3_hybrid_Top1_Exact = 0.3890
"""

import time
import numpy as np
import pandas as pd
from ctab import load_attack, get_canonical_benchmark, group_split, HierarchicalMapper, evaluate_mapping

def run_seed(df_techniques, bench_df, seeds, seed):
    tr, te, manifest = group_split(bench_df, seeds=seeds, seed=seed)
    t0 = time.perf_counter()
    mapper = HierarchicalMapper(df_techniques, tr, alpha=0.65, C=8.0, auto_tune_alpha=True)
    fit_time = time.perf_counter() - t0
    
    fn = lambda text, top_k=5: mapper.ranked(text, top_k=top_k)
    met, rows = evaluate_mapping(te, fn)
    top1 = met["Top1_Exact"]
    print(f"seed {seed:3d}: train={len(tr)}, test={len(te)}, alpha*={mapper.best_alpha:.2f}, fit={fit_time:.2f}s, Top1_Exact={top1:.4f}")
    return top1

def main():
    print("=" * 60)
    print("REPRODUCTION VERIFICATION: Standalone ctab HierarchicalMapper")
    print("=" * 60)
    
    df_techniques, _, _, _ = load_attack()
    bench_df = get_canonical_benchmark()
    seeds = (7, 21, 42, 84, 123)
    
    scores = {}
    for s in seeds:
        scores[s] = run_seed(df_techniques, bench_df, seeds, s)
        if s == 7:
            assert scores[7] == 0.4128, f"Seed 7 mismatch! Expected 0.4128, got {scores[7]}"
            print("  --> SEED 7 VERIFIED: EXACT MATCH 0.4128")
    
    mean_score = round(float(np.mean(list(scores.values()))), 4)
    print("=" * 60)
    print(f"5-Seed Scores: {scores}")
    print(f"5-Seed Mean  : {mean_score:.4f} (target: 0.3890)")
    assert mean_score == 0.3890, f"5-seed mean mismatch! Expected 0.3890, got {mean_score}"
    print("  --> 5-SEED MEAN VERIFIED: EXACT MATCH 0.3890!")
    print("=" * 60)

if __name__ == "__main__":
    main()
