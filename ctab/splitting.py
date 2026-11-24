"""
ctab.splitting
==============
Deterministic hashed dataset splitting and OOD holdout partitioning.
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
