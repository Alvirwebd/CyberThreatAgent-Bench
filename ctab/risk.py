"""
ctab.risk
=========
Stage 8 Learned Calibration and Risk Scoring.
"""

import numpy as np
from scipy.optimize import lsq_linear

BASE_WEIGHTS_V5 = {
    "cvss": 0.40,
    "actor": 0.25,
    "stage": 0.20,
    "graph": 0.15,
    "conf": 0.00,
}

def normalize_cvss(val):
    if val is None:
        return 0.0
    try:
        return float(np.clip(float(val) / 10.0, 0.0, 1.0))
    except Exception:
        return 0.0

def compute_risk_score(components, weights=None):
    weights = weights or BASE_WEIGHTS_V5
    available = {}
    for k, w in weights.items():
        v = components.get(k)
        if v is None:
            continue
        try:
            available[k] = (w, float(np.clip(v, 0.0, 1.0)))
        except Exception:
            continue

    if not available:
        return {
            "risk_score": 0.0,
            "risk_level": "Low",
            "components_used": [],
            "weight_coverage": 0.0,
        }

    total_w = sum(w for w, _ in available.values())
    raw = sum(w * v for w, v in available.values()) / total_w if total_w > 0 else 0.0
    score = round(raw * 100.0, 2)
    level = "Low" if score < 25 else "Medium" if score < 50 else "High" if score < 75 else "Critical"

    return {
        "risk_score": score,
        "risk_level": level,
        "components_used": sorted(available.keys()),
        "weight_coverage": round(total_w / max(1e-6, sum(weights.values())), 4),
    }

def fit_chain_risk_weights(X, y, bounds=(0.05, 0.45)):
    """
    Fits non-negative bounded component weights strictly on training folds.
    """
    X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0)
    y_arr = np.nan_to_num(np.array(y, dtype=np.float64), nan=0.0)
    res = lsq_linear(X_arr, y_arr, bounds=bounds)
    total = res.x.sum()
    w_norm = res.x / total if total > 0 else np.array([0.20] * len(res.x))
    keys = ["cvss", "actor", "stage", "graph", "conf"][:len(w_norm)]
    return {k: round(float(v), 4) for k, v in zip(keys, w_norm)}
