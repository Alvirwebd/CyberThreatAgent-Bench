"""
CyberThreatAgent-Bench (CTAB) Core Framework
============================================
A standardized, multi-regime evaluation benchmark and neuro-symbolic
architecture for autonomous cybersecurity incident triage.
"""

__version__ = "1.0.0"

from .normalizer import ingest_alerts, normalize_alert
from .dataset import load_attack, load_sigma_rules, build_benchmark, get_canonical_benchmark
from .splitting import deterministic_split, group_split
from .mappers import (
    HierarchicalMapper,
    SecBERTHybridMapper,
    ZeroShotMapper,
    BM25Mapper,
    DenseRetrieverMapper,
    SupervisedDenseMapper,
    HybridDenseRouter,
    fit_mappers,
    evaluate_mapping,
    technique_match,
)
from .cascade import MarginGatedRouter, HybridLLMCascade
from .graph import build_attack_graph, compute_graph_metrics
from .risk import compute_risk_score, fit_chain_risk_weights
from .agent import AgentToolDispatcher, PlanAndExecuteAgent, AutonomousReActAgent

__all__ = [
    "__version__",
    "ingest_alerts",
    "normalize_alert",
    "load_attack",
    "load_sigma_rules",
    "build_benchmark",
    "get_canonical_benchmark",
    "deterministic_split",
    "group_split",
    "HierarchicalMapper",
    "SecBERTHybridMapper",
    "ZeroShotMapper",
    "BM25Mapper",
    "DenseRetrieverMapper",
    "SupervisedDenseMapper",
    "HybridDenseRouter",
    "fit_mappers",
    "evaluate_mapping",
    "technique_match",
    "MarginGatedRouter",
    "HybridLLMCascade",
    "build_attack_graph",
    "compute_graph_metrics",
    "compute_risk_score",
    "fit_chain_risk_weights",
    "AgentToolDispatcher",
    "PlanAndExecuteAgent",
    "AutonomousReActAgent",
]
