"""
CyberThreatAgent-Bench (CTAB) Core Framework
============================================
A standardized, multi-regime evaluation benchmark and neuro-symbolic
architecture for autonomous cybersecurity incident triage.
"""

__version__ = "1.0.0"

from .normalizer import ingest_alerts, normalize_alert
from .splitting import deterministic_split
from .mappers import HierarchicalMapper, SecBERTHybridMapper
from .cascade import MarginGatedRouter, HybridLLMCascade
from .graph import build_attack_graph, compute_graph_metrics
from .risk import compute_risk_score, fit_chain_risk_weights
from .agent import AgentToolDispatcher, PlanAndExecuteAgent, AutonomousReActAgent

__all__ = [
    "__version__",
    "ingest_alerts",
    "normalize_alert",
    "deterministic_split",
    "HierarchicalMapper",
    "SecBERTHybridMapper",
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
