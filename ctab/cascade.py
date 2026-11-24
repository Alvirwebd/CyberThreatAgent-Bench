"""
ctab.cascade
============
Margin-Gated Confidence Cascade and Selective Escalation Router.
"""

import numpy as np

class MarginGatedRouter:
    """
    Evaluates top-1/top-2 prediction margins against learned confidence thresholds.
    Escalates uncertain alerts to higher-tier models while maintaining low benign false escalation.
    """
    def __init__(self, margin_threshold=0.10, tau_abstain=0.12):
        self.margin_threshold = float(margin_threshold)
        self.tau_abstain = float(tau_abstain)

    def route(self, ranked_predictions, confidence_score=0.5):
        """
        Determines routing decision for an alert:
        - 'abstain': if confidence < tau_abstain or no valid candidate
        - 'escalate': if (top1 - top2) < margin_threshold
        - 'accept_local': local top1 accepted with high confidence
        """
        if not ranked_predictions or confidence_score < self.tau_abstain:
            return {"decision": "abstain", "candidate": None, "margin": 0.0}

        top1 = ranked_predictions[0]
        if top1.get("attack_id") is None:
            return {"decision": "abstain", "candidate": None, "margin": 0.0}

        top2 = ranked_predictions[1] if len(ranked_predictions) > 1 else None
        margin = float(top1.get("score", 0.0) - (top2.get("score", 0.0) if top2 else 0.0))

        if margin < self.margin_threshold:
            return {"decision": "escalate", "candidate": top1.get("attack_id"), "margin": round(margin, 4)}

        return {"decision": "accept_local", "candidate": top1.get("attack_id"), "margin": round(margin, 4)}

class HybridLLMCascade:
    """
    Cascaded execution wrapper combining local neuro-symbolic mapper and cloud LLM fallback.
    """
    def __init__(self, local_mapper, router=None, llm_fn=None):
        self.local_mapper = local_mapper
        self.router = router or MarginGatedRouter()
        self.llm_fn = llm_fn

    def predict(self, text):
        ranked = self.local_mapper.ranked(text, top_k=2, allow_abstain=True)
        conf = self.local_mapper.confidence(text) if hasattr(self.local_mapper, "confidence") else 0.5
        decision = self.router.route(ranked, confidence_score=conf)

        if decision["decision"] == "escalate" and self.llm_fn is not None:
            llm_pred = self.llm_fn(text)
            return llm_pred or decision["candidate"]
        elif decision["decision"] == "abstain":
            return None
        return decision["candidate"]
