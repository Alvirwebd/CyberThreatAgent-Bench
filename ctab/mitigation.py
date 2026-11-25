"""
ctab.mitigation
===============
MITRE ATT&CK STIX Course-of-Action (Mitigation) Recommender and NDCG@k Evaluator.
Maps techniques into valid M1xxx mitigation ID space using authentic STIX 'mitigates' relationships.
Reproduces Stage 9 of Project2-CyberThreatAgent_Bench.ipynb.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT_DIR / "ctab_bench" / "snapshots"

# Tactic playbook as fallback / supplementary guidance
PLAYBOOK = {
    "reconnaissance": [("Harden external footprint", "M1056"), ("Rate-limit public endpoints", "M1037")],
    "resource-development": [("Monitor for brand impersonation", "M1056"), ("Takedown domain lookalikes", "M1021")],
    "initial-access": [("Enforce phishing-resistant MFA", "M1032"), ("Block macro execution from downloads", "M1042")],
    "execution": [("Constrain PowerShell to Constrained Language Mode", "M1042"), ("Block unsigned binaries via AppLocker", "M1038")],
    "persistence": [("Audit Run keys and scheduled tasks weekly", "M1047"), ("Alert on new services", "M1047")],
    "privilege-escalation": [("Patch local privilege-escalation CVEs on a 7-day SLA", "M1051"), ("Remove unnecessary admin rights", "M1026")],
    "defense-evasion": [("Enable tamper protection on EDR", "M1040"), ("Centralise audit logs off-host", "M1047")],
    "credential-access": [("Enable Credential Guard / LSASS protection", "M1043"), ("Tier admin accounts", "M1026")],
    "discovery": [("Restrict directory enumeration to authorised tooling", "M1018")],
    "lateral-movement": [("Segment management networks", "M1030"), ("Restrict RDP to jump hosts", "M1035")],
    "collection": [("Apply DLP to archive and staging paths", "M1057")],
    "command-and-control": [("Egress filtering with domain allowlists", "M1037"), ("Sinkhole known C2 domains", "M1037")],
    "exfiltration": [("Cap outbound volume per host", "M1037"), ("Alert on unusual TLS destinations", "M1037")],
    "impact": [("Immutable offline backups", "M1053"), ("Test ransomware recovery quarterly", "M1053")],
    "unknown": [("Escalate to a human analyst for scoping", "M1056")],
}

class MitigationEngine:
    def __init__(self, attack_path=None, attack_release="16.1"):
        if attack_path is None:
            attack_path = SNAPSHOT_DIR / "attack" / f"enterprise-attack-{attack_release}.json"
        self.attack_path = Path(attack_path)
        self._load()

    def _load(self):
        with open(self.attack_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        objects = bundle.get("objects", [])

        self.mit_id_by_stix = {}
        self.mit_name_by_id = {}
        for o in objects:
            if o.get("type") == "course-of-action" and not o.get("revoked"):
                for ref in o.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        mid = ref.get("external_id")
                        if mid:
                            self.mit_id_by_stix[o["id"]] = mid
                            self.mit_name_by_id[mid] = o.get("name") or mid

        self.tech_stix_by_aid = {}
        self.tech_tactics = {}
        for o in objects:
            if o.get("type") == "attack-pattern" and not o.get("revoked"):
                for ref in o.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        aid = ref.get("external_id")
                        if aid:
                            self.tech_stix_by_aid[aid] = o["id"]
                            phases = [p["phase_name"] for p in (o.get("kill_chain_phases") or [])
                                      if p.get("kill_chain_name") == "mitre-attack"]
                            self.tech_tactics[aid] = phases[0] if phases else "unknown"

        self.tech_mitigated_by = defaultdict(set)
        for rel in objects:
            if rel.get("type") == "relationship" and rel.get("relationship_type") == "mitigates":
                src = rel.get("source_ref")
                tgt = rel.get("target_ref")
                if src in self.mit_id_by_stix:
                    self.tech_mitigated_by[tgt].add(self.mit_id_by_stix[src])

        # All valid mitigations — strictly M1xxx only (filter out deprecated T1xxx course-of-action objects)
        self.valid_mitigations = {mid for mid in self.mit_id_by_stix.values() if mid.startswith("M")}

    def gold_mitigation_ids(self, attack_id):
        """Return ONLY authentic MITRE ATT&CK STIX M1xxx mitigation IDs (zero PLAYBOOK padding)."""
        if not attack_id:
            return []
        stix = self.tech_stix_by_aid.get(attack_id)
        if not stix and "." in attack_id:
            stix = self.tech_stix_by_aid.get(attack_id.split(".")[0])
        if not stix:
            return []
        mits = {m for m in self.tech_mitigated_by.get(stix, set()) if m.startswith("M")}
        if not mits and "." in attack_id:
            parent_stix = self.tech_stix_by_aid.get(attack_id.split(".")[0])
            if parent_stix:
                mits = {m for m in self.tech_mitigated_by.get(parent_stix, set()) if m.startswith("M")}
        return sorted(mits)

    def gold_mitigation_counts(self, gold_techniques):
        """
        Graded relevance map exclusively from authentic MITRE ATT&CK STIX 'mitigates' edges
        (ZERO PLAYBOOK padding in gold):
            rel(m) = |{g in gold_techniques : m directly mitigates g in STIX}|
        """
        gold_mit_rel = defaultdict(int)
        for g in (gold_techniques or []):
            for m in self.gold_mitigation_ids(g):
                gold_mit_rel[m] += 1
        return dict(gold_mit_rel)

    def recommend_mitigations(self, attack_id, top_k=3):
        if not attack_id:
            return [mid for _, mid in PLAYBOOK["unknown"][:top_k]]
        stix = self.tech_stix_by_aid.get(attack_id)
        if not stix and "." in attack_id:
            stix = self.tech_stix_by_aid.get(attack_id.split(".")[0])
        stix_mits = sorted(m for m in self.tech_mitigated_by.get(stix, set()) if m.startswith("M"))
        out = list(stix_mits)
        if len(out) < top_k:
            tactic = self.tech_tactics.get(attack_id, "unknown")
            for _, mid in PLAYBOOK.get(tactic, PLAYBOOK["unknown"]):
                if mid not in out:
                    out.append(mid)
        return out[:top_k]

    def compute_mitigation_ndcg_from_ids(self, pred_mits, gold_techniques, k=3):
        """
        Strict STIX-only graded nDCG@k given predicted mitigation IDs and gold_techniques.
        Returns None when gold_techniques is empty (benign) or has zero STIX mitigates edges
        (never returns 1.0 on empty gold!).
        """
        if not gold_techniques:
            return None
        gold_mit_rel = self.gold_mitigation_counts(gold_techniques)
        if not gold_mit_rel:
            return None

        eval_preds = []
        for m in (pred_mits or []):
            if m and m not in eval_preds:
                eval_preds.append(m)
            if len(eval_preds) >= k:
                break
        if not eval_preds:
            return 0.0

        gains = [gold_mit_rel.get(m, 0) for m in eval_preds]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal_gains = sorted(gold_mit_rel.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
        return min(1.0, (dcg / idcg) if idcg > 0 else 0.0)

    def compute_mitigation_ndcg(self, pred_techniques, gold_techniques, k=3):
        """
        Strict STIX-only graded nDCG@k from predicted techniques.
        Returns None when gold_techniques is empty or has zero STIX mitigates edges.
        """
        if not gold_techniques:
            return None
        gold_mit_rel = self.gold_mitigation_counts(gold_techniques)
        if not gold_mit_rel:
            return None

        pred_mits = []
        for p in (pred_techniques or []):
            for m in self.recommend_mitigations(p, top_k=k):
                if m not in pred_mits:
                    pred_mits.append(m)
        return self.compute_mitigation_ndcg_from_ids(pred_mits, gold_techniques, k=k)

_DEFAULT_ENGINE = None

def get_mitigation_engine():
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = MitigationEngine()
    return _DEFAULT_ENGINE
