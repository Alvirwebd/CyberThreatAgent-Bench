#!/usr/bin/env python3
"""
Autonomous ReAct Tool-Calling Agent Baseline vs. Static & Cascaded Pipelines
CyberThreatAgent-Bench (Phase 4)
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path("ctab_bench/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "react_agent_cache.jsonl"
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR = OUT_DIR / "latex"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# 8 Structured Tool Schemas
TOOL_SCHEMAS = [
    {
        "name": "ingest_alert",
        "description": "Validates schema of raw security log and extracts timestamp, host, and clean text.",
        "parameters": {"type": "object", "properties": {"raw_log": {"type": "string"}}, "required": ["raw_log"]}
    },
    {
        "name": "extract_iocs",
        "description": "Extracts refanged IOC indicators (IPv4, domain, SHA-256, MD5, process, registry).",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    },
    {
        "name": "map_attack_technique",
        "description": "Maps alert text to MITRE ATT&CK technique with confidence-gated abstention for benign telemetry.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    },
    {
        "name": "attribute_threat_actor",
        "description": "Attributes candidate threat actors based on observed ATT&CK technique IDs.",
        "parameters": {"type": "object", "properties": {"technique_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["technique_ids"]}
    },
    {
        "name": "retrieve_cves",
        "description": "Normalizes CVE IDs mentioned in alert and retrieves CVSS scores.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    },
    {
        "name": "build_attack_graph",
        "description": "Constructs temporal-causal attack graph linking technique nodes by host and shared IOCs.",
        "parameters": {"type": "object", "properties": {"alerts": {"type": "array", "items": {"type": "object"}}}, "required": ["alerts"]}
    },
    {
        "name": "score_incident_risk",
        "description": "Computes calibrated multi-factor risk score (0-100) and severity rating.",
        "parameters": {"type": "object", "properties": {"cvss": {"type": "number"}, "actor_score": {"type": "number"}, "stage_score": {"type": "number"}, "graph_score": {"type": "number"}, "confidence": {"type": "number"}}, "required": []}
    },
    {
        "name": "recommend_mitigations",
        "description": "Retrieves ranked MITRE mitigations for confirmed ATT&CK technique IDs.",
        "parameters": {"type": "object", "properties": {"technique_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["technique_ids"]}
    }
]

print("Agentic benchmark harness loaded.")
