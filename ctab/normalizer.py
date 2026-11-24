"""
ctab.normalizer
===============
Stage 1 alert ingestion, de-duplication, schema validation, and normalization.
"""

import re
import json
from datetime import datetime

REQUIRED_FIELDS = ("alert_id", "timestamp", "raw_text")

def normalize_alert(record):
    """
    Validates schema, strips noise, parses ISO timestamp, and normalizes defanged indicators.
    """
    for field in REQUIRED_FIELDS:
        if field not in record or not record[field]:
            return None, f"missing:{field}"

    ts = str(record["timestamp"]).strip()
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        iso_ts = dt.isoformat()
    except Exception:
        return None, "bad_timestamp"

    raw = str(record["raw_text"]).strip()
    if not raw:
        return None, "missing:raw_text"

    # Normalize defanged indicators: hxxp -> http, [.] -> .
    clean_text = raw.replace("hxxp://", "http://").replace("hxxps://", "https://").replace("[.]", ".")

    normalized = dict(record)
    normalized["timestamp"] = iso_ts
    normalized["raw_text_clean"] = clean_text
    return normalized, "ok"

def ingest_alerts(raw_records):
    """
    Ingests and de-duplicates alerts. Returns (accepted, summary_metrics).
    """
    accepted = []
    seen_ids = set()
    reject_reasons = {}
    dups = 0

    for r in raw_records:
        norm, status = normalize_alert(r)
        if status != "ok":
            reject_reasons[status] = reject_reasons.get(status, 0) + 1
            continue
        aid = norm["alert_id"]
        if aid in seen_ids:
            dups += 1
            continue
        seen_ids.add(aid)
        accepted.append(norm)

    total = len(raw_records)
    summary = {
        "rows_in": total,
        "rows_accepted": len(accepted),
        "rows_rejected": sum(reject_reasons.values()),
        "duplicates_removed": dups,
        "schema_valid_rate": round(len(accepted) / total, 4) if total > 0 else 0.0,
        "reject_reasons": reject_reasons,
    }
    return accepted, summary
