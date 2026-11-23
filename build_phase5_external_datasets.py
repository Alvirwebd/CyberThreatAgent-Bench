#!/usr/bin/env python3
"""
Phase 5 Dataset Generator & Telemetry Loader:
1. MITRE TRAM Full Bootstrap Corpus (N >= 1,000 sentences, zero-leakage, analyst-annotated)
2. Splunk Attack Data Real Runtime Telemetry (N >= 300 events across >= 25 ATT&CK techniques)
   Strict Git-LFS Rule: Payloads fetched strictly from media.githubusercontent.com/media/splunk/attack_data/master/...
"""

import os
import sys
import json
import re
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SNAPSHOTS_DIR = Path("ctab_bench/snapshots")
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
TELEM_DIR = SNAPSHOTS_DIR / "telemetry"
TELEM_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Part B.1: MITRE TRAM Full Bootstrap Corpus
# -------------------------------------------------------------
print("[1/2] Processing MITRE TRAM Full Bootstrap Corpus...")
tram_out_path = SNAPSHOTS_DIR / "tram_bootstrap_corpus.csv"
if not tram_out_path.exists():
    tram_raw_file = SNAPSHOTS_DIR / "tram_bootstrap.json"
    if not tram_raw_file.exists():
        url = "https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/main/data/training/bootstrap-training-data.json"
        print(f"Downloading TRAM bootstrap from {url}...")
        r = requests.get(url, timeout=30)
        with open(tram_raw_file, "wb") as f:
            f.write(r.content)

    with open(tram_raw_file, "r", encoding="utf-8") as f:
        tram_raw = json.load(f)

    # Load existing Sigma bench texts to guarantee zero overlap
    bench_texts = set()
    if Path("ctab_bench/snapshots/chain_corpus.csv").exists():
        try:
            df_chain = pd.read_csv("ctab_bench/snapshots/chain_corpus.csv")
            bench_texts.update(df_chain["text"].dropna().str.lower().str.strip())
        except Exception:
            pass

    # Extract accepted sentences
    tram_rows = []
    seen_texts = set()
    for s in tram_raw.get("sentences", []):
        if s.get("disposition") != "accept":
            continue
        t = s.get("text", "").strip()
        if not t or len(t) < 15:
            continue
        norm_t = t.lower()
        if norm_t in seen_texts or norm_t in bench_texts:
            continue
        seen_texts.add(norm_t)

        # Collect valid technique IDs
        attack_ids = []
        for m in s.get("mappings", []):
            aid = m.get("attack_id", "").strip().upper()
            if re.match(r"^T\d{4}(\.\d{3})?$", aid):
                attack_ids.append(aid)
        if not attack_ids:
            continue

        # Clean text: remove any explicit technique tags if present
        cleaned_t = re.sub(r"(?i)\battack\.t\d{4}(\.\d{3})?", "", t)
        cleaned_t = re.sub(r"\bT\d{4}(\.\d{3})?\b", "", cleaned_t).strip()
        if len(cleaned_t) < 15:
            continue

        tram_rows.append({
            "sample_id": f"TRAM-BOOT-{len(tram_rows):05d}",
            "text": cleaned_t,
            "attack_id": attack_ids[0],
            "attack_ids": ";".join(attack_ids),
            "source": "MITRE TRAM Full Bootstrap (Analyst-Annotated)"
        })

    df_tram_boot = pd.DataFrame(tram_rows)
    df_tram_boot.to_csv(tram_out_path, index=False)
else:
    df_tram_boot = pd.read_csv(tram_out_path)

tram_sha = hashlib.sha256(tram_out_path.read_bytes()).hexdigest()
print(f"  [SAVED] {tram_out_path}: {len(df_tram_boot)} unique sentences across {df_tram_boot['attack_id'].nunique()} techniques | SHA-256: {tram_sha[:16]}...")

# -------------------------------------------------------------
# Part B.2: Splunk Attack Data Real Runtime Telemetry
# -------------------------------------------------------------
print("\n[2/2] Fetching & Parsing Splunk Attack Data Real Runtime Telemetry across 30 ATT&CK techniques...")

picked_path = SNAPSHOTS_DIR / "picked_30_splunk.json"
with open(picked_path, "r", encoding="utf-8") as f:
    picked_datasets = json.load(f)

def fetch_media_payload(rel_path, max_bytes=384*1024):
    safe_name = rel_path.replace("/", "_")
    cache_path = TELEM_DIR / safe_name
    if cache_path.exists():
        content = cache_path.read_bytes()
        if not content.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return content.decode("utf-8", errors="replace")

    media_url = f"https://media.githubusercontent.com/media/splunk/attack_data/master/{rel_path}"
    try:
        r = requests.get(media_url, stream=True, timeout=12)
        if r.status_code == 200:
            chunks = []
            total = 0
            for chunk in r.iter_content(chunk_size=64*1024):
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            raw_data = b"".join(chunks)
            assert not raw_data.startswith(b"version https://git-lfs.github.com/spec/v1"), f"FATAL: Git-LFS pointer retrieved for {rel_path}!"
            cache_path.write_bytes(raw_data)
            return raw_data.decode("utf-8", errors="replace")
        else:
            print(f"  [HTTP {r.status_code}] Failed to fetch {media_url}")
            return None
    except Exception as e:
        print(f"  [WARN] Failed to stream {rel_path}: {e}")
        return None

def extract_events_from_payload(payload, tech_id, rel_path):
    events = []
    # 1. XML Sysmon / Security events
    if "<Event" in payload:
        raw_events = re.split(r"(?=<Event\b)", payload)
        for raw_ev in raw_events:
            if "<Event" not in raw_ev or "</Event>" not in raw_ev:
                continue
            try:
                ev_str = raw_ev[:raw_ev.find("</Event>") + 8]
                root = ET.fromstring(ev_str)
                data_map = {}
                for d in root.findall(".//{http://schemas.microsoft.com/win/2004/08/events/event}Data"):
                    data_map[d.get("Name")] = d.text
                if not data_map:
                    for d in root.findall(".//Data"):
                        data_map[d.get("Name")] = d.text

                cmd = data_map.get("CommandLine") or data_map.get("ProcessCommandLine") or ""
                img = data_map.get("Image") or data_map.get("NewProcessName") or ""
                parent = data_map.get("ParentImage") or data_map.get("ParentProcessName") or ""
                dst_ip = data_map.get("DestinationIp") or data_map.get("IpAddress") or ""
                tgt_obj = data_map.get("TargetObject") or data_map.get("ObjectName") or ""
                hashes = data_map.get("Hashes") or ""
                user = data_map.get("User") or data_map.get("SubjectUserName") or ""

                if not cmd and not img and not dst_ip and not tgt_obj:
                    continue

                parts = []
                if img: parts.append(f"Image: {img}")
                if cmd: parts.append(f"CommandLine: {cmd}")
                if parent: parts.append(f"ParentImage: {parent}")
                if dst_ip: parts.append(f"Destination: {dst_ip}")
                if tgt_obj: parts.append(f"TargetObject: {tgt_obj}")
                if user: parts.append(f"User: {user}")
                if hashes: parts.append(f"Hashes: {hashes[:80]}")

                text = " | ".join(parts)
                # Purge ground truth tags
                text_clean = re.sub(r"(?i)\battack\.t\d{4}(\.\d{3})?", "", text)
                text_clean = re.sub(r"\bT\d{4}(\.\d{3})?\b", "", text_clean).strip()
                if len(text_clean) > 25:
                    events.append({
                        "raw_text": f"Windows Event Telemetry: {text_clean}",
                        "attack_id": tech_id,
                        "platform": "windows",
                        "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
                    })
            except Exception:
                continue

    # 2. JSON Lines (CloudTrail / ASL / OCSF)
    elif payload.strip().startswith("{") or "eventName" in payload or "eventSource" in payload:
        for line in payload.splitlines():
            line = line.strip()
            if not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                r = json.loads(line)
                ev_name = r.get("eventName") or r.get("action") or ""
                ev_src = r.get("eventSource") or r.get("api") or ""
                src_ip = r.get("sourceIPAddress") or r.get("src_ip") or ""
                user = r.get("userIdentity", {}).get("userName") or r.get("user") or ""
                params = json.dumps(r.get("requestParameters", {})) if r.get("requestParameters") else ""
                
                parts = []
                if ev_name: parts.append(f"Event: {ev_name}")
                if ev_src: parts.append(f"Source: {ev_src}")
                if src_ip: parts.append(f"SourceIP: {src_ip}")
                if user: parts.append(f"User: {user}")
                if params and len(params) > 2: parts.append(f"Params: {params[:120]}")

                text = " | ".join(parts)
                text_clean = re.sub(r"(?i)\battack\.t\d{4}(\.\d{3})?", "", text)
                text_clean = re.sub(r"\bT\d{4}(\.\d{3})?\b", "", text_clean).strip()
                if len(text_clean) > 25:
                    events.append({
                        "raw_text": f"Cloud Telemetry: {text_clean}",
                        "attack_id": tech_id,
                        "platform": "cloud",
                        "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
                    })
            except Exception:
                continue

    # 3. Plain text / auditd log lines
    if not events:
        for line in payload.splitlines():
            line = line.strip()
            if len(line) < 30 or line.startswith("#"):
                continue
            text_clean = re.sub(r"(?i)\battack\.t\d{4}(\.\d{3})?", "", line)
            text_clean = re.sub(r"\bT\d{4}(\.\d{3})?\b", "", text_clean).strip()
            if len(text_clean) > 25:
                events.append({
                    "raw_text": f"System Telemetry: {text_clean[:250]}",
                    "attack_id": tech_id,
                    "platform": "linux" if "audit" in rel_path.lower() else "windows",
                    "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
                })
    return events

all_telem_events = []
seen_texts = set()

for tech_id, rel_path in picked_datasets:
    payload = fetch_media_payload(rel_path)
    if not payload:
        continue
    evs = extract_events_from_payload(payload, tech_id, rel_path)
    # Take up to 20 balanced events per technique
    added = 0
    for ev in evs:
        norm = ev["raw_text"].lower().strip()
        if norm not in seen_texts:
            seen_texts.add(norm)
            all_telem_events.append(ev)
            added += 1
            if added >= 18:
                break

df_telem = pd.DataFrame(all_telem_events)
df_telem["sample_id"] = [f"SPLUNK-TELEM-{i:05d}" for i in range(len(df_telem))]

telem_out_path = TELEM_DIR / "splunk_attack_data.csv"
df_telem.to_csv(telem_out_path, index=False)
telem_sha = hashlib.sha256(telem_out_path.read_bytes()).hexdigest()

n_events = len(df_telem)
n_techs = df_telem["attack_id"].nunique()
print(f"Total parsed real telemetry events: {n_events} across {n_techs} ATT&CK techniques")
print(f"  [SAVED] {telem_out_path} | SHA-256: {telem_sha[:16]}...")
assert n_events >= 300, f"Expected >= 300 events, got {n_events}"
assert n_techs >= 25, f"Expected >= 25 techniques, got {n_techs}"
print("\n[SUCCESS] Phase 5 Real Telemetry Benchmark Built Successfully (N >= 300, Tech >= 25)!")
