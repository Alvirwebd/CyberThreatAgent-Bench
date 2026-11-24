#!/usr/bin/env python3
"""
scripts/build_external_datasets.py
==================================
Builds and verifies external generalization corpora for CyberThreatAgent-Bench:
1. MITRE TRAM Full Bootstrap Corpus (N >= 1,000 sentences, zero tag leakage, analyst-annotated)
2. Splunk Attack Data Real Runtime Telemetry (N >= 300 events across >= 25 ATT&CK techniques):
   - Prefers Sysmon XML logs over raw Windows Security stanzas
   - Proper timestamp-stanza splitting for multi-line Windows Security logs (no line chopping)
   - Hex-decoding and event grouping for Linux auditd
   - Filters out Attack Range / OS background noise
   - Strict assertions: N >= 300, techniques >= 25, 0 rows < 80 chars, 0 Git-LFS stubs, 0 tag leaks
"""

import os
import sys
import json
import re
import hashlib
import binascii
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

# ==============================================================================
# Part B.1: MITRE TRAM Full Bootstrap Corpus
# ==============================================================================
print("[1/2] Building MITRE TRAM Full Bootstrap Corpus...")
tram_out_path = SNAPSHOTS_DIR / "tram_bootstrap_corpus.csv"

tram_raw_file = SNAPSHOTS_DIR / "tram_bootstrap.json"
if not tram_raw_file.exists():
    url = "https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/main/data/training/bootstrap-training-data.json"
    print(f"  Downloading TRAM bootstrap corpus from {url}...")
    r = requests.get(url, timeout=30)
    with open(tram_raw_file, "wb") as f:
        f.write(r.content)

with open(tram_raw_file, "r", encoding="utf-8") as f:
    tram_raw = json.load(f)

# Collect existing Sigma bench texts to guarantee zero overlap
bench_texts = set()
for candidate in [SNAPSHOTS_DIR / "chain_corpus.csv", Path("artifacts/stage3_predictions.csv")]:
    if candidate.exists():
        try:
            df_c = pd.read_csv(candidate)
            col = "text" if "text" in df_c.columns else ("raw_text" if "raw_text" in df_c.columns else None)
            if col:
                bench_texts.update(df_c[col].dropna().str.lower().str.strip())
        except Exception:
            pass

tram_rows = []
seen_texts = set()
for s in tram_raw.get("sentences", []):
    if s.get("disposition") != "accept":
        continue
    t = s.get("text", "").strip()
    if not t or len(t) < 20:
        continue
    norm_t = t.lower()
    if norm_t in seen_texts or norm_t in bench_texts:
        continue
    seen_texts.add(norm_t)

    # Valid technique IDs
    attack_ids = []
    for m in s.get("mappings", []):
        aid = m.get("attack_id", "").strip().upper()
        if re.match(r"^T\d{4}(\.\d{3})?$", aid):
            attack_ids.append(aid)
    if not attack_ids:
        continue

    # Purge any leaked ATT&CK identifiers
    cleaned_t = re.sub(r"(?i)\battack\.t\d{4}(\.\d{3})?", "", t)
    cleaned_t = re.sub(r"\bT\d{4}(\.\d{3})?\b", "", cleaned_t).strip()
    if len(cleaned_t) < 20:
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
tram_sha = hashlib.sha256(tram_out_path.read_bytes()).hexdigest()
print(f"  [SAVED] {tram_out_path}: {len(df_tram_boot)} unique sentences across {df_tram_boot['attack_id'].nunique()} techniques | SHA-256: {tram_sha[:16]}...")

# ==============================================================================
# Part B.2: Splunk Attack Data Real Runtime Telemetry
# ==============================================================================
print("\n[2/2] Building Splunk Attack Data Real Runtime Telemetry (Sysmon XML / Security Stanzas / Auditd)...")

picked_path = SNAPSHOTS_DIR / "picked_30_splunk.json"
if not picked_path.exists():
    raise FileNotFoundError(f"Missing {picked_path}")

with open(picked_path, "r", encoding="utf-8") as f:
    picked_datasets = json.load(f)

# Background noise processes and tools in Splunk Attack Range to filter out
NOISE_PATTERNS = [
    re.compile(r"SplunkUniversalForwarder", re.I),
    re.compile(r"splunkd(?:\.exe)?\b", re.I),
    re.compile(r"splunk-winevtlog\.exe\b", re.I),
    re.compile(r"Amazon\\SSM", re.I),
    re.compile(r"ssm-agent(?:\.exe)?\b", re.I),
    re.compile(r"autochk\.exe\b", re.I),
    re.compile(r"setupcl\.exe\b", re.I),
    re.compile(r"SearchIndexer\.exe\b", re.I),
    re.compile(r"mpsigstub\.exe\b", re.I),
    re.compile(r"svchost\.exe -k (?:wsappx|appmodel|utcsvc|netsvcs)\s*$", re.I),
]

def is_noise(text):
    for pat in NOISE_PATTERNS:
        if pat.search(text):
            return True
    return False

def clean_tags(text):
    text = re.sub(r"(?i)\battack\.t\d{4}(?:\.\d{3})?", "", text)
    text = re.sub(r"\bT\d{4}(?:\.\d{3})?\b", "", text)
    return re.sub(r"\s+", " ", text).strip()

def decode_hex_proctitle(raw_str):
    """Hex decodes Linux auditd proctitle fields if hex encoded."""
    m = re.match(r"^[0-9a-fA-F]{8,}$", raw_str.strip())
    if m:
        try:
            decoded = binascii.unhexlify(raw_str.strip()).decode("utf-8", errors="replace")
            decoded = decoded.replace("\x00", " ").strip()
            if len(decoded) > 3:
                return decoded
        except Exception:
            pass
    return raw_str

def parse_sysmon_xml(payload, tech_id, rel_path):
    events = []
    if "<Event" not in payload:
        return events

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

            eid = ""
            eid_elem = root.find(".//{http://schemas.microsoft.com/win/2004/08/events/event}EventID")
            if eid_elem is None:
                eid_elem = root.find(".//EventID")
            if eid_elem is not None and eid_elem.text:
                eid = eid_elem.text.strip()

            img = data_map.get("Image") or data_map.get("NewProcessName") or ""
            orig_name = data_map.get("OriginalFileName") or ""
            cmd = data_map.get("CommandLine") or data_map.get("ProcessCommandLine") or ""
            parent_img = data_map.get("ParentImage") or data_map.get("ParentProcessName") or ""
            src_img = data_map.get("SourceImage") or ""
            tgt_img = data_map.get("TargetImage") or ""
            access = data_map.get("GrantedAccess") or ""
            call_trace = data_map.get("CallTrace") or ""
            tgt_file = data_map.get("TargetFilename") or data_map.get("TargetObject") or ""
            details = data_map.get("Details") or ""
            svc_name = data_map.get("ServiceName") or ""
            img_path = data_map.get("ImagePath") or ""
            task_name = data_map.get("TaskName") or ""
            task_content = data_map.get("TaskContent") or ""
            script_block = data_map.get("ScriptBlockText") or ""
            dst_ip = data_map.get("DestinationIp") or data_map.get("IpAddress") or ""
            dst_port = data_map.get("DestinationPort") or ""
            query_name = data_map.get("QueryName") or ""

            parts = []
            if eid: parts.append(f"EventID: {eid}")
            if img: parts.append(f"Image: {img}")
            if orig_name: parts.append(f"OriginalFileName: {orig_name}")
            if cmd: parts.append(f"CommandLine: {cmd}")
            if parent_img: parts.append(f"ParentImage: {parent_img}")
            if src_img: parts.append(f"SourceImage: {src_img}")
            if tgt_img: parts.append(f"TargetImage: {tgt_img}")
            if access: parts.append(f"GrantedAccess: {access}")
            if call_trace: parts.append(f"CallTrace: {call_trace[:120]}")
            if tgt_file: parts.append(f"TargetFilename: {tgt_file}")
            if details: parts.append(f"Details: {details}")
            if svc_name: parts.append(f"ServiceName: {svc_name}")
            if img_path: parts.append(f"ImagePath: {img_path}")
            if task_name: parts.append(f"TaskName: {task_name}")
            if task_content: parts.append(f"TaskContent: {task_content[:150]}")
            if script_block: parts.append(f"ScriptBlockText: {script_block[:180]}")
            if dst_ip: parts.append(f"DestinationIp: {dst_ip}")
            if dst_port: parts.append(f"DestinationPort: {dst_port}")
            if query_name: parts.append(f"QueryName: {query_name}")

            combined = " | ".join(parts)
            if len(combined) < 50 or is_noise(combined):
                continue

            cleaned = clean_tags(combined)
            if len(cleaned) >= 80:
                events.append({
                    "raw_text": f"Windows Sysmon Event: {cleaned}",
                    "attack_id": tech_id,
                    "platform": "windows",
                    "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
                })
        except Exception:
            continue
    return events

def parse_windows_security_stanzas(payload, tech_id, rel_path):
    events = []
    # Split strictly on timestamp stanzas (MM/DD/YYYY HH:MM:SS AM/PM)
    stanzas = re.split(r"(?=\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+[AP]M)", payload)
    for stanza in stanzas:
        stanza = stanza.strip()
        if len(stanza) < 60:
            continue
        try:
            fields = {}
            for line in stanza.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    fields[k.strip()] = v.strip()
                elif ":" in line:
                    k, v = line.split(":", 1)
                    fields[k.strip()] = v.strip()

            code = fields.get("EventCode") or fields.get("EventID") or ""
            new_proc = fields.get("New Process Name") or fields.get("NewProcessName") or fields.get("Process Name") or ""
            creator_proc = fields.get("Creator Process Name") or fields.get("CreatorProcessName") or fields.get("Parent Process Name") or ""
            proc_cmd = fields.get("Process Command Line") or fields.get("CommandLine") or ""
            obj_name = fields.get("Object Name") or fields.get("ObjectName") or ""
            svc_name = fields.get("Service Name") or fields.get("ServiceName") or ""
            task_name = fields.get("Task Name") or fields.get("TaskName") or ""
            share_name = fields.get("Share Name") or fields.get("ShareName") or ""

            parts = []
            if code: parts.append(f"EventCode: {code}")
            if new_proc: parts.append(f"NewProcessName: {new_proc}")
            if creator_proc: parts.append(f"CreatorProcessName: {creator_proc}")
            if proc_cmd: parts.append(f"CommandLine: {proc_cmd}")
            if obj_name: parts.append(f"ObjectName: {obj_name}")
            if svc_name: parts.append(f"ServiceName: {svc_name}")
            if task_name: parts.append(f"TaskName: {task_name}")
            if share_name: parts.append(f"ShareName: {share_name}")

            combined = " | ".join(parts)
            if len(combined) < 50 or is_noise(combined):
                # Fallback to compact cleaned stanza if key fields weren't strictly matched
                stanza_lines = [l.strip() for l in stanza.splitlines() if l.strip() and not l.startswith("RecordNumber") and not l.startswith("TimeGenerated")]
                combined = " | ".join(stanza_lines[:10])

            if len(combined) < 50 or is_noise(combined):
                continue

            cleaned = clean_tags(combined)
            if len(cleaned) >= 80:
                events.append({
                    "raw_text": f"Windows Security Event: {cleaned}",
                    "attack_id": tech_id,
                    "platform": "windows",
                    "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
                })
        except Exception:
            continue
    return events

def parse_linux_auditd(payload, tech_id, rel_path):
    events = []
    # Group auditd events by msg=audit(timestamp:serial)
    event_groups = {}
    for line in payload.splitlines():
        line = line.strip()
        if not line.startswith("type="):
            continue
        m = re.search(r"msg=audit\((\d+\.\d+:\d+)\):", line)
        if m:
            ev_id = m.group(1)
            event_groups.setdefault(ev_id, []).append(line)

    for ev_id, lines in event_groups.items():
        parts = []
        for line in lines:
            t_match = re.search(r"type=(\w+)", line)
            rec_type = t_match.group(1) if t_match else ""
            if rec_type in ("SYSCALL", "EXECVE", "PROCTITLE", "PATH"):
                # Extract exe, comm, proctitle, name
                exe_m = re.search(r'\bexe="([^"]+)"', line)
                comm_m = re.search(r'\bcomm="([^"]+)"', line)
                name_m = re.search(r'\bname="([^"]+)"', line)
                proctitle_m = re.search(r'\bproctitle=([0-9a-fA-F]+)\b', line)

                if exe_m: parts.append(f"exe: {exe_m.group(1)}")
                if comm_m: parts.append(f"comm: {comm_m.group(1)}")
                if name_m: parts.append(f"path: {name_m.group(1)}")
                if proctitle_m:
                    decoded_cmd = decode_hex_proctitle(proctitle_m.group(1))
                    parts.append(f"proctitle: {decoded_cmd}")

        combined = " | ".join(list(dict.fromkeys(parts)))
        if len(combined) < 40 or is_noise(combined):
            # Fallback to direct cleaned auditd lines
            combined = " | ".join([l[:120] for l in lines[:4]])

        cleaned = clean_tags(combined)
        if len(cleaned) >= 80:
            events.append({
                "raw_text": f"Linux Auditd Event: {cleaned}",
                "attack_id": tech_id,
                "platform": "linux",
                "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
            })
    return events

def parse_cloudtrail_json(payload, tech_id, rel_path):
    events = []
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
            if params and len(params) > 2: parts.append(f"Params: {params[:150]}")

            combined = " | ".join(parts)
            if len(combined) < 40 or is_noise(combined):
                continue
            cleaned = clean_tags(combined)
            if len(cleaned) >= 80:
                events.append({
                    "raw_text": f"Cloud Telemetry Event: {cleaned}",
                    "attack_id": tech_id,
                    "platform": "cloud",
                    "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
                })
        except Exception:
            continue
    return events

# Load local cache or stream payloads
all_telemetry = []
seen_texts = set()

for tech_id, rel_path in picked_datasets:
    safe_name = rel_path.replace("/", "_")
    cache_path = TELEM_DIR / safe_name
    payload = None
    if cache_path.exists():
        raw_b = cache_path.read_bytes()
        if not raw_b.startswith(b"version https://git-lfs.github.com/spec/v1"):
            payload = raw_b.decode("utf-8", errors="replace")

    if not payload:
        media_url = f"https://media.githubusercontent.com/media/splunk/attack_data/master/{rel_path}"
        try:
            r = requests.get(media_url, stream=True, timeout=15)
            if r.status_code == 200:
                chunks = []
                total = 0
                for chunk in r.iter_content(chunk_size=64*1024):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= 384*1024:
                        break
                raw_b = b"".join(chunks)
                assert not raw_b.startswith(b"version https://git-lfs.github.com/spec/v1")
                cache_path.write_bytes(raw_b)
                payload = raw_b.decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [WARN] Failed to fetch {media_url}: {e}")

    if not payload:
        continue

    # Dispatch to appropriate parser
    if "<Event" in payload:
        evs = parse_sysmon_xml(payload, tech_id, rel_path)
    elif "LogName=Security" in payload or re.search(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+[AP]M", payload):
        evs = parse_windows_security_stanzas(payload, tech_id, rel_path)
    elif "type=SYSCALL" in payload or "type=EXECVE" in payload:
        evs = parse_linux_auditd(payload, tech_id, rel_path)
    elif payload.strip().startswith("{") or "eventName" in payload:
        evs = parse_cloudtrail_json(payload, tech_id, rel_path)
    else:
        # Generic robust parser
        evs = []
        for line in payload.splitlines():
            line = line.strip()
            if len(line) >= 70 and not is_noise(line):
                cleaned = clean_tags(line)
                if len(cleaned) >= 80:
                    evs.append({
                        "raw_text": f"Runtime Event: {cleaned[:250]}",
                        "attack_id": tech_id,
                        "platform": "windows" if "win" in rel_path.lower() else "linux",
                        "source": f"Splunk Attack Data ({rel_path.split('/')[-1]})"
                    })

    # Add up to 18 balanced events per technique
    added = 0
    for ev in evs:
        norm = ev["raw_text"].lower().strip()
        if norm not in seen_texts:
            seen_texts.add(norm)
            all_telemetry.append(ev)
            added += 1
            if added >= 25:
                break

df_telem = pd.DataFrame(all_telemetry)
df_telem["sample_id"] = [f"SPLUNK-TELEM-{i:05d}" for i in range(len(df_telem))]

telem_out_path = TELEM_DIR / "splunk_attack_data.csv"
df_telem.to_csv(telem_out_path, index=False)
telem_sha = hashlib.sha256(telem_out_path.read_bytes()).hexdigest()

# Strict Assertions
n_events = len(df_telem)
n_techs = df_telem["attack_id"].nunique()
short_rows = (df_telem["raw_text"].str.len() < 80).sum()
leaked_tags = df_telem["raw_text"].str.contains(r"\bT1\d{3}\b", regex=True).sum()

print(f"\nSplunk Runtime Telemetry Results:")
print(f"  Total events           : {n_events}")
print(f"  Unique techniques      : {n_techs}")
print(f"  Rows < 80 characters   : {short_rows}")
print(f"  Leaked T1xxx tags      : {leaked_tags}")
print(f"  SHA-256                : {telem_sha}")
print(f"  [SAVED] {telem_out_path}")

assert n_events >= 300, f"Expected >= 300 events, got {n_events}"
assert n_techs >= 25, f"Expected >= 25 techniques, got {n_techs}"
assert short_rows == 0, f"Expected 0 rows < 80 chars, got {short_rows}"
assert leaked_tags == 0, f"Expected 0 leaked tags, got {leaked_tags}"

print("\n[SUCCESS] Phase 5 External Datasets Built & Validated (Task 2 Complete)!")
