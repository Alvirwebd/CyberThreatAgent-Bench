import json
import re
import random
import hashlib
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, nnls
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
from IPython.core.interactiveshell import InteractiveShell

shell = InteractiveShell.instance()

with open('Project2-CyberThreatAgent_Bench.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']

# Run cells up to Cell 13 (before chain corpus)
print("Running setup cells...")
for i, c in enumerate(code_cells[:7]):
    src = ''.join(c.get('source', []))
    if 'class Config:' in src:
        src = src.replace('fast: bool = False', 'fast: bool = True')
    clean = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('%matplotlib'))
    shell.run_cell(clean)

print("Setup complete. Testing multi-alert benign chains...")

CFG = shell.user_ns['CFG']
CHAIN_TEMPLATES_V5 = shell.user_ns['CHAIN_TEMPLATES_V5']
HOST_POOL = shell.user_ns['HOST_POOL']
_PUBLIC_OCTETS = shell.user_ns['_PUBLIC_OCTETS']
_rand_ip = shell.user_ns['_rand_ip']
_rand_domain = shell.user_ns['_rand_domain']
_rand_sha256 = shell.user_ns['_rand_sha256']
PLACEHOLDER_IOC_TYPE = shell.user_ns['PLACEHOLDER_IOC_TYPE']

BENIGN_ADMIN_TEMPLATES = [
    "Scheduled backup completed successfully on {host}, 1200 files, no egress",
    "Antivirus signature update finished on {host}, definitions 2026.03.14",
    "User {user} logged in interactively to {host} from internal corporate subnet {src_ip}",
    "Software inventory collection on {host}: 42 applications, no changes",
    "System telemetry health check from {host} to monitoring server {dst_ip} on port 8080",
    "Routine log rotation and archive compression completed on {host} by systemd",
    "NTP time synchronization succeeded on {host} with local domain controller",
]

def build_chain_corpus_multialert(cfg, seed):
    rng = random.Random(seed)
    alerts, truth = [], []
    for inc in range(cfg.n_incidents):
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
            if rng.random() < cfg.defang_rate:
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

    # Benign multi-alert administrative sessions (3-5 alerts per incident)
    n_benign_inc = max(4, cfg.n_incidents // 5)
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

df_a, df_t = build_chain_corpus_multialert(CFG, CFG.seeds[0])
print(f"Generated {len(df_a)} alerts across {len(df_t)} incidents.")
benign_alerts = df_a[df_a['label'] == 'benign']
print(f"Benign alerts: {len(benign_alerts)} across {benign_alerts['incident_id'].nunique()} incidents.")
for inc_id, grp in benign_alerts.groupby('incident_id'):
    print(f"  {inc_id}: {len(grp)} alerts")
