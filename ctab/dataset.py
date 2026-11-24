"""
ctab.dataset
============
ATT&CK STIX loader, Sigma benchmark corpus builder, and snapshot dataset access.
Reproduces Cells 2, 3, and 5 of Project2-CyberThreatAgent_Bench.ipynb.
"""

import os
import re
import json
import yaml
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

ATTACK_TAG_RE = re.compile(r"^attack\.(t\d{4}(?:\.\d{3})?)$", re.I)
RULE_OS_OF_DIR = (("windows", "windows"), ("linux", "linux"),
                  ("macos", "macos"), ("cloud", "cloud"))

ROOT_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT_DIR / "ctab_bench" / "snapshots"


def get_attack_id(obj):
    """Extracts ATT&CK external technique or group ID from STIX external references."""
    for ref in (obj.get("external_references") or []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def get_tactics(obj):
    """Extracts MITRE ATT&CK tactic kill-chain phases."""
    return [p["phase_name"] for p in (obj.get("kill_chain_phases") or [])
            if p.get("kill_chain_name") == "mitre-attack"]


def load_attack(attack_path=None, attack_release="16.1"):
    """
    Loads frozen MITRE ATT&CK STIX bundle and returns structured DataFrames and graph indexes.
    """
    if attack_path is None:
        attack_path = SNAPSHOT_DIR / "attack" / f"enterprise-attack-{attack_release}.json"
    attack_path = Path(attack_path)
    if not attack_path.exists():
        raise FileNotFoundError(f"ATT&CK bundle not found at: {attack_path}")

    with open(attack_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    stix_by_id = {o["id"]: o for o in objects}

    attack_version = "unknown"
    for o in objects:
        if o.get("type") == "x-mitre-collection":
            attack_version = o.get("x_mitre_version") or "unknown"
            break

    tech_rows = []
    for o in objects:
        if o.get("type") != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        aid = get_attack_id(o)
        if not aid:
            continue
        tech_rows.append({
            "stix_id": o["id"],
            "attack_id": aid,
            "parent_id": aid.split(".")[0],
            "is_subtechnique": "." in aid,
            "name": o.get("name") or "",
            "tactics": get_tactics(o),
            "description": (o.get("description") or "")[:1500],
            "platforms": o.get("x_mitre_platforms") or [],
        })
    df_techniques = pd.DataFrame(tech_rows).reset_index(drop=True)

    group_rows = []
    for o in objects:
        if o.get("type") != "intrusion-set" or o.get("revoked"):
            continue
        aid = get_attack_id(o)
        if aid:
            group_rows.append({
                "stix_id": o["id"],
                "attack_id": aid,
                "name": o.get("name") or "",
                "aliases": o.get("aliases") or []
            })
    df_groups = pd.DataFrame(group_rows).reset_index(drop=True)

    mit_rows = []
    for o in objects:
        if o.get("type") != "course-of-action" or o.get("revoked"):
            continue
        aid = get_attack_id(o)
        mit_rows.append({
            "stix_id": o["id"],
            "attack_id": aid,
            "name": o.get("name") or "",
            "description": (o.get("description") or "")[:700]
        })
    df_mitigations = pd.DataFrame(mit_rows).reset_index(drop=True)

    # Graph indexes
    parents = sorted({r["parent_id"] for _, r in df_techniques.iterrows()})
    children = defaultdict(list)
    for _, r in df_techniques.iterrows():
        children[r["parent_id"]].append(r["attack_id"])

    name_of = {r["attack_id"]: r["name"] for _, r in df_techniques.iterrows()}
    tactics_of = {r["attack_id"]: r["tactics"] for _, r in df_techniques.iterrows()}
    mit_name_of = {r["attack_id"]: r["name"] for _, r in df_mitigations.iterrows() if r["attack_id"]}

    group_uses_technique = defaultdict(set)
    technique_used_by_group = defaultdict(set)
    technique_mitigated_by = defaultdict(set)
    mitigation_mitigates = defaultdict(set)

    for rel in objects:
        if rel.get("type") != "relationship" or rel.get("revoked"):
            continue
        src, tgt, rtype = rel.get("source_ref"), rel.get("target_ref"), rel.get("relationship_type")
        s_type = (stix_by_id.get(src) or {}).get("type")
        t_type = (stix_by_id.get(tgt) or {}).get("type")
        if rtype == "uses" and s_type == "intrusion-set" and t_type == "attack-pattern":
            group_uses_technique[src].add(tgt)
            technique_used_by_group[tgt].add(src)
        if rtype == "mitigates" and s_type == "course-of-action" and t_type == "attack-pattern":
            technique_mitigated_by[tgt].add(src)
            mitigation_mitigates[src].add(tgt)

    indexes = {
        "attack_version": attack_version,
        "parents": parents,
        "children": children,
        "NAME_OF": name_of,
        "TACTICS_OF": tactics_of,
        "MIT_NAME_OF": mit_name_of,
        "group_uses_technique": group_uses_technique,
        "technique_used_by_group": technique_used_by_group,
        "technique_mitigated_by": technique_mitigated_by,
        "mitigation_mitigates": mitigation_mitigates,
    }
    return df_techniques, df_mitigations, df_groups, indexes


def _scalar(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return str(v)


def _rule_text(rule):
    parts = [_scalar(rule.get("title")), _scalar(rule.get("description")),
             _scalar(rule.get("logsource")), _scalar(rule.get("detection"))]
    raw = " ".join(p for p in parts if p).strip()
    scrubbed = re.sub(r"(?i)\battack\.(?:t\d{4}(?:\.\d{3})?|[a-z0-9_-]+)\b", "", raw)
    return " ".join(scrubbed.split())


def _parse_rule_date(v):
    if not v:
        return None
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return pd.to_datetime(str(v), format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(str(v))
    except Exception:
        return None


def load_sigma_rules(sigma_dir=None):
    """
    Parses human-authored Sigma YAML detection rules from snapshot directory.
    """
    if sigma_dir is None:
        sigma_dir = SNAPSHOT_DIR / "sigma"
    sigma_dir = Path(sigma_dir)
    if not sigma_dir.exists():
        raise FileNotFoundError(f"Sigma rules snapshot directory not found: {sigma_dir}")

    files = list(sigma_dir.rglob("*.yml")) + list(sigma_dir.rglob("*.yaml"))
    rows, bad = [], 0
    for fp in files:
        try:
            rule = yaml.safe_load(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            bad += 1
            continue
        if not isinstance(rule, dict) or str(rule.get("status") or "").lower() == "deprecated":
            continue
        tags = []
        for tag in (rule.get("tags") or []):
            m = ATTACK_TAG_RE.match(str(tag).strip())
            if m:
                tags.append(m.group(1).upper())
        tags = list(dict.fromkeys(tags))
        if not tags:
            continue
        rel = str(fp.parent.relative_to(sigma_dir)).replace("\\", "/")
        os_tag = next((o for k, o in RULE_OS_OF_DIR if k in rel), "other")
        rows.append({
            "rule_id": str(rule.get("id") or fp.stem),
            "text": _rule_text(rule),
            "attack_ids": tags,
            "attack_id": tags[0],
            "n_tags": len(tags),
            "level": str(rule.get("level") or "unknown").lower(),
            "status": str(rule.get("status") or "unknown").lower(),
            "rule_date": _parse_rule_date(rule.get("date")),
            "source_family": rel,
            "rule_os": os_tag,
        })
    return pd.DataFrame(rows), bad


def _norm_for_dedup(t):
    return re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).strip()


def build_benchmark(df_rules=None, min_label_count=2, max_samples=None, seeds=(7, 21, 42, 84, 123)):
    """
    Builds canonical benchmark corpus with near-duplicate removal and minimum label filtering.
    """
    if df_rules is None:
        df_rules, _ = load_sigma_rules()

    d = df_rules.drop_duplicates("rule_id").copy()
    n_before = len(d)
    d["_dedup"] = d["text"].map(_norm_for_dedup)
    d = d.drop_duplicates("_dedup").drop(columns=["_dedup"])
    n_dupes = n_before - len(d)

    vc = d["attack_id"].value_counts()
    d = d[d["attack_id"].isin(vc[vc >= min_label_count].index)].copy()
    n_dropped_labels = int((vc < min_label_count).sum())

    if max_samples is not None and len(d) > max_samples:
        per_label = max(6, max_samples // max(1, d["attack_id"].nunique()))
        d = pd.concat([g.sample(min(len(g), per_label), random_state=seeds[0])
                       for _, g in d.groupby("attack_id", group_keys=False)])
        d = d.sample(frac=1, random_state=seeds[0]).head(max_samples)

    d = d.sort_values("rule_id").reset_index(drop=True)
    d.insert(0, "sample_id", ["CTA-%04d" % (i + 1) for i in range(len(d))])
    stats = {
        "rules_loaded": int(len(df_rules)),
        "after_dup_removal": int(len(d)),
        "exact_and_normalised_duplicates_removed": int(n_dupes),
        "labels_below_min_count_dropped": n_dropped_labels,
        "samples": int(len(d)),
        "labels": int(d["attack_id"].nunique()),
        "multi_tag_share": round(float((d["n_tags"] > 1).mean()), 4),
        "mean_tags_per_rule": round(float(d["n_tags"].mean()), 3),
        "by_os": dict(Counter(d["rule_os"])),
    }
    return d, stats


def get_canonical_benchmark(csv_path=None, force_rebuild=False):
    """
    Loads or generates the canonical N=2,530 Sigma benchmark dataset.
    """
    if csv_path is None:
        csv_path = ROOT_DIR / "artifacts" / "sigma_benchmark_2530.csv"
    csv_path = Path(csv_path)

    if csv_path.exists() and not force_rebuild:
        df = pd.read_csv(csv_path)
        if "attack_ids" in df.columns and isinstance(df["attack_ids"].iloc[0], str):
            import ast
            try:
                df["attack_ids"] = df["attack_ids"].apply(ast.literal_eval)
            except Exception:
                pass
        return df

    bench_df, _ = build_benchmark(min_label_count=2, max_samples=None)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(csv_path, index=False)
    return bench_df
