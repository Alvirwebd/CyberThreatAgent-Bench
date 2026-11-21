#!/usr/bin/env python3
"""
Multi-LLM Scaling Benchmark Harness for CyberThreatAgent-Bench (Phase 3.1 Hotfix)
Evaluates 4 LLM scales across Groq LPU and Google Cloud API on the exact same
n=200 stratified leak-free held-out test set (artifacts/test_200_fold0.csv):
  1. openai/gpt-oss-20b   ("GPT-OSS 20B", Groq LPU)
  2. qwen/qwen3.8-27b     ("Qwen 3.8 27B", Groq LPU)
  3. openai/gpt-oss-120b  ("GPT-OSS 120B", Groq LPU)
  4. gemini-3.5-flash-lite ("Gemini 3.5 Flash-Lite", Google Cloud API)

Includes ranked Top-5 predictions, persistent JSONL disk cache, strict rate pacing
outside latency timers (3.5s Groq, 4.5s Gemini), exponential backoff on 429/503,
and strict think-tag stripping.
"""

import os
import sys
import re
import ast
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path("c:/Research/Aiworks/CyberThreatAgent_Bench")
load_dotenv(ROOT / ".env")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

CACHE_DIR = ROOT / "ctab_bench" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_responses.jsonl"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load JSONL Cache from all cache files
RESPONSE_CACHE = {}
for p in CACHE_DIR.glob("*.jsonl"):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    k = entry.get("cache_key")
                    if k:
                        RESPONSE_CACHE[k] = entry
                except Exception:
                    pass
print(f"[CACHE] Loaded {len(RESPONSE_CACHE)} cached LLM responses from {CACHE_DIR}")

def get_cache_key(model_name: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}::ranked_top5::{h}"

def save_cache_entry(entry: dict):
    k = entry.get("cache_key")
    RESPONSE_CACHE[k] = entry
    mid_safe = str(entry.get("model_id", "general")).replace("/", "_").replace(":", "_")
    target_file = CACHE_DIR / f"llm_responses_{mid_safe}.jsonl"
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def strip_thinking(text: str) -> str:
    if not text:
        return ""
    # Strip <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()

ATTACK_ID_PATTERN = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)

def parse_predictions(raw_text: str) -> list:
    """Parses ranked list of up to 5 MITRE ATT&CK technique IDs from LLM response."""
    if not raw_text:
        return []
    clean = strip_thinking(raw_text)
    
    # 1. Try JSON parse
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            cand_list = data.get("predictions") or data.get("attack_ids") or [data.get("attack_id")]
            if isinstance(cand_list, list):
                valid = []
                for item in cand_list:
                    if isinstance(item, str):
                        val = item.strip().upper()
                        if ATTACK_ID_PATTERN.match(val) and val not in valid:
                            valid.append(val)
                if valid:
                    return valid[:5]
        elif isinstance(data, list):
            valid = []
            for item in data:
                if isinstance(item, str):
                    val = item.strip().upper()
                    if ATTACK_ID_PATTERN.match(val) and val not in valid:
                        valid.append(val)
            if valid:
                return valid[:5]
    except Exception:
        pass

    # 2. Regex fallback for any ATT&CK IDs
    matches = re.findall(r"\b(T\d{4}(?:\.\d{3})?)\b", clean, re.IGNORECASE)
    valid = []
    for m in matches:
        val = m.upper()
        if val not in valid:
            valid.append(val)
    return valid[:5]

# Google GenAI Client
_genai_client = None
if GEMINI_KEY:
    try:
        from google import genai
        _genai_client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"[WARNING] google.genai initialization failed: {e}")

SYS_PROMPT = (
    "You are an expert cybersecurity analyst. Map the following security alert to the top 5 most relevant "
    "MITRE ATT&CK techniques or sub-techniques in order of confidence.\n"
    'Respond ONLY with a valid JSON object in this exact schema: {"predictions": ["TXXXX.XXX", "TYYYY", "TZZZZ", "TAAAA", "TBBBB"]}'
)

def call_groq(model_id: str, text: str, max_retries: int = 10) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }
    max_tokens = 1024 if "gpt-oss" in model_id else 512
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    if "gpt-oss" in model_id:
        payload["reasoning_effort"] = "low"

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=35)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                preds = parse_predictions(raw)
                usage = data.get("usage", {})
                rem_tok = int(resp.headers.get("x-ratelimit-remaining-tokens", 8000))
                if rem_tok < 1500:
                    time.sleep(1.0)
                return {
                    "status": "ok",
                    "raw": raw,
                    "predictions": preds,
                    "latency_ms": dt_ms,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "parse_failure": (len(preds) == 0)
                }
            elif resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 5.0 * (attempt + 1)))
                print(f"  [Groq 429 Rate Limit] Backing off {retry_after + 1.0:.1f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(retry_after + 1.0)
            elif resp.status_code in (500, 502, 503, 504):
                backoff = 3.0 * (attempt + 1)
                print(f"  [Groq HTTP {resp.status_code}] Backing off {backoff:.1f}s...", flush=True)
                time.sleep(backoff)
            else:
                print(f"  [Groq HTTP {resp.status_code}] {resp.text[:100]}", flush=True)
                time.sleep(3.0)
        except Exception as e:
            print(f"  [Groq Exception] {e}", flush=True)
            time.sleep(3.0 * (attempt + 1))

    return {"status": "error", "predictions": [], "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "parse_failure": True}

def call_gemini(model_id: str, text: str, max_retries: int = 10) -> dict:
    if not _genai_client:
        return {"status": "error_no_key", "predictions": [], "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "parse_failure": True}

    prompt = f"{SYS_PROMPT}\n\nSecurity Alert:\n{text}"

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = _genai_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            raw = resp.text
            preds = parse_predictions(raw)
            return {
                "status": "ok",
                "raw": raw,
                "predictions": preds,
                "latency_ms": dt_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "parse_failure": (len(preds) == 0)
            }
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                backoff = 4.0 * (attempt + 1)
                print(f"  [Gemini Transient {type(e).__name__}] Backing off {backoff:.1f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(backoff)
            else:
                print(f"  [Gemini Exception] {e}", flush=True)
                time.sleep(3.0)

    return {"status": "error", "predictions": [], "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "parse_failure": True}

def run_benchmark(target_models=None, limit=None):
    test_csv = ROOT / "artifacts" / "test_200_fold0.csv"
    if not test_csv.exists():
        print(f"[ERROR] Test set not found at {test_csv}. Please run extract_test_200.py first!")
        return
    df_test = pd.read_csv(test_csv)
    if limit:
        df_test = df_test.head(limit)
    n_alerts = len(df_test)
    print(f"\n[BENCHMARK] Evaluating {n_alerts} held-out test alerts across LLM models...")

    all_models = [
        {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "regime": "Groq LPU", "provider": "groq", "p_cost": 0.075, "c_cost": 0.30, "pace": 3.5},
        {"id": "qwen/qwen3.8-27b", "name": "Qwen 3.8 27B", "regime": "Groq LPU", "provider": "groq", "p_cost": 0.10, "c_cost": 0.40, "pace": 3.5},
        {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "regime": "Groq LPU", "provider": "groq", "p_cost": 0.60, "c_cost": 2.40, "pace": 3.5},
        {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash-Lite", "regime": "Google Cloud API", "provider": "gemini", "p_cost": 0.075, "c_cost": 0.30, "pace": 4.5},
    ]
    if target_models:
        models = [m for m in all_models if m["id"] in target_models or m["name"] in target_models]
    else:
        models = all_models

    all_summary = []
    all_pred_records = []

    for m_info in models:
        mid = m_info["id"]
        mname = m_info["name"]
        provider = m_info["provider"]
        regime = m_info["regime"]
        pace = m_info["pace"]
        print(f"\n==================================================")
        print(f"Evaluating Model: {mid} ({mname}) [{regime}]")
        print(f"==================================================")

        top1_exact = []
        top1_any_tag = []
        top1_parent = []
        top3_exact = []
        mrr5_exact = []
        parse_fails = []
        latencies = []

        for idx, row in df_test.iterrows():
            text = row["text"]
            gt_id = str(row["attack_id"]).strip().upper()
            
            # parse attack_ids list
            attack_ids = []
            if "attack_ids" in row and pd.notna(row["attack_ids"]):
                try:
                    val_tags = row["attack_ids"]
                    if isinstance(val_tags, str):
                        attack_ids = ast.literal_eval(val_tags)
                    elif isinstance(val_tags, (list, tuple, np.ndarray, set)):
                        attack_ids = list(val_tags)
                except Exception:
                    attack_ids = [str(row["attack_ids"]).strip()]
            if not attack_ids:
                attack_ids = [gt_id]
            attack_ids_set = {str(x).strip().upper() for x in attack_ids}

            cache_key = get_cache_key(mid, text)
            if cache_key in RESPONSE_CACHE:
                res = RESPONSE_CACHE[cache_key]
                preds = res.get("predictions", [])
                lat_ms = res.get("latency_ms", 500.0)
                pf = res.get("parse_failure", (len(preds) == 0))
            else:
                if provider == "groq":
                    res = call_groq(mid, text)
                    time.sleep(pace)  # Rate pacing strictly outside latency timer
                else:
                    res = call_gemini(mid, text)
                    time.sleep(pace)

                res["cache_key"] = cache_key
                res["sample_id"] = row.get("sample_id", f"CTA-{idx}")
                res["model_id"] = mid
                res["model_name"] = mname
                save_cache_entry(res)
                preds = res.get("predictions", [])
                lat_ms = res.get("latency_ms", 500.0)
                pf = res.get("parse_failure", (len(preds) == 0))

            pred_top1 = preds[0] if preds else None
            is_exact = (pred_top1 == gt_id) if pred_top1 else False
            is_any_tag = (pred_top1 in attack_ids_set) if pred_top1 else False
            gt_parent = gt_id.split(".")[0]
            pred_parent = pred_top1.split(".")[0] if pred_top1 else ""
            is_parent = (pred_parent == gt_parent) if pred_top1 else False
            is_top3 = (gt_id in preds[:3]) if preds else False

            # MRR@5
            rr = 0.0
            for rk, p in enumerate(preds[:5], 1):
                if p == gt_id:
                    rr = 1.0 / rk
                    break

            top1_exact.append(float(is_exact))
            top1_any_tag.append(float(is_any_tag))
            top1_parent.append(float(is_parent))
            top3_exact.append(float(is_top3))
            mrr5_exact.append(rr)
            parse_fails.append(float(pf))
            latencies.append(lat_ms)

            all_pred_records.append({
                "sample_id": row["sample_id"],
                "model_id": mid,
                "model_name": mname,
                "regime": regime,
                "gt_attack_id": gt_id,
                "pred_top1": pred_top1,
                "predictions": json.dumps(preds),
                "is_top1_exact": is_exact,
                "is_top1_any_tag": is_any_tag,
                "is_top1_parent": is_parent,
                "is_top3_exact": is_top3,
                "rr_at_5": rr,
                "latency_ms": lat_ms,
                "parse_failure": pf
            })

            if (idx + 1) % 25 == 0 or (idx + 1) == n_alerts:
                print(f"  [{mname}] Alert {idx+1}/{n_alerts} -> Top-1 Exact: {np.mean(top1_exact):.3f} | Top-3: {np.mean(top3_exact):.3f} | MRR@5: {np.mean(mrr5_exact):.3f} | p50 Lat: {np.median(latencies):.1f}ms", flush=True)

        summary_row = {
            "model_id": mid,
            "model_name": mname,
            "regime": regime,
            "n_alerts": n_alerts,
            "top1_exact": float(np.mean(top1_exact)),
            "top1_any_tag": float(np.mean(top1_any_tag)),
            "top1_parent": float(np.mean(top1_parent)),
            "top3_exact": float(np.mean(top3_exact)),
            "mrr_at_5": float(np.mean(mrr5_exact)),
            "parse_failure_rate": float(np.mean(parse_fails)),
            "latency_p50_ms": float(np.median(latencies)),
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
        }
        all_summary.append(summary_row)

    sum_path = OUT_DIR / "llm_eval_200.csv"
    pred_path = OUT_DIR / "llm_predictions_200.csv"

    if sum_path.exists():
        existing_sum = pd.read_csv(sum_path)
        existing_sum = existing_sum[~existing_sum["model_id"].isin([m["id"] for m in models])]
        df_sum = pd.concat([existing_sum, pd.DataFrame(all_summary)], ignore_index=True)
    else:
        df_sum = pd.DataFrame(all_summary)

    if pred_path.exists():
        existing_preds = pd.read_csv(pred_path)
        existing_preds = existing_preds[~existing_preds["model_id"].isin([m["id"] for m in models])]
        df_preds = pd.concat([existing_preds, pd.DataFrame(all_pred_records)], ignore_index=True)
    else:
        df_preds = pd.DataFrame(all_pred_records)

    df_sum.to_csv(sum_path, index=False)
    df_preds.to_csv(pred_path, index=False)

    print("\n==================================================")
    print("FINAL MULTI-LLM BENCHMARK SUMMARY (n=200 Held-Out Fold 0)")
    print("==================================================")
    print(df_sum[["model_name", "regime", "top1_exact", "top1_any_tag", "top3_exact", "mrr_at_5", "parse_failure_rate", "latency_p50_ms"]].to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model IDs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of alerts for smoke testing")
    args = parser.parse_args()
    target = [x.strip() for x in args.models.split(",")] if args.models else None
    run_benchmark(target, args.limit)
