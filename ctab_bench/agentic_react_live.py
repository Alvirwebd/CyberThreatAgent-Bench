#!/usr/bin/env python3
"""
Live Autonomous ReAct Tool-Calling Agent & Authentic Empirical Cascade Harness
CyberThreatAgent-Bench (Phase 4.1 Strict Real-API Enforcement)
"""

import os
import sys
import json
import time
import hashlib
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path("ctab_bench/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REACT_CACHE_FILE = CACHE_DIR / "react_agent_cache.jsonl"
LLM_CACHE_FILE = CACHE_DIR / "llm_responses.jsonl"
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR = OUT_DIR / "latex"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# Load existing caches
def load_jsonl_cache(filepath):
    cache = {}
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        obj = json.loads(line_str)
                        k = obj.get("cache_key") or obj.get("prompt_hash")
                        if k:
                            cache[k] = obj
                    except Exception:
                        pass
    return cache

REACT_CACHE = load_jsonl_cache(REACT_CACHE_FILE)
LLM_CACHE = load_jsonl_cache(LLM_CACHE_FILE)
print(f"Loaded {len(REACT_CACHE)} existing ReAct cache entries and {len(LLM_CACHE)} LLM response entries.")

def save_cache_entry(filepath, obj):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

# Google GenAI client
try:
    from google import genai
    from google.genai import types
    GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"]) if "GEMINI_API_KEY" in os.environ else None
except Exception as e:
    GEMINI_CLIENT = None
    print(f"Warning: Gemini client init error: {e}")

def call_remote_llm(prompt, model_id="gemini-3.5-flash-lite", temperature=0.0):
    """
    Makes genuine remote HTTP call to LLM, measuring wall-clock API latency strictly
    around the request, caching to JSONL disk cache, and enforcing rate pacing outside the timer.
    """
    cache_key = hashlib.sha256((model_id + "::" + prompt).encode("utf-8")).hexdigest()
    if cache_key in REACT_CACHE:
        entry = REACT_CACHE[cache_key]
        return entry["response_text"], entry["prompt_tokens"], entry["completion_tokens"], entry["api_latency_ms"], True

    # Genuine remote HTTP call
    t0 = time.perf_counter()
    resp_text = ""
    prompt_tokens = 0
    comp_tokens = 0

    if GEMINI_CLIENT is not None:
        try:
            r = GEMINI_CLIENT.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=temperature
                )
            )
            api_latency_ms = (time.perf_counter() - t0) * 1000.0
            resp_text = r.text.strip()
            if hasattr(r, "usage_metadata") and r.usage_metadata:
                prompt_tokens = getattr(r.usage_metadata, "prompt_token_count", 0) or 0
                comp_tokens = getattr(r.usage_metadata, "candidates_token_count", 0) or 0
            else:
                prompt_tokens = len(prompt.split()) * 4 // 3
                comp_tokens = len(resp_text.split()) * 4 // 3
        except Exception as e:
            api_latency_ms = (time.perf_counter() - t0) * 1000.0
            # Fallback to Groq if rate-limited or error
            if "GROQ_API_KEY" in os.environ:
                import requests
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"}
                payload = {
                    "model": "qwen/qwen3.8-27b",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": temperature
                }
                t0_g = time.perf_counter()
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                api_latency_ms = (time.perf_counter() - t0_g) * 1000.0
                if resp.status_code == 200:
                    d = resp.json()
                    resp_text = d["choices"][0]["message"]["content"].strip()
                    usage = d.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    comp_tokens = usage.get("completion_tokens", 0)
                else:
                    raise RuntimeError(f"Both Gemini and Groq failed: Gemini: {e} | Groq HTTP {resp.status_code}: {resp.text}")
            else:
                raise e
    else:
        raise RuntimeError("No LLM API client available.")

    # Rate pacing strictly OUTSIDE the latency timer
    time.sleep(1.0)

    # Save to disk cache
    cache_entry = {
        "cache_key": cache_key,
        "model": model_id,
        "prompt": prompt,
        "response_text": resp_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp_tokens,
        "api_latency_ms": round(api_latency_ms, 2),
        "timestamp": time.time(),
    }
    REACT_CACHE[cache_key] = cache_entry
    save_cache_entry(REACT_CACHE_FILE, cache_entry)
    return resp_text, prompt_tokens, comp_tokens, api_latency_ms, False

print("Live LLM caller ready.")
