import os
import sys
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

CACHE_DIR = Path("ctab_bench/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REACT_CACHE_FILE = CACHE_DIR / "react_agent_cache.jsonl"

def load_react_cache():
    cache = {}
    if REACT_CACHE_FILE.exists():
        with open(REACT_CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        k = d.get("cache_key")
                        if k:
                            cache[k] = d
                    except Exception:
                        pass
    return cache

REACT_CACHE = load_react_cache()

def call_llm_react(prompt, model_id="qwen/qwen3.8-27b", temperature=0.0):
    cache_key = hashlib.sha256((model_id + "::" + prompt).encode("utf-8")).hexdigest()
    if cache_key in REACT_CACHE:
        entry = REACT_CACHE[cache_key]
        return entry["response_text"], entry["prompt_tokens"], entry["completion_tokens"], entry["api_latency_ms"], True

    # Real HTTP call
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    t0 = time.perf_counter()
    resp_text = ""
    prompt_tokens = 0
    comp_tokens = 0

    if groq_key and "qwen" in model_id.lower():
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are an autonomous Tier-3 SOC cybersecurity ReAct triage agent. You reason step-by-step and select tools strictly via JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        api_lat_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            data = resp.json()
            resp_text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            comp_tokens = usage.get("completion_tokens", 0)
        else:
            raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text}")
    elif gemini_key:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_key)
        r = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=temperature)
        )
        api_lat_ms = (time.perf_counter() - t0) * 1000.0
        resp_text = r.text.strip()
        if hasattr(r, "usage_metadata") and r.usage_metadata:
            prompt_tokens = getattr(r.usage_metadata, "prompt_token_count", 0) or 0
            comp_tokens = getattr(r.usage_metadata, "candidates_token_count", 0) or 0
    else:
        raise RuntimeError("No valid API key available!")

    # Rate pacing OUTSIDE latency timer
    time.sleep(1.2)

    entry = {
        "cache_key": cache_key,
        "model": model_id,
        "prompt": prompt,
        "response_text": resp_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp_tokens,
        "api_latency_ms": round(api_lat_ms, 2),
        "timestamp": time.time()
    }
    REACT_CACHE[cache_key] = entry
    with open(REACT_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return resp_text, prompt_tokens, comp_tokens, round(api_lat_ms, 2), False

print("Testing single call to call_llm_react...")
test_prompt = '{"instruction": "Choose the first tool to triage this alert", "alert": "Scheduled backup completed successfully on WKSTN-001, 1200 files, no egress", "output_format": {"thought": "str", "action": "extract_iocs|map_attack_technique|finish_triage", "action_input": {}}}'
text, pt, ct, lat, hit = call_llm_react(test_prompt)
print(f"Result: hit={hit}, lat={lat}ms, tokens={pt}+{ct}")
print("Response text:", text)
