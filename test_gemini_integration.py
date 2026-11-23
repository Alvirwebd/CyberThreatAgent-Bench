from dotenv import load_dotenv
import os, requests, json, time, hashlib
from pathlib import Path

load_dotenv()
gk = os.environ.get("GEMINI_API_KEY")
PINNED_MODEL = "gemini-3.5-flash-lite"
CACHE_FILE = Path("ctab_bench/cache/react_agent_cache.jsonl")

def call_gemini(prompt):
    cache_key = hashlib.sha256((PINNED_MODEL + "::" + prompt).encode("utf-8")).hexdigest()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={gk}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=30)
    api_lat_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code == 200:
        data = resp.json()
        resp_text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        comp_tokens = usage.get("candidatesTokenCount", 0)
        record = {
            "cache_key": cache_key,
            "model": PINNED_MODEL,
            "prompt": prompt,
            "response_text": resp_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "api_latency_ms": round(api_lat_ms, 2),
            "timestamp": time.time()
        }
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return resp_text, prompt_tokens, comp_tokens, api_lat_ms
    else:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

# Test with a prompt
prompt = "You are a cybersecurity SOC agent. Select a tool as strict JSON: {\"action\": \"map_attack_technique\", \"action_input\": {\"text\": \"powershell encoded command\"}}"
txt, pt, ct, lat = call_gemini(prompt)
print("SUCCESS:", txt)
print(f"Tokens: pt={pt}, ct={ct}, lat={lat:.1f}ms")
