from dotenv import load_dotenv
import os, requests, json, time

load_dotenv()
gk = os.environ.get("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={gk}"

cache = {json.loads(l)["cache_key"]: json.loads(l) for l in open("ctab_bench/cache/react_agent_cache.jsonl", encoding="utf-8") if l.strip()}
prompt = [v["prompt"] for v in cache.values() if "INC-000" in v.get("prompt", "")][-1]

payload = {
    "contents": [{"parts": [{"text": prompt}]}],
    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
}
t0 = time.perf_counter()
r = requests.post(url, json=payload, timeout=25)
lat_ms = (time.perf_counter() - t0) * 1000.0

print("status:", r.status_code)
print(f"Latency: {lat_ms:.1f} ms")
if r.status_code == 200:
    res = r.json()
    usage = res.get("usageMetadata", {})
    pt = usage.get("promptTokenCount", 0)
    ct = usage.get("candidatesTokenCount", 0)
    print(f"Tokens: pt={pt}, ct={ct}, total={pt+ct}")
    txt = res["candidates"][0]["content"]["parts"][0]["text"]
    print("Parsed JSON test:", json.loads(txt))
else:
    print("error:", r.text[:300])
