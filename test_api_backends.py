import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Test Gemini 3.5 Flash-Lite
print("Testing Gemini 3.5 Flash-Lite...")
try:
    from google import genai
    from google.genai import types
    client_g = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    t0 = time.perf_counter()
    r = client_g.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents='Respond strictly with JSON: {"status": "ok", "provider": "gemini-3.5-flash-lite"}',
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    lat = (time.perf_counter() - t0) * 1000
    print(f"Gemini 3.5 Flash-Lite: SUCCESS in {lat:.1f} ms | Text: {r.text.strip()}")
    print("Tokens:", r.usage_metadata.prompt_token_count, r.usage_metadata.candidates_token_count)
except Exception as e:
    print("Gemini 3.5 Flash-Lite error:", e)

# Test Groq models
print("\nTesting Groq models...")
for model_candidate in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_candidate,
            "messages": [{"role": "user", "content": 'Respond strictly with JSON: {"status": "ok"}'}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        t0 = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        lat2 = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            print(f"Groq {model_candidate}: SUCCESS in {lat2:.1f} ms | Text: {resp.json()['choices'][0]['message']['content'].strip()}")
        else:
            print(f"Groq {model_candidate} HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"Groq {model_candidate} error:", e)
