from dotenv import load_dotenv
import os, requests
load_dotenv()

print("--- Testing Gemini ---")
try:
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    r = client.models.generate_content(model="gemini-3.5-flash-lite", contents='Respond strictly with valid json: {"msg": "hello"}')
    print("Gemini 3.5 Flash-Lite OK:", r.text.strip()[:60])
except Exception as e:
    print("Gemini error:", e)

print("--- Testing Groq Qwen ---")
try:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}", "Content-Type": "application/json"}
    payload = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": 'Respond strictly with valid json: {"msg": "hello"}'}], "response_format": {"type": "json_object"}}
    res = requests.post(url, headers=headers, json=payload, timeout=10)
    print("Groq Qwen status:", res.status_code, res.json()["choices"][0]["message"]["content"][:60])
except Exception as e:
    print("Groq error:", e)
