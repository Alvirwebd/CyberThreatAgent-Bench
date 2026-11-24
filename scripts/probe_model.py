import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
for m in ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"]:
    try:
        r = client.models.generate_content(
            model=m,
            contents="Return a JSON with status: ok",
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        print(f"{m}: SUCCESS -> {r.text.strip()[:40]}")
    except Exception as e:
        print(f"{m}: FAILED -> {e}")
