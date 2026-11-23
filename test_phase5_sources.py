import requests, json

print("=== Checking MITRE TRAM Bootstrap Corpus ===")
tram_url = "https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/main/data/training/bootstrap-training-data.json"
try:
    r = requests.get(tram_url, timeout=15)
    print("TRAM HTTP status:", r.status_code, "Size bytes:", len(r.content))
    if r.status_code == 200:
        data = r.json()
        print("TRAM JSON type:", type(data), "Length:", len(data))
        if isinstance(data, list) and len(data) > 0:
            print("First item keys:", list(data[0].keys()))
            print("First item sample:", str(data[0])[:300])
except Exception as e:
    print("TRAM error:", e)

print("\n=== Checking Splunk Attack Data Manifests & Payloads ===")
# Check attack_data repo
splunk_manifest_url = "https://raw.githubusercontent.com/splunk/attack_data/master/datasets/attack_techniques/T1059.001/atomic_red_team/windows-sysmon.log"
splunk_media_url = "https://media.githubusercontent.com/media/splunk/attack_data/master/datasets/attack_techniques/T1059.001/atomic_red_team/windows-sysmon.log"
try:
    r_raw = requests.get(splunk_manifest_url, timeout=15)
    print("Raw URL status:", r_raw.status_code, "Content-Type:", r_raw.headers.get("content-type"), "Len:", len(r_raw.content))
    print("Raw URL snippet:", r_raw.text[:150])
    
    r_media = requests.get(splunk_media_url, timeout=15)
    print("Media URL status:", r_media.status_code, "Len:", len(r_media.content))
    print("Media URL snippet:", r_media.text[:200])
except Exception as e:
    print("Splunk test error:", e)
