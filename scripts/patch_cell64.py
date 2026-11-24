import json
from pathlib import Path

nb_path = Path("Project2-CyberThreatAgent_Bench.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c64 = nb["cells"][64]
src = "".join(c64["source"])

# Update router display name
src = src.replace('"display": "Agentic Router"', '"display": "Margin-Gated Local Router"')

# Update llm_csv and field names
old_block = '''    llm_csv = OUT / "llm_scaling_benchmark.csv"
    llm_pts = []
    if llm_csv.exists():
        df_llm = pd.read_csv(llm_csv)
        palette = {
            "openai/gpt-oss-20b": ("#8E24AA", "D", 100),
            "qwen/qwen3.8-27b": ("#5E35B1", "D", 110),
            "openai/gpt-oss-120b": ("#3949AB", "D", 130),
            "gemini-3.5-flash": ("#00838F", "h", 120),
        }
        for _, row in df_llm.iterrows():
            mid = row["model_id"]
            mname = row["model_name"]
            lat = float(row["p50_latency_ms"])
            mrr = float(row["Top1_Exact"])
            color, marker, size = palette.get(mid, ("#8E24AA", "D", 110))
            llm_pts.append({
                "name": mid,
                "display": f"{mname} ({row['hardware_regime']})",
                "lat": lat,
                "mrr": mrr,
                "color": color,
                "marker": marker,
                "size": size,
                "regime": row["hardware_regime"]
            })'''

new_block = '''    llm_csv = OUT / "llm_eval_200.csv"
    llm_pts = []
    if llm_csv.exists():
        df_llm = pd.read_csv(llm_csv)
        palette = {
            "openai/gpt-oss-20b": ("#8E24AA", "D", 100),
            "qwen/qwen3.8-27b": ("#5E35B1", "D", 110),
            "openai/gpt-oss-120b": ("#3949AB", "D", 130),
            "gemini-3.5-flash-lite": ("#00838F", "h", 120),
        }
        for _, row in df_llm.iterrows():
            mid = row["model_id"]
            mname = row["model_name"]
            lat = float(row.get("latency_p50_ms", row.get("p50_latency_ms", 500.0)))
            mrr = float(row.get("top1_exact", row.get("Top1_Exact", 0.35)))
            reg_val = str(row.get("regime", row.get("hardware_regime", "Cloud LLM")))
            color, marker, size = palette.get(mid, ("#8E24AA", "D", 110))
            llm_pts.append({
                "name": mid,
                "display": f"{mname} ({reg_val})",
                "lat": lat,
                "mrr": mrr,
                "color": color,
                "marker": marker,
                "size": size,
                "regime": reg_val
            })'''

assert old_block in src, "old_block not found in Cell 64"
src = src.replace(old_block, new_block)

c64["source"] = [l + "\n" for l in src.splitlines()]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

with open(Path("../Project2-CyberThreatAgent_Bench.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[PASS] Cell 64 updated with llm_eval_200.csv and Margin-Gated Local Router.")
