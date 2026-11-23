import json
from pathlib import Path

NB_PATH = Path("Project2-CyberThreatAgent_Bench.ipynb")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 18 ranked method
c18_src = "".join(nb["cells"][18]["source"])
old_c18 = """    def ranked(self, text, top_k=None, allow_abstain=True):
        if allow_abstain and self.confidence(text) < getattr(self, "tau_abstain", 0.12):
            return []
        out = self._ranked_from(self.scores(text), top_k=None)
        out = self._apply_parent_fallback(out)
        return out[:top_k] if top_k else out"""

new_c18 = """    def ranked(self, text, top_k=None, allow_abstain=True):
        if allow_abstain and self.confidence(text) < getattr(self, "tau_abstain", 0.12):
            return [{"attack_id": None, "name": "Abstain (Benign/OOD)", "score": 0.0, "tactics": []}]
        out = self._ranked_from(self.scores(text), top_k=None)
        out = self._apply_parent_fallback(out)
        return out[:top_k] if top_k else out"""

if old_c18 in c18_src:
    c18_src = c18_src.replace(old_c18, new_c18)
    nb["cells"][18]["source"] = [l + "\n" for l in c18_src.splitlines()[:-1]] + [c18_src.splitlines()[-1]]
    print("[UPDATED] Cell 18 ranked() to return sentinel None dict on abstention.")

# 2. Update Cell 20 benign abstention check
c20_src = "".join(nb["cells"][20]["source"])
c20_src = c20_src.replace(
    'ben_abstains = [len(MAPPERS["hybrid"].ranked(t, top_k=1, allow_abstain=True)) == 0 for t in benign_texts]',
    'ben_abstains = [(MAPPERS["hybrid"].ranked(t, top_k=1, allow_abstain=True)[0]["attack_id"] is None) for t in benign_texts]'
)
c20_src = c20_src.replace(
    'test_recalls = [len(MAPPERS["hybrid"].ranked(t, top_k=1, allow_abstain=True)) > 0 for t in test_b["text"]]',
    'test_recalls = [(MAPPERS["hybrid"].ranked(t, top_k=1, allow_abstain=True)[0]["attack_id"] is not None) for t in test_b["text"]]'
)
nb["cells"][20]["source"] = [l + "\n" for l in c20_src.splitlines()[:-1]] + [c20_src.splitlines()[-1]]
print("[UPDATED] Cell 20 benign abstention checks.")

# 3. Update Cell 58 smoke test
c58_src = "".join(nb["cells"][58]["source"])
c58_src = c58_src.replace(
    'assert len(r) == 0, f"Expected empty prediction on benign text, got: {r}"',
    'assert r[0]["attack_id"] is None, f"Expected None prediction on benign text, got: {r}"'
)
nb["cells"][58]["source"] = [l + "\n" for l in c58_src.splitlines()[:-1]] + [c58_src.splitlines()[-1]]
print("[UPDATED] Cell 58 smoke test assertion.")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Abstention contract cleanly updated across notebook.")
