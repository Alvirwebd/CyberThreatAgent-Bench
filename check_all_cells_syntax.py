import json
from pathlib import Path

NB_PATH = Path("Project2-CyberThreatAgent_Bench.ipynb")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
print(f"Checking syntax of all {len(code_cells)} code cells...")

errors = 0
for i, cell in enumerate(code_cells, start=1):
    src = "".join(cell.get("source", []))
    clean_src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("%matplotlib"))
    try:
        compile(clean_src, f"<cell_{i}>", "exec")
    except Exception as e:
        print(f"[SYNTAX ERROR] Cell {i}: {e}")
        errors += 1

if errors == 0:
    print(f"ALL {len(code_cells)} CODE CELLS COMPILED CLEANLY WITH ZERO SYNTAX ERRORS!")
else:
    print(f"Found {errors} syntax errors.")
