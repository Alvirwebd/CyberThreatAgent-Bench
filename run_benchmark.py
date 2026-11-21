#!/usr/bin/env python3
"""
CyberThreatAgent-Bench CLI Runner
=================================
Reproduces the complete multi-stage cybersecurity agent benchmark from the manuscript.

Usage:
    python run_benchmark.py --fast     # Quick smoke test run (2 seeds, subset alerts)
    python run_benchmark.py --full     # Full publication run (5 seeds, full corpus)
    python run_benchmark.py --help     # Show options
"""

import sys
import os
import argparse
import time
import json
import hashlib
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CyberThreatAgent-Bench Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --fast          # Quick verification run (~1-2 minutes)
  python run_benchmark.py --full          # Full 5-seed publication benchmark
        """
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true", help="Quick run with 2 seeds and alert sub-sampling")
    group.add_argument("--full", action="store_true", help="Complete publication run with 5 seeds across full corpus")
    parser.add_argument("--notebook", type=str, default="Project2-CyberThreatAgent_Bench.ipynb",
                        help="Path to benchmark notebook (default: Project2-CyberThreatAgent_Bench.ipynb)")
    parser.add_argument("--output-dir", type=str, default="artifacts",
                        help="Path to output directory (default: ctab_bench/outputs)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Propagate output directory to environment for notebook cells
    if args.output_dir:
        os.environ["CTAB_OUT_DIR"] = str(Path(args.output_dir).resolve())
    
    # Default to fast mode if neither specified for quick safe inspection
    fast_mode = True if args.fast or not args.full else False
    mode_str = "FAST (Sanity Smoke Test)" if fast_mode else "FULL (5-Seed Publication Run)"
    
    nb_path = Path(args.notebook).resolve()
    if not nb_path.exists():
        print(f"[ERROR] Notebook not found at: {nb_path}")
        sys.exit(1)

    print("=" * 70)
    print("CyberThreatAgent-Bench: Autonomous Cybersecurity Agent Benchmark")
    print("=" * 70)
    print(f"Target Notebook : {nb_path.name}")
    print(f"Execution Mode  : {mode_str}")
    print(f"Working Dir     : {Path.cwd()}")
    print("=" * 70)

    # Set non-interactive matplotlib backend
    import matplotlib
    matplotlib.use("Agg")
    from IPython.core.interactiveshell import InteractiveShell

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update Config cell fast flag in memory
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "class Config:" in src and "fast: bool =" in src:
                src_lines = []
                for line in src.splitlines():
                    if "fast: bool =" in line:
                        src_lines.append(f"    fast: bool = {fast_mode}                     # Auto-configured by CLI")
                    else:
                        src_lines.append(line)
                cell["source"] = [l + "\n" for l in src_lines[:-1]] + [src_lines[-1]]
                break

    shell = InteractiveShell.instance()
    t_start = time.perf_counter()
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    total = len(code_cells)

    print(f"\nExecuting {total} code cells sequentially...\n")
    for i, cell in enumerate(code_cells, start=1):
        src = "".join(cell.get("source", []))
        clean_src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("%matplotlib"))
        
        # Extract title or first comment line
        header = f"Cell {i}"
        for line in src.splitlines():
            line_s = line.strip()
            if line_s.startswith("# CELL") or line_s.startswith("# Stage"):
                header = line_s.lstrip("# ").strip()
                break

        t0 = time.perf_counter()
        print(f"[{i:02d}/{total:02d}] {header[:55]:<55} ... ", end="", flush=True)
        res = shell.run_cell(clean_src)
        dur = time.perf_counter() - t0

        if not res.success:
            print(f"FAILED ({dur:.2f}s)")
            if res.error_in_exec:
                print(f"\n[ERROR] Exception encountered:\n{res.error_in_exec}")
            sys.exit(1)
        else:
            print(f"PASS ({dur:.2f}s)")

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY in {total_time:.1f}s ({total_time/60:.2f} min)")
    print("=" * 70)
    
    out_dir = Path(args.output_dir)
    print(f"\nArtifacts and exports generated in: {out_dir}")
    if (out_dir / "benchmark_summary.csv").exists():
        print(f"  * {out_dir / 'benchmark_summary.csv'}")
    if (out_dir / "latex" / "table_baselines.tex").exists():
        print(f"  * {out_dir / 'latex' / 'table_baselines.tex'}")
    if (out_dir / "figures").exists():
        n_figs = len(list((out_dir / "figures").glob("*.png")))
        print(f"  * {out_dir / 'figures'} ({n_figs} publication figures rendered)")

if __name__ == "__main__":
    main()
