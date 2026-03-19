import os
import subprocess
import sys
import argparse
import time

def run_grid_search():
    parser = argparse.ArgumentParser(description="Grid Search for Traditional SAGE Personas")
    parser.add_argument("--start_p", type=int, default=5)
    parser.add_argument("--end_p", type=int, default=15)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--l1", type=float, default=1.0)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--word_csv", type=str, default=None)
    
    args = parser.parse_args()
    persona_range = range(args.start_p, args.end_p + 1)
    
    print("="*60)
    print(f">>> TRADITIONAL SAGE GRID SEARCH STARTED")
    print(f">>> Range: Personas {args.start_p} to {args.end_p}")
    print("="*60)

    total_start_time = time.time()
    python_exe = sys.executable
    
    for p in persona_range:
        run_start_time = time.time()
        print(f"\n[EXPERIMENT START] n_personas = {p}")
        
        # Correct path to run_traditional_full.py within its directory
        cmd = [
            python_exe, "experiments/traditional-sage/run_traditional_full.py",
            "--n_personas", str(p),
            "--iters", str(args.iters),
            "--l1", str(args.l1)
        ]
        if args.subset: cmd.extend(["--subset", str(args.subset)])
        if args.data_file: cmd.extend(["--data_file", args.data_file])
        if args.word_csv: cmd.extend(["--word_csv", args.word_csv])

        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[EXPERIMENT SUCCESS] n_personas = {p} | Time: {time.time() - run_start_time:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"[EXPERIMENT FAILED] n_personas = {p} | Error: {e}")

    print("\n" + "="*60)
    print(f">>> GRID SEARCH COMPLETED | Total Time: {(time.time() - total_start_time)/3600:.2f} hours")
    print("="*60)

if __name__ == "__main__":
    run_grid_search()
