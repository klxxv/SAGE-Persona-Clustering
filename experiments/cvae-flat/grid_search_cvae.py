import os
import subprocess
import sys
import argparse
import time

def run_grid_search():
    parser = argparse.ArgumentParser(description="Grid Search for CVAE-SAGE Personas (Serially)")
    parser.add_argument("--start_p", type=int, default=5, help="Starting number of personas")
    parser.add_argument("--end_p", type=int, default=15, help="Ending number of personas (inclusive)")
    parser.add_argument("--iters", type=int, default=1000, help="Iterations per experiment")
    parser.add_argument("--l1", type=float, default=1e-6, help="L1 penalty lambda")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--subset", type=int, default=None, help="Subset of characters (optional)")
    parser.add_argument("--data_file", type=str, default=None, help="Path to main all_words.csv")
    parser.add_argument("--word_csv", type=str, default=None, help="Path to word2vec_clusters.csv")
    
    args = parser.parse_args()

    persona_range = range(args.start_p, args.end_p + 1)
    
    print("="*60)
    print(f">>> GRID SEARCH STARTED")
    print(f">>> Range: Personas {args.start_p} to {args.end_p}")
    print(f">>> Params: iters={args.iters}, l1={args.l1}, lr={args.lr}")
    print("="*60)

    total_start_time = time.time()
    python_exe = sys.executable
    
    for p in persona_range:
        run_start_time = time.time()
        print(f"\n\n[EXPERIMENT START] n_personas = {p}")
        print("-" * 40)
        
        cmd = [
            python_exe, "experiments/cvae-flat/run_cvae_full.py",
            "--n_personas", str(p),
            "--iters", str(args.iters),
            "--l1", str(args.l1),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size)
        ]
        
        if args.subset: cmd.extend(["--subset", str(args.subset)])
        if args.data_file: cmd.extend(["--data_file", args.data_file])
        if args.word_csv: cmd.extend(["--word_csv", args.word_csv])

        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        try:
            subprocess.run(cmd, env=env, check=True)
            elapsed = time.time() - run_start_time
            print("-" * 40)
            print(f"[EXPERIMENT SUCCESS] n_personas = {p} | Time: {elapsed:.2f}s")
            
        except subprocess.CalledProcessError as e:
            print("-" * 40)
            print(f"[EXPERIMENT FAILED] n_personas = {p} with exit code {e.returncode}")
            print("Moving to next experiment...")
            continue

    total_elapsed = time.time() - total_start_time
    print("\n" + "="*60)
    print(f">>> GRID SEARCH COMPLETED | Total Time: {total_elapsed/3600:.2f} hours")
    print("="*60)

if __name__ == "__main__":
    run_grid_search()
