import os
import subprocess
import sys
import argparse
import time

def run_cmd(cmd, name):
    print(f"\n" + "="*60)
    print(f">>> STARTING SUB-TASK: {name}")
    print(f">>> Command: {' '.join(cmd)}")
    print("="*60 + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"\n>>> SUCCESS: {name}")
    except subprocess.CalledProcessError as e:
        print(f"\n>>> FAILED: {name} with exit code {e.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Master Grid Search for SAGE Persona models")
    parser.add_argument("--start_p",     type=int,   default=5)
    parser.add_argument("--end_p",       type=int,   default=15)
    parser.add_argument("--iters",       type=int,   default=500)
    parser.add_argument("--subset",      type=int,   default=None)
    parser.add_argument("--l1",          type=float, default=1.0)
    parser.add_argument("--labels",      nargs='+',  default=None,
                        help="Labels to run, e.g. W2V-Role  (omit to run all)")

    args = parser.parse_args()
    python_exe = sys.executable

    # Per-role W2V (one shared cluster_dir, 4 role CSVs inside)
    per_role_types = [
        ("W2V-Role", "data/sage_cluster_dataset/w2v-role"),
    ]

    for label, cluster_dir in per_role_types:
        if args.labels and label not in args.labels:
            continue
        if not os.path.isdir(cluster_dir):
            print(f"Warning: Cluster dir not found, skipping {label}: {cluster_dir}")
            continue

        for p in range(args.start_p, args.end_p + 1):
            cmd = [
                python_exe, "experiments/traditional-sage/run_traditional_full.py",
                "--n_personas",  str(p),
                "--iters",       str(args.iters),
                "--l1",          str(args.l1),
                "--cluster_dir", cluster_dir,
                "--label",       label,
            ]
            if args.subset:
                cmd.extend(["--subset", str(args.subset)])
            run_cmd(cmd, f"{label} SAGE Grid Search (P={p})")

    print("\n" + "#"*60)
    print(">>> ALL GRID SEARCH TASKS COMPLETED")
    print("#"*60)

if __name__ == "__main__":
    main()
