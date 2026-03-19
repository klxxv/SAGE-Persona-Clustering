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
    parser.add_argument("--start_p", type=int, default=5)
    parser.add_argument("--end_p", type=int, default=15)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--l1", type=float, default=1.0)
    
    args = parser.parse_args()
    python_exe = sys.executable

    # 1. CVAE Grid Search
    # cvae_cmd = [
    #     python_exe, "experiments/cvae-flat/grid_search_cvae.py",
    #     "--start_p", str(args.start_p),
    #     "--end_p", str(args.end_p),
    #     "--iters", str(args.iters),
    #     "--l1", str(args.l1)
    # ]
    # if args.subset: cvae_cmd.extend(["--subset", str(args.subset)])
    # run_cmd(cvae_cmd, "CVAE-Flat Word-based Grid Search")

    # 2. Traditional SAGE Grid Searches (4 Cluster Types)
    cluster_types = [
        ("BERT-256", "data/sage_cluster_dataset/bert_256/word2vec_clusters.csv"),
        ("BERT-512", "data/sage_cluster_dataset/bert_512/word2vec_clusters.csv"),
        ("W2V-256",  "data/sage_cluster_dataset/w2v_256/word2vec_clusters.csv"),
        ("W2V-512",  "data/sage_cluster_dataset/w2v_512/word2vec_clusters.csv")
    ]

    for label, cluster_path in cluster_types:
        if not os.path.exists(cluster_path):
            print(f"Warning: Cluster file not found, skipping {label}: {cluster_path}")
            continue
            
        trad_cmd = [
            python_exe, "experiments/traditional-sage/grid_search_traditional_sage.py",
            "--start_p", str(args.start_p),
            "--end_p", str(args.end_p),
            "--iters", str(args.iters),
            "--l1", str(args.l1),
            "--word_csv", cluster_path,
            "--label", label # PASS UNIQUE LABEL
        ]
        if args.subset: trad_cmd.extend(["--subset", str(args.subset)])
        run_cmd(trad_cmd, f"Traditional SAGE Grid Search ({label})")

    print("\n" + "#"*60)
    print(">>> ALL GRID SEARCH TASKS COMPLETED")
    print("#"*60)

if __name__ == "__main__":
    main()
