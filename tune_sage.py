import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from SAGE_L1_clustering import LiteraryPersonaSAGE
from sage_metrics import calculate_metrics, calculate_perplexity
import json
import time

def train_and_eval(n_personas, em_iters_list, train_df, test_df, num_internal_nodes, cluster_centers, args, temp_model_meta):
    """
    Sub-process function to handle one specific n_personas across multiple iteration milestones.
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    results = []
    checkpoint_dir = os.path.join(args.output_dir, f"P{n_personas}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Sort em_iters to ensure incremental training
    sorted_iters = sorted(em_iters_list)
    
    current_resume_state = None
    last_iter = 0
    
    # Try to find the latest existing checkpoint
    for it in reversed(sorted_iters):
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_it{it}.pt")
        if os.path.exists(ckpt_path):
            print(f"  [P={n_personas}] Found existing checkpoint for iteration {it}. Loading...")
            current_resume_state = torch.load(ckpt_path, map_location=device)
            last_iter = it
            break

    for target_iter in sorted_iters:
        if target_iter <= last_iter:
            # Already calculated, but we need to load it for the next step or for results
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_it{target_iter}.pt")
            current_resume_state = torch.load(ckpt_path, map_location=device)
            
            # Calculate metrics if not already in global results (handled by main process usually)
            # For simplicity, we re-calculate or just skip if we assume global JSON is the source of truth
            continue 
        
        model_engine = LiteraryPersonaSAGE(
            n_personas=n_personas, 
            em_iters=target_iter, 
            l1_lambda=args.l1_lambda
        )
        model_engine.device = device
        
        # Sync meta info
        model_engine.R = temp_model_meta['R']
        model_engine.r_map = temp_model_meta['r_map']
        model_engine.word_paths = temp_model_meta['word_paths'].to(device)
        model_engine.word_signs = temp_model_meta['word_signs'].to(device)
        model_engine.vocab_clusters = temp_model_meta['vocab_clusters']
        model_engine.c_map = temp_model_meta['c_map']
        
        # Train
        model_engine.fit(train_df, num_internal_nodes, resume_state=current_resume_state)
        
        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_it{target_iter}.pt")
        model_engine.save_checkpoint(ckpt_path, target_iter)
        
        # Evaluation
        V_total = len(model_engine.vocab_clusters)
        char_word_counts_df = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().unstack(fill_value=0)
        char_word_counts_df = char_word_counts_df.reindex(columns=range(V_total), fill_value=0)
        char_dist = char_word_counts_df.values / (char_word_counts_df.values.sum(axis=1, keepdims=True) + 1e-10)
        
        silhouette = calculate_metrics(char_dist, model_engine.p_assignments, cluster_centers, n_personas)
        
        # Perplexity
        perplexity = calculate_perplexity(
            model_engine.model, 
            test_df, 
            model_engine.word_paths, 
            model_engine.word_signs, 
            device
        )
        
        res = {
            "n_personas": n_personas,
            "em_iters": target_iter,
            "silhouette": silhouette,
            "perplexity": perplexity,
            "timestamp": time.time()
        }
        
        # Save local result
        with open(os.path.join(checkpoint_dir, f"result_it{target_iter}.json"), 'w') as f:
            json.dump(res, f)
            
        current_resume_state = torch.load(ckpt_path, map_location=device)
        last_iter = target_iter

def run_optimized_grid_search(args):
    print(">>> Initializing Optimized Grid Search...")
    # Use a dummy model to load data and build tree once
    temp_model = LiteraryPersonaSAGE(n_personas=args.n_personas_list[0], em_iters=1)
    df, num_internal_nodes = temp_model.load_and_preprocess_data(args.data_file, args.word_csv_file)
    
    # Meta info to pass to sub-processes
    temp_model_meta = {
        'R': temp_model.R,
        'r_map': temp_model.r_map,
        'word_paths': temp_model.word_paths.cpu(),
        'word_signs': temp_model.word_signs.cpu(),
        'vocab_clusters': temp_model.vocab_clusters,
        'c_map': temp_model.c_map
    }

    df_words = pd.read_csv(args.word_csv_file)
    df_words['vector'] = df_words['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
    cluster_centers = np.vstack(df_words.groupby('cluster_id')['vector'].apply(lambda x: np.mean(np.vstack(x), axis=0)).values)
    
    # Pre-mapping
    authors = sorted(df["author"].unique())
    m_map = {a: i for i, a in enumerate(authors)}
    df['m_idx'] = df['author'].map(m_map)
    roles = ['agent', 'patient', 'possessive', 'predicative']
    r_map = {r: i for i, r in enumerate(roles)}
    df['r_idx'] = df['role'].map(r_map)
    df['w_idx'] = df['cluster_id'].map(temp_model.c_map)
    char_keys = sorted(df["char_key"].unique())
    c_map = {ck: i for i, ck in enumerate(char_keys)}
    df['c_idx'] = df['char_key'].map(c_map)

    np.random.seed(42)
    test_chars = np.random.choice(char_keys, size=int(len(char_keys) * 0.2), replace=False)
    train_df = df[~df['char_key'].isin(test_chars)].copy()
    test_df = df[df['char_key'].isin(test_chars)].copy()

    # Multiprocessing
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for n_personas in args.n_personas_list:
        p = mp.Process(target=train_and_eval, args=(
            n_personas, args.em_iters_list, train_df, test_df, 
            num_internal_nodes, cluster_centers, args, temp_model_meta
        ))
        p.start()
        processes.append(p)
        # Small delay to avoid concurrent disk/IO heavy initialization if needed
        time.sleep(2)

    for p in processes:
        p.join()

    # Final Merge
    all_results = []
    for n_personas in args.n_personas_list:
        checkpoint_dir = os.path.join(args.output_dir, f"P{n_personas}")
        for it in args.em_iters_list:
            res_path = os.path.join(checkpoint_dir, f"result_it{it}.json")
            if os.path.exists(res_path):
                with open(res_path, 'r') as f:
                    all_results.append(json.load(f))
    
    with open(args.output_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f">>> Grid search complete. Results saved to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="checkpoints")
    parser.add_argument('--output_json', type=str, default="grid_search_results.json")
    parser.add_argument('--n_personas_list', type=int, nargs='+', default=[4, 8, 12, 16])
    parser.add_argument('--em_iters_list', type=int, nargs='+', default=[50, 100, 150, 200])
    parser.add_argument('--l1_lambda', type=float, default=1e-6)
    
    args = parser.parse_args()
    run_optimized_grid_search(args)
