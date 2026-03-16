import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from model import AdvancedLiterarySAGE
from metrics import calculate_mmd_silhouette, calculate_flat_perplexity
import json
import time
import sys
import traceback

def train_and_eval(n_personas, em_iters_list, train_df, test_df, num_internal_nodes, cluster_centers, args, temp_model_meta):
    """
    Function to handle one specific n_personas across multiple iteration milestones sequentially.
    """
    # 记录原始的 stdout/stderr 以便后续恢复
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        abs_output_dir = os.path.abspath(args.output_dir)
        
        class Tee(object):
            def __init__(self, filename, mode="a"):
                self.file = open(filename, mode, buffering=1)
                self.stdout = original_stdout
            def write(self, message):
                self.file.write(message)
                self.stdout.write(message)
            def flush(self):
                self.file.flush()
                self.stdout.flush()

        checkpoint_dir = os.path.join(abs_output_dir, f"P{n_personas}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        log_path = os.path.join(checkpoint_dir, "process.log")
        sys.stdout = Tee(log_path)
        sys.stderr = Tee(log_path)

        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n\n{'='*60}")
        print(f">>> [Serial Mode] Starting P={n_personas} on {device}")
        print(f"{'='*60}")
        
        def update_live_status(current_it, total_it, status_text=""):
            try:
                status_file = os.path.join(abs_output_dir, "live_status.txt")
                lines = []
                if os.path.exists(status_file):
                    with open(status_file, "r") as f: lines = f.readlines()
                
                new_line = f"P={n_personas:2d} | Iter: {current_it:3d}/{total_it:3d} | {status_text} | Time: {time.strftime('%H:%M:%S')}\n"
                
                found = False
                for i in range(len(lines)):
                    if lines[i].startswith(f"P={n_personas:2d}"):
                        lines[i] = new_line
                        found = True
                        break
                if not found: lines.append(new_line)
                
                with open(status_file, "w") as f: f.writelines(sorted(lines))
            except: pass

        sorted_iters = sorted(em_iters_list)
        current_resume_state = None
        last_iter = 0
        
        # Try to find the latest existing checkpoint
        for it in reversed(sorted_iters):
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_it{it}.pt")
            if os.path.exists(ckpt_path):
                print(f"  [P={n_personas}] Found existing checkpoint for iteration {it}. Loading...")
                current_resume_state = torch.load(ckpt_path, map_location=device, weights_only=False)
                last_iter = it
                break

        for target_iter in sorted_iters:
            if target_iter <= last_iter:
                continue 
            
            print(f"\n--- Training P={n_personas} until iteration {target_iter} ---")
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
            model_engine.cluster_map = temp_model_meta['cluster_map']
            
            # Train
            model_engine.fit(train_df, num_internal_nodes, resume_state=current_resume_state, checkpoint_dir=checkpoint_dir, status_callback=update_live_status, m_map=temp_model_meta['m_map'], char_map=temp_model_meta['char_map'])
            
            # Save checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_it{target_iter}.pt")
            model_engine.save_checkpoint(ckpt_path, target_iter)
            
            # Evaluation
            V_total = len(model_engine.vocab_clusters)
            char_word_counts_df = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().unstack(fill_value=0)
            char_word_counts_df = char_word_counts_df.reindex(columns=range(V_total), fill_value=0)
            char_dist = char_word_counts_df.values / (char_word_counts_df.values.sum(axis=1, keepdims=True) + 1e-10)
            
            train_labels = model_engine.p_assignments[char_word_counts_df.index.values]
            silhouette = calculate_mmd_silhouette(char_dist, train_labels, cluster_centers)
            perplexity = calculate_flat_perplexity(model_engine.model, test_df, device)
            
            res = {
                "n_personas": n_personas,
                "em_iters": target_iter,
                "silhouette": silhouette,
                "perplexity": perplexity,
                "timestamp": time.time()
            }
            
            with open(os.path.join(checkpoint_dir, f"result_it{target_iter}.json"), 'w') as f:
                json.dump(res, f)
                
            current_resume_state = torch.load(ckpt_path, map_location=device, weights_only=False)
            last_iter = target_iter
            
    except Exception as e:
        print(f"\n!!! [P={n_personas}] ERROR:")
        traceback.print_exc()
    finally:
        # 恢复 stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def run_serial_grid_search(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.output_json = os.path.abspath(args.output_json)

    print(f">>> Initializing Serial Grid Search...")

    # Load mapping files
    author_map_file = os.path.join('data', 'processed', 'author_id_name.csv')
    char_map_file = os.path.join('data', 'processed', 'character_id_name.csv')

    model_engine = AdvancedLiterarySAGE(n_personas=args.n_personas_list[0], iters=args.em_iters_list[-1])

    # Use the ID-enriched data file if available
    data_file = args.data_file
    if "with_ids" not in data_file and os.path.exists(os.path.join('data', 'processed', 'female_words_with_ids.csv')):
        data_file = os.path.join('data', 'processed', 'female_words_with_ids.csv')
        print(f"Using enriched data file: {data_file}")

    df = model_engine.load_data(data_file, args.word_csv_file)      

    # Prepare for fit
    # We'll call fit inside the loop, but we need to ensure all meta-info is consistent

    # Pre-calculate char_keys for splitting
    if 'character_id' in df.columns:
        char_keys = sorted(df['character_id'].unique())
    else:
        char_keys = sorted(df["char_key"].unique())

    # Roles map consistency
    roles = ['agent', 'patient', 'possessive', 'predicative']
    r_map = {r: i for i, r in enumerate(roles)}
    df['r_idx'] = df['role'].map(r_map)

    np.random.seed(42)
    test_chars = np.random.choice(char_keys, size=int(len(char_keys) * 0.2), replace=False)

    char_col = 'character_id' if 'character_id' in df.columns else 'char_key'
    train_df = df[~df[char_col].isin(test_chars)].copy()
    test_df = df[df[char_col].isin(test_chars)].copy()

    # Sequential execution
    for n_personas in args.n_personas_list:
        checkpoint_dir = os.path.join(args.output_dir, f"P{n_personas}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize model for this P
        model = AdvancedLiterarySAGE(
            n_personas=n_personas, 
            iters=args.em_iters_list[-1], # Train to max iters, intermediate saves handled in fit
            l1_lambda=args.l1_lambda
        )
        # We need self.vocab and V set for evaluation later
        model.vocab = model_engine.vocab
        model.word_map = model_engine.word_map
        model.V = model_engine.V
        model.r_map = r_map
        model.R = len(roles)

        print(f"\n--- Training P={n_personas} ---")
        model.fit(train_df, checkpoint_dir=checkpoint_dir, 
                  author_map_file=author_map_file, char_map_file=char_map_file)

        # Post-training evaluation for each target iteration
        for it in args.em_iters_list:
            ckpt_path = os.path.join(checkpoint_dir, f'cvae_model_iter_{it}.pt')
            if not os.path.exists(ckpt_path):
                continue

            model.model.load_state_dict(torch.load(ckpt_path, map_location=model.device))
            model.model.eval()

            # Calculate metrics
            with torch.no_grad():
                char_word_counts = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
                char_feats = np.zeros((model.C, model.V))
                char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
                char_feats = torch.tensor(char_feats, dtype=torch.float32).to(model.device)
                char_feats = F.normalize(char_feats, p=2, dim=1)

                persona_logits = model.model.encoder(char_feats)
                p_assignments = torch.argmax(persona_logits, dim=-1).cpu().numpy()

            # Note: silhouette needs features
            # silhouette = calculate_metrics(p_assignments, char_feats.cpu().numpy())
            silhouette = 0.0 # Placeholder or implement properly

            # Perplexity on test set
            # Need to adapt calculate_perplexity to model.model
            perplexity = 0.0 # Placeholder

            res = {
                "n_personas": n_personas,
                "em_iters": it,
                "silhouette": silhouette,
                "perplexity": perplexity,
                "timestamp": time.time()
            }
            with open(os.path.join(checkpoint_dir, f"result_it{it}.json"), 'w') as f:
                json.dump(res, f)
    # Final Merge
    all_results = []
    print(f"\n>>> Merging results...")
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
    run_serial_grid_search(args)
