import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from sage.model import AdvancedLiterarySAGE
from sage.metrics import calculate_silhouette, calculate_latent_silhouette, calculate_flat_perplexity

def check_cuda_environment():
    if torch.cuda.is_available():
        print(f">>> CUDA is available: {torch.version.cuda}")
        return True
    else:
        print(">>> CUDA not available, using CPU.")
        return False

def run_full_cvae(n_personas=8, iters=1000, batch_size=8192, subset_chars=None, lr=1e-3, l1_lambda=1e-6, 
                  resume_path=None, data_file=None, word_csv=None, use_clusters=False):
    check_cuda_environment()
    
    if word_csv is None: word_csv = "data/processed/female_vocab_map.csv"
    if data_file is None: data_file = "data/processed/female_words_with_ids.csv"

    print(f">>> CVAE-SAGE (Word-based): P={n_personas}, iters={iters}, l1={l1_lambda}")
    trainer = AdvancedLiterarySAGE(n_personas=n_personas, mode='cvae_flat', iters=iters, l1_lambda=l1_lambda)
    df_full = trainer.load_data(data_file, word_csv, use_clusters=use_clusters)
    
    if subset_chars is not None:
        unique_chars = df_full['char_key'].unique()
        # Randomized subset for better representation
        np.random.seed(42)
        subset_keys = np.random.choice(unique_chars, size=min(subset_chars, len(unique_chars)), replace=False)
        df_full = df_full[df_full['char_key'].isin(subset_keys)].copy()
        print(f"    Randomly subsetted to {len(subset_keys)} characters.")

    res_dir = os.path.abspath(f"data/results/cvae_results/P{n_personas}_L{l1_lambda}")
    ckpt_dir = os.path.abspath(f"checkpoints/cvae_P{n_personas}_L{l1_lambda}")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_time = time.time()
    train_df_mapped = trainer.fit(df_full, batch_size=batch_size, checkpoint_dir=ckpt_dir, lr=lr, resume_path=resume_path)
    print(f">>> Training completed in {time.time() - start_time:.2f}s")
    
    torch.save(trainer.model.state_dict(), os.path.join(res_dir, "final_model.pt"))

    df_results = trainer.char_info_df.copy()
    df_results["persona"] = trainer.p_assignments
    assign_path = os.path.join(res_dir, "char_assignments.csv")
    df_results.to_csv(assign_path, index=False)
    print(f">>> Assignments saved to {assign_path}")

    print(">>> Evaluating Metrics...")
    latent_s_score, labels = calculate_latent_silhouette(trainer.model, train_df_mapped, trainer.device)
    print(f"    Silhouette Score: {latent_s_score:.4f}")

    vocab = trainer.vocab
    eta_persona = trainer.model.decoder.eta_persona.detach().cpu().numpy()
    role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}
    records = []
    for p in range(n_personas):
        assigned_count = (trainer.p_assignments == p).sum()
        for r_idx, r_name in role_names.items():
            if r_idx < trainer.R:
                weights = eta_persona[p, r_idx, :]
                top_indices = np.argsort(weights)[-15:][::-1]
                for i in top_indices:
                    if weights[i] > 0:
                        records.append({'persona': p, 'role': r_name, 'word': vocab[i], 'weight': weights[i], 'chars': assigned_count})
    
    kw_path = os.path.join(res_dir, "keywords.csv")
    pd.DataFrame(records).to_csv(kw_path, index=False)
    print(f">>> Keywords saved to {kw_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_personas", type=int, default=8)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1.0)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--word_csv", type=str, default=None)
    args = parser.parse_args()
    run_full_cvae(n_personas=args.n_personas, iters=args.iters, batch_size=args.batch_size, 
                  subset_chars=args.subset, lr=args.lr, l1_lambda=args.l1, resume_path=args.resume,
                  data_file=args.data_file, word_csv=args.word_csv)
