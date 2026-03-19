import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from sage.model import AdvancedLiterarySAGE
from sage.metrics import calculate_all_silhouettes

def check_cuda_environment():
    if torch.cuda.is_available():
        print(f">>> CUDA is available: {torch.version.cuda}")
        return True
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
        np.random.seed(42)
        subset_keys = np.random.choice(unique_chars, size=min(subset_chars, len(unique_chars)), replace=False)
        df_full = df_full[df_full['char_key'].isin(subset_keys)].copy()

    res_dir = os.path.abspath(f"data/results/cvae_results/P{n_personas}_L{l1_lambda}")
    ckpt_dir = os.path.abspath(f"checkpoints/cvae_P{n_personas}_L{l1_lambda}")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_time = time.time()
    train_df_mapped = trainer.fit(df_full, batch_size=batch_size, checkpoint_dir=ckpt_dir, lr=lr, resume_path=resume_path)
    print(f">>> Training completed in {time.time() - start_time:.2f}s")
    
    torch.save(trainer.model.state_dict(), os.path.join(res_dir, "final_model.pt"))

    # 1. Export Assignments
    df_results = trainer.char_info_df.copy()
    df_results["persona"] = trainer.p_assignments
    df_results.to_csv(os.path.join(res_dir, "char_assignments.csv"), index=False)

    # 2. Comprehensive Silhouette Analysis
    print(">>> Performing Multi-dimensional Silhouette Analysis...")
    trainer.model.eval()
    with torch.no_grad():
        # Get raw features (normalized bow)
        C = train_df_mapped['c_idx'].max() + 1
        V = trainer.V
        char_word_counts = train_df_mapped.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        raw_features = np.zeros((C, V), dtype=np.float32)
        raw_features[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        row_sums = raw_features.sum(axis=1, keepdims=True)
        raw_features = np.divide(raw_features, row_sums, out=np.zeros_like(raw_features), where=row_sums!=0)
        
        # Get persona probabilities
        char_feats_tensor = torch.tensor(raw_features, dtype=torch.float32).to(trainer.device)
        char_feats_tensor = F.normalize(char_feats_tensor, p=2, dim=1)
        persona_probs = F.softmax(trainer.model.encoder(char_feats_tensor), dim=-1).cpu().numpy()
        
        # Get averaged persona effects [P, V]
        # model.decoder.eta_persona is [P, R, V]
        persona_effects = trainer.model.decoder.eta_persona.mean(dim=1).detach().cpu().numpy()
        
        scores = calculate_all_silhouettes(raw_features, persona_probs, persona_effects)
        for k, v in scores.items():
            print(f"    {k:20s}: {v:.4f}")

    # 3. Extract Keywords
    vocab = trainer.vocab
    role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}
    records = []
    eta_persona_np = trainer.model.decoder.eta_persona.detach().cpu().numpy()
    for p in range(n_personas):
        assigned_count = (trainer.p_assignments == p).sum()
        for r_idx, r_name in role_names.items():
            if r_idx < trainer.R:
                weights = eta_persona_np[p, r_idx, :]
                top_indices = np.argsort(weights)[-15:][::-1]
                for i in top_indices:
                    if weights[i] > 0:
                        records.append({'persona': p, 'role': r_name, 'word': vocab[i], 'weight': weights[i], 'chars': assigned_count})
    pd.DataFrame(records).to_csv(os.path.join(res_dir, "keywords.csv"), index=False)
    print(f">>> Results saved to {res_dir}")

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
