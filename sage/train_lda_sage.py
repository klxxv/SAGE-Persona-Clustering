import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from lda_sage import LDASAGE, lda_sage_loss
from tqdm import tqdm
import json
import time

class CharacterRoleDataset(Dataset):
    def __init__(self, char_indices, author_indices, role_indices, bow_matrix):
        self.char_indices = char_indices
        self.author_indices = author_indices
        self.role_indices = role_indices
        self.bow_matrix = bow_matrix
    def __len__(self):
        return len(self.char_indices)
    def __getitem__(self, idx):
        return self.char_indices[idx], self.author_indices[idx], self.role_indices[idx], self.bow_matrix[idx]

def train_lda_sage(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training Dirichlet-regularized LDA-SAGE on {device}")

    # 1. Load Data
    df_raw = pd.read_csv(args.data_file)
    df_clusters = pd.read_csv(args.word_csv_file)
    df_raw['char_key'] = df_raw['book'].astype(str) + "_" + df_raw['char_id'].astype(str)
    
    # 2. Filtering for Target Scale (Target: ~4860 chars)
    print(f">>> Filtering characters with < {args.min_mentions} mentions...")
    char_totals = df_raw.groupby('char_key')['count'].sum()
    valid_chars = char_totals[char_totals >= args.min_mentions].index
    df_filtered = df_raw[df_raw['char_key'].isin(valid_chars)].copy()
    
    # 3. Mapping
    df = df_filtered.merge(df_clusters[['word', 'cluster_id']], on='word', how='inner')
    vocab_clusters = sorted(df['cluster_id'].unique())
    V = len(vocab_clusters)
    cluster_map = {c: i for i, c in enumerate(vocab_clusters)}
    
    authors = sorted(df['author'].unique())
    M = len(authors)
    author_map = {a: i for i, a in enumerate(authors)}
    
    roles = ['agent', 'patient', 'possessive', 'predicative']
    R = len(roles)
    r_map = {r: i for i, r in enumerate(roles)}
    df = df[df['role'].isin(roles)].copy()
    
    char_keys = sorted(df['char_key'].unique())
    C = len(char_keys)
    char_map = {ck: i for i, ck in enumerate(char_keys)}
    
    print(f">>> Dataset Scale: {C} characters, {M} authors, {V} clusters")

    # 4. Prepare Tensors
    grouped = df.groupby(['char_key', 'author', 'role'])
    char_indices, author_indices, role_indices, bow_list = [], [], [], []
    
    for (ck, auth, role), group in tqdm(grouped, desc="Building Batch Data"):
        vec = torch.zeros(V)
        w_indices = group['cluster_id'].map(cluster_map).values
        counts = group['count'].values
        vec[w_indices] = torch.tensor(counts, dtype=torch.float32)
        char_indices.append(char_map[ck]); author_indices.append(author_map[auth]); role_indices.append(r_map[role]); bow_list.append(vec)
        
    char_indices = torch.tensor(char_indices, dtype=torch.long)
    author_indices = torch.tensor(author_indices, dtype=torch.long)
    role_indices = torch.tensor(role_indices, dtype=torch.long)
    bow_matrix = torch.stack(bow_list)

    bg_log_freq = torch.log((bow_matrix.sum(dim=0) + 1) / (bow_matrix.sum() + V))

    # 5. Initialize Model
    model = LDASAGE(V, C, M, n_personas=args.n_personas, n_roles=R).to(device)
    model.bg_log_freq.copy_(bg_log_freq.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    loader = DataLoader(CharacterRoleDataset(char_indices, author_indices, role_indices, bow_matrix), batch_size=args.batch_size, shuffle=True)

    # 6. Training Loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for c_idx, m_idx, r_idx, batch_bow in pbar:
            c_idx, m_idx, r_idx, batch_bow = c_idx.to(device), m_idx.to(device), r_idx.to(device), batch_bow.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # Forward now returns log_probs and log_theta
                log_probs, log_theta = model(c_idx, m_idx, r_idx)
                loss = lda_sage_loss(
                    log_probs, batch_bow, log_theta, 
                    alpha=args.alpha, 
                    l1_lambda_eta=args.l1_lambda_eta, 
                    model=model
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # 7. Save Results
    print(">>> Saving Advanced Results...")
    model.eval()
    with torch.no_grad():
        theta = model.get_persona_dist().cpu().numpy()
    df_res = pd.DataFrame(theta, columns=[f"P{i}" for i in range(args.n_personas)])
    df_res['char_key'] = char_keys
    df_res['persona_label'] = np.argmax(theta, axis=1)
    df_res.to_csv(os.path.join(args.output_dir, "lda_persona_assignments.csv"), index=False)
    
    persona_keywords = {}
    eta_p_np = model.eta_p.detach().cpu().numpy()
    for p in range(args.n_personas):
        top_indices = np.argsort(eta_p_np[p])[-20:][::-1]
        persona_keywords[f"P{p}"] = [str(vocab_clusters[i]) for i in top_indices]
    with open(os.path.join(args.output_dir, "lda_keywords.json"), 'w') as f:
        json.dump(persona_keywords, f, indent=4)
    print(f">>> Dirichlet-regularized Training Complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="data/results/lda_results")
    parser.add_argument('--n_personas', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--l1_lambda_eta', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1, help="Dirichlet prior alpha (Sparsity if < 1.0)")
    parser.add_argument('--min_mentions', type=int, default=15)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_lda_sage(args)
