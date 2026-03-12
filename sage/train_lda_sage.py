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

class CharacterDataset(Dataset):
    """
    每个样本是一个角色及其所有的词频
    """
    def __init__(self, bow_matrix, role_labels):
        self.bow_matrix = bow_matrix
        self.role_labels = role_labels
    def __len__(self):
        return self.bow_matrix.size(0)
    def __getitem__(self, idx):
        return idx, self.bow_matrix[idx], self.role_labels[idx]

def train_lda_sage(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training LDA-SAGE (Mixed Persona Model) on {device}")

    # 1. 加载数据 (类似于 VAE 但需要 ID 索引)
    print(">>> Loading and Preprocessing data...")
    df = pd.read_csv(args.data_file)
    df_words = pd.read_csv(args.word_csv_file)
    
    vocab_clusters = sorted(df_words['cluster_id'].unique())
    V = len(vocab_clusters)
    cluster_map = {c: i for i, c in enumerate(vocab_clusters)}
    
    roles = ['agent', 'patient', 'possessive', 'predicative']
    r_map = {r: i for i, r in enumerate(roles)}
    df = df[df['role'].isin(roles)].copy()
    
    char_keys = sorted((df['book'] + "_" + df['char_id'].astype(str)).unique())
    C = len(char_keys)
    char_map = {ck: i for i, ck in enumerate(char_keys)}
    
    df['c_idx'] = (df['book'] + "_" + df['char_id'].astype(str)).map(char_map)
    df['r_idx'] = df['role'].map(r_map)
    df['w_idx'] = df['cluster_id'].map(cluster_map)
    
    # 构建 BoW 矩阵 (C x V)
    bow_matrix = torch.zeros((C, V))
    char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    bow_matrix[char_word_counts['c_idx'].values, char_word_counts['w_idx'].values] = torch.tensor(char_word_counts['count'].values, dtype=torch.float32)
    
    char_main_role = df.groupby('c_idx')['r_idx'].agg(lambda x: x.value_counts().index[0]).reindex(range(C), fill_value=0)
    role_labels = torch.tensor(char_main_role.values, dtype=torch.long)

    # 计算背景词频
    total_counts = bow_matrix.sum(dim=0)
    bg_log_freq = torch.log(total_counts / total_counts.sum() + 1e-10)

    # 2. 初始化模型
    model = LDASAGE(vocab_size=V, n_chars=C, n_personas=args.n_personas, n_roles=len(roles)).to(device)
    model.bg_log_freq.copy_(bg_log_freq.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    dataset = CharacterDataset(bow_matrix, role_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. 训练循环
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for char_idx, batch_bow, batch_role in pbar:
            char_idx = char_idx.to(device)
            batch_bow = batch_bow.to(device)
            batch_role = batch_role.to(device)
            
            optimizer.zero_grad()
            
            # Forward: 计算每个词的混合对数似然
            log_probs = model(char_idx, batch_role)
            
            # Loss: -sum(log_probs * counts) + Regularization
            loss = lda_sage_loss(
                log_probs, batch_bow, 
                model.theta_logits[char_idx], 
                l1_lambda=args.l1_lambda
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.2f}"})

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "lda_sage_best.pt"))

    # 4. 保存结果
    print(">>> Saving LDA-SAGE results...")
    model.eval()
    with torch.no_grad():
        theta = model.get_persona_dist().cpu().numpy()
        
    df_res = pd.DataFrame(theta, columns=[f"P{i}" for i in range(args.n_personas)])
    df_res['char_key'] = char_keys
    df_res['persona_label'] = np.argmax(theta, axis=1)
    df_res.to_csv(os.path.join(args.output_dir, "lda_persona_assignments.csv"), index=False)
    
    # 提取 Persona 关键词
    persona_keywords = {}
    eta_np = model.eta_p.detach().cpu().numpy()
    for p in range(args.n_personas):
        top_indices = np.argsort(eta_np[p])[-20:][::-1]
        persona_keywords[f"P{p}"] = [vocab_clusters[i] for i in top_indices]
        
    with open(os.path.join(args.output_dir, "lda_keywords.json"), 'w') as f:
        json.dump(persona_keywords, f, indent=4)
        
    print(f">>> LDA-SAGE Training Complete. Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="../results/lda_results")
    parser.add_argument('--n_personas', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l1_lambda', type=float, default=1e-4)
    
    args = parser.parse_args()
    train_lda_sage(args)
