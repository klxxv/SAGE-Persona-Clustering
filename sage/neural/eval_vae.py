import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from vae_model import NeuralSAGE
from tqdm import tqdm
import json

def calculate_perplexity(model, bow_matrix, role_labels, device):
    """
    计算 VAE 模型在给定数据上的困惑度 (Perplexity)
    """
    model.eval()
    with torch.no_grad():
        bow_matrix = bow_matrix.to(device)
        role_labels = role_labels.to(device)
        
        # Forward pass
        log_recon, mu, logvar, theta = model(bow_matrix, role_labels)
        
        # log_prob * counts
        log_likelihood = (log_recon * bow_matrix).sum(dim=-1)
        total_tokens = bow_matrix.sum(dim=-1)
        
        # Per-token log-likelihood
        avg_ll = (log_likelihood / (total_tokens + 1e-10)).mean().item()
        perplexity = np.exp(-avg_ll)
        return perplexity

def calculate_silhouette(embeddings, labels):
    """
    计算轮廓系数
    """
    from sklearn.metrics import silhouette_score
    # 过滤掉样本数过少的聚类以防报错
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        return 0.0
    return silhouette_score(embeddings, labels)

def evaluate_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Evaluating Neural SAGE on {device}")

    # 1. 加载数据
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

    # 构建 BoW
    bow_matrix = torch.zeros((C, V))
    char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    bow_matrix[char_word_counts['c_idx'].values, char_word_counts['w_idx'].values] = torch.tensor(char_word_counts['count'].values, dtype=torch.float32)
    
    char_main_role = df.groupby('c_idx')['r_idx'].agg(lambda x: x.value_counts().index[0]).reindex(range(C), fill_value=0)
    role_labels = torch.tensor(char_main_role.values, dtype=torch.long)

    # 2. 初始化并加载模型
    model = NeuralSAGE(vocab_size=V, n_personas=args.n_personas, n_roles=len(roles)).to(device)
    model_path = os.path.join(args.model_dir, "vae_best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, "vae_model.pt")
    
    print(f">>> Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 执行评估
    print(">>> Running Metrics Calculation...")
    
    # 获取潜空间嵌入 (Inference)
    with torch.no_grad():
        h = model.encoder(bow_matrix.to(device))
        mu = model.fc_mu(h)
        theta = torch.softmax(mu, dim=-1).cpu().numpy()
        labels = np.argmax(theta, axis=1)

    # 轮廓系数 (基于角色在各个 Persona 上的分布向量)
    silhouette = calculate_silhouette(theta, labels)
    
    # 困惑度
    perplexity = calculate_perplexity(model, bow_matrix, role_labels, device)

    print(f"\n{'='*40}")
    print(f"VAE-SAGE Evaluation Results (P={args.n_personas}):")
    print(f"{'-'*40}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Perplexity:      {perplexity:.2f}")
    print(f"{'='*40}\n")

    # 保存评估指标
    metrics = {
        "n_personas": args.n_personas,
        "silhouette": silhouette,
        "perplexity": perplexity
    }
    with open(os.path.join(args.model_dir, "vae_eval_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True, help="Directory where vae_best_model.pt is located")
    parser.add_argument('--n_personas', type=int, default=8)
    
    args = parser.parse_args()
    evaluate_vae(args)
