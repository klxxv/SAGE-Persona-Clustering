
import torch
import pandas as pd
import numpy as np
from sage.metrics import calculate_metrics
import json
import os

def verify_and_analyze():
    ckpt_path = 'checkpoints/P4/checkpoint_it1000.pt'
    print(f">>> Verifying {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    p_assignments = ckpt['p_assignments']
    
    # Load mapping data
    word_csv = 'data/processed/word2vec_clusters.csv'
    data_file = 'data/processed/all_words.csv'
    df_words = pd.read_csv(word_csv)
    df_words['vector'] = df_words['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
    cluster_centers = np.vstack(df_words.groupby('cluster_id')['vector'].apply(lambda x: np.mean(np.vstack(x), axis=0)).values)
    
    # Extract assignments and distributions
    # Replicate filtering from model.py
    df = pd.read_csv(data_file)
    roles = ['agent', 'patient', 'possessive', 'predicative']
    df = df[df['role'].isin(roles)].copy()
    word_to_cluster = dict(zip(df_words.word, df_words.cluster_id))
    df['cluster_id'] = df['word'].map(word_to_cluster)
    df.dropna(subset=['cluster_id'], inplace=True)
    df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
    
    char_totals = df.groupby("char_key")["count"].sum()
    valid_keys = char_totals[char_totals >= 10].index
    df = df[df["char_key"].isin(valid_keys)].copy()
    
    char_keys = sorted(df["char_key"].unique())
    char_map = {ck: i for i, ck in enumerate(char_keys)}
    
    # Calculate distributions
    V_total = 1000
    df['w_idx'] = df['cluster_id'].map({c: i for i, c in enumerate(sorted(df_words.cluster_id.unique()))})
    char_word_counts = df.groupby(['char_key', 'w_idx'])['count'].sum().unstack(fill_value=0)
    char_word_counts = char_word_counts.reindex(columns=range(V_total), fill_value=0)
    char_dist = char_word_counts.values / (char_word_counts.values.sum(axis=1, keepdims=True) + 1e-10)
    
    # Align labels
    labels = np.array([p_assignments[char_map[ck]] for ck in char_word_counts.index])
    
    # Verify Silhouette
    print(">>> Calculating Silhouette Score (EMD)...")
    silhouette = calculate_metrics(char_dist, labels, cluster_centers, 4)
    print(f"Verified Silhouette Score: {silhouette:.4f}")
    
    # If silhouette is high, export the distinctive keywords
    # Based on traditional SAGE importance: we look at eta_pers
    eta_pers = ckpt['model_weights']['eta_pers'] # [P, R, V]
    # In traditional SAGE, personas are distinctive based on how they deviate from background
    # We average over roles to find general importance
    importance = eta_pers.mean(dim=1).numpy() # [P, 1000]
    
    vocab_clusters = sorted(df_words.cluster_id.unique())
    
    results = []
    for p in range(4):
        # Get top clusters by weight
        top_idx = np.argsort(importance[p])[-50:][::-1]
        for idx in top_idx:
            cid = vocab_clusters[idx]
            # Get most frequent word in this cluster for this persona to make it readable
            rep_word = df[(df['cluster_id'] == cid) & (df['char_key'].isin(char_word_counts.index[labels == p]))]['word'].mode()
            word = rep_word.values[0] if not rep_word.empty else f"C{cid}"
            results.append({"persona": p, "cluster_id": cid, "word": word, "weight": float(importance[p, idx])})
            
    pd.DataFrame(results).to_csv('data/results/p4_traditional_distinctive_keywords.csv', index=False)
    print(">>> Exported distinctive keywords to data/results/p4_traditional_distinctive_keywords.csv")

if __name__ == "__main__":
    verify_and_analyze()
