import os
import pandas as pd
import numpy as np
import torch
from model import AdvancedLiterarySAGE
from metrics import calculate_silhouette, calculate_latent_silhouette, calculate_flat_perplexity
import time

def test_flat_cvae_workflow():
    word_csv = "fullset_data/word2vec_clusters.csv" 
    data_file = "fullset_data/all_words.csv"
    
    if not os.path.exists(word_csv):
        word_csv = "data/processed/word2vec_clusters.csv"
        data_file = "data/processed/all_words.csv"

    n_personas = 8
    iters = 200 # 为快速验证折中设为 200
    
    # 稍微降低 L1 惩罚，防止坍缩过于严重；稍微提高 KL 权重，鼓励均匀分配
    print(f">>> Testing Flat CVAE-SAGE model with n_personas={n_personas}, iters={iters}")
    trainer = AdvancedLiterarySAGE(n_personas=n_personas, mode='cvae_flat', iters=iters, l1_lambda=5e-6)
    
    print(">>> Loading data...")
    df_full = trainer.load_data(data_file, word_csv)
    char_keys = sorted(df_full["char_key"].unique())
    
    np.random.seed(42)
    sample_chars = np.random.choice(char_keys, size=min(200, len(char_keys)), replace=False)
    df_processed = df_full[df_full['char_key'].isin(sample_chars)].copy()
    
    print(f"    Sampled chars: {len(df_processed['char_key'].unique())}")
    
    sampled_keys_list = list(df_processed['char_key'].unique())
    train_keys = sampled_keys_list[:int(len(sampled_keys_list) * 0.8)]
    test_keys = sampled_keys_list[int(len(sampled_keys_list) * 0.8):]
    
    train_df = df_processed[df_processed['char_key'].isin(train_keys)].copy()
    test_df = df_processed[df_processed['char_key'].isin(test_keys)].copy()

    print(f">>> Starting Flat CVAE Training ({iters} iters)...")
    start_time = time.time()
    trainer.fit(train_df)
    end_time = time.time()
    print(f">>> Training completed in {end_time - start_time:.2f}s")
    
    # --- 聚类指标评估 ---
    print("\n>>> Evaluating Clustering Metrics...")
    C = trainer.C
    V = trainer.V
    char_word_counts = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_feats = np.zeros((C, V))
    char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
    row_sums = char_feats.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    char_dists = char_feats / row_sums
    
    labels = trainer.p_assignments
    
    # 1. 原始空间轮廓系数 (充满噪音)
    raw_s_score = calculate_silhouette(char_dists, labels)
    print(f"    [Raw Space] Silhouette Score: {raw_s_score:.4f} (Expected to be low due to author/bg noise)")
    
    # 2. 潜空间轮廓系数 (提纯后的人格概率)
    latent_s_score, _ = calculate_latent_silhouette(trainer.model, train_df, trainer.device)
    print(f"    [Latent Space] Silhouette Score: {latent_s_score:.4f} (Measures actual persona separation)")
    
    perp = calculate_flat_perplexity(trainer.model, train_df, trainer.device)
    print(f"    Perplexity: {perp:.4f}")

    # --- 核心特性展示：提取人格的灵魂词汇 ---
    print("\n>>> Extracting Top Words for Learned Personas (Disentanglement check)...")
    vocab = trainer.vocab
    # 获取人格效应矩阵 (eta_persona: [P, R, V])
    eta_persona = trainer.model.decoder.eta_persona.detach().cpu().numpy()
    
    # 我们找出真正被分配了角色的人格
    active_personas = np.unique(labels)
    
    for p in active_personas:
        print(f"\n  Persona {p} (Assigned to {(labels == p).sum()} characters):")
        # 针对 4 种语法角色 (agent, patient, poss, pred) 打印前 5 个特征词
        role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}
        for r_idx, r_name in role_names.items():
            if r_idx < trainer.R:
                # 提取该人格在特定角色下的权重向量
                weights = eta_persona[p, r_idx, :]
                # 排序获取 top-5 的词汇索引
                top_indices = np.argsort(weights)[-5:][::-1]
                top_words = [vocab[i] for i in top_indices if weights[i] > 0]
                if top_words:
                    print(f"    {r_name}: {', '.join(top_words)}")
                else:
                    print(f"    {r_name}: (No strong positive features, suppressed by L1)")

    print("\n>>> TEST FINISHED.")

if __name__ == "__main__":
    test_flat_cvae_workflow()
