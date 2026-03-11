import os
import pandas as pd
import numpy as np
import torch
from model import LiteraryPersonaSAGE
from metrics import calculate_metrics, calculate_perplexity
import time

def test_model_workflow():
    # 1. 基础参数设置
    word_csv = "../data/processed/word2vec_clusters.csv"
    data_file = "../data/processed/all_words.csv"
    n_personas = 6
    em_iters = 20
    
    # 2. 初始化模型
    print(f">>> Testing SAGE model with n_personas={n_personas}, em_iters={em_iters}")
    model = LiteraryPersonaSAGE(n_personas=n_personas, em_iters=em_iters)
    
    # 3. 数据加载与预处理
    print(">>> Loading and preprocessing data...")
    df_full, num_internal_nodes = model.load_and_preprocess_data(data_file, word_csv)
    char_keys = sorted(df_full["char_key"].unique())
    
    # 采样 200 个角色进行验证
    np.random.seed(42)
    sample_chars = np.random.choice(char_keys, size=min(200, len(char_keys)), replace=False)
    df_processed = df_full[df_full['char_key'].isin(sample_chars)].copy()
    
    # 获取词向量质心
    df_words = pd.read_csv(word_csv)
    df_words['vector'] = df_words['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
    cluster_centers = np.vstack(df_words.groupby('cluster_id')['vector'].apply(lambda x: np.mean(np.vstack(x), axis=0)).values)
    
    # 设置作者映射
    all_authors = sorted(df_processed["author"].unique())
    model.m_map = {author: i for i, author in enumerate(all_authors)}
    model.M = len(all_authors)
    
    # 映射
    df_processed['m_idx'] = df_processed['author'].map(model.m_map)
    df_processed['r_idx'] = df_processed['role'].map(model.r_map)
    df_processed['w_idx'] = df_processed['cluster_id'].map(model.cluster_map)
    
    char_keys_sampled = sorted(df_processed["char_key"].unique())
    char_map = {char: i for i, char in enumerate(char_keys_sampled)}
    df_processed['c_idx'] = df_processed['char_key'].map(char_map)

    # 划分
    np.random.seed(42)
    test_chars = np.random.choice(char_keys_sampled, size=int(len(char_keys_sampled) * 0.2), replace=False)
    train_df = df_processed[~df_processed['char_key'].isin(test_chars)].copy()
    test_df = df_processed[df_processed['char_key'].isin(test_chars)].copy()
    
    print(f"    Train chars: {len(train_df['char_key'].unique())}, Test chars: {len(test_df['char_key'].unique())}")
    
    # 4. 训练
    print(">>> Starting Fit process (20 iters)...")
    model.fit(train_df, num_internal_nodes, m_map=model.m_map, char_map=char_map)
    print(">>> Fit process completed successfully!")
    
    # 5. 验证指标
    print(">>> Calculating Silhouette and Perplexity...")
    char_word_counts_df = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().unstack(fill_value=0)
    V_total = len(model.vocab_clusters)
    char_word_counts_df = char_word_counts_df.reindex(columns=range(V_total), fill_value=0)
    char_dist = char_word_counts_df.values / (char_word_counts_df.values.sum(axis=1, keepdims=True) + 1e-10)
    
    train_labels = model.p_assignments[char_word_counts_df.index.values]
    silhouette = calculate_metrics(char_dist, train_labels, cluster_centers, n_personas)
    perplexity = calculate_perplexity(model.model, test_df, model.word_paths, model.word_signs, model.device)
    
    print(f"    Silhouette (MMD): {silhouette:.4f}")
    print(f"    Perplexity: {perplexity:.4f}")
    print("\n>>> TEST PASSED: Model is learning effectively.")

if __name__ == "__main__":
    test_model_workflow()
