import torch
import numpy as np
import pandas as pd
import ast
import os
import sys

# 确保能导入 sage 目录下的模块
sys.path.append(os.getcwd())
from sage.metrics import calculate_metrics

def run_evaluation():
    print(">>> Loading results for Metric Calculation...")
    
    # 1. 加载分配结果和词簇数据
    assign_df = pd.read_csv('data/results/lda_results/lda_persona_assignments.csv')
    cluster_df = pd.read_csv('data/processed/word2vec_clusters.csv')
    
    # 2. 解析词簇向量 (V, dim)
    print(">>> Parsing and Aggregating word vectors per cluster...")
    def parse_vector(v_str):
        try:
            return np.array(ast.literal_eval(v_str))
        except:
            return np.fromstring(v_str.strip('[]'), sep=' ')

    cluster_df['parsed_vec'] = cluster_df['vector'].apply(parse_vector)
    
    # 按照 cluster_id 聚合，取均值作为词簇中心
    cluster_centers = cluster_df.groupby('cluster_id')['parsed_vec'].apply(lambda x: np.mean(np.stack(x), axis=0))
    
    # 确保顺序与 vocab_clusters 一致
    vocab_clusters = sorted(cluster_df['cluster_id'].unique())
    vectors = np.stack([cluster_centers[cid] for cid in vocab_clusters])
    
    # 3. 构建角色的词分布 P (C, V)
    # 我们需要重新构建训练时的 BoW 矩阵并归一化
    print(">>> Reconstructing normalized BoW matrix...")
    df_raw = pd.read_csv('data/processed/all_words.csv')
    
    # 简单过滤和映射
    vocab_clusters = sorted(cluster_df['cluster_id'].unique())
    cluster_map = {c: i for i, c in enumerate(vocab_clusters)}
    char_keys = sorted(assign_df['char_key'].unique())
    char_map = {ck: i for i, ck in enumerate(char_keys)}
    
    df_raw['char_key'] = df_raw['book'].astype(str) + "_" + df_raw['char_id'].astype(str)
    df_filtered = df_raw[df_raw['char_key'].isin(char_map)].copy()
    df_filtered['w_idx'] = df_filtered['word'].map(cluster_df.set_index('word')['cluster_id']).map(cluster_map)
    df_filtered = df_filtered.dropna(subset=['w_idx'])
    
    C, V = len(char_keys), len(vocab_clusters)
    counts_matrix = np.zeros((C, V))
    
    # 聚合计数
    char_word_counts = df_filtered.groupby(['char_key', 'w_idx'])['count'].sum().reset_index()
    for row in char_word_counts.itertuples():
        counts_matrix[char_map[row.char_key], int(row.w_idx)] = row.count
        
    # 归一化为概率分布 (和为1)
    row_sums = counts_matrix.sum(axis=1, keepdims=True)
    char_distributions = counts_matrix / (row_sums + 1e-10)
    
    # 4. 获取标签
    labels = assign_df.set_index('char_key').reindex(char_keys)['persona_label'].values
    
    # 5. 调用 calculate_metrics
    print(">>> Calculating MMD-based Silhouette Score...")
    # 由于 4 万个角色计算 MMD 全矩阵极其耗时且显存易溢出，我们采样 5000 个进行计算
    sample_idx = np.random.choice(C, min(5000, C), replace=False)
    
    avg_silhouette = calculate_metrics(
        char_distributions[sample_idx], 
        labels[sample_idx], 
        vectors, 
        n_personas=8
    )
    
    print(f"\n{'='*40}")
    print(f"MMD Silhouette Score: {avg_silhouette:.4f}")
    print(f"(Range: -1 to 1, higher is better)")
    print(f"{'='*40}")

if __name__ == "__main__":
    run_evaluation()
