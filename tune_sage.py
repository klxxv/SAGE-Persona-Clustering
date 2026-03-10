import os
import argparse
import pandas as pd
import numpy as np
import torch
from SAGE_L1_clustering import LiteraryPersonaSAGE
from sage_metrics import calculate_metrics, calculate_perplexity
import json

def run_grid_search(args):
    # 1. 加载数据
    print(">>> Loading data for Grid Search...")
    # 这里我们复用 LiteraryPersonaSAGE 的加载逻辑
    temp_model = LiteraryPersonaSAGE(n_personas=args.n_personas_list[0], em_iters=10)
    df, num_internal_nodes = temp_model.load_and_preprocess_data(args.data_file, args.word_csv_file)
    
    # 获取词簇向量 (1000, dim)
    df_words = pd.read_csv(args.word_csv_file)
    df_words['vector'] = df_words['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
    cluster_centers = np.vstack(df_words.groupby('cluster_id')['vector'].apply(lambda x: np.mean(np.vstack(x), axis=0)).values)
    
    # 2. 划分训练集和测试集 (按角色划分)
    char_keys = df['char_key'].unique()
    np.random.seed(42)
    test_chars = np.random.choice(char_keys, size=int(len(char_keys) * 0.2), replace=False)
    
    train_df = df[~df['char_key'].isin(test_chars)].copy()
    test_df = df[df['char_key'].isin(test_chars)].copy()
    
    results = []
    
    # 3. 网格搜索
    for n_personas in args.n_personas_list:
        for em_iters in args.em_iters_list:
            print(f"\n===== Testing: n_personas={n_personas}, em_iters={em_iters} =====")
            
            model_engine = LiteraryPersonaSAGE(
                n_personas=n_personas, 
                em_iters=em_iters, 
                l1_lambda=args.l1_lambda
            )
            
            # 训练模型 (仅在训练集上)
            model_engine.fit(train_df, num_internal_nodes)
            
            # --- 指标 A: EMD 轮廓系数 ---
            # 提取所有角色的分布 [C, V]
            # 我们直接从模型内部或 df 中统计
            char_word_counts = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().unstack(fill_value=0).values
            # 归一化
            char_dist = char_word_counts / (char_word_counts.sum(axis=1, keepdims=True) + 1e-10)
            labels = model_engine.p_assignments # 训练集上的分配
            
            silhouette = calculate_metrics(char_dist, labels, cluster_centers, n_personas)
            
            # --- 指标 B: 困惑度 ---
            # 首先为测试集分配最可能的 persona (E-step logic)
            # 为了简化，我们直接在 fit 之后调用困惑度函数，该函数需要处理 test_df
            # 我们需要把测试集的 c_idx, m_idx 等重新映射
            # 此处略作简化：使用训练好的权重直接计算
            perplexity = calculate_perplexity(
                model_engine.model, 
                test_df, 
                model_engine.word_paths, 
                model_engine.word_signs, 
                model_engine.device
            )
            
            print(f"  >> Silhouette (EMD): {silhouette:.4f}")
            print(f"  >> Perplexity: {perplexity:.4f}")
            
            res = {
                "n_personas": n_personas,
                "em_iters": em_iters,
                "silhouette": silhouette,
                "perplexity": perplexity
            }
            results.append(res)
            
            # 实时保存结果
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_json', type=str, default="grid_search_results.json")
    parser.add_argument('--n_personas_list', type=int, nargs='+', default=[4, 6, 8, 12])
    parser.add_argument('--em_iters_list', type=int, nargs='+', default=[20, 50])
    parser.add_argument('--l1_lambda', type=float, default=1e-6)
    
    args = parser.parse_args()
    run_grid_search(args)
