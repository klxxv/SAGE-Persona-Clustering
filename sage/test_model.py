import os
import pandas as pd
import numpy as np
import torch
from model import LiteraryCVAE_SAGE
# from metrics import calculate_metrics, calculate_perplexity # 暂时注释，避免依赖冲突
import time

def test_cvae_workflow():
    # 1. 基础参数设置 (假设在项目根目录运行，路径使用相对根目录的路径)
    word_csv = "fullset_data/word2vec_clusters.csv"
    data_file = "fullset_data/all_words.csv"
    
    # 如果 fullset_data 不存在，尝试 data/processed
    if not os.path.exists(word_csv):
        word_csv = "data/processed/word2vec_clusters.csv"
        data_file = "data/processed/all_words.csv"

    if not os.path.exists(word_csv):
        print(f"Error: Data files not found at {word_csv}. Please check paths.")
        return

    n_personas = 6
    iters = 20 # 测试运行，迭代次数少一点
    
    # 2. 初始化模型
    print(f">>> Testing CVAE-SAGE model with n_personas={n_personas}, iters={iters}")
    model = LiteraryCVAE_SAGE(n_personas=n_personas, iters=iters, l1_lambda=1e-5)
    
    # 3. 数据加载与预处理
    print(">>> Loading and preprocessing data...")
    df_full, num_internal_nodes = model.load_and_preprocess_data(data_file, word_csv)
    char_keys = sorted(df_full["char_key"].unique())
    
    # 采样 100 个角色进行快速验证
    np.random.seed(42)
    sample_chars = np.random.choice(char_keys, size=min(100, len(char_keys)), replace=False)
    df_processed = df_full[df_full['char_key'].isin(sample_chars)].copy()
    
    print(f"    Sampled chars: {len(df_processed['char_key'].unique())}")
    
    # 4. 训练
    print(f">>> Starting CVAE Training ({iters} iters)...")
    start_time = time.time()
    model.fit(df_processed, num_internal_nodes)
    end_time = time.time()
    
    print(f">>> Training completed in {end_time - start_time:.2f}s")
    
    # 5. 验证基本输出
    print(f">>> Personas Assigned: {len(np.unique(model.p_assignments))}")
    print(f">>> Assignment distribution: {np.bincount(model.p_assignments)}")
    
    # 保存结果测试
    test_output_dir = "checkpoints/test_cvae_output"
    model.save_results(test_output_dir)
    
    if os.path.exists(os.path.join(test_output_dir, "cvae_character_personas.csv")):
        print("\n>>> TEST PASSED: CVAE-SAGE model workflow is functional.")
    else:
        print("\n>>> TEST FAILED: Output files not found.")

if __name__ == "__main__":
    test_cvae_workflow()
