import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from model import AdvancedLiterarySAGE
from metrics import calculate_silhouette, calculate_latent_silhouette, calculate_flat_perplexity

def check_cuda_environment():
    print("="*50)
    print(">>> CUDA Environment Check")
    print("="*50)
    if torch.cuda.is_available():
        print(f"CUDA is available: YES")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        print("="*50 + "\n")
        return True
    else:
        print("CUDA is available: NO")
        print("Warning: Training on CPU with full data will be extremely slow.")
        print("="*50 + "\n")
        return False

def run_full_cvae():
    # 1. 检查环境
    has_gpu = check_cuda_environment()
    
    # 2. 基础参数设置 (全量运行建议增加迭代次数)
    word_csv = "fullset_data/word2vec_clusters.csv" 
    data_file = "fullset_data/all_words.csv"
    
    if not os.path.exists(word_csv):
        word_csv = "data/processed/word2vec_clusters.csv"
        data_file = "data/processed/all_words.csv"

    if not os.path.exists(word_csv):
        print(f"Error: Data files not found at {word_csv}. Please check paths.")
        return

    n_personas = 8
    # GPU 下可以跑得更多，比如 1000 或 2000
    iters = 1000 if has_gpu else 50
    
    # 稍微降低 L1 惩罚以鼓励模型提取更多特征 (5e-6 -> 1e-6)
    print(f">>> Initializing Full Flat CVAE-SAGE with n_personas={n_personas}, iters={iters}")
    trainer = AdvancedLiterarySAGE(n_personas=n_personas, mode='cvae_flat', iters=iters, l1_lambda=1e-6)
    
    # 3. 数据加载 (不再限制采样，加载全量角色)
    print(">>> Loading full dataset...")
    df_full = trainer.load_data(data_file, word_csv)
    print(f"    Total characters loaded: {len(df_full['char_key'].unique())}")
    print(f"    Total unique tokens (V): {trainer.V}")
    
    # 划分 90% Train / 10% Test
    char_keys = list(df_full['char_key'].unique())
    np.random.seed(42)
    np.random.shuffle(char_keys)
    
    split_idx = int(len(char_keys) * 0.9)
    train_keys = char_keys[:split_idx]
    test_keys = char_keys[split_idx:]
    
    train_df = df_full[df_full['char_key'].isin(train_keys)].copy()
    test_df = df_full[df_full['char_key'].isin(test_keys)].copy()
    
    print(f"    Train characters: {len(train_keys)} | Test characters: {len(test_keys)}")

    # 4. 训练
    print(f"\n>>> Starting GPU Training ({iters} iters)...")
    start_time = time.time()
    trainer.fit(train_df)
    end_time = time.time()
    print(f">>> Training completed in {end_time - start_time:.2f}s")
    
    # 保存模型
    os.makedirs('data/results', exist_ok=True)
    torch.save(trainer.model.state_dict(), 'data/results/cvae_flat_full_model.pt')
    print(">>> Model weights saved to data/results/cvae_flat_full_model.pt")

    # 5. 验证基本输出
    labels = trainer.p_assignments
    unique_personas = np.unique(labels)
    print(f"\n>>> Personas Assigned: {len(unique_personas)}/{n_personas}")
    counts = np.bincount(labels, minlength=n_personas)
    print(f">>> Assignment distribution: {counts}")
    
    # 6. 评估
    print("\n>>> Evaluating Clustering Metrics...")
    C = trainer.C
    V = trainer.V
    char_word_counts = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_feats = np.zeros((C, V))
    char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
    
    # L1归一化
    row_sums = char_feats.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    char_dists = char_feats / row_sums
    
    raw_s_score = calculate_silhouette(char_dists, labels)
    latent_s_score, _ = calculate_latent_silhouette(trainer.model, train_df, trainer.device)
    perp = calculate_flat_perplexity(trainer.model, test_df, trainer.device)
    
    print(f"    [Raw Space] Silhouette Score   : {raw_s_score:.4f}")
    print(f"    [Latent Space] Silhouette Score: {latent_s_score:.4f}")
    print(f"    [Test Set] Perplexity          : {perp:.4f}")

    # 7. 提取特征词
    print("\n>>> Extracting Top Words for Learned Personas...")
    vocab = trainer.vocab
    eta_persona = trainer.model.decoder.eta_persona.detach().cpu().numpy()
    
    role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}
    
    # 将提取的关键词保存为 CSV 以便后续生成报告
    records = []
    for p in unique_personas:
        print(f"\n  Persona {p} (Assigned to {(labels == p).sum()} characters):")
        for r_idx, r_name in role_names.items():
            if r_idx < trainer.R:
                weights = eta_persona[p, r_idx, :]
                top_indices = np.argsort(weights)[-10:][::-1] # 提取前 10 个词
                top_words = []
                for i in top_indices:
                    if weights[i] > 0:
                        top_words.append(vocab[i])
                        records.append({'persona': p, 'role': r_name, 'word': vocab[i], 'weight': weights[i]})
                        
                if top_words:
                    print(f"    {r_name}: {', '.join(top_words)}")
                else:
                    print(f"    {r_name}: (No strong positive features)")

    pd.DataFrame(records).to_csv('data/results/cvae_full_keywords.csv', index=False)
    print("\n>>> Feature keywords saved to data/results/cvae_full_keywords.csv")
    print(">>> FULL RUN FINISHED.")

if __name__ == "__main__":
    run_full_cvae()
