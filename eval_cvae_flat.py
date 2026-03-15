import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sage.model import AdvancedLiterarySAGE, SAGE_CVAE_Flat
from sage.metrics import calculate_silhouette_custom, calculate_mmd_silhouette, calculate_flat_perplexity

def run_eval():
    # 1. 路径设置
    word_csv = "fullset_data/word2vec_clusters.csv" 
    bert_csv = "fullset_data/bert_clusters.csv"
    data_file = "fullset_data/all_words.csv"
    checkpoint_path = "checkpoints/cvae_flat_full/cvae_flat_full_model.pt"
    
    if not os.path.exists(word_csv):
        word_csv = "data/processed/word2vec_clusters.csv"
        data_file = "data/processed/all_words.csv"

    n_personas = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> [EVAL] Starting Multi-Dimensional Evaluation on {device}")

    # 2. 初始化环境与加载数据
    trainer = AdvancedLiterarySAGE(n_personas=n_personas, mode='cvae_flat')
    df_full = trainer.load_data(data_file, word_csv)
    
    # 手动对齐映射逻辑
    authors = sorted(df_full["author"].unique())
    trainer.m_map = {a: i for i, a in enumerate(authors)}
    char_keys = sorted(df_full["char_key"].unique())
    trainer.char_map = {ck: i for i, ck in enumerate(char_keys)}
    df_full['m_idx'] = df_full['author'].map(trainer.m_map)
    df_full['c_idx'] = df_full['char_key'].map(trainer.char_map)
    df_full['r_idx'] = df_full['role'].map(trainer.r_map)
    df_full['w_idx'] = df_full['word'].map(trainer.word_map)
    trainer.M, trainer.C = len(trainer.m_map), len(trainer.char_map)

    # 3. 加载模型
    trainer.model = SAGE_CVAE_Flat(trainer.V, trainer.M, trainer.P, trainer.R).to(device)
    print(f">>> Loading weights from {checkpoint_path}...")
    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    trainer.model.eval()

    # 4. 提取特征与人格分配
    print(">>> Running inference...")
    char_word_counts = df_full.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_feats_np = np.zeros((trainer.C, trainer.V), dtype=np.float32)
    char_feats_np[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
    
    char_feats_tensor = F.normalize(torch.tensor(char_feats_np), p=2, dim=1).to(device)
    
    with torch.no_grad():
        logits = trainer.model.encoder(char_feats_tensor)
        labels = torch.argmax(logits, dim=-1).cpu().numpy()
        logits_np = logits.cpu().numpy()
        probs_np = F.softmax(logits, dim=-1).cpu().numpy()

    # 5. 加载词向量用于语义评估
    def load_vectors(csv_path, vocab):
        df_v = pd.read_csv(csv_path)
        df_v['vector'] = df_v['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
        v_map = dict(zip(df_v['word'], df_v['vector']))
        return np.array([v_map[w] for w in vocab])

    print(">>> Loading Word2Vec and BERT embeddings...")
    w2v_vectors = load_vectors(word_csv, trainer.vocab)
    bert_vectors = load_vectors(bert_csv, trainer.vocab) if os.path.exists(bert_csv) else None

    # 6. 开始计算各指标
    print("\n" + "="*60)
    print(f"{'Space / Metric':<30} | {'Silhouette Score':<20}")
    print("-" * 60)

    # (1) 原始空间 (BoW + Cosine)
    # 使用 L1 归一化的分布来算 Cosine，更符合概率特性
    row_sums = char_feats_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    char_dists = char_feats_np / row_sums
    s_raw = calculate_silhouette_custom(char_dists, labels, metric='cosine')
    print(f"{'[Raw] BoW Distribution (Cos)':<30} | {s_raw:>18.4f}")

    # (2) Logits 空间 (Encoder Output + Cosine)
    s_logits = calculate_silhouette_custom(logits_np, labels, metric='cosine')
    print(f"{'[Latent] Raw Logits (Cos)':<30} | {s_logits:>18.4f}")

    # (3) Softmax 空间 (Encoder Probs + Cosine)
    s_probs = calculate_silhouette_custom(probs_np, labels, metric='cosine')
    print(f"{'[Latent] Softmax Probs (Cos)':<30} | {s_probs:>18.4f}")

    # (4) 语义空间: Word2Vec + Cosine
    # 角色在 W2V 空间中的表征 = 词分布与 W2V 矩阵的乘积
    char_w2v_feats = np.dot(char_dists, w2v_vectors)
    s_w2v_cos = calculate_silhouette_custom(char_w2v_feats, labels, metric='cosine')
    print(f"{'[Semantic] W2V Weighted (Cos)':<30} | {s_w2v_cos:>18.4f}")

    # (5) 语义空间: BERT + Cosine
    if bert_vectors is not None:
        char_bert_feats = np.dot(char_dists, bert_vectors)
        s_bert_cos = calculate_silhouette_custom(char_bert_feats, labels, metric='cosine')
        print(f"{'[Semantic] BERT Weighted (Cos)':<30} | {s_bert_cos:>18.4f}")

    # (6) 语义空间: Word2Vec + MMD (推土机距离近似)
    # 注意：MMD 计算量极大，我们对 42k 角色进行 2000 样本采样评估
    sample_size = 2000
    if trainer.C > sample_size:
        idx = np.random.choice(trainer.C, sample_size, replace=False)
        s_w2v_mmd = calculate_mmd_silhouette(char_dists[idx], labels[idx], w2v_vectors)
    else:
        s_w2v_mmd = calculate_mmd_silhouette(char_dists, labels, w2v_vectors)
    print(f"{'[Semantic] W2V + MMD (EMD)':<30} | {s_w2v_mmd:>18.4f}")

    print("="*60)

if __name__ == "__main__":
    run_eval()
