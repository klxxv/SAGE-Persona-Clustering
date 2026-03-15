import torch
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

# ==========================================
# 1. 核心 MMD 距离计算 (推土机距离的高效近似)
# ==========================================
def calculate_mmd_distance_matrix(P, Q, K):
    """
    P: [N, V] - 样本分布
    Q: [M, V] - 质心分布 (可选)
    K: [V, V] - 词与词之间的核矩阵 (基于 W2V/BERT 距离)
    """
    # 如果 Q 是 None，计算 N 样本两两之间的距离
    if Q is None: Q = P
    
    PK = torch.matmul(P, K)
    pKp = torch.sum(PK * P, dim=1)
    
    QK = torch.matmul(Q, K)
    qKq = torch.sum(QK * Q, dim=1)
    
    pKq = torch.matmul(PK, Q.t())
    
    mmd_sq = pKp.unsqueeze(1) + qKq.unsqueeze(0) - 2 * pKq
    return torch.sqrt(torch.clamp(mmd_sq, min=0.0))

# ==========================================
# 2. 多维度轮廓系数计算函数
# ==========================================

def calculate_silhouette_custom(features, labels, metric='cosine'):
    """
    通用轮廓系数计算
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: return -1.0
    return silhouette_score(features, labels, metric=metric)

def calculate_mmd_silhouette(char_distributions, labels, word_vectors, sigma=None):
    """
    在语义空间通过 MMD 计算轮廓系数
    char_distributions: [C, V] 角色的词分布
    labels: [C] 人格分配
    word_vectors: [V, dim] 词向量 (BERT/W2V)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 计算词语间的 RBF 核矩阵 K
    print(f"    [Metrics] Building Kernel matrix for {word_vectors.shape[0]} tokens...")
    # 为了计算效率，如果词表太大，建议使用部分采样或 linear kernel
    # 这里使用简单的欧式距离 RBF
    dists_sq = squareform(pdist(word_vectors, metric='sqeuclidean'))
    if sigma is None:
        sigma = np.sqrt(np.median(dists_sq))
        if sigma == 0: sigma = 1.0
    
    K = torch.from_numpy(np.exp(-dists_sq / (2 * sigma**2))).float().to(device)
    P = torch.from_numpy(char_distributions).float().to(device)
    
    # 2. 计算 Persona 质心
    unique_labels = np.unique(labels)
    centroids = []
    for l in unique_labels:
        mask = (labels == l)
        centroids.append(P[mask].mean(dim=0))
    Q = torch.stack(centroids)
    
    # 3. 计算 MMD 距离矩阵并得到轮廓分数
    print("    [Metrics] Computing Batch MMD...")
    with torch.no_grad():
        all_dists = calculate_mmd_distance_matrix(P, Q, K).cpu().numpy()
    
    a = np.zeros(len(labels))
    b = np.zeros(len(labels))
    
    for i, label in enumerate(labels):
        # 映射到 unique_labels 索引
        label_idx = np.where(unique_labels == label)[0][0]
        a[i] = all_dists[i, label_idx]
        other_dists = np.delete(all_dists[i], label_idx)
        b[i] = np.min(other_dists)
        
    s_scores = (b - a) / np.maximum(a, b + 1e-10)
    return np.mean(s_scores)

def calculate_flat_perplexity(model, df, device):
    """计算扁平解码器的困惑度"""
    model.eval()
    with torch.no_grad():
        # 获取 batch 数据
        batch_m = torch.tensor(df['m_idx'].values, device=device)
        batch_c = torch.tensor(df['c_idx'].values, device=device)
        batch_r = torch.tensor(df['r_idx'].values, device=device)
        batch_w = torch.tensor(df['w_idx'].values, device=device)
        batch_count = torch.tensor(df['count'].values, dtype=torch.float32, device=device)
        
        total_tokens = batch_count.sum().item()
        if total_tokens == 0: return 0.0

        # 由于 C 很大，我们不一次性生成 char_feats 以免内存溢出
        # 直接使用模型的 Encoder 对当前 batch 里的角色进行预测
        # 注意：这里需要传入当前 batch 对应的 char_feats
        # 为了评估准确，我们预先计算好全量 char_feats (仅用于 eval)
        V_size = model.decoder.V
        C = df['c_idx'].max() + 1
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((C, V_size), dtype=np.float32)
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats = F.normalize(torch.tensor(char_feats), p=2, dim=1).to(device)
        
        persona_logits = model.encoder(char_feats)
        z_persona = F.one_hot(torch.argmax(persona_logits, dim=-1), num_classes=model.P).float()
        
        # 使用对应的 z_persona 进行解码
        current_z = z_persona[batch_c]
        log_probs = model.decoder(batch_m, current_z, batch_r)
        
        word_log_probs = log_probs[torch.arange(len(batch_w)), batch_w]
        total_log_likelihood = torch.sum(word_log_probs * batch_count).item()
        
        return np.exp(-total_log_likelihood / total_tokens)
