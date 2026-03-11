import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def mmd_distance_batch(P, Q, K):
    """
    使用最大均值差异 (MMD) 计算分布 P 与 质心 Q 之间的距离。
    P: [C, V] - C个角色的词分布 (已经归一化，和为1)
    Q: [K, V] - K个Persona的中心词分布 (已经归一化，和为1)
    K: [V, V] - 词簇间的核矩阵 (Kernel Matrix)，如 RBF 核
    
    公式: MMD^2(p, q) = p^T K p + q^T K q - 2 p^T K q
    """
    device = P.device
    
    # 1. 计算 P K P^T 的对角线项: [C]
    # (P K) * P -> sum over V
    PK = torch.matmul(P, K)
    pKp = torch.sum(PK * P, dim=1)
    
    # 2. 计算 Q K Q^T 的对角线项: [K]
    QK = torch.matmul(Q, K)
    qKq = torch.sum(QK * Q, dim=1)
    
    # 3. 计算交互项 P K Q^T: [C, K]
    pKq = torch.matmul(PK, Q.t())
    
    # 4. 组合得到 MMD^2 矩阵: [C, K]
    # 广播机制: [C, 1] + [1, K] - 2 * [C, K]
    mmd_sq = pKp.unsqueeze(1) + qKq.unsqueeze(0) - 2 * pKq
    
    # 确保数值稳定性，由于浮点误差可能产生极小的负数
    mmd_sq = torch.clamp(mmd_sq, min=0.0)
    
    # 返回 MMD 距离 (开方)
    return torch.sqrt(mmd_sq)

def calculate_metrics(char_distributions, labels, cluster_centers_v2w, n_personas, sigma=None):
    """
    计算基于 MMD 的轮廓系数 和 困惑度
    char_distributions: numpy [C, V]
    labels: numpy [C] (persona assignments)
    cluster_centers_v2w: numpy [V, dim] (word2vec center of 1000 clusters)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 计算地面距离矩阵 M 并转换为核矩阵 K
    print("  [Metrics] Computing RBF Kernel matrix K for MMD...")
    # 计算平方欧式距离
    M_sq = squareform(pdist(cluster_centers_v2w, metric='sqeuclidean'))
    
    # 启发式设置 sigma: 如果未提供，使用距离中位数的平方根
    if sigma is None:
        # 使用部分样本估计中位数以节省时间
        sigma = np.sqrt(np.median(M_sq))
        if sigma == 0: sigma = 1.0 # 防止除零
    
    K = np.exp(-M_sq / (2 * sigma**2))
    K = torch.from_numpy(K).float().to(device)
    
    P = torch.from_numpy(char_distributions).float().to(device)
    
    # 2. 计算 Persona 质心 (词频分布的均值)
    print("  [Metrics] Computing Persona centroids...")
    centroids = []
    for p in range(n_personas):
        mask = (labels == p)
        if mask.any():
            centroids.append(P[mask].mean(dim=0))
        else:
            # 机制回退：赋予均匀分布
            centroids.append(torch.ones(P.shape[1], device=device) / P.shape[1])

    Q = torch.stack(centroids)
    
    # 3. 计算所有样本到所有质心的 MMD 距离
    print("  [Metrics] Running Batch MMD...")
    with torch.no_grad():
        all_dists = mmd_distance_batch(P, Q, K)
    
    # 4. 轮廓系数计算 (基于 MMD 距离)
    all_dists = all_dists.cpu().numpy()
    a = np.zeros(len(labels)) # 到自己质心的距离
    b = np.zeros(len(labels)) # 到最近其他质心的距离
    
    for i, label in enumerate(labels):
        a[i] = all_dists[i, label]
        other_dists = np.delete(all_dists[i], label)
        b[i] = np.min(other_dists)
        
    s_scores = (b - a) / np.maximum(a, b + 1e-10)
    avg_silhouette = np.mean(s_scores)
    
    return avg_silhouette

def calculate_perplexity(model, test_df, word_paths, word_signs, device):
    """
    在留出集上计算困惑度。
    """
    model.eval()
    with torch.no_grad():
        char_groups = test_df.groupby('char_key')
        total_log_likelihood = 0.0
        total_tokens = 0.0
        
        P = model.eta_pers.shape[0]
        
        for char_key, char_df in char_groups:
            m_idx = torch.tensor(char_df['m_idx'].values, device=device)
            r_idx = torch.tensor(char_df['r_idx'].values, device=device)
            w_idx = torch.tensor(char_df['w_idx'].values, device=device)
            counts = torch.tensor(char_df['count'].values, dtype=torch.float32, device=device)
            
            node_paths = word_paths[w_idx]
            node_signs = word_signs[w_idx]
            
            char_persona_lls = []
            for p in range(P):
                p_idx = torch.full_like(m_idx, p)
                log_probs = model(m_idx, p_idx, r_idx, node_paths, node_signs)
                char_persona_lls.append(torch.sum(log_probs * counts).item())
            
            max_ll = max(char_persona_lls)
            total_log_likelihood += max_ll
            total_tokens += counts.sum().item()
        
        if total_tokens == 0:
            return 0.0
            
        perplexity = np.exp(-total_log_likelihood / total_tokens)
        return perplexity
