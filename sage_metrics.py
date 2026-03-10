import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def sinkhorn_distance_batch(P, Q, M, eps=0.01, max_iter=100):
    """
    高效并行计算分布 P 与 质心 Q 之间的最优传输距离 (Sinkhorn 近似)
    P: [C, V] - C个角色的词分布
    Q: [K, V] - K个Persona的中心词分布
    M: [V, V] - 词簇间的地面距离矩阵
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = P.to(device)
    Q = Q.to(device)
    M = M.to(device)
    
    K_kernel = torch.exp(-M / eps)
    
    # 结果容器 [C, K]
    results = torch.zeros((P.shape[0], Q.shape[0]), device=device)
    
    # 由于内存限制，分批处理
    batch_size = 500
    for i in range(0, P.shape[0], batch_size):
        P_batch = P[i:i+batch_size] # [B, V]
        
        # 对每一个 Persona 质心进行计算
        for j in range(Q.shape[0]):
            Q_j = Q[j].expand(P_batch.shape[0], -1) # [B, V]
            
            u = torch.ones_like(P_batch) / P_batch.shape[-1]
            for _ in range(30): # 简化迭代次数加速搜索
                v = Q_j / (torch.matmul(u, K_kernel) + 1e-8)
                u = P_batch / (torch.matmul(v, K_kernel.t()) + 1e-8)
            
            dist = torch.sum(u * torch.matmul(v * K_kernel, M.t()), dim=-1)
            results[i:i+batch_size, j] = dist
            
    return results

def calculate_metrics(char_distributions, labels, cluster_centers_v2w, n_personas):
    """
    计算 EMD 轮廓系数 和 困惑度
    char_distributions: numpy [C, V]
    labels: numpy [C] (persona assignments)
    cluster_centers_v2w: numpy [V, dim] (word2vec center of 1000 clusters)
    """
    # 1. 计算地面距离矩阵 M (词簇间的距离)
    print("  [Metrics] Computing ground distance matrix M...")
    M = squareform(pdist(cluster_centers_v2w, metric='euclidean'))
    M = torch.from_numpy(M).float()
    
    P = torch.from_numpy(char_distributions).float()
    
    # 2. 计算 Persona 质心 (词频分布的均值)
    print("  [Metrics] Computing Persona centroids...")
    centroids = []
    for k in range(n_personas):
        mask = (labels == k)
        if mask.any():
            centroids.append(P[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros(P.shape[1]))
    Q = torch.stack(centroids)
    
    # 3. 计算所有样本到所有质心的 EMD
    print("  [Metrics] Running Batch Sinkhorn...")
    with torch.no_grad():
        all_dists = sinkhorn_distance_batch(P, Q, M)
    
    # 4. 轮廓系数计算 (基于质心)
    all_dists = all_dists.cpu().numpy()
    a = np.zeros(len(labels)) # 到自己质心的距离
    b = np.zeros(len(labels)) # 到最近其他质心的距离
    
    for i, label in enumerate(labels):
        a[i] = all_dists[i, label]
        other_dists = np.delete(all_dists[i], label)
        b[i] = np.min(other_dists)
        
    s_scores = (b - a) / np.maximum(a, b)
    avg_silhouette = np.mean(s_scores)
    
    return avg_silhouette

def calculate_perplexity(model, test_df, word_paths, word_signs, device):
    """
    在留出集上计算困惑度
    """
    model.eval()
    with torch.no_grad():
        m_idx = torch.tensor(test_df['m_idx'].values, device=device)
        p_idx = torch.tensor(test_df['p_idx'].values, device=device)
        r_idx = torch.tensor(test_df['r_idx'].values, device=device)
        w_idx = torch.tensor(test_df['w_idx'].values, device=device)
        counts = torch.tensor(test_df['count'].values, dtype=torch.float32, device=device)
        
        node_paths = word_paths[w_idx]
        node_signs = word_signs[w_idx]
        
        log_probs = model(m_idx, p_idx, r_idx, node_paths, node_signs)
        total_log_likelihood = torch.sum(log_probs * counts).item()
        total_tokens = counts.sum().item()
        
        perplexity = np.exp(-total_log_likelihood / total_tokens)
        return perplexity
