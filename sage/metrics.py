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

def calculate_silhouette(features, labels, metric='cosine'):
    """
    Standard silhouette score. Returns -1.0 if only 1 cluster exists.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
    try:
        return silhouette_score(features, labels, metric=metric)
    except:
        return 0.0

def calculate_latent_silhouette(model, df, device):
    """
    Calculate silhouette score in the latent space of the VAE.
    """
    model.eval()
    with torch.no_grad():
        C = df['c_idx'].max() + 1
        V = model.decoder.V
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((C, V), dtype=np.float32)
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats_tensor = torch.tensor(char_feats, dtype=torch.float32).to(device)
        char_feats_tensor = F.normalize(char_feats_tensor, p=2, dim=1)
        
        persona_logits = model.encoder(char_feats_tensor)
        persona_probs = F.softmax(persona_logits, dim=-1).cpu().numpy()
        labels = np.argmax(persona_probs, axis=1)
        
        score = calculate_silhouette(persona_probs, labels)
        return score, labels

def calculate_distribution_silhouette(df, labels, V):
    """
    Calculate silhouette score based on raw word distributions of characters.
    Used for Traditional SAGE which has no latent space.
    """
    C = df['c_idx'].max() + 1
    char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_dist = np.zeros((C, V), dtype=np.float32)
    char_dist[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
    # Normalize to probabilities
    row_sums = char_dist.sum(axis=1, keepdims=True)
    char_dist = np.divide(char_dist, row_sums, out=np.zeros_like(char_dist), where=row_sums!=0)
    
    return calculate_silhouette(char_dist, labels)

def calculate_flat_perplexity(model, df, device):
    """计算扁平解码器的困惑度"""
    model.eval()
    with torch.no_grad():
        batch_m = torch.tensor(df['m_idx'].values, device=device)
        batch_c = torch.tensor(df['c_idx'].values, device=device)
        batch_r = torch.tensor(df['r_idx'].values, device=device)
        batch_w = torch.tensor(df['w_idx'].values, device=device)
        batch_count = torch.tensor(df['count'].values, dtype=torch.float32, device=device)
        
        total_tokens = batch_count.sum().item()
        if total_tokens == 0: return 0.0

        V_size = model.decoder.V
        C = df['c_idx'].max() + 1
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((C, V_size), dtype=np.float32)
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats = F.normalize(torch.tensor(char_feats), p=2, dim=1).to(device)
        
        persona_logits = model.encoder(char_feats)
        z_persona = F.one_hot(torch.argmax(persona_logits, dim=-1), num_classes=model.P).float()
        
        current_z = z_persona[batch_c]
        log_probs = model.decoder(batch_m, current_z, batch_r)
        
        word_log_probs = log_probs[torch.arange(len(batch_w)), batch_w]
        total_log_likelihood = torch.sum(word_log_probs * batch_count).item()
        
        return np.exp(-total_log_likelihood / total_tokens)
