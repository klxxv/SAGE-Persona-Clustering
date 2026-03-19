import torch
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import torch.nn.functional as F

# ==========================================
# 1. 核心轮廓系数计算工具
# ==========================================

def compute_silhouette(features, labels, metric='cosine'):
    """通用轮廓系数计算，处理单类别异常"""
    if len(np.unique(labels)) < 2:
        return -1.0
    try:
        return silhouette_score(features, labels, metric=metric)
    except Exception as e:
        return 0.0

# ==========================================
# 2. 三大维度轮廓系数分析
# ==========================================

def calculate_all_silhouettes(raw_features, persona_probs, persona_effects):
    """
    计算三个维度的轮廓系数：
    1. Raw Semantic: 原始语义空间 (基于原始输入特征)
    2. Weighted Semantic: 分类加权语义空间 (基于模型学习到的预期 Persona 效应)
    3. Latent Persona: Persona 概率空间 (基于编码器输出)
    
    Args:
        raw_features: [C, V] 归一化后的原始特征 (词频或词簇频率)
        persona_probs: [C, P] 每个角色属于每个 Persona 的概率
        persona_effects: [P, V] 每个 Persona 的语义偏移向量 (如果是多 Role，需预先平均或选择)
    """
    labels = np.argmax(persona_probs, axis=1)
    
    # 1. Raw Semantic Silhouette
    raw_s = compute_silhouette(raw_features, labels)
    
    # 2. Weighted Semantic Silhouette (Classification Weighted)
    # X_weighted = Probs * Effects -> 得到每个角色“预期”的 Persona 语义表现
    weighted_features = np.matmul(persona_probs, persona_effects)
    weighted_s = compute_silhouette(weighted_features, labels)
    
    # 3. Latent Persona Silhouette
    latent_s = compute_silhouette(persona_probs, labels)
    
    return {
        "raw_semantic": raw_s,
        "weighted_semantic": weighted_s,
        "latent_persona": latent_s
    }

# ==========================================
# 3. 后向兼容与专用辅助函数
# ==========================================

def calculate_latent_silhouette(model, df, device):
    """
    针对 CVAE 的快速计算 (Latent Space Only)
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
        
        score = compute_silhouette(persona_probs, labels)
        return score, labels

def calculate_distribution_silhouette(df, labels, V):
    """
    针对传统 SAGE 的快速计算 (Raw Space Only)
    """
    C = df['c_idx'].max() + 1
    char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_dist = np.zeros((C, V), dtype=np.float32)
    char_dist[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
    row_sums = char_dist.sum(axis=1, keepdims=True)
    char_dist = np.divide(char_dist, row_sums, out=np.zeros_like(char_dist), where=row_sums!=0)
    return compute_silhouette(char_dist, labels)

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
