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
    except Exception:
        return 0.0

# ==========================================
# 2. 四大维度轮廓系数分析
# ==========================================

def calculate_all_silhouettes(raw_features, persona_probs, persona_effects, vocab_embeddings=None):
    """
    计算多个维度的轮廓系数：
    1. Raw Semantic: 原始全量词频空间
    2. Filtered Semantic: 过滤公共词后的语义嵌入空间 (考虑词频 + 语义距离)
    3. Weighted Semantic: 分类加权语义空间 (Probs * Effects)
    4. Latent Persona: Persona 概率空间
    """
    labels = np.argmax(persona_probs, axis=1)
    results = {}
    
    # 1. Raw Semantic Silhouette (Full BoW)
    results["raw_semantic"] = compute_silhouette(raw_features, labels)
    
    # 2. Filtered Semantic Silhouette (The New Patch)
    if vocab_embeddings is not None:
        # A. 识别 Persona 间的公共词 (基于 persona_effects [P, V])
        P, V = persona_effects.shape
        top_k = 20
        # 统计每个词在多少个 Persona 中名列前茅
        hit_counts = np.zeros(V)
        for p in range(P):
            top_indices = np.argsort(persona_effects[p])[::-1][:top_k]
            hit_counts[top_indices] += 1
        
        # 定义公共词：在超过 50% 的 Persona 中出现的词
        common_mask = (hit_counts >= max(2, P * 0.5))
        
        # B. 过滤并投影到语义空间 (Weighted by Count)
        # filtered_raw: [C, V], 将公共词位置设为 0
        filtered_raw = raw_features.copy()
        filtered_raw[:, common_mask] = 0
        
        # C. 投影到 Embedding 空间: [C, V] * [V, D] -> [C, D]
        # vocab_embeddings 形状应为 [V, D]
        semantic_features = np.matmul(filtered_raw, vocab_embeddings)
        
        # 归一化语义向量以便计算余弦距离
        norms = np.linalg.norm(semantic_features, axis=1, keepdims=True)
        semantic_features = np.divide(semantic_features, norms, out=np.zeros_like(semantic_features), where=norms!=0)
        
        results["filtered_semantic"] = compute_silhouette(semantic_features, labels)
    else:
        results["filtered_semantic"] = 0.0

    # 3. Weighted Semantic Silhouette
    weighted_features = np.matmul(persona_probs, persona_effects)
    results["weighted_semantic"] = compute_silhouette(weighted_features, labels)
    
    # 4. Latent Persona Silhouette
    results["latent_persona"] = compute_silhouette(persona_probs, labels)
    
    return results

# ==========================================
# 3. 后向兼容辅助函数
# ==========================================

def calculate_latent_silhouette(model, df, device):
    model.eval()
    with torch.no_grad():
        C = df['c_idx'].max() + 1
        V = model.decoder.V
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((C, V), dtype=np.float32)
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats_tensor = torch.tensor(char_feats, dtype=torch.float32).to(device)
        char_feats_tensor = F.normalize(char_feats_tensor, p=2, dim=1)
        persona_probs = F.softmax(model.encoder(char_feats_tensor), dim=-1).cpu().numpy()
        labels = np.argmax(persona_probs, axis=1)
        return compute_silhouette(persona_probs, labels), labels

def calculate_flat_perplexity(model, df, device):
    model.eval()
    with torch.no_grad():
        batch_count = torch.tensor(df['count'].values, dtype=torch.float32, device=device)
        total_tokens = batch_count.sum().item()
        if total_tokens == 0: return 0.0
        
        # Re-fetch persona assignments for the character IDs in batch
        C = df['c_idx'].max() + 1
        V_size = model.decoder.V
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((C, V_size), dtype=np.float32)
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats = F.normalize(torch.tensor(char_feats), p=2, dim=1).to(device)
        persona_logits = model.encoder(char_feats)
        z_persona = F.one_hot(torch.argmax(persona_logits, dim=-1), num_classes=model.P).float()
        
        batch_m = torch.tensor(df['m_idx'].values, device=device)
        batch_c = torch.tensor(df['c_idx'].values, device=device)
        batch_r = torch.tensor(df['r_idx'].values, device=device)
        batch_w = torch.tensor(df['w_idx'].values, device=device)
        
        current_z = z_persona[batch_c]
        log_probs = model.decoder(batch_m, current_z, batch_r)
        word_log_probs = log_probs[torch.arange(len(batch_w)), batch_w]
        total_log_likelihood = torch.sum(word_log_probs * batch_count).item()
        return np.exp(-total_log_likelihood / total_tokens)
