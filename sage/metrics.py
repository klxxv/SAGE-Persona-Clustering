import torch
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import torch.nn.functional as F

def calculate_silhouette(char_distributions, labels):
    """
    基于扁平词表分布（原始空间），计算角色的轮廓系数 (Silhouette Score)。
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
    return silhouette_score(char_distributions, labels, metric='cosine')

def calculate_latent_silhouette(model, df, device):
    """
    在潜空间 (Latent Space) 计算轮廓系数。
    这是衡量 VAE 编码器是否成功将角色在潜在流形上聚类的标准方法。
    """
    model.eval()
    with torch.no_grad():
        V_size = model.decoder.V
        C = df['c_idx'].max() + 1
        
        # 1. 重建特征矩阵
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((C, V_size))
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats = torch.tensor(char_feats, dtype=torch.float32).to(device)
        char_feats = F.normalize(char_feats, p=2, dim=1)
        
        # 2. 提取潜空间特征 (Encoder 输出的人格概率分布)
        persona_logits = model.encoder(char_feats)
        latent_probs = F.softmax(persona_logits, dim=-1).cpu().numpy()
        labels = np.argmax(latent_probs, axis=-1)
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return -1.0, labels
            
        # 在潜空间计算轮廓系数
        latent_s_score = silhouette_score(latent_probs, labels, metric='cosine')
        return latent_s_score, labels

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
        P = model.P
        C = df['c_idx'].max() + 1
        
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((C, V_size))
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats = torch.tensor(char_feats, dtype=torch.float32).to(device)
        char_feats = F.normalize(char_feats, p=2, dim=1)
        
        persona_logits = model.encoder(char_feats)
        z_persona = F.one_hot(torch.argmax(persona_logits, dim=-1), num_classes=P).float()
        
        current_z = z_persona[batch_c]
        log_probs = model.decoder(batch_m, current_z, batch_r)
        word_log_probs = log_probs[torch.arange(len(batch_w)), batch_w]
        total_log_likelihood = torch.sum(word_log_probs * batch_count).item()
        
        return np.exp(-total_log_likelihood / total_tokens)
