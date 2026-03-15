import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 连续潜变量编码器 (Continuous Encoder)
# ==========================================
class CharacterContinuousEncoder(nn.Module):
    """
    Encoder: 将角色的全量 Token 词袋特征映射为 K 维连续潜空间的正态分布参数 (mu, logvar)
    """
    def __init__(self, input_dim, hidden_dim, K):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, K)
        self.fc_logvar = nn.Linear(hidden_dim, K)
        
    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.dropout(h, p=0.2, training=self.training)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# ==========================================
# 2. 连续特征轴解码器 (Continuous Axes Decoder)
# ==========================================
class FlatContinuousDecoder(nn.Module):
    """
    解码器: Logits = E_bg + E_author[m] + E_axes[z]
    使用连续的 z 向量 (坐标) 对 K 个特征轴进行线性组合
    """
    def __init__(self, V, M, K, R):
        super().__init__()
        self.V = V
        self.eta_bg = nn.Parameter(torch.zeros(R, V))
        self.eta_author = nn.Parameter(torch.zeros(M, R, V))
        
        # 这里的 eta_axes 不再是互斥的人格，而是 K 个独立的语义特征轴
        self.eta_axes = nn.Parameter(torch.zeros(K, R, V))
        
    def forward(self, m_idx, z, r_idx):
        """
        z: [batch, K] (从正态分布中采样出的连续坐标)
        """
        batch_size = m_idx.shape[0]
        
        bg = self.eta_bg[r_idx] # [batch, V]
        author_w = self.eta_author[m_idx, r_idx] # [batch, V]
        
        # 线性组合特征轴: z * eta_axes
        axes_flat = self.eta_axes.view(z.shape[1], -1)
        sampled_axes = torch.matmul(z, axes_flat) # [batch, R*V]
        sampled_axes = sampled_axes.view(batch_size, -1, self.V) # [batch, R, V]
        
        batch_indices = torch.arange(batch_size, device=m_idx.device)
        axes_w = sampled_axes[batch_indices, r_idx] # [batch, V]
        
        logits = bg + author_w + axes_w
        return F.log_softmax(logits, dim=-1)

# ==========================================
# 3. Beta-VAE 模型总成
# ==========================================
class SAGE_BetaVAE(nn.Module):
    def __init__(self, input_dim, M, K, R, hidden_dim=512):
        super().__init__()
        self.encoder = CharacterContinuousEncoder(input_dim, hidden_dim, K)
        self.decoder = FlatContinuousDecoder(input_dim, M, K, R)
        self.K = K

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu # 推理时直接使用均值

    def forward(self, char_feats, m_idx, r_idx):
        mu, logvar = self.encoder(char_feats)
        z = self.reparameterize(mu, logvar)
        log_probs = self.decoder(m_idx, z, r_idx)
        return log_probs, mu, logvar, z

# ==========================================
# 4. 训练器
# ==========================================
class BetaLiterarySAGE:
    def __init__(self, n_axes=5, beta=2.0, l1_lambda=1e-5, iters=100):
        self.K = n_axes
        self.beta = beta  # β-VAE 的核心参数，>1 鼓励强解耦
        self.l1_lambda = l1_lambda
        self.iters = iters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | Mode: Beta-VAE (Continuous) | Beta={self.beta}")

    def load_data(self, data_file, word_csv_file):
        df_words = pd.read_csv(word_csv_file)
        self.vocab = df_words['word'].tolist()
        self.word_map = {w: i for i, w in enumerate(self.vocab)}
        self.V = len(self.vocab)
        
        df = pd.read_csv(data_file)
        roles = ['agent', 'patient', 'possessive', 'predicative']
        df = df[df['role'].isin(roles)].copy()
        self.r_map = {r: i for i, r in enumerate(roles)}
        self.R = len(roles)
        
        df = df[df['word'].isin(self.word_map)].copy()
        df['w_idx'] = df['word'].map(self.word_map)
        df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
        return df

    def fit(self, df):
        authors = sorted(df["author"].unique())
        self.m_map = {a: i for i, a in enumerate(authors)}
        char_keys = sorted(df["char_key"].unique())
        self.char_map = {ck: i for i, ck in enumerate(char_keys)}
        
        df['m_idx'] = df['author'].map(self.m_map)
        df['c_idx'] = df['char_key'].map(self.char_map)
        df['r_idx'] = df['role'].map(self.r_map)
        
        self.M, self.C = len(self.m_map), len(self.char_map)

        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((self.C, self.V))
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats = torch.tensor(char_feats, dtype=torch.float32).to(self.device)
        char_feats = F.normalize(char_feats, p=2, dim=1)

        self.model = SAGE_BetaVAE(self.V, self.M, self.K, self.R).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        batch_m = torch.tensor(df['m_idx'].values, device=self.device)
        batch_c = torch.tensor(df['c_idx'].values, device=self.device)
        batch_r = torch.tensor(df['r_idx'].values, device=self.device)
        batch_w = torch.tensor(df['w_idx'].values, device=self.device)
        batch_count = torch.tensor(df['count'].values, dtype=torch.float32, device=self.device)

        dataset_size = len(batch_m)
        batch_size = 10000

        print(f">>> Training Beta-VAE with {self.K} continuous axes...")
        pbar = tqdm(range(self.iters))
        for it in pbar:
            self.model.train()
            permutation = torch.randperm(dataset_size, device=self.device)
            epoch_recon_loss = 0.0; epoch_kl_loss = 0.0; epoch_l1_loss = 0.0
            steps = 0
            
            for i in range(0, dataset_size, batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                b_m, b_c, b_r, b_w, b_count = batch_m[indices], batch_c[indices], batch_r[indices], batch_w[indices], batch_count[indices]
                
                log_probs, mu, logvar, z = self.model(char_feats[b_c], b_m, b_r)
                
                # 1. 重构损失
                recon_loss = -torch.sum(log_probs[torch.arange(len(b_w)), b_w] * b_count) / b_count.sum()
                
                # 2. 连续正态分布的 KL 散度 (逼近 N(0, I))
                # 均值计算时考虑 batch 大小以保持稳定
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
                
                # 3. L1 稀疏约束
                l1_loss = torch.sum(torch.abs(self.model.decoder.eta_author)) + \
                          torch.sum(torch.abs(self.model.decoder.eta_axes))
                
                # 引入 beta 权重
                loss = recon_loss + self.beta * kl_loss + self.l1_lambda * l1_loss
                loss.backward()
                optimizer.step()
                
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_l1_loss += l1_loss.item()
                steps += 1
            
            pbar.set_postfix({
                "Recon": f"{epoch_recon_loss/steps:.4f}", 
                "KL": f"{epoch_kl_loss/steps:.4f}", 
                "L1": f"{epoch_l1_loss/steps:.4f}"
            })

        self.model.eval()
        with torch.no_grad():
            self.z_assignments, _ = self.model.encoder(char_feats)
            self.z_assignments = self.z_assignments.cpu().numpy()
