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
        self.eta_axes = nn.Parameter(torch.zeros(K, R, V))
        
    def forward(self, m_idx, z):
        """
        m_idx: [Batch_size] 批次角色的作者ID
        z: [Batch_size, K] 批次角色的连续坐标
        """
        bg = self.eta_bg # [R, V]
        author_w = self.eta_author[m_idx] # [B, R, V]
        
        # 爱因斯坦求和约定 (einsum)：完美的高维矩阵乘法，避免巨大的显存开销
        # b: batch, k: axes, r: role, v: vocab
        axes_w = torch.einsum('bk, krv -> brv', z, self.eta_axes) # [B, R, V]
        
        # 混合效应叠加并计算概率
        logits = bg.unsqueeze(0) + author_w + axes_w # [B, R, V]
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

    def forward(self, char_feats, m_idx):
        mu, logvar = self.encoder(char_feats)
        z = self.reparameterize(mu, logvar)
        log_probs = self.decoder(m_idx, z)
        return log_probs, mu, logvar, z

# ==========================================
# 4. 训练器 (Character-Level Mini-batching)
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

        # 1. 构建角色的特征矩阵 [C, V] (CPU内存安全)
        print(">>> Preparing memory-efficient character features...")
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((self.C, self.V), dtype=np.float32)
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats_cpu = F.normalize(torch.tensor(char_feats), p=2, dim=1)

        # 2. 高效的数据索引预处理 (防止构建 5GB 的巨型目标矩阵导致 OOM)
        df_sorted = df.sort_values('c_idx')
        c_idx_np = df_sorted['c_idx'].values
        r_idx_arr = torch.tensor(df_sorted['r_idx'].values, dtype=torch.long)
        w_idx_arr = torch.tensor(df_sorted['w_idx'].values, dtype=torch.long)
        count_arr = torch.tensor(df_sorted['count'].values, dtype=torch.float32)
        
        char_start_idx = np.searchsorted(c_idx_np, np.arange(self.C), side='left')
        char_end_idx = np.searchsorted(c_idx_np, np.arange(self.C), side='right')

        char_to_author_cpu = torch.tensor(df.groupby('c_idx')['m_idx'].first().values, dtype=torch.long)

        # 3. 初始化模型
        self.model = SAGE_BetaVAE(self.V, self.M, self.K, self.R).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # 按照角色 (Character) 进行 Batch，而不是 Token！
        batch_size = 256 # 每次处理 256 个角色，极度节省显存

        print(f">>> Training Beta-VAE over {self.C} characters with Character-Level batch size {batch_size}...")
        pbar = tqdm(range(self.iters))
        for it in pbar:
            self.model.train()
            permutation = torch.randperm(self.C)
            epoch_recon_loss = 0.0; epoch_kl_loss = 0.0; epoch_l1_loss = 0.0
            steps = 0
            
            for i in range(0, self.C, batch_size):
                optimizer.zero_grad()
                b_c = permutation[i:i+batch_size] # 抽取 B 个角色
                
                # 将该批次的特征推入 GPU
                b_m = char_to_author_cpu[b_c].to(self.device)
                b_feats = char_feats_cpu[b_c].to(self.device)
                
                # 动态构建真实标签张量 [B, R, V]，只占用极小内存 (如 30MB)
                b_counts = torch.zeros((len(b_c), self.R, self.V), device=self.device)
                for batch_idx, char_id in enumerate(b_c):
                    start, end = char_start_idx[char_id], char_end_idx[char_id]
                    if start < end:
                        roles = r_idx_arr[start:end].to(self.device)
                        words = w_idx_arr[start:end].to(self.device)
                        cnts = count_arr[start:end].to(self.device)
                        b_counts[batch_idx, roles, words] = cnts
                
                total_words_in_batch = b_counts.sum()
                if total_words_in_batch == 0: continue
                
                # 前向传播 (极其迅速)
                log_probs, mu, logvar, z = self.model(b_feats, b_m)
                
                # 1. 准确的 ELBO 重构损失 (按 Token 均摊以稳定梯度)
                recon_loss = -torch.sum(log_probs * b_counts) / total_words_in_batch
                
                # 2. 准确的 KL 散度 (每个角色仅计算一次，不再跟单词数量挂钩)
                kl_divergences = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                kl_loss = torch.sum(kl_divergences) / total_words_in_batch
                
                # 3. L1 稀疏约束
                l1_loss = torch.sum(torch.abs(self.model.decoder.eta_author)) + \
                          torch.sum(torch.abs(self.model.decoder.eta_axes))
                
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
            self.z_assignments, _ = self.model.encoder(char_feats_cpu.to(self.device))
            self.z_assignments = self.z_assignments.cpu().numpy()
