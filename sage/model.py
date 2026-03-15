import os
import pandas as pd
import numpy as np
import heapq
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 哈夫曼树构建器 (Huffman Tree Builder)
# ==========================================

class HuffmanTreeBuilder:
    def __init__(self, word_counts):
        """
        word_counts: Dict[word_id, frequency]
        """
        self.word_counts = word_counts
        self.V = len(word_counts)
        self.parent_map = {}
        self.sign_map = {}
        self.next_node_id = self.V

    def build(self):
        print(f">>> Building Huffman Tree for {self.V} tokens...")
        # 优先级队列：(频率, 节点ID)
        heap = [[count, i] for i, count in self.word_counts.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            count1, node1 = heapq.heappop(heap)
            count2, node2 = heapq.heappop(heap)
            
            parent_node = self.next_node_id
            self.next_node_id += 1
            
            self.parent_map[node1] = parent_node
            self.sign_map[node1] = -1.0 # Left
            self.parent_map[node2] = parent_node
            self.sign_map[node2] = 1.0  # Right
            
            heapq.heappush(heap, [count1 + count2, parent_node])

        num_internal_nodes = self.next_node_id - self.V
        paths = {}
        for i in range(self.V):
            path, signs = [], []
            curr = i
            while curr in self.parent_map:
                p_node = self.parent_map[curr]
                path.append(p_node - self.V)
                signs.append(self.sign_map[curr])
                curr = p_node
            paths[i] = (path[::-1], signs[::-1])

        max_len = max(len(p[0]) for p in paths.values())
        word_paths = torch.full((self.V, max_len), num_internal_nodes, dtype=torch.long)
        word_signs = torch.zeros((self.V, max_len))
        for i in range(self.V):
            p, s = paths[i]
            word_paths[i, :len(p)] = torch.tensor(p)
            word_signs[i, :len(s)] = torch.tensor(s)
            
        return word_paths, word_signs, num_internal_nodes

# ==========================================
# 2. 现代扁平化解码器 (Flat Softmax Decoder)
# ==========================================

class FlatMixedEffectsDecoder(nn.Module):
    """
    扁平化解码器：Logits = E_bg + E_author[m] + E_persona[z]
    直接对全量词表输出 Softmax，不使用树。
    """
    def __init__(self, V, M, P, R):
        super().__init__()
        self.V = V
        # 背景偏置：[R, V]
        self.eta_bg = nn.Parameter(torch.zeros(R, V))
        # 作者效应：[M, R, V]
        self.eta_author = nn.Parameter(torch.zeros(M, R, V))
        # 人格效应：[P, R, V]
        self.eta_persona = nn.Parameter(torch.zeros(P, R, V))
        
    def forward(self, m_idx, z_persona, r_idx):
        """
        z_persona: [batch, P] (Gumbel-Softmax output)
        """
        batch_size = m_idx.shape[0]
        
        # 1. 背景
        bg = self.eta_bg[r_idx] # [batch, V]
        
        # 2. 作者
        # 使用 batch_indices 提取每个样本对应的作者
        author_w = self.eta_author[m_idx, r_idx] # [batch, V]
        
        # 3. 人格 (矩阵乘法解耦)
        # self.eta_persona: [P, R, V] -> reshape [P, R*V]
        pers_flat = self.eta_persona.view(z_persona.shape[1], -1)
        sampled_pers = torch.matmul(z_persona, pers_flat) # [batch, R*V]
        sampled_pers = sampled_pers.view(batch_size, -1, self.V) # [batch, R, V]
        
        # 提取对应 R 的 persona 向量
        batch_indices = torch.arange(batch_size, device=m_idx.device)
        persona_w = sampled_pers[batch_indices, r_idx] # [batch, V]
        
        logits = bg + author_w + persona_w
        return F.log_softmax(logits, dim=-1)

# ==========================================
# 3. 核心 VAE 模型 (结合 Flat Decoder)
# ==========================================

class SAGE_CVAE_Flat(nn.Module):
    def __init__(self, input_dim, M, P, R, hidden_dim=512):
        super().__init__()
        # Encoder: BoW -> Persona Logits
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, P)
        )
        # Decoder: Mixed Effects Flat
        self.decoder = FlatMixedEffectsDecoder(input_dim, M, P, R)
        self.P = P

    def forward(self, char_feats, m_idx, r_idx, temp=1.0, hard=True):
        persona_logits = self.encoder(char_feats)
        z_persona = F.gumbel_softmax(persona_logits, tau=temp, hard=hard)
        log_probs = self.decoder(m_idx, z_persona, r_idx)
        return log_probs, persona_logits, z_persona

# ==========================================
# 4. 统一训练器 (支持多种模式)
# ==========================================

class AdvancedLiterarySAGE:
    def __init__(self, n_personas=16, mode='cvae_flat', iters=100, l1_lambda=1e-5):
        self.P = n_personas
        self.mode = mode # 'cvae_flat' 或 'huffman_tree'
        self.iters = iters
        self.l1_lambda = l1_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | Mode: {self.mode}")

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

    def fit(self, df, batch_size=8192, checkpoint_dir='data/results/checkpoints'):
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 1. 基础索引映射
        authors = sorted(df["author"].unique())
        self.m_map = {a: i for i, a in enumerate(authors)}
        char_keys = sorted(df["char_key"].unique())
        self.char_map = {ck: i for i, ck in enumerate(char_keys)}
        
        df['m_idx'] = df['author'].map(self.m_map)
        df['c_idx'] = df['char_key'].map(self.char_map)
        df['r_idx'] = df['role'].map(self.r_map)
        
        self.M, self.C = len(self.m_map), len(self.char_map)

        # 2. 准备特征
        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats = np.zeros((self.C, self.V))
        char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
        char_feats = torch.tensor(char_feats, dtype=torch.float32).to(self.device)
        char_feats = F.normalize(char_feats, p=2, dim=1)

        # 3. 初始化模型
        if self.mode == 'cvae_flat':
            self.model = SAGE_CVAE_Flat(self.V, self.M, self.P, self.R).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # 准备数据数组
        all_m = torch.tensor(df['m_idx'].values, device=self.device)
        all_c = torch.tensor(df['c_idx'].values, device=self.device)
        all_r = torch.tensor(df['r_idx'].values, device=self.device)
        all_w = torch.tensor(df['w_idx'].values, device=self.device)
        all_count = torch.tensor(df['count'].values, dtype=torch.float32, device=self.device)

        n_samples = len(df)
        print(f">>> Training {self.mode} with {n_samples} samples (batch_size={batch_size})...")
        pbar = tqdm(range(self.iters))
        
        # 设定检查点保存频率
        save_interval = max(1, self.iters // 10)
        
        for it in pbar:
            self.model.train()
            
            # Shuffle data indices for epoch
            indices = torch.randperm(n_samples, device=self.device)
            
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_l1_loss = 0.0
            total_count = 0.0
            
            temp = max(0.5, 1.0 * np.exp(-0.01 * it))
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                
                batch_m = all_m[batch_idx]
                batch_c = all_c[batch_idx]
                batch_r = all_r[batch_idx]
                batch_w = all_w[batch_idx]
                batch_count = all_count[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward
                log_probs, persona_logits, z_persona = self.model(char_feats[batch_c], batch_m, batch_r, temp=temp)
                
                # Reconstruction Loss: NLL
                recon_loss = -torch.sum(log_probs[torch.arange(len(batch_w)), batch_w] * batch_count) / batch_count.sum()
                
                # KL Loss
                p_soft = F.softmax(persona_logits, dim=-1)
                kl_loss = torch.sum(p_soft * (torch.log(p_soft + 1e-10) - torch.log(torch.tensor(1.0/self.P))), dim=-1).mean()
                
                # L1 Loss (Disentanglement)
                l1_loss = torch.sum(torch.abs(self.model.decoder.eta_author)) + \
                          torch.sum(torch.abs(self.model.decoder.eta_persona))
                
                loss = recon_loss + 0.1 * kl_loss + self.l1_lambda * l1_loss
                loss.backward()
                optimizer.step()
                
                batch_sum_count = batch_count.sum().item()
                total_recon_loss += recon_loss.item() * batch_sum_count
                total_kl_loss += kl_loss.item() * batch_sum_count
                total_l1_loss += l1_loss.item() * batch_sum_count
                total_count += batch_sum_count

            avg_recon = total_recon_loss / total_count if total_count > 0 else 0
            avg_kl = total_kl_loss / total_count if total_count > 0 else 0
            avg_l1 = total_l1_loss / total_count if total_count > 0 else 0
            
            pbar.set_postfix({"Recon": f"{avg_recon:.4f}", "L1": f"{avg_l1:.4f}"})
            
            if (it + 1) % save_interval == 0 or (it + 1) == self.iters:
                ckpt_path = os.path.join(checkpoint_dir, f'cvae_model_iter_{it+1}.pt')
                torch.save(self.model.state_dict(), ckpt_path)

        # 分配人格
        self.model.eval()
        with torch.no_grad():
            self.p_assignments = torch.argmax(self.model.encoder(char_feats), dim=-1).cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cvae_flat', choices=['cvae_flat', 'huffman'])
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()
    
    # 示例运行逻辑可参考测试脚本
    print(f"Initializing model in {args.mode} mode...")
