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
# 1. Huffman Tree Builder
# ==========================================

class HuffmanTreeBuilder:
    def __init__(self, word_counts):
        self.word_counts = word_counts
        self.V = len(word_counts)
        self.parent_map = {}
        self.sign_map = {}
        self.next_node_id = self.V

    def build(self):
        print(f">>> Building Huffman Tree for {self.V} tokens...")
        heap = [[count, i] for i, count in self.word_counts.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            count1, node1 = heapq.heappop(heap)
            count2, node2 = heapq.heappop(heap)
            parent_node = self.next_node_id
            self.next_node_id += 1
            self.parent_map[node1] = parent_node
            self.sign_map[node1] = -1.0
            self.parent_map[node2] = parent_node
            self.sign_map[node2] = 1.0
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
# 2. Flat Mixed Effects Decoder
# ==========================================

class FlatMixedEffectsDecoder(nn.Module):
    def __init__(self, V, M, P, R, role_mask=None, log_bg=None):
        super().__init__()
        self.V = V
        self.eta_bg = nn.Parameter(torch.zeros(R, V))
        if log_bg is not None:
            self.eta_bg.data.copy_(log_bg.unsqueeze(0).expand(R, V))
            self.eta_bg.requires_grad = False
            
        self.eta_author = nn.Parameter(torch.zeros(M, R, V))
        self.eta_persona = nn.Parameter(torch.zeros(P, R, V))
        self.register_buffer('role_mask', role_mask)

    def forward(self, m_idx, z_persona, r_idx):
        batch_size = m_idx.shape[0]
        bg = self.eta_bg[r_idx]
        author_w = self.eta_author[m_idx, r_idx]
        
        pers_flat = self.eta_persona.view(z_persona.shape[1], -1)
        sampled_pers = torch.matmul(z_persona, pers_flat)
        sampled_pers = sampled_pers.view(batch_size, -1, self.V)
        
        batch_indices = torch.arange(batch_size, device=m_idx.device)
        persona_w = sampled_pers[batch_indices, r_idx]
        
        logits = bg + author_w + persona_w
        if self.role_mask is not None:
            current_mask = self.role_mask[r_idx]
            logits = logits.masked_fill(current_mask == 0, -100.0) 
        return F.log_softmax(logits, dim=-1)

# ==========================================
# 3. SAGE_CVAE_Flat Model
# ==========================================

class SAGE_CVAE_Flat(nn.Module):
    def __init__(self, input_dim, M, P, R, role_mask=None, hidden_dim=512, log_bg=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, P)
        )
        self.decoder = FlatMixedEffectsDecoder(input_dim, M, P, R, role_mask=role_mask, log_bg=log_bg)
        self.P = P

    def forward(self, char_feats, m_idx, r_idx, temp=1.0, hard=True):
        persona_logits = self.encoder(char_feats)
        z_persona = F.gumbel_softmax(persona_logits, tau=temp, hard=hard)
        log_probs = self.decoder(m_idx, z_persona, r_idx)
        return log_probs, persona_logits, z_persona

# ==========================================
# 4. AdvancedLiterarySAGE Trainer
# ==========================================

class AdvancedLiterarySAGE:
    def __init__(self, n_personas=16, mode='cvae_flat', iters=100, l1_lambda=1.0):
        self.P = n_personas
        self.mode = mode
        self.iters = iters
        self.l1_lambda = l1_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.role_mask = None
        print(f"Using device: {self.device} | Mode: {self.mode}")

    def load_data(self, data_file, word_csv_file, use_clusters=False):
        df_words = pd.read_csv(word_csv_file)
        if use_clusters and 'cluster' in df_words.columns:
            self.word_to_cluster = dict(zip(df_words['word'], df_words['cluster']))
            self.vocab = sorted(df_words['cluster'].unique().tolist())
            self.word_map = {c: i for i, c in enumerate(self.vocab)}
            self.V = len(self.vocab)
        else:
            self.vocab = df_words['word'].tolist()
            self.word_map = {w: i for i, w in enumerate(self.vocab)}
            self.V = len(self.vocab)
            self.word_to_cluster = None

        df = pd.read_csv(data_file)
        roles = ['agent', 'patient', 'possessive', 'predicative']
        df = df[df['role'].isin(roles)].copy()
        self.r_map = {r: i for i, r in enumerate(roles)}
        self.R = len(roles)

        if self.word_to_cluster:
            df = df[df['word'].isin(self.word_to_cluster)].copy()
            df['w_idx'] = df['word'].map(self.word_to_cluster).map(self.word_map)
        else:
            df = df[df['word'].isin(self.word_map)].copy()
            df['w_idx'] = df['word'].map(self.word_map)
        
        word_counts_all = df.groupby('w_idx')['count'].sum()
        full_counts = np.zeros(self.V)
        for idx, count in word_counts_all.items():
            if idx < self.V: full_counts[int(idx)] = count
        word_probs = (full_counts + 1e-5) / (full_counts.sum() + 1e-5 * self.V)
        self.log_bg = torch.tensor(np.log(word_probs), dtype=torch.float32).to(self.device)

        role_mask = torch.zeros(self.R, self.V, device=self.device)
        for r_name, r_idx in self.r_map.items():
            valid_words = df[df['role'] == r_name]['w_idx'].unique()
            valid_words = [int(w) for w in valid_words if 0 <= w < self.V]
            role_mask[r_idx, valid_words] = 1.0
        self.role_mask = role_mask

        if 'book' in df.columns and 'char_id' in df.columns:
            df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
        elif 'book' in df.columns and 'character_id' in df.columns:
            df["char_key"] = df["book"] + "_" + df["character_id"].astype(str)
        else:
            df["char_key"] = "char_" + df.index.astype(str)
        return df

    def prepare_df(self, df):
        df = df.copy()
        if 'author' in df.columns: df['m_idx'] = df['author'].map(self.m_map)
        elif 'author_id' in df.columns: df['m_idx'] = df['author_id']
        if 'char_key' in df.columns and hasattr(self, 'char_map'): df['c_idx'] = df['char_key'].map(self.char_map)
        elif 'character_id' in df.columns: df['c_idx'] = df['character_id']
        if 'role' in df.columns: df['r_idx'] = df['role'].map(self.r_map)
        df = df.dropna(subset=['m_idx', 'c_idx', 'r_idx'])
        df['m_idx'] = df['m_idx'].astype(int)
        df['c_idx'] = df['c_idx'].astype(int)
        df['r_idx'] = df['r_idx'].astype(int)
        return df

    def fit(self, df, batch_size=8192, checkpoint_dir='data/results/checkpoints', 
            author_map_file=None, char_map_file=None, lr=1e-3, resume_path=None):
        import os
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if not hasattr(self, 'm_map'):
            authors = sorted(df["author"].unique())
            self.m_map = {a: i for i, a in enumerate(authors)}
            self.M = len(self.m_map)
        if not hasattr(self, 'char_map'):
            char_keys = sorted(df["char_key"].unique())
            self.char_map = {ck: i for i, ck in enumerate(char_keys)}
            self.C = len(self.char_map)
        df = self.prepare_df(df)

        # Save metadata info for post-processing
        temp_info = df.groupby("c_idx")[["author", "book", "m_idx"]].first().reindex(range(self.C)).ffill().bfill()
        self.char_info_df = temp_info

        char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
        char_feats_np = np.zeros((self.C, self.V))
        for _, row in char_word_counts.iterrows():
            c, w, cnt = int(row['c_idx']), int(row['w_idx']), row['count']
            if c < self.C and w < self.V: char_feats_np[c, w] = cnt
        
        char_feats = torch.tensor(char_feats_np, dtype=torch.float32).to(self.device)
        char_feats = F.normalize(char_feats, p=2, dim=1)

        if self.mode == 'cvae_flat':
            self.model = SAGE_CVAE_Flat(self.V, self.M, self.P, self.R, role_mask=self.role_mask, log_bg=self.log_bg).to(self.device)
        
        start_iter = 0
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                start_iter = checkpoint.get('iteration', 0)
            else: self.model.load_state_dict(checkpoint)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        all_m = torch.tensor(df['m_idx'].values, device=self.device)
        all_c = torch.tensor(df['c_idx'].values, device=self.device)
        all_r = torch.tensor(df['r_idx'].values, device=self.device)
        all_w = torch.tensor(df['w_idx'].values, device=self.device)
        all_count = torch.tensor(df['count'].values, dtype=torch.float32, device=self.device)

        n_samples = len(df)
        print(f">>> Training {self.mode} with {n_samples} samples...")
        pbar = tqdm(range(start_iter, self.iters), ascii=True)
        best_recon = float('inf')
        
        for it in pbar:
            self.model.train()
            indices = torch.randperm(n_samples, device=self.device)
            total_recon, n_tok = 0.0, 0.0
            temp = max(0.5, 1.0 * np.exp(-0.01 * it))
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                b_m, b_c, b_r, b_w, b_cnt = all_m[batch_idx], all_c[batch_idx], all_r[batch_idx], all_w[batch_idx], all_count[batch_idx]
                optimizer.zero_grad()
                log_probs, p_logits, z_pers = self.model(char_feats[b_c], b_m, b_r, temp=temp)
                recon_loss = -torch.sum(log_probs[torch.arange(len(b_w)), b_w] * b_cnt) / b_cnt.sum()
                p_soft = F.softmax(p_logits, dim=-1)
                kl_loss = torch.sum(p_soft * (torch.log(p_soft + 1e-10) - torch.log(torch.tensor(1.0/self.P))), dim=-1).mean()
                l1_loss = self.model.decoder.eta_author.abs().mean() + self.model.decoder.eta_persona.abs().mean()
                loss = recon_loss + 0.1 * kl_loss + self.l1_lambda * l1_loss
                loss.backward()
                optimizer.step()
                batch_sum = b_cnt.sum().item()
                total_recon += recon_loss.item() * batch_sum
                n_tok += batch_sum
            
            avg_recon = total_recon / n_tok
            pers_std = self.model.decoder.eta_persona.std().item()
            pbar.set_postfix({"Recon": f"{avg_recon:.4f}", "P_Std": f"{pers_std:.5f}"})
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            latest_file = os.path.join(checkpoint_dir, 'latest_model.pt')
            torch.save(self.model.state_dict(), latest_file)
            if avg_recon < best_recon:
                best_recon = avg_recon
                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
                    
        self.model.eval()
        with torch.no_grad():
            self.p_assignments = torch.argmax(self.model.encoder(char_feats), dim=-1).cpu().numpy()
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cvae_flat', choices=['cvae_flat', 'huffman'])
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()
    print(f"Initializing model in {args.mode} mode...")
