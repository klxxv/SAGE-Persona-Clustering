import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.cluster.hierarchy as sch
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 修复后的层次化 SAGE 模型 (处理了 Padding)
# ==========================================
class HierarchicalSAGE(nn.Module):
    def __init__(self, M, P, num_internal_nodes):
        super().__init__()
        # 【修复1】多加 1 个维度作为 Padding 节点 (索引为 num_internal_nodes)
        self.padding_idx = num_internal_nodes
        self.eta_bg = nn.Parameter(torch.zeros(self.padding_idx + 1))
        self.eta_meta = nn.Parameter(torch.zeros(M, self.padding_idx + 1))
        self.eta_pers = nn.Parameter(torch.zeros(P, self.padding_idx + 1))
        
    def forward(self, m_idx, p_idx, node_paths, node_signs):
        batch_size, max_path_len = node_paths.shape

        m_idx_expanded = m_idx.unsqueeze(1).expand(-1, max_path_len)
        p_idx_expanded = p_idx.unsqueeze(1).expand(-1, max_path_len)

        bg = self.eta_bg[node_paths]
        meta = self.eta_meta[m_idx_expanded, node_paths]
        pers = self.eta_pers[p_idx_expanded, node_paths]
        
        logits = bg + meta + pers
        
        # 【修复1】使用正确的 padding_idx 进行掩码
        path_mask = (node_paths != self.padding_idx).float()
        
        log_probs = F.logsigmoid(node_signs * logits) * path_mask
        word_log_probs = log_probs.sum(dim=1)
        
        return word_log_probs

# ==========================================
# 2. 核心 SAGE 训练器
# ==========================================
class LiteraryPersonaSAGE:
    def __init__(self, n_personas=8, alpha=1.0, l1_lambda=0.1, m_epochs=10, em_iters=50, min_mentions=10):
        self.P = n_personas
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.m_epochs = m_epochs # 【修复2】M-步的内循环轮数
        self.iters = em_iters
        self.min_mentions = min_mentions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _build_semantic_tree(self, vocab, word_vectors=None):
        """
        【修复3】基于词向量的凝聚聚类 (Agglomerative Clustering) 构建二叉树
        """
        print(">>> Building a semantic binary tree for Hierarchical Softmax...")
        V = len(vocab)
        
        # 如果没有提供词向量，为了脚本能跑，生成随机向量（真实场景请传入 Word2Vec 向量）
        if word_vectors is None:
            print("    [Warning] No word vectors provided. Using random vectors for clustering.")
            word_vectors = np.random.randn(V, 50) 
            
        
        
        # 使用 Ward 链接法进行层次聚类
        Z = sch.linkage(word_vectors, method='ward')
        
        # Z 的形状是 (V-1, 4)。每行代表一次合并，产生一个内部节点
        num_internal_nodes = V - 1
        
        # 记录每个节点的父节点和走向 (左=-1, 右=1)
        parent_map = {}
        sign_map = {}
        
        for i, row in enumerate(Z):
            internal_node_idx = i # 内部节点 ID (0 到 V-2)
            left_child = int(row[0])
            right_child = int(row[1])
            
            parent_map[left_child] = internal_node_idx
            sign_map[left_child] = -1.0
            
            parent_map[right_child] = internal_node_idx
            sign_map[right_child] = 1.0

        # 为每个词汇生成路径
        paths = {}
        for i in range(V):
            path = []
            signs = []
            curr = i
            while curr in parent_map:
                p_node = parent_map[curr]
                path.append(p_node)
                signs.append(sign_map[curr])
                curr = p_node
            
            # 路径是从叶子到根，通常反转为从根到叶子
            paths[i] = (path[::-1], signs[::-1])

        max_len = max(len(p[0]) for p in paths.values())
        
        # 【修复1】用 num_internal_nodes 作为 Padding 填充空白
        self.word_paths = torch.full((V, max_len), num_internal_nodes, dtype=torch.long)
        self.word_signs = torch.zeros((V, max_len))

        for i in range(V):
            path, signs = paths[i]
            self.word_paths[i, :len(path)] = torch.tensor(path)
            self.word_signs[i, :len(signs)] = torch.tensor(signs)
            
        print(f"    Tree built. Vocab size: {V}, Internal nodes: {num_internal_nodes}, Max Depth: {max_len}")
        return num_internal_nodes

    def load_and_preprocess_data(self, data_file, cluster_file):
        df = pd.read_csv(data_file)
        df_clusters = pd.read_csv(cluster_file)
        word_to_cluster = dict(zip(df_clusters.word, df_clusters.cluster_id))
        df['cluster_id'] = df['word'].map(word_to_cluster)
        df.dropna(subset=['cluster_id'], inplace=True)
        df['cluster_id'] = df['cluster_id'].astype(int)
        df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
        char_totals = df.groupby("char_key")["count"].sum()
        valid_keys = char_totals[char_totals >= self.min_mentions].index
        df = df[df["char_key"].isin(valid_keys)].copy()
        df["feat"] = df["role"] + ":" + df["cluster_id"].astype(str)
        return df

    def fit(self, df, pre_trained_word_vectors=None):
        self.vocab = sorted(df["feat"].unique())
        self.v_map = {feat: i for i, feat in enumerate(self.vocab)}
        self.V = len(self.vocab)
        
        num_internal_nodes = self._build_semantic_tree(self.vocab, pre_trained_word_vectors)
        self.word_paths = self.word_paths.to(self.device)
        self.word_signs = self.word_signs.to(self.device)

        authors = sorted(df["author"].unique())
        self.m_map = {author: i for i, author in enumerate(authors)}
        self.M = len(authors)

        char_keys = sorted(df["char_key"].unique())
        c_map = {ck: i for i, ck in enumerate(char_keys)}
        self.C = len(char_keys)
        
        print(f"Clustered Vocab (V): {self.V}, Authors (M): {self.M}, Chars (C): {self.C}")

        temp_info = df.groupby("char_key")[["author", "book"]].first().reindex(char_keys)
        self.char_info_df = temp_info.reset_index()
        char_to_m = np.array([self.m_map[a] for a in self.char_info_df['author']])
        char_to_book = self.char_info_df['book'].values
        unique_books = self.char_info_df['book'].unique()

        row_indices = df["char_key"].map(c_map).values
        col_indices = df["feat"].map(self.v_map).values
        data = df["count"].values
        self.X = sp.csr_matrix((data, (row_indices, col_indices)), shape=(self.C, self.V))

        np.random.seed(42)
        self.p_assignments = np.random.randint(0, self.P, size=self.C)
        self.book_persona_counts = defaultdict(lambda: np.zeros(self.P, dtype=int))
        for c_idx in range(self.C):
            self.book_persona_counts[char_to_book[c_idx]][self.p_assignments[c_idx]] += 1

        self.model = HierarchicalSAGE(self.M, self.P, num_internal_nodes).to(self.device)
        # 必须是纯 SGD 才能配合 Proximal 截断
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0, momentum=0.0)

        print(f">>> Starting Stochastic EM training ({self.iters} rounds)...")
        for it in range(self.iters):
            print(f"\n--- Round {it+1}/{self.iters} ---")
            
            # --- M-STEP ---
            self.model.train()
            u_indices = char_to_m * self.P + self.p_assignments
            U_total = self.M * self.P
            G = sp.csr_matrix((np.ones(self.C), (u_indices, np.arange(self.C))), shape=(U_total, self.C))
            Y = G.dot(self.X)
            
            active_u_pairs, word_indices = Y.nonzero()
            counts = Y.data
            
            # 【修复2】增加 M-步的多 Epoch 内循环
            batch_size = 1024
            for m_ep in range(self.m_epochs):
                perm = np.random.permutation(len(active_u_pairs))
                shuffled_pairs = active_u_pairs[perm]
                shuffled_words = word_indices[perm]
                shuffled_counts = counts[perm]
                
                total_loss, total_tokens = 0, 0
                
                for i in range(0, len(shuffled_pairs), batch_size):
                    optimizer.zero_grad()
                    
                    batch_indices = shuffled_pairs[i:i+batch_size]
                    batch_words = shuffled_words[i:i+batch_size]
                    batch_counts = torch.tensor(shuffled_counts[i:i+batch_size], dtype=torch.float32, device=self.device)

                    m_idx = torch.tensor(batch_indices // self.P, device=self.device)
                    p_idx = torch.tensor(batch_indices % self.P, device=self.device)
                    
                    node_paths = self.word_paths[batch_words]
                    node_signs = self.word_signs[batch_words]

                    word_log_probs = self.model(m_idx, p_idx, node_paths, node_signs)
                    batch_total_tokens = batch_counts.sum()
                    
                    if batch_total_tokens > 0:
                        nll_loss = -torch.sum(word_log_probs * batch_counts) / batch_total_tokens
                        nll_loss.backward()
                        optimizer.step()
                        
                        # 【修复4】真正的近端梯度下降 (Proximal L1 Soft-thresholding)
                        with torch.no_grad():
                            lr = optimizer.param_groups[0]['lr']
                            # 动态调整惩罚，使其与均值化的 Loss 处于同一量级
                            penalty = lr * (self.l1_lambda / batch_total_tokens.item())
                            
                            # 仅对 meta 和 pers 截断，保留 padding 节点的 0.0 不变
                            eta_meta = self.model.eta_meta.data
                            self.model.eta_meta.data.copy_(torch.sign(eta_meta) * F.relu(eta_meta.abs() - penalty))
                            
                            eta_pers = self.model.eta_pers.data
                            self.model.eta_pers.data.copy_(torch.sign(eta_pers) * F.relu(eta_pers.abs() - penalty))
                            
                        total_loss += nll_loss.item() * batch_total_tokens.item()
                        total_tokens += batch_total_tokens.item()

            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            l1_norm = (self.model.eta_meta.abs().sum() + self.model.eta_pers.abs().sum()).item()
            print(f"  [M-Step] Avg NLL/Token: {avg_loss:.4f} | L1 Norm: {l1_norm:.4f}")

            # --- E-STEP ---
            self.model.eval()
            with torch.no_grad():
                all_word_log_probs = torch.zeros((self.M, self.P, self.V), device=self.device)
                
                # 预计算极大提升吉布斯采样效率
                for m_idx in range(self.M):
                    for p_idx in range(self.P):
                        m_tensor = torch.full((self.V,), m_idx, device=self.device, dtype=torch.long)
                        p_tensor = torch.full((self.V,), p_idx, device=self.device, dtype=torch.long)
                        all_word_log_probs[m_idx, p_idx, :] = self.model(m_tensor, p_tensor, self.word_paths, self.word_signs)

                for book in tqdm(unique_books, desc="  [E-Step] Gibbs Sampling"):
                    char_indices_in_book = np.where(char_to_book == book)[0]
                    if len(char_indices_in_book) == 0: continue
                    m_d = char_to_m[char_indices_in_book[0]]
                    
                    log_probs_for_author = all_word_log_probs[m_d, :, :] 

                    for c in char_indices_in_book:
                        old_p = self.p_assignments[c]
                        self.book_persona_counts[book][old_p] -= 1
                        
                        prior = np.log(self.book_persona_counts[book] + self.alpha)
                        
                        c_word_vector = torch.tensor(self.X[c].toarray().flatten(), dtype=torch.float32, device=self.device)
                        ll = (log_probs_for_author * c_word_vector).sum(dim=1)
                        
                        post_logits = torch.tensor(prior, device=self.device) + ll
                        post_probs = F.softmax(post_logits, dim=0).cpu().numpy()
                        
                        new_p = np.random.choice(self.P, p=post_probs)
                        self.p_assignments[c] = new_p
                        self.book_persona_counts[book][new_p] += 1
                        
    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n>>> Saving model and results to {output_dir}/")
        torch.save(self.model.state_dict(), os.path.join(output_dir, "sage_model_weights.pt"))
        metadata = {
            "vocab": self.vocab, "m_map": self.m_map, "P": self.P,
            "word_paths": self.word_paths.cpu(),
            "word_signs": self.word_signs.cpu()
        }
        with open(os.path.join(output_dir, "sage_metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        df_results = self.char_info_df.copy()
        df_results["persona"] = self.p_assignments
        df_results.to_csv(os.path.join(output_dir, "sage_character_personas.csv"), index=False)
        print(">>> All files saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hierarchical SAGE model.")
    parser.add_argument('--cluster_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_file', type=str, default='fullset_data/all_words.csv')
    parser.add_argument('--n_personas', type=int, default=8)
    parser.add_argument('--em_iters', type=int, default=10) 
    parser.add_argument('--m_epochs', type=int, default=10) # 控制 M-Step 的收敛
    parser.add_argument('--l1_lambda', type=float, default=0.1) # 开启 L1
    
    args = parser.parse_args()

    if os.path.exists(args.cluster_file):
        model = LiteraryPersonaSAGE(
            n_personas=args.n_personas, 
            em_iters=args.em_iters,
            m_epochs=args.m_epochs,
            l1_lambda=args.l1_lambda
        )
        df_processed = model.load_and_preprocess_data(args.data_file, args.cluster_file)
        
        # NOTE: 实际应用中，你需要从你的特征库中提前构建一个 len(vocab) x embedding_dim 的 numpy 矩阵传入
        # 这里为了兼容测试不崩溃，如果不传，内部会生成随机向量
        dummy_vectors = None 
        
        model.fit(df_processed, pre_trained_word_vectors=dummy_vectors)
        model.save_results(args.output_dir)
    else:
        print(f"Error: Cluster file not found at {args.cluster_file}")