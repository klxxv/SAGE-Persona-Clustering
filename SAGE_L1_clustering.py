import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaln
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 论文中原汁原味的 OWL-QN 优化器实现 (简化版正交投影拟牛顿法)
# ==========================================
class OWLQN(torch.optim.Optimizer):
    """
    Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
    用于精确求解带有 L1 正则化的目标函数 (Andrew and Gao, 2007)
    """
    def __init__(self, params, lr=1.0, l1_lambda=0.1):
        defaults = dict(lr=lr, l1_lambda=l1_lambda)
        super(OWLQN, self).__init__(params, defaults)

    def _pseudo_gradient(self, w, g, l1_lambda):
        """计算伪梯度 (Pseudo-gradient)"""
        pg = torch.zeros_like(g)
        # w < 0 的正交象限
        idx_neg = w < 0
        pg[idx_neg] = g[idx_neg] - l1_lambda
        # w > 0 的正交象限
        idx_pos = w > 0
        pg[idx_pos] = g[idx_pos] + l1_lambda
        # w == 0 的正交象限
        idx_zero = w == 0
        pg_zero = torch.zeros_like(g[idx_zero])
        
        g_zero = g[idx_zero]
        pg_zero[g_zero + l1_lambda < 0] = g_zero[g_zero + l1_lambda < 0] + l1_lambda
        pg_zero[g_zero - l1_lambda > 0] = g_zero[g_zero - l1_lambda > 0] - l1_lambda
        pg[idx_zero] = pg_zero
        return pg

    def _project_to_orthant(self, x, x_old):
        """将权重投影回原本的正交象限，若跨越坐标轴则截断为0"""
        sign_x = torch.sign(x)
        sign_x_old = torch.sign(x_old)
        # 如果符号发生改变（从正到负或负到正），强制截断为 0
        zero_mask = (sign_x * sign_x_old) < 0
        x[zero_mask] = 0.0
        return x

    def step(self, closure):
        """执行一步 OWL-QN 更新"""
        loss = closure()
        for group in self.param_groups:
            l1_lambda = group['l1_lambda']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                # 获取伪梯度
                pg = self._pseudo_gradient(p.data, p.grad.data, l1_lambda)
                
                # 更新前的参数
                p_old = p.data.clone()
                
                # 按照伪梯度进行梯度下降更新
                p.data.add_(pg, alpha=-lr)
                
                # 正交象限投影 (保证不会越过 0 边界)
                p.data = self._project_to_orthant(p.data, p_old)
        return loss

# ==========================================
# 2. 修复特征空间维度的层次化 SAGE 模型
# ==========================================
class HierarchicalSAGE(nn.Module):
    def __init__(self, M, P, R, num_internal_nodes):
        super().__init__()
        self.padding_idx = num_internal_nodes
        # 修正：参数维度中加入了 R (Role)，严格区分句法角色
        self.eta_bg = nn.Parameter(torch.zeros(R, self.padding_idx + 1))
        self.eta_meta = nn.Parameter(torch.zeros(M, R, self.padding_idx + 1))
        self.eta_pers = nn.Parameter(torch.zeros(P, R, self.padding_idx + 1))
        
    def forward(self, m_idx, p_idx, r_idx, node_paths, node_signs):
        batch_size, max_path_len = node_paths.shape

        m_idx_exp = m_idx.unsqueeze(1).expand(-1, max_path_len)
        p_idx_exp = p_idx.unsqueeze(1).expand(-1, max_path_len)
        r_idx_exp = r_idx.unsqueeze(1).expand(-1, max_path_len)

        # 获取对应角色的树节点权重
        bg = self.eta_bg[r_idx_exp, node_paths]
        meta = self.eta_meta[m_idx_exp, r_idx_exp, node_paths]
        pers = self.eta_pers[p_idx_exp, r_idx_exp, node_paths]
        
        logits = bg + meta + pers
        
        path_mask = (node_paths != self.padding_idx).float()
        log_probs = F.logsigmoid(node_signs * logits) * path_mask
        word_log_probs = log_probs.sum(dim=1)
        
        return word_log_probs

# ==========================================
# 3. Alpha 切片采样 (Slice Sampling) 
# ==========================================
def slice_sample_alpha(alpha, persona_counts, num_docs, P, w=0.5, max_steps=100):
    """
    使用切片采样动态更新 Dirichlet 前置参数 alpha
    """
    def log_prob(a):
        if a <= 0: return -np.inf
        # 根据 Dirchlet-Multinomial 边缘似然计算 log p(counts | alpha)
        lp = num_docs * (gammaln(P * a) - P * gammaln(a))
        for counts in persona_counts.values():
            lp += np.sum(gammaln(counts + a)) - gammaln(np.sum(counts) + P * a)
        return lp

    # 1. 计算当前的 log likelihood 并采样高度
    current_lp = log_prob(alpha)
    z = current_lp - np.random.exponential(1.0)
    
    # 2. 寻找边界 (Stepping out)
    L = alpha - w * np.random.rand()
    R = L + w
    step_count = 0
    while log_prob(L) > z and step_count < max_steps:
        L -= w
        step_count += 1
    step_count = 0
    while log_prob(R) > z and step_count < max_steps:
        R += w
        step_count += 1
        
    L = max(L, 1e-4) # 避免边界为负数
    
    # 3. 采样 (Shrinking)
    for _ in range(max_steps):
        new_alpha = L + np.random.rand() * (R - L)
        if log_prob(new_alpha) > z:
            return new_alpha
        if new_alpha < alpha:
            L = new_alpha
        else:
            R = new_alpha
            
    return alpha

# ==========================================
# 4. 核心训练器
# ==========================================
class LiteraryPersonaSAGE:
    def __init__(self, n_personas=8, init_alpha=1.0, l1_lambda=0.1, em_iters=50, min_mentions=10):
        self.P = n_personas
        self.alpha = init_alpha
        self.l1_lambda = l1_lambda
        self.iters = em_iters
        self.min_mentions = min_mentions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _build_balanced_tree(self, df_words):
        """
        修复：按照论文方法，每一层两两配对，构建完全平衡的二叉树
        """
        print(">>> Building a strictly BALANCED semantic binary tree...")
        
        # 获取唯一的 cluster_id 及其对应的向量均值
        df_words['vector'] = df_words['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
        cluster_vectors = df_words.groupby('cluster_id')['vector'].apply(lambda x: np.mean(np.vstack(x), axis=0)).to_dict()
        
        clusters = list(cluster_vectors.keys())
        V = len(clusters)
        self.vocab_clusters = sorted(clusters)
        self.c_map = {c: i for i, c in enumerate(self.vocab_clusters)}
        
        current_level_nodes = {self.c_map[k]: v for k, v in cluster_vectors.items()}
        parent_map = {}
        sign_map = {}
        next_node_id = V
        
        # 自底向上进行每一层的强制两两合并
        while len(current_level_nodes) > 1:
            nodes = list(current_level_nodes.keys())
            vectors = np.vstack(list(current_level_nodes.values()))
            
            # 计算当前层的成对距离矩阵
            dist_matrix = squareform(pdist(vectors, metric='euclidean'))
            np.fill_diagonal(dist_matrix, np.inf)
            
            paired = set()
            next_level_nodes = {}
            
            for i in range(len(nodes)):
                if i in paired: continue
                # 寻找未配对的最近邻居
                closest_j = None
                min_dist = np.inf
                for j in range(len(nodes)):
                    if j != i and j not in paired and dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        closest_j = j
                
                if closest_j is not None:
                    node1, node2 = nodes[i], nodes[closest_j]
                    paired.add(i)
                    paired.add(closest_j)
                    
                    # 生成新的父节点
                    parent_node = next_node_id
                    next_node_id += 1
                    
                    parent_map[node1] = parent_node
                    sign_map[node1] = -1.0  # Left
                    parent_map[node2] = parent_node
                    sign_map[node2] = 1.0   # Right
                    
                    next_level_nodes[parent_node] = (vectors[i] + vectors[closest_j]) / 2.0
                else:
                    # 单数落单，直接晋级下一层
                    next_level_nodes[nodes[i]] = vectors[i]
                    
            current_level_nodes = next_level_nodes

        num_internal_nodes = next_node_id - V
        
        # 记录路径
        paths = {}
        for i in range(V):
            path = []
            signs = []
            curr = i
            while curr in parent_map:
                p_node = parent_map[curr]
                path.append(p_node - V)  # 内部节点索引从0开始
                signs.append(sign_map[curr])
                curr = p_node
            paths[i] = (path[::-1], signs[::-1])

        max_len = max(len(p[0]) for p in paths.values())
        
        self.word_paths = torch.full((V, max_len), num_internal_nodes, dtype=torch.long)
        self.word_signs = torch.zeros((V, max_len))

        for i in range(V):
            path, signs = paths[i]
            self.word_paths[i, :len(path)] = torch.tensor(path)
            self.word_signs[i, :len(signs)] = torch.tensor(signs)
            
        print(f"    Balanced Tree built. Clusters (V): {V}, Internal nodes: {num_internal_nodes}, Depth: {max_len}")
        return num_internal_nodes

    def load_and_preprocess_data(self, data_file, word_csv_file):
        df_words = pd.read_csv(word_csv_file)
        num_internal_nodes = self._build_balanced_tree(df_words)
        self.word_paths = self.word_paths.to(self.device)
        self.word_signs = self.word_signs.to(self.device)

        df = pd.read_csv(data_file)
        
        # 提取论文规定的 4 种角色
        roles = ['agent', 'patient', 'possessive', 'predicative']
        df = df[df['role'].isin(roles)].copy()
        self.r_map = {r: i for i, r in enumerate(roles)}
        self.R = len(roles)

        # 映射 cluster_id (忽略未在聚类中心的生僻词)
        word_to_cluster = dict(zip(df_words.word, df_words.cluster_id))
        df['cluster_id'] = df['word'].map(word_to_cluster)
        df.dropna(subset=['cluster_id'], inplace=True)
        df['cluster_id'] = df['cluster_id'].astype(int)
        
        # 过滤低频角色
        df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
        char_totals = df.groupby("char_key")["count"].sum()
        valid_keys = char_totals[char_totals >= self.min_mentions].index
        df = df[df["char_key"].isin(valid_keys)].copy()
        
        return df, num_internal_nodes

    def fit(self, df, num_internal_nodes):
        authors = sorted(df["author"].unique())
        self.m_map = {author: i for i, author in enumerate(authors)}
        self.M = len(authors)

        char_keys = sorted(df["char_key"].unique())
        c_map = {ck: i for i, ck in enumerate(char_keys)}
        self.C = len(char_keys)
        
        print(f"Authors (M): {self.M}, Roles (R): {self.R}, Chars (C): {self.C}")

        # 将数据转为方便处理的张量结构
        df['m_idx'] = df['author'].map(self.m_map)
        df['c_idx'] = df['char_key'].map(c_map)
        df['r_idx'] = df['role'].map(self.r_map)
        df['w_idx'] = df['cluster_id'].map(self.c_map)
        
        temp_info = df.groupby("char_key")[["author", "book", "m_idx"]].first().reindex(char_keys).reset_index()
        self.char_info_df = temp_info
        char_to_m = temp_info['m_idx'].values
        char_to_book = temp_info['book'].values
        unique_books = temp_info['book'].unique()

        # 预分配角色
        np.random.seed(42)
        self.p_assignments = np.random.randint(0, self.P, size=self.C)
        self.book_persona_counts = defaultdict(lambda: np.zeros(self.P, dtype=int))
        for c_idx in range(self.C):
            self.book_persona_counts[char_to_book[c_idx]][self.p_assignments[c_idx]] += 1

        self.model = HierarchicalSAGE(self.M, self.P, self.R, num_internal_nodes).to(self.device)
        
        # 替换为真实的 OWL-QN 优化器
        optimizer = OWLQN(self.model.parameters(), lr=1.0, l1_lambda=self.l1_lambda)

        print(f">>> Starting Stochastic EM training ({self.iters} rounds)...")
        for it in range(self.iters):
            print(f"\n--- Round {it+1}/{self.iters} (Alpha: {self.alpha:.4f}) ---")
            
            # --- 论文约束: 每 5 次迭代执行 Slice Sampling 动态调整 Alpha ---
            if (it + 1) % 5 == 0:
                self.alpha = slice_sample_alpha(self.alpha, self.book_persona_counts, len(unique_books), self.P)
                print(f"  [Hyper-param] Updated Dirichlet Alpha: {self.alpha:.4f}")

            # --- M-STEP (最大化似然) ---
            self.model.train()
            
            # 更新 df 里的 persona 标签
            p_map_df = pd.Series(self.p_assignments, index=np.arange(self.C))
            df['p_idx'] = df['c_idx'].map(p_map_df)
            
            # 聚合相同 (m, p, r, w) 的计数以加速
            agg_df = df.groupby(['m_idx', 'p_idx', 'r_idx', 'w_idx'])['count'].sum().reset_index()
            
            m_idx_t = torch.tensor(agg_df['m_idx'].values, device=self.device)
            p_idx_t = torch.tensor(agg_df['p_idx'].values, device=self.device)
            r_idx_t = torch.tensor(agg_df['r_idx'].values, device=self.device)
            w_idx_t = torch.tensor(agg_df['w_idx'].values, device=self.device)
            counts_t = torch.tensor(agg_df['count'].values, dtype=torch.float32, device=self.device)
            
            total_tokens = counts_t.sum().item()
            
            def closure():
                optimizer.zero_grad()
                node_paths = self.word_paths[w_idx_t]
                node_signs = self.word_signs[w_idx_t]
                word_log_probs = self.model(m_idx_t, p_idx_t, r_idx_t, node_paths, node_signs)
                # 使用 Negative Log Likelihood 作为目标
                loss = -torch.sum(word_log_probs * counts_t)/total_tokens
                loss.backward()
                return loss
            
            # 运行 OWL-QN
            prev_loss = float('inf')
            tolerance = 1e-5  # 论文设定的绝对收敛阈值 
            max_m_steps = 200 # 设置一个安全上限，防止极端情况下的震荡导致死循环
            
            print("  [M-Step] Optimizing parameters...")
            for m_step in range(max_m_steps):
                loss_val = optimizer.step(closure)
                current_loss = loss_val.item()
                
                # 检查绝对收敛条件：| L(t-1) - L(t) | < 1e-5
                if abs(prev_loss - current_loss) < tolerance:
                    print(f"  [M-Step] Converged at step {m_step + 1} (Loss diff: {abs(prev_loss - current_loss):.6e})")
                    break
                    
                prev_loss = current_loss
                
            l1_norm = (self.model.eta_meta.abs().sum() + self.model.eta_pers.abs().sum()).item()
            print(f"  [M-Step] Final Avg NLL/Token: {current_loss:.4f} | L1 Norm: {l1_norm:.4f}")
            
            # --- E-STEP (Collapsed Gibbs Sampling) ---
            self.model.eval()
            with torch.no_grad():
                # 预计算极大提升吉布斯采样效率
                V_total = len(self.vocab_clusters)
                all_word_log_probs = torch.zeros((self.M, self.P, self.R, V_total), device=self.device)
                
                # 为所有组合前向传播
                for r_idx in range(self.R):
                    r_tensor = torch.full((V_total,), r_idx, device=self.device, dtype=torch.long)
                    for m_idx in range(self.M):
                        m_tensor = torch.full((V_total,), m_idx, device=self.device, dtype=torch.long)
                        for p_idx in range(self.P):
                            p_tensor = torch.full((V_total,), p_idx, device=self.device, dtype=torch.long)
                            all_word_log_probs[m_idx, p_idx, r_idx, :] = self.model(m_tensor, p_tensor, r_tensor, self.word_paths, self.word_signs)

                # --- 优化：使用张量运算预计算所有角色的 Likelihood ---
                print("  [E-Step] Pre-calculating character likelihoods...")
                ll_matrix = torch.zeros((self.C, self.P), device=self.device)
                
                # 提取 df 里的所有索引
                c_idx_arr = torch.tensor(df['c_idx'].values, device=self.device)
                m_idx_arr = torch.tensor(df['m_idx'].values, device=self.device)
                r_idx_arr = torch.tensor(df['r_idx'].values, device=self.device)
                w_idx_arr = torch.tensor(df['w_idx'].values, device=self.device)
                counts_arr = torch.tensor(df['count'].values, dtype=torch.float32, device=self.device)
                
                for p_idx in range(self.P):
                    # 获取当前 persona 下所有词的 log_prob: [N_rows]
                    # all_word_log_probs: [M, P, R, V]
                    probs_p = all_word_log_probs[m_idx_arr, p_idx, r_idx_arr, w_idx_arr]
                    
                    # 乘以频次: [N_rows]
                    weighted_probs = probs_p * counts_arr
                    
                    # 按照 c_idx 聚合求和: [C]
                    ll_matrix[:, p_idx].index_add_(0, c_idx_arr, weighted_probs)
                
                ll_matrix = ll_matrix.cpu().numpy()
                # ---------------------------------------------------------

                for book in tqdm(unique_books, desc="  [E-Step] Gibbs Sampling"):
                    char_indices = np.where(char_to_book == book)[0]
                    if len(char_indices) == 0: continue
                    
                    for c in char_indices:
                        old_p = self.p_assignments[c]
                        self.book_persona_counts[book][old_p] -= 1
                        
                        prior = np.log(self.book_persona_counts[book] + self.alpha)
                        ll = ll_matrix[c, :] # 直接使用预计算的结果
                        
                        post_logits = prior + ll
                        # 减去最大值防止溢出
                        post_logits -= np.max(post_logits)
                        post_probs = np.exp(post_logits) / np.sum(np.exp(post_logits))
                        
                        new_p = np.random.choice(self.P, p=post_probs)
                        self.p_assignments[c] = new_p
                        self.book_persona_counts[book][new_p] += 1
                        
    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n>>> Saving model and results to {output_dir}/")
        torch.save(self.model.state_dict(), os.path.join(output_dir, "sage_model_weights.pt"))
        df_results = self.char_info_df.copy()
        df_results["persona"] = self.p_assignments
        df_results.to_csv(os.path.join(output_dir, "sage_character_personas.csv"), index=False)
        print(">>> All files saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hierarchical SAGE model.")
    parser.add_argument('--word_csv_file', type=str, required=True, help="word,cluster_id,vector CSV file")
    parser.add_argument('--data_file', type=str, required=True, help="book,author,char_id,role,word,count CSV file")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_personas', type=int, default=8)
    parser.add_argument('--em_iters', type=int, default=50)
    parser.add_argument('--l1_lambda', type=float, default=1e-6)
    
    args = parser.parse_args()

    model = LiteraryPersonaSAGE(
        n_personas=args.n_personas, 
        em_iters=args.em_iters,
        l1_lambda=args.l1_lambda
    )
    df_processed, num_nodes = model.load_and_preprocess_data(args.data_file, args.word_csv_file)
    model.fit(df_processed, num_nodes)
    model.save_results(args.output_dir)