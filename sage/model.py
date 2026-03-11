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
    改进版 Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) 近似实现
    结合了伪梯度(Pseudo-gradient)、正交象限投影(Orthant Projection) 以及简单的回溯线搜索(Backtracking Line Search)
    """
    def __init__(self, params, lr=1.0, l1_lambda=1.0, c1=1e-4, beta=0.5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, l1_lambda=l1_lambda, c1=c1, beta=beta)
        super(OWLQN, self).__init__(params, defaults)

    def _pseudo_gradient(self, w, g, l1_lambda):
        """
        计算次梯度/伪梯度 (Pseudo-gradient)
        处理 L1 正则化在 0 点不可导的问题
        """
        if l1_lambda == 0.0:
            return g # 对于没有正则化的参数组（如 eta_bg），直接返回原始梯度

        pg = torch.zeros_like(g)
        
        # w < 0 的象限
        idx_neg = w < 0
        pg[idx_neg] = g[idx_neg] - l1_lambda
        
        # w > 0 的象限
        idx_pos = w > 0
        pg[idx_pos] = g[idx_pos] + l1_lambda
        
        # w == 0 的象限
        idx_zero = w == 0
        g_zero = g[idx_zero]
        pg_zero = torch.zeros_like(g_zero)
        
        # 只有当原始梯度大于 L1 惩罚时，才会在 0 点产生向左或向右的梯度
        pg_zero[g_zero + l1_lambda < 0] = g_zero[g_zero + l1_lambda < 0] + l1_lambda
        pg_zero[g_zero - l1_lambda > 0] = g_zero[g_zero - l1_lambda > 0] - l1_lambda
        
        pg[idx_zero] = pg_zero
        return pg

    def _project_to_orthant(self, x, x_old, pg):
        """
        将更新后的权重投影回原本的正交象限
        防止跨越坐标轴改变符号，如果符号改变则强制截断为 0
        """
        # 定义目标象限：由当前变量所在的象限或伪梯度的反方向决定
        orthant = torch.sign(x_old)
        orthant[orthant == 0] = torch.sign(-pg[orthant == 0])
        
        # 将跨越象限的值截断为 0
        x_projected = x.clone()
        cross_mask = (torch.sign(x_projected) * orthant) < 0
        x_projected[cross_mask] = 0.0
        
        return x_projected

    @torch.no_grad()
    def step(self, closure):
        """
        执行单步 OWL-QN 更新，包含闭包求值和线搜索
        """
        if closure is None:
            raise RuntimeError("OWL-QN requires a closure to evaluate loss for line search.")
            
        # 1. 计算初始 Loss 和梯度
        with torch.enable_grad():
            loss = closure()
            
        initial_loss = loss.item()

        # 保存当前参数和计算出的伪梯度
        p_olds = []
        pgs = []
        
        for group in self.param_groups:
            l1_lambda = group['l1_lambda']
            for p in group['params']:
                if p.grad is None:
                    p_olds.append(None)
                    pgs.append(None)
                    continue
                    
                p_olds.append(p.clone())
                pg = self._pseudo_gradient(p, p.grad, l1_lambda)
                pgs.append(pg)

        # 2. 回溯线搜索 (Backtracking Line Search) 确保充分下降
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta'] # 步长衰减率
            l1_lambda = group['l1_lambda']
            
            for p in group['params']:
                if p.grad is None:
                    idx += 1
                    continue
                
                p_old = p_olds[idx]
                pg = pgs[idx]
                
                current_lr = lr
                max_ls_iters = 10 # 最大线搜索次数
                
                for ls_iter in range(max_ls_iters):
                    # 尝试走一步
                    p.copy_(p_old)
                    p.add_(pg, alpha=-current_lr)
                    
                    # 正交投影
                    p.copy_(self._project_to_orthant(p, p_old, pg))
                    
                    # 计算尝试步之后的 loss
                    with torch.enable_grad():
                        new_loss = closure()
                        
                    # L1 正则化的目标值计算
                    l1_penalty_old = l1_lambda * p_old.abs().sum()
                    l1_penalty_new = l1_lambda * p.abs().sum()
                    
                    # 简单的充分下降条件检查 (Armijo rule 简化版)
                    if new_loss.item() + l1_penalty_new <= initial_loss + l1_penalty_old:
                        break # 找到了合适的步长
                        
                    # 否则衰减学习率
                    current_lr *= beta
                    
                idx += 1

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
        self.cluster_map = {c: i for i, c in enumerate(self.vocab_clusters)}
        
        current_level_nodes = {self.cluster_map[k]: v for k, v in cluster_vectors.items()}
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

    def fit(self, df, num_internal_nodes, resume_state=None, checkpoint_dir=None, status_callback=None, m_map=None, char_map=None):
        # Metadata setup
        if m_map is not None:
            self.m_map = m_map
        else:
            authors = sorted(df["author"].unique())
            self.m_map = {author: i for i, author in enumerate(authors)}
        self.M = len(self.m_map)

        char_keys = sorted(df["char_key"].unique())
        if char_map is not None:
            local_char_map = char_map
        else:
            local_char_map = {ck: i for i, ck in enumerate(char_keys)}
        
        # 将数据转为方便处理的张量结构
        df['m_idx'] = df['author'].map(self.m_map)
        df['c_idx'] = df['char_key'].map(local_char_map)
        df['r_idx'] = df['role'].map(self.r_map)
        df['w_idx'] = df['cluster_id'].map(self.cluster_map)
        
        # 统计当前输入中涉及的角色
        present_char_indices = sorted(df['c_idx'].unique())
        self.C = max(present_char_indices) + 1 if present_char_indices else 0
        
        temp_info = df.groupby("c_idx")[["author", "book", "m_idx"]].first().reindex(range(self.C)).fillna(method='ffill').fillna(method='bfill')
        self.char_info_df = temp_info
        char_to_book = temp_info['book'].values
        unique_books = df['book'].unique()

        self.model = HierarchicalSAGE(self.M, self.P, self.R, num_internal_nodes).to(self.device)

        if resume_state:
            print(f">>> Resuming from existing state (P={self.P})...")
            self.model.load_state_dict(resume_state['model_weights'])
            self.p_assignments = resume_state['p_assignments']
            self.alpha = resume_state['alpha']
            self.book_persona_counts = resume_state['book_persona_counts']
            start_iter = resume_state['current_iter']
        else:
            # 预分配角色
            np.random.seed(42)
            self.p_assignments = np.random.randint(0, self.P, size=self.C)
            self.book_persona_counts = defaultdict(lambda: np.zeros(self.P, dtype=int))
            for c_idx in range(self.C):
                self.book_persona_counts[char_to_book[c_idx]][self.p_assignments[c_idx]] += 1
            start_iter = 0

        # ---------------------------------------------------------
        # 【极致优化】：预计算 M-Step 静态索引
        # ---------------------------------------------------------
        print(">>> Pre-calculating static indices for M-Step optimization...")
        # 1. 预先按 (c_idx, m_idx, r_idx, w_idx) 聚合，这是静态的
        m_step_static = df.groupby(['c_idx', 'm_idx', 'r_idx', 'w_idx'])['count'].sum().reset_index()
        
        # 2. 转为张量并移至设备
        ms_c_idx = torch.tensor(m_step_static['c_idx'].values, device=self.device)
        ms_m_idx = torch.tensor(m_step_static['m_idx'].values, device=self.device)
        ms_r_idx = torch.tensor(m_step_static['r_idx'].values, device=self.device)
        ms_w_idx = torch.tensor(m_step_static['w_idx'].values, device=self.device)
        ms_count = torch.tensor(m_step_static['count'].values, dtype=torch.float32, device=self.device)
        total_tokens = ms_count.sum().item()

        print(f"Authors (M): {self.M}, Roles (R): {self.R}, Chars (C): {self.C}")

        regularized_params = [self.model.eta_meta, self.model.eta_pers]
        unregularized_params = [self.model.eta_bg]
        
        optimizer = OWLQN([
            {'params': regularized_params, 'l1_lambda': self.l1_lambda},
            {'params': unregularized_params, 'l1_lambda': 0.0} 
        ], lr=1.0)

        print(f">>> Starting Stochastic EM training ({start_iter} to {self.iters} rounds)...")
        em_pbar = tqdm(range(start_iter, self.iters), desc=f"[EM P={self.P}]")
        for it in em_pbar:
            current_alpha = self.alpha
            em_pbar.set_postfix({"Alpha": f"{current_alpha:.4f}"})
            
            # 回报进度给父进程/仪表盘
            if status_callback:
                status_callback(it + 1, self.iters, f"Alpha={current_alpha:.4f}")

            # 实时保存进度
            if checkpoint_dir and it > start_iter and it % 5 == 0:
                latest_path = os.path.join(checkpoint_dir, f"checkpoint_it{it}_temp.pt")
                self.save_checkpoint(latest_path, it)
                print(f"  [P={self.P}] Periodic checkpoint saved at iteration {it}")
            
            if (it + 1) % 5 == 0:
                self.alpha = slice_sample_alpha(self.alpha, self.book_persona_counts, len(unique_books), self.P)

            # --- M-STEP (Optimized) ---
            self.model.train()
            
            # 这里的 p_assignments 张量化
            p_assignments_t = torch.from_numpy(self.p_assignments).to(self.device)
            # 获取当前所有静态行对应的 p_idx
            ms_p_idx = p_assignments_t[ms_c_idx]
            
            def closure():
                optimizer.zero_grad()
                node_paths = self.word_paths[ms_w_idx]
                node_signs = self.word_signs[ms_w_idx]
                # 直接在 ms_idx 上计算概率
                word_log_probs = self.model(ms_m_idx, ms_p_idx, ms_r_idx, node_paths, node_signs)
                # 加权求和得到负对数似然
                loss = -torch.sum(word_log_probs * ms_count) / total_tokens
                loss.backward()
                return loss
            
            prev_loss = float('inf')
            tolerance = 1e-4  # 稍微放宽收敛条件以加速
            max_m_steps = 50  # 减少最大 M-Step 步数
            
            m_step_pbar = tqdm(range(max_m_steps), desc=f"  It {it+1} M-Step", leave=False)
            for m_step in m_step_pbar:
                # 实时更新状态到仪表盘
                if status_callback and m_step % 10 == 0:
                    status_callback(it + 1, self.iters, f"M-Step: {m_step}/{max_m_steps} | Loss={prev_loss:.4f}")
                
                loss_val = optimizer.step(closure)
                current_loss = loss_val.item()
                m_step_pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
                if abs(prev_loss - current_loss) < tolerance: break
                prev_loss = current_loss
            
            # --- E-STEP ---
            self.model.eval()
            with torch.no_grad():
                if status_callback: status_callback(it + 1, self.iters, "E-Step: Precalculating...")
                V_total = len(self.vocab_clusters)
                all_word_log_probs = torch.zeros((self.M, self.P, self.R, V_total), device=self.device)
                
                # 优化：向量化前向传播，分块处理以控制显存
                m_indices = torch.arange(self.M, device=self.device)
                p_indices = torch.arange(self.P, device=self.device)
                r_indices = torch.arange(self.R, device=self.device)
                
                grid_m, grid_p, grid_r = torch.meshgrid(m_indices, p_indices, r_indices, indexing='ij')
                flat_m = grid_m.reshape(-1)
                flat_p = grid_p.reshape(-1)
                flat_r = grid_r.reshape(-1)
                
                num_combinations = flat_m.shape[0]
                # 在 GPU 上，chunk_size 需要根据显存动态调整
                chunk_size = 200 if self.device.type == 'cuda' else 500
                
                for start_idx in range(0, num_combinations, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_combinations)
                    batch_m = flat_m[start_idx:end_idx]
                    batch_p = flat_p[start_idx:end_idx]
                    batch_r = flat_r[start_idx:end_idx]
                    
                    b_size = batch_m.shape[0]
                    # 确保扩展张量也在同一设备
                    m_exp = batch_m.unsqueeze(1).expand(-1, V_total).reshape(-1)
                    p_exp = batch_p.unsqueeze(1).expand(-1, V_total).reshape(-1)
                    r_exp = batch_r.unsqueeze(1).expand(-1, V_total).reshape(-1)
                    w_exp = torch.arange(V_total, device=self.device).repeat(b_size)
                    
                    node_paths = self.word_paths[w_exp]
                    node_signs = self.word_signs[w_exp]
                    
                    chunk_probs = self.model(m_exp, p_exp, r_exp, node_paths, node_signs)
                    chunk_probs = chunk_probs.view(b_size, V_total)
                    
                    all_word_log_probs[batch_m, batch_p, batch_r, :] = chunk_probs

                # 预计算 Likelihood Matrix
                if status_callback: status_callback(it + 1, self.iters, "E-Step: Likelihood Matrix...")
                c_idx_arr = torch.tensor(df['c_idx'].values, device=self.device)
                m_idx_arr = torch.tensor(df['m_idx'].values, device=self.device)
                r_idx_arr = torch.tensor(df['r_idx'].values, device=self.device)
                w_idx_arr = torch.tensor(df['w_idx'].values, device=self.device)
                counts_arr = torch.tensor(df['count'].values, dtype=torch.float32, device=self.device)
                
                # 初始化 Likelihood Matrix
                ll_matrix = torch.zeros((self.C, self.P), device=self.device)
                
                for p_idx in range(self.P):
                    probs_p = all_word_log_probs[m_idx_arr, p_idx, r_idx_arr, w_idx_arr]
                    ll_matrix[:, p_idx].index_add_(0, c_idx_arr, probs_p * counts_arr)
                
                ll_matrix = ll_matrix.cpu().numpy()

                # 使用 joblib 并行化 Gibbs 采样
                from joblib import Parallel, delayed
                
                def sample_single_book(book_id, b_unique_books, b_char_to_book, b_ll_matrix, b_p_assignments, b_book_persona_counts, b_alpha, b_P):
                    char_indices = np.where(b_char_to_book == book_id)[0]
                    local_assignments = {}
                    local_counts = b_book_persona_counts[book_id].copy()
                    
                    for c in char_indices:
                        old_p = b_p_assignments[c]
                        local_counts[old_p] -= 1
                        
                        prior = np.log(local_counts + b_alpha)
                        post_logits = prior + b_ll_matrix[c, :]
                        post_logits -= np.max(post_logits)
                        post_probs = np.exp(post_logits) / np.sum(np.exp(post_logits))
                        
                        new_p = np.random.choice(b_P, p=post_probs)
                        local_assignments[c] = new_p
                        local_counts[new_p] += 1
                    return book_id, local_assignments, local_counts

                # 预填充所有书籍的计数以确保线程安全
                for b in unique_books: _ = self.book_persona_counts[b]

                parallel_results = Parallel(n_jobs=-1, backend="threading")(
                    delayed(sample_single_book)(
                        book, unique_books, char_to_book, ll_matrix, self.p_assignments, 
                        self.book_persona_counts, self.alpha, self.P
                    ) for book in unique_books
                )

                # 写回结果
                for book_id, local_assignments, local_counts in parallel_results:
                    self.book_persona_counts[book_id] = local_counts
                    for c, new_p in local_assignments.items():
                        self.p_assignments[c] = new_p
                        
    def save_checkpoint(self, path, current_iter):
        checkpoint = {
            'model_weights': self.model.state_dict(),
            'p_assignments': self.p_assignments,
            'alpha': self.alpha,
            'book_persona_counts': dict(self.book_persona_counts),
            'current_iter': current_iter,
            'P': self.P
        }
        torch.save(checkpoint, path)

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