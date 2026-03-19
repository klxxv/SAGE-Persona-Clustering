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
# 1. OWL-QN Optimizer
# ==========================================
class OWLQN(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, l1_lambda=1.0,
                 history_size=10, max_ls_iters=20, beta=0.5, c1=1e-4):
        defaults = dict(lr=lr, l1_lambda=l1_lambda)
        super().__init__(params, defaults)
        self.history_size = history_size
        self.max_ls_iters = max_ls_iters
        self.beta = beta
        self.c1 = c1
        self._s_hist   = []
        self._y_hist   = []
        self._rho_hist = []

    def reset_history(self):
        self._s_hist.clear()
        self._y_hist.clear()
        self._rho_hist.clear()

    def _flat_params(self):
        return torch.cat([p.data.view(-1) for g in self.param_groups for p in g['params']])

    def _flat_grads(self):
        return torch.cat([
            (p.grad.view(-1) if p.grad is not None else p.data.new_zeros(p.numel()))
            for g in self.param_groups for p in g['params']
        ])

    def _set_params(self, flat):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                n = p.numel()
                p.data.copy_(flat[offset:offset + n].view_as(p))
                offset += n

    def _flat_pseudo_grad(self):
        vecs = []
        for group in self.param_groups:
            l1 = group['l1_lambda']
            for p in group['params']:
                g = p.grad.view(-1) if p.grad is not None else p.data.new_zeros(p.numel())
                w = p.data.view(-1)
                if l1 == 0.0:
                    vecs.append(g)
                    continue
                pg = g.clone()
                pg[w > 0] = g[w > 0] + l1
                pg[w < 0] = g[w < 0] - l1
                m0 = (w == 0)
                g0 = g[m0]
                pg0 = g0.new_zeros(g0.shape)
                pg0[g0 < -l1] = g0[g0 < -l1] + l1
                pg0[g0 >  l1] = g0[g0 >  l1] - l1
                pg[m0] = pg0
                vecs.append(pg)
        return torch.cat(vecs)

    def _l1_penalty(self):
        total = 0.0
        for group in self.param_groups:
            l1 = group['l1_lambda']
            if l1 > 0:
                for p in group['params']:
                    total += l1 * p.data.abs().sum().item()
        return total

    def _lbfgs_direction(self, pg):
        q = pg.clone()
        alphas = []
        for s, y, rho in zip(reversed(self._s_hist),
                              reversed(self._y_hist),
                              reversed(self._rho_hist)):
            a = rho * torch.dot(s, q)
            alphas.append(a)
            q.add_(y, alpha=-a.item())
        if self._s_hist:
            s_l, y_l = self._s_hist[-1], self._y_hist[-1]
            gamma = (torch.dot(s_l, y_l) / torch.dot(y_l, y_l).clamp(min=1e-12)).clamp(1e-8, 1e8)
            r = q * gamma
        else:
            r = q.clone()
        for s, y, rho, a in zip(self._s_hist, self._y_hist,
                                  self._rho_hist, reversed(alphas)):
            b = rho * torch.dot(y, r)
            r.add_(s, alpha=(a - b).item())
        return -r

    def _project_direction(self, d, pg):
        d = d.clone()
        d[(d * pg) > 0] = 0.0
        return d

    def _project_params(self, x_new, x_old, pg):
        orthant = torch.sign(x_old)
        orthant[x_old == 0] = -torch.sign(pg[x_old == 0])
        x_new = x_new.clone()
        x_new[(torch.sign(x_new) * orthant) < 0] = 0.0
        return x_new

    @torch.no_grad()
    def step(self, closure):
        if closure is None: raise RuntimeError("OWL-QN requires a closure.")
        with torch.enable_grad(): loss = closure()
        x0 = self._flat_params()
        g0 = self._flat_grads()
        pg = self._flat_pseudo_grad()
        h0 = loss.item() + self._l1_penalty()
        d = self._lbfgs_direction(pg)
        d = self._project_direction(d, pg)
        pg_dot_d = torch.dot(pg, d).item()
        if pg_dot_d >= 0:
            d = -pg
            pg_dot_d = torch.dot(pg, d).item()
        lr = self.param_groups[0]['lr']
        found = False
        x1 = x0
        for _ in range(self.max_ls_iters):
            x1 = self._project_params(x0 + lr * d, x0, pg)
            self._set_params(x1)
            with torch.enable_grad(): new_loss = closure(backward=False)
            h1 = new_loss.item() + self._l1_penalty()
            if h1 <= h0 + self.c1 * lr * pg_dot_d:
                found = True
                break
            lr *= self.beta
        if not found:
            self._set_params(x0)
            return loss
        with torch.enable_grad(): closure(backward=True)
        g1 = self._flat_grads()
        s, y = x1 - x0, g1 - g0
        sy = torch.dot(s, y).item()
        if sy > 1e-10:
            self._s_hist.append(s.clone())
            self._y_hist.append(y.clone())
            self._rho_hist.append(1.0 / sy)
            if len(self._s_hist) > self.history_size:
                self._s_hist.pop(0); self._y_hist.pop(0); self._rho_hist.pop(0)
        return loss

# ==========================================
# 2. Hierarchical SAGE Model
# ==========================================
class HierarchicalSAGE(nn.Module):
    def __init__(self, M, P, R, num_internal_nodes):
        super().__init__()
        self.padding_idx = num_internal_nodes
        self.eta_bg = nn.Parameter(torch.zeros(R, self.padding_idx + 1))
        self.eta_meta = nn.Parameter(torch.zeros(M, R, self.padding_idx + 1))
        self.eta_pers = nn.Parameter(torch.zeros(P, R, self.padding_idx + 1))
        
    def forward(self, m_idx, p_idx, r_idx, node_paths, node_signs):
        batch_size, max_path_len = node_paths.shape
        m_idx_exp = m_idx.unsqueeze(1).expand(-1, max_path_len)
        p_idx_exp = p_idx.unsqueeze(1).expand(-1, max_path_len)
        r_idx_exp = r_idx.unsqueeze(1).expand(-1, max_path_len)
        bg = self.eta_bg[r_idx_exp, node_paths]
        meta = self.eta_meta[m_idx_exp, r_idx_exp, node_paths]
        pers = self.eta_pers[p_idx_exp, r_idx_exp, node_paths]
        logits = bg + meta + pers
        path_mask = (node_paths != self.padding_idx).float()
        log_probs = F.logsigmoid(node_signs * logits) * path_mask
        return log_probs.sum(dim=1)

# ==========================================
# 3. Alpha Slice Sampling
# ==========================================
def slice_sample_alpha(alpha, persona_counts, num_docs, P, w=0.5, max_steps=100):
    def log_prob(a):
        if a <= 0: return -np.inf
        lp = num_docs * (gammaln(P * a) - P * gammaln(a))
        for counts in persona_counts.values():
            lp += np.sum(gammaln(counts + a)) - gammaln(np.sum(counts) + P * a)
        return lp
    current_lp = log_prob(alpha)
    z = current_lp - np.random.exponential(1.0)
    L = alpha - w * np.random.rand()
    R = L + w
    step_count = 0
    while log_prob(L) > z and step_count < max_steps:
        L -= w; step_count += 1
    step_count = 0
    while log_prob(R) > z and step_count < max_steps:
        R += w; step_count += 1
    L = max(L, 1e-4)
    for _ in range(max_steps):
        new_alpha = L + np.random.rand() * (R - L)
        if log_prob(new_alpha) > z: return new_alpha
        if new_alpha < alpha: L = new_alpha
        else: R = new_alpha
    return alpha

# ==========================================
# 4. LiteraryPersonaSAGE Trainer
# ==========================================
class LiteraryPersonaSAGE:
    def __init__(self, n_personas=8, init_alpha=1.0, l1_lambda=0.01, em_iters=50, min_mentions=10):
        self.P = n_personas
        self.alpha = init_alpha
        self.l1_lambda = l1_lambda
        self.iters = em_iters
        self.min_mentions = min_mentions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | L1 Lambda: {self.l1_lambda}")

    def _build_balanced_tree(self, df_words):
        print(">>> Building a strictly BALANCED semantic binary tree...")
        df_words['vector'] = df_words['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
        cluster_vectors = df_words.groupby('cluster_id')['vector'].apply(lambda x: np.mean(np.vstack(x), axis=0)).to_dict()
        clusters = list(cluster_vectors.keys())
        V = len(clusters)
        self.vocab_clusters = sorted(clusters)
        self.cluster_map = {c: i for i, c in enumerate(self.vocab_clusters)}
        current_level_nodes = {self.cluster_map[k]: v for k, v in cluster_vectors.items()}
        parent_map, sign_map, next_node_id = {}, {}, V
        while len(current_level_nodes) > 1:
            nodes = list(current_level_nodes.keys())
            vectors = np.vstack(list(current_level_nodes.values()))
            dist_matrix = squareform(pdist(vectors, metric='euclidean'))
            np.fill_diagonal(dist_matrix, np.inf)
            paired, next_level_nodes = set(), {}
            for i in range(len(nodes)):
                if i in paired: continue
                closest_j, min_dist = None, np.inf
                for j in range(len(nodes)):
                    if j != i and j not in paired and dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]; closest_j = j
                if closest_j is not None:
                    node1, node2 = nodes[i], nodes[closest_j]
                    paired.add(i); paired.add(closest_j)
                    parent_node = next_node_id; next_node_id += 1
                    parent_map[node1] = parent_node; sign_map[node1] = -1.0
                    parent_map[node2] = parent_node; sign_map[node2] = 1.0
                    next_level_nodes[parent_node] = (vectors[i] + vectors[closest_j]) / 2.0
                else: next_level_nodes[nodes[i]] = vectors[i]
            current_level_nodes = next_level_nodes
        num_internal_nodes = next_node_id - V
        paths = {}
        for i in range(V):
            path, signs, curr = [], [], i
            while curr in parent_map:
                p_node = parent_map[curr]
                path.append(p_node - V); signs.append(sign_map[curr]); curr = p_node
            paths[i] = (path[::-1], signs[::-1])
        max_len = max(len(p[0]) for p in paths.values())
        self.word_paths = torch.full((V, max_len), num_internal_nodes, dtype=torch.long)
        self.word_signs = torch.zeros((V, max_len))
        for i in range(V):
            path, signs = paths[i]
            self.word_paths[i, :len(path)] = torch.tensor(path)
            self.word_signs[i, :len(signs)] = torch.tensor(signs)
        print(f"    Balanced Tree built. clusters: {V}, depth: {max_len}")
        return num_internal_nodes

    def load_and_preprocess_data(self, data_file, word_csv_file):
        df_words = pd.read_csv(word_csv_file)
        num_internal_nodes = self._build_balanced_tree(df_words)
        self.word_paths = self.word_paths.to(self.device)
        self.word_signs = self.word_signs.to(self.device)
        df = pd.read_csv(data_file)
        roles = ['agent', 'patient', 'possessive', 'predicative']
        df = df[df['role'].isin(roles)].copy()
        self.r_map = {r: i for i, r in enumerate(roles)}; self.R = len(roles)
        word_to_cluster = dict(zip(df_words.word, df_words.cluster_id))
        df['cluster_id'] = df['word'].map(word_to_cluster)
        df.dropna(subset=['cluster_id'], inplace=True); df['cluster_id'] = df['cluster_id'].astype(int)
        df["char_key"] = df["book"].astype(str) + "_" + df["char_id"].astype(str)
        char_totals = df.groupby("char_key")["count"].sum()
        valid_keys = char_totals[char_totals >= self.min_mentions].index
        df = df[df["char_key"].isin(valid_keys)].copy()
        return df, num_internal_nodes

    def fit(self, df, num_internal_nodes, resume_state=None, checkpoint_dir=None, status_callback=None, m_map=None, char_map=None):
        checkpoint_dir = os.path.abspath(checkpoint_dir) if checkpoint_dir else None
        if m_map: self.m_map = m_map
        else:
            authors = sorted(df["author"].unique())
            self.m_map = {author: i for i, author in enumerate(authors)}
        self.M = len(self.m_map)
        char_keys = sorted(df["char_key"].unique())
        local_char_map = char_map if char_map else {ck: i for i, ck in enumerate(char_keys)}
        df['m_idx'] = df['author'].map(self.m_map)
        df['c_idx'] = df['char_key'].map(local_char_map)
        df['r_idx'] = df['role'].map(self.r_map)
        df['w_idx'] = df['cluster_id'].map(self.cluster_map)
        present_char_indices = sorted(df['c_idx'].unique())
        self.C = max(present_char_indices) + 1 if present_char_indices else 0
        temp_info = df.groupby("c_idx")[["author", "book", "m_idx"]].first().reindex(range(self.C)).ffill().bfill()
        self.char_info_df = temp_info
        char_to_book = temp_info['book'].values; unique_books = df['book'].unique()
        self.model = HierarchicalSAGE(self.M, self.P, self.R, num_internal_nodes).to(self.device)
        if resume_state:
            self.model.load_state_dict(resume_state['model_weights'])
            self.p_assignments = resume_state['p_assignments']
            self.alpha = resume_state['alpha']
            self.book_persona_counts = resume_state['book_persona_counts']
            start_iter = resume_state['current_iter']
        else:
            np.random.seed(42); self.p_assignments = np.random.randint(0, self.P, size=self.C)
            self.book_persona_counts = defaultdict(lambda: np.zeros(self.P, dtype=int))
            for c_idx in range(self.C): self.book_persona_counts[char_to_book[c_idx]][self.p_assignments[c_idx]] += 1
            start_iter = 0
        m_step_static = df.groupby(['c_idx', 'm_idx', 'r_idx', 'w_idx'])['count'].sum().reset_index()
        ms_c_idx = torch.tensor(m_step_static['c_idx'].values, device=self.device)
        ms_m_idx = torch.tensor(m_step_static['m_idx'].values, device=self.device)
        ms_r_idx = torch.tensor(m_step_static['r_idx'].values, device=self.device)
        ms_w_idx = torch.tensor(m_step_static['w_idx'].values, device=self.device)
        ms_count = torch.tensor(m_step_static['count'].values, dtype=torch.float32, device=self.device)
        total_tokens = ms_count.sum().item(); effective_l1 = self.l1_lambda / total_tokens
        if not resume_state:
            warmup_optimizer = OWLQN([{'params':[self.model.eta_meta], 'l1_lambda':effective_l1}, {'params':[self.model.eta_bg], 'l1_lambda':0.0}], lr=1.0)
            ms_p_idx_dummy = torch.zeros_like(ms_c_idx)
            def warmup_closure(backward=True):
                warmup_optimizer.zero_grad()
                word_log_probs = self.model(ms_m_idx, ms_p_idx_dummy, ms_r_idx, self.word_paths[ms_w_idx], self.word_signs[ms_w_idx])
                loss = -torch.sum(word_log_probs * ms_count) / total_tokens
                if backward: loss.backward()
                return loss
            for _ in range(10): warmup_optimizer.step(warmup_closure)
        optimizer = OWLQN([{'params':[self.model.eta_meta, self.model.eta_pers], 'l1_lambda':effective_l1}, {'params':[self.model.eta_bg], 'l1_lambda':0.0}], lr=1.0)
        em_pbar = tqdm(range(start_iter, self.iters), desc=f"[EM P={self.P}]", ascii=True)
        best_loss = float('inf')
        for it in em_pbar:
            if (it + 1) % 5 == 0: self.alpha = slice_sample_alpha(self.alpha, self.book_persona_counts, len(unique_books), self.P)
            self.model.train(); optimizer.reset_history()
            p_assignments_t = torch.from_numpy(self.p_assignments).to(self.device); ms_p_idx = p_assignments_t[ms_c_idx]
            def closure(backward=True):
                optimizer.zero_grad()
                word_log_probs = self.model(ms_m_idx, ms_p_idx, ms_r_idx, self.word_paths[ms_w_idx], self.word_signs[ms_w_idx])
                loss = -torch.sum(word_log_probs * ms_count) / total_tokens
                if backward: loss.backward()
                return loss
            prev_m_loss = float('inf')
            for m_step in range(50):
                loss_val = optimizer.step(closure); current_m_loss = loss_val.item()
                if abs(prev_m_loss - current_m_loss) < 1e-5: break
                prev_m_loss = current_m_loss
            
            # File strategy: Only keep latest and best
            if checkpoint_dir:
                self.save_checkpoint(os.path.join(checkpoint_dir, 'latest_model.pt'), it)
                if prev_m_loss < best_loss:
                    best_loss = prev_m_loss
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'), it)

            self.model.eval()
            with torch.no_grad():
                V_total = len(self.vocab_clusters)
                all_word_log_probs = torch.zeros((self.M, self.P, self.R, V_total), device=self.device)
                m_indices, p_indices, r_indices = torch.arange(self.M, device=self.device), torch.arange(self.P, device=self.device), torch.arange(self.R, device=self.device)
                grid_m, grid_p, grid_r = torch.meshgrid(m_indices, p_indices, r_indices, indexing='ij')
                flat_m, flat_p, flat_r = grid_m.reshape(-1), grid_p.reshape(-1), grid_r.reshape(-1)
                chunk_size = 200 if self.device.type == 'cuda' else 500
                for start_idx in range(0, flat_m.shape[0], chunk_size):
                    end_idx = min(start_idx + chunk_size, flat_m.shape[0])
                    b_m, b_p, b_r = flat_m[start_idx:end_idx], flat_p[start_idx:end_idx], flat_r[start_idx:end_idx]
                    b_size = b_m.shape[0]
                    m_exp, p_exp, r_exp = b_m.unsqueeze(1).expand(-1, V_total).reshape(-1), b_p.unsqueeze(1).expand(-1, V_total).reshape(-1), b_r.unsqueeze(1).expand(-1, V_total).reshape(-1)
                    w_exp = torch.arange(V_total, device=self.device).repeat(b_size)
                    chunk_probs = self.model(m_exp, p_exp, r_exp, self.word_paths[w_exp], self.word_signs[w_exp])
                    all_word_log_probs[b_m, b_p, b_r, :] = chunk_probs.view(b_size, V_total)
                c_idx_arr, m_idx_arr, r_idx_arr, w_idx_arr = torch.tensor(df['c_idx'].values, device=self.device), torch.tensor(df['m_idx'].values, device=self.device), torch.tensor(df['r_idx'].values, device=self.device), torch.tensor(df['w_idx'].values, device=self.device)
                counts_arr, ll_matrix = torch.tensor(df['count'].values, dtype=torch.float32, device=self.device), torch.zeros((self.C, self.P), device=self.device)
                for p_idx in range(self.P):
                    probs_p = all_word_log_probs[m_idx_arr, p_idx, r_idx_arr, w_idx_arr]
                    ll_matrix[:, p_idx].index_add_(0, c_idx_arr, probs_p * counts_arr)
                ll_matrix = ll_matrix.cpu().numpy()
                from joblib import Parallel, delayed
                def sample_single_book(book_id, b_ll_matrix, b_p_assignments, b_book_persona_counts, b_alpha, b_P):
                    char_indices = np.where(char_to_book == book_id)[0]
                    local_counts = b_book_persona_counts[book_id].copy()
                    local_assigns = {}
                    for c in char_indices:
                        old_p = b_p_assignments[c]; local_counts[old_p] -= 1
                        post_logits = np.log(local_counts + b_alpha) + b_ll_matrix[c, :]
                        post_logits -= np.max(post_logits)
                        post_probs = np.exp(post_logits) / np.sum(np.exp(post_logits))
                        new_p = np.random.choice(b_P, p=post_probs)
                        local_assigns[c] = new_p; local_counts[new_p] += 1
                    return book_id, local_assigns, local_counts
                for b in unique_books: _ = self.book_persona_counts[b]
                parallel_results = Parallel(n_jobs=-1, backend="threading")(delayed(sample_single_book)(book, ll_matrix, self.p_assignments, self.book_persona_counts, self.alpha, self.P) for book in unique_books)
                for book_id, local_assigns, local_counts in parallel_results:
                    self.book_persona_counts[book_id] = local_counts
                    for c, new_p in local_assigns.items(): self.p_assignments[c] = new_p
        return df

    def save_checkpoint(self, path, current_iter):
        checkpoint = {'model_weights': self.model.state_dict(), 'p_assignments': self.p_assignments, 'alpha': self.alpha, 'book_persona_counts': dict(self.book_persona_counts), 'current_iter': current_iter, 'P': self.P}
        torch.save(checkpoint, path)

    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "sage_model_weights.pt"))
        df_results = self.char_info_df.copy()
        df_results["persona"] = self.p_assignments
        df_results.to_csv(os.path.join(output_dir, "sage_character_personas.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_personas', type=int, default=8)
    parser.add_argument('--em_iters', type=int, default=50)
    parser.add_argument('--l1_lambda', type=float, default=1.0)
    args = parser.parse_args()
    model = LiteraryPersonaSAGE(n_personas=args.n_personas, em_iters=args.em_iters, l1_lambda=args.l1_lambda)
    df_processed, num_nodes = model.load_and_preprocess_data(args.data_file, args.word_csv_file)
    model.fit(df_processed, num_nodes, checkpoint_dir=args.output_dir)
    model.save_results(args.output_dir)
