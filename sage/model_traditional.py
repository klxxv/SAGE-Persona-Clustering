import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaln
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

ROLES = ['agent', 'patient', 'possessive', 'predicative']

W2V_EMB_FILE   = "data/processed/female_word2vec_embedding.csv"
W2V_VOCAB_FILE = "data/processed/female_vocab_map.csv"


def _hydrate_cluster_csv(df_lite, role):
    """
    Given a lightweight cluster CSV (word, cluster_id), add 'vector' column
    by computing per-cluster centroids from W2V embeddings.
    Returns a full DataFrame (word, cluster_id, vector-string).
    """
    print(f"  [{role}] No vector column found – hydrating from W2V embeddings ...")
    df_vocab = pd.read_csv(W2V_VOCAB_FILE)
    df_vocab.columns = df_vocab.columns.str.lstrip('\ufeff')
    word_to_id = dict(zip(df_vocab['word'], df_vocab['word_id']))

    df_emb = pd.read_csv(W2V_EMB_FILE)
    df_emb.columns = df_emb.columns.str.lstrip('\ufeff')
    emb_col = [c for c in df_emb.columns if c != 'word_id'][0]
    df_emb['vec'] = df_emb[emb_col].apply(lambda x: np.fromstring(x, sep=','))
    id_to_vec = dict(zip(df_emb['word_id'], df_emb['vec']))

    df = df_lite.copy()
    df['vec'] = df['word'].map(lambda w: id_to_vec.get(word_to_id.get(w)))
    df = df.dropna(subset=['vec'])

    # Centroid per cluster
    centroids = (df.groupby('cluster_id')['vec']
                 .apply(lambda x: np.mean(np.vstack(x), axis=0))
                 .to_dict())
    df['vector'] = df['cluster_id'].map(
        lambda c: ','.join(f'{v:.6f}' for v in centroids[c])
    )
    # Also save the full version alongside the lite file for future runs
    return df[['word', 'cluster_id', 'vector']]

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
# 2a. Legacy shared-tree model (kept for old checkpoints)
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
# 2b. Per-role independent trees (different vocab & depth per role)
# ==========================================
class PerRoleHierarchicalSAGE(nn.Module):
    """
    Each role r has its own binary tree with V_r leaves and N_r internal nodes.
    Parameters per role:
        eta_bg[r]   : [N_r + 1]        (no L1)
        eta_meta[r] : [M, N_r + 1]     (L1 regularised)
        eta_pers[r] : [P, N_r + 1]     (L1 regularised)
    The +1 slot is the padding index (used for path-length alignment).
    """
    def __init__(self, M, P, num_internal_nodes_per_role):
        """
        num_internal_nodes_per_role : list of R ints
        """
        super().__init__()
        self.R = len(num_internal_nodes_per_role)
        self.padding_idxs = list(num_internal_nodes_per_role)   # padding_idx[r] = N_r

        self.eta_bg   = nn.ParameterList([
            nn.Parameter(torch.zeros(n + 1)) for n in num_internal_nodes_per_role
        ])
        self.eta_meta = nn.ParameterList([
            nn.Parameter(torch.zeros(M, n + 1)) for n in num_internal_nodes_per_role
        ])
        self.eta_pers = nn.ParameterList([
            nn.Parameter(torch.zeros(P, n + 1)) for n in num_internal_nodes_per_role
        ])

    def forward_role(self, r, m_idx, p_idx, node_paths, node_signs):
        """
        r          : int, role index
        m_idx      : [B] long  – author indices
        p_idx      : [B] long  – persona indices
        node_paths : [B, L_r]  – internal-node indices for role r's tree
        node_signs : [B, L_r]  – ±1 direction signs
        Returns    : [B] log-probabilities
        """
        B, L = node_paths.shape
        pad   = self.padding_idxs[r]
        m_exp = m_idx.unsqueeze(1).expand(B, L)
        p_exp = p_idx.unsqueeze(1).expand(B, L)

        bg   = self.eta_bg[r][node_paths]           # [B, L]
        meta = self.eta_meta[r][m_exp, node_paths]  # [B, L]
        pers = self.eta_pers[r][p_exp, node_paths]  # [B, L]

        logits    = bg + meta + pers
        path_mask = (node_paths != pad).float()
        return (F.logsigmoid(node_signs * logits) * path_mask).sum(dim=1)  # [B]


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
# 4. LiteraryPersonaSAGE  (per-role mode)
# ==========================================
class LiteraryPersonaSAGE:
    def __init__(self, n_personas=8, init_alpha=1.0, l1_lambda=0.01,
                 em_iters=50, min_mentions=10):
        self.P           = n_personas
        self.alpha       = init_alpha
        self.l1_lambda   = l1_lambda
        self.iters       = em_iters
        self.min_mentions = min_mentions
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | L1 Lambda: {self.l1_lambda}")

    # ------------------------------------------------------------------
    # Tree builder  (called once per role)
    # ------------------------------------------------------------------
    def _build_balanced_tree(self, df_words, role_label=""):
        """
        df_words must have columns: word, cluster_id, vector
        Returns:
            word_paths        [V, max_depth]  long
            word_signs        [V, max_depth]  float
            num_internal_nodes  int
            vocab_clusters    sorted list of cluster_ids  (length V)
            cluster_map       {cluster_id: vocab_index}
            word_to_cluster   {word: cluster_id}
        """
        tag = f"[{role_label}] " if role_label else ""
        print(f"  {tag}Building balanced binary tree ...")

        df_words = df_words.copy()
        df_words['vector'] = df_words['vector'].apply(
            lambda x: np.array([float(v) for v in x.split(',')])
        )

        # Centroid per cluster (one row per cluster_id after K-Means, so mean = itself)
        cluster_vectors = (
            df_words.groupby('cluster_id')['vector']
            .apply(lambda x: np.mean(np.vstack(x), axis=0))
            .to_dict()
        )

        vocab_clusters = sorted(cluster_vectors.keys())
        cluster_map    = {c: i for i, c in enumerate(vocab_clusters)}
        word_to_cluster = dict(zip(df_words['word'], df_words['cluster_id']))

        V = len(vocab_clusters)
        current_level = {cluster_map[k]: v for k, v in cluster_vectors.items()}

        parent_map, sign_map, next_node_id = {}, {}, V
        while len(current_level) > 1:
            nodes   = list(current_level.keys())
            vectors = np.vstack(list(current_level.values()))
            dist    = squareform(pdist(vectors, metric='euclidean'))
            np.fill_diagonal(dist, np.inf)
            paired, next_level = set(), {}
            for i in range(len(nodes)):
                if i in paired: continue
                j_min = int(np.argmin(dist[i]))
                if j_min not in paired:
                    n1, n2 = nodes[i], nodes[j_min]
                    paired.update({i, j_min})
                    p_node = next_node_id; next_node_id += 1
                    parent_map[n1] = p_node; sign_map[n1] = -1.0
                    parent_map[n2] = p_node; sign_map[n2] =  1.0
                    next_level[p_node] = (vectors[i] + vectors[j_min]) / 2.0
                else:
                    next_level[nodes[i]] = vectors[i]
            current_level = next_level

        num_internal_nodes = next_node_id - V

        # Build path tensors
        paths = {}
        for i in range(V):
            path, signs, curr = [], [], i
            while curr in parent_map:
                p_node = parent_map[curr]
                path.append(p_node - V)
                signs.append(sign_map[curr])
                curr = p_node
            paths[i] = (path[::-1], signs[::-1])

        max_len    = max(len(p[0]) for p in paths.values())
        word_paths = torch.full((V, max_len), num_internal_nodes, dtype=torch.long)
        word_signs = torch.zeros((V, max_len))
        for i, (path, signs) in paths.items():
            word_paths[i, :len(path)] = torch.tensor(path)
            word_signs[i, :len(signs)] = torch.tensor(signs)

        print(f"  {tag}Tree built: V={V} clusters, depth={max_len}, "
              f"internal_nodes={num_internal_nodes}")
        return word_paths, word_signs, num_internal_nodes, vocab_clusters, cluster_map, word_to_cluster

    # ------------------------------------------------------------------
    # Data loading  (per-role mode)
    # ------------------------------------------------------------------
    def load_and_preprocess_data(self, data_file, role_cluster_csvs):
        """
        role_cluster_csvs : dict  {role_name: csv_path}
                            csv must have columns: word, cluster_id, vector
        Returns processed df with columns including r_idx, w_idx (role-local).
        """
        self.r_map = {r: i for i, r in enumerate(ROLES)}
        self.R     = len(ROLES)

        # Build one tree per role
        self.word_paths_list          = []   # [R] tensors  [V_r, L_r]
        self.word_signs_list          = []   # [R] tensors  [V_r, L_r]
        self.num_internal_nodes_list  = []   # [R] ints
        self.vocab_clusters_list      = []   # [R] sorted cluster_id lists
        self.cluster_map_list         = []   # [R] {cluster_id: vocab_idx}
        self.word_to_cluster_list     = []   # [R] {word: cluster_id}

        for role in ROLES:
            csv_path = role_cluster_csvs[role]
            df_r     = pd.read_csv(csv_path)
            if 'vector' not in df_r.columns:
                df_r = _hydrate_cluster_csv(df_r, role)
            wp, ws, nin, vc, cm, w2c = self._build_balanced_tree(df_r, role_label=role)
            self.word_paths_list.append(wp.to(self.device))
            self.word_signs_list.append(ws.to(self.device))
            self.num_internal_nodes_list.append(nin)
            self.vocab_clusters_list.append(vc)
            self.cluster_map_list.append(cm)
            self.word_to_cluster_list.append(w2c)

        # Load training data
        df = pd.read_csv(data_file)
        if 'char_id' not in df.columns and 'character_id' in df.columns:
            df.rename(columns={'character_id': 'char_id'}, inplace=True)
        df['role'] = df['role'].astype(str).str.lower().str.strip()
        df = df[df['role'].isin(ROLES)].copy()
        df['r_idx'] = df['role'].map(self.r_map)

        # Map each word to its role-local cluster vocab index
        df['w_idx'] = pd.NA
        for r, role in enumerate(ROLES):
            mask = df['role'] == role
            cluster_ids = df.loc[mask, 'word'].map(self.word_to_cluster_list[r])
            df.loc[mask, 'w_idx'] = cluster_ids.map(self.cluster_map_list[r])

        df = df.dropna(subset=['w_idx']).copy()
        df['w_idx'] = df['w_idx'].astype(int)

        # Character filtering
        df['char_key'] = df['book'].astype(str) + '_' + df['char_id'].astype(str)
        char_totals    = df.groupby('char_key')['count'].sum()
        valid_keys     = char_totals[char_totals >= self.min_mentions].index
        df = df[df['char_key'].isin(valid_keys)].copy()

        print(f"\nProcessed data role counts:\n{df['role'].value_counts()}")
        print(f"Total observations after filtering: {len(df)}")
        return df, self.num_internal_nodes_list

    # ------------------------------------------------------------------
    # EM Training
    # ------------------------------------------------------------------
    def fit(self, df, num_internal_nodes_list,
            resume_state=None, checkpoint_dir=None, m_map=None, char_map=None):

        checkpoint_dir = os.path.abspath(checkpoint_dir) if checkpoint_dir else None

        # Author map
        if m_map:
            self.m_map = m_map
        else:
            self.m_map = {a: i for i, a in enumerate(sorted(df['author'].unique()))}
        self.M = len(self.m_map)

        # Character map
        char_keys       = sorted(df['char_key'].unique())
        local_char_map  = char_map if char_map else {ck: i for i, ck in enumerate(char_keys)}
        df['m_idx'] = df['author'].map(self.m_map)
        df['c_idx'] = df['char_key'].map(local_char_map)

        present_char_indices = sorted(df['c_idx'].unique())
        self.C = max(present_char_indices) + 1 if present_char_indices else 0

        temp_info = (
            df.groupby('c_idx')[['author', 'book', 'm_idx', 'char_id']]
            .first().reindex(range(self.C)).ffill().bfill()
        )
        self.char_info_df = temp_info
        char_to_book = temp_info['book'].values
        unique_books = df['book'].unique()

        # Build model
        self.model = PerRoleHierarchicalSAGE(
            self.M, self.P, num_internal_nodes_list
        ).to(self.device)

        if resume_state:
            self.model.load_state_dict(resume_state['model_weights'])
            self.p_assignments       = resume_state['p_assignments']
            self.alpha               = resume_state['alpha']
            self.book_persona_counts = resume_state['book_persona_counts']
            start_iter               = resume_state['current_iter']
        else:
            np.random.seed(42)
            self.p_assignments = np.random.randint(0, self.P, size=self.C)
            self.book_persona_counts = defaultdict(lambda: np.zeros(self.P, dtype=int))
            for c_idx in range(self.C):
                self.book_persona_counts[char_to_book[c_idx]][self.p_assignments[c_idx]] += 1
            start_iter = 0

        # Static M-step tensors (aggregated counts)
        m_step_static = (
            df.groupby(['c_idx', 'm_idx', 'r_idx', 'w_idx'])['count']
            .sum().reset_index()
        )
        ms_c_idx = torch.tensor(m_step_static['c_idx'].values, device=self.device)
        ms_m_idx = torch.tensor(m_step_static['m_idx'].values, device=self.device)
        ms_r_idx = torch.tensor(m_step_static['r_idx'].values, device=self.device)
        ms_w_idx = torch.tensor(m_step_static['w_idx'].values, device=self.device)
        ms_count = torch.tensor(m_step_static['count'].values,
                                dtype=torch.float32, device=self.device)
        total_tokens  = ms_count.sum().item()
        effective_l1  = self.l1_lambda / total_tokens

        # Pre-group observations by role (avoids repeated masking in closure)
        role_groups = {}
        for r in range(self.R):
            mask = (ms_r_idx == r)
            if mask.sum() == 0:
                role_groups[r] = None
                continue
            role_groups[r] = {
                'm':   ms_m_idx[mask],
                'c':   ms_c_idx[mask],
                'w':   ms_w_idx[mask],
                'cnt': ms_count[mask],
            }

        # ── Warmup (meta + bg only, persona frozen at 0) ──────────────────────
        if not resume_state:
            warmup_opt = OWLQN(
                [{'params': list(self.model.eta_meta.parameters()), 'l1_lambda': effective_l1},
                 {'params': list(self.model.eta_bg.parameters()),   'l1_lambda': 0.0}],
                lr=1.0
            )
            p_dummy = torch.zeros(ms_c_idx.shape, dtype=torch.long, device=self.device)
            warmup_role_groups = {}
            for r in range(self.R):
                mask = (ms_r_idx == r)
                if mask.sum() == 0:
                    warmup_role_groups[r] = None
                    continue
                warmup_role_groups[r] = {
                    'm':   ms_m_idx[mask],
                    'p':   p_dummy[mask],
                    'w':   ms_w_idx[mask],
                    'cnt': ms_count[mask],
                }

            def warmup_closure(backward=True):
                warmup_opt.zero_grad()
                loss_t = None
                for r in range(self.R):
                    g = warmup_role_groups[r]
                    if g is None: continue
                    paths = self.word_paths_list[r][g['w']]
                    signs = self.word_signs_list[r][g['w']]
                    lp    = self.model.forward_role(r, g['m'], g['p'], paths, signs)
                    rl    = -torch.sum(lp * g['cnt']) / total_tokens
                    loss_t = rl if loss_t is None else loss_t + rl
                if loss_t is None:
                    loss_t = torch.tensor(0.0, device=self.device)
                if backward: loss_t.backward()
                return loss_t

            for _ in range(10):
                warmup_opt.step(warmup_closure)

        # ── Main EM ───────────────────────────────────────────────────────────
        optimizer = OWLQN(
            [{'params': list(self.model.eta_meta.parameters()) +
                        list(self.model.eta_pers.parameters()), 'l1_lambda': effective_l1},
             {'params': list(self.model.eta_bg.parameters()), 'l1_lambda': 0.0}],
            lr=1.0
        )

        em_pbar        = tqdm(range(start_iter, self.iters), desc=f"[EM P={self.P}]", ascii=True)
        best_loss      = float('inf')
        current_p_std  = 0.0

        # E-step tensors (full df, not aggregated)
        c_idx_arr  = torch.tensor(df['c_idx'].values,  device=self.device)
        m_idx_arr  = torch.tensor(df['m_idx'].values,  device=self.device)
        r_idx_arr  = torch.tensor(df['r_idx'].values,  device=self.device)
        w_idx_arr  = torch.tensor(df['w_idx'].values,  device=self.device)
        counts_arr = torch.tensor(df['count'].values,  dtype=torch.float32, device=self.device)

        for it in em_pbar:
            if (it + 1) % 5 == 0:
                self.alpha = slice_sample_alpha(
                    self.alpha, self.book_persona_counts, len(unique_books), self.P
                )
            em_pbar.set_postfix({"Alpha": f"{self.alpha:.4f}",
                                  "P_Std": f"{current_p_std:.5f}"})

            # ── M-step ────────────────────────────────────────────────────────
            self.model.train()
            optimizer.reset_history()
            p_assignments_t = torch.from_numpy(self.p_assignments).to(self.device)
            # Materialise persona index per role-group
            for r in range(self.R):
                g = role_groups[r]
                if g is None: continue
                g['p'] = p_assignments_t[g['c']]

            def closure(backward=True):
                optimizer.zero_grad()
                loss_t = None
                for r in range(self.R):
                    g = role_groups[r]
                    if g is None: continue
                    paths = self.word_paths_list[r][g['w']]
                    signs = self.word_signs_list[r][g['w']]
                    lp    = self.model.forward_role(r, g['m'], g['p'], paths, signs)
                    rl    = -torch.sum(lp * g['cnt']) / total_tokens
                    loss_t = rl if loss_t is None else loss_t + rl
                if loss_t is None:
                    loss_t = torch.tensor(0.0, device=self.device)
                if backward: loss_t.backward()
                return loss_t

            prev_m_loss = float('inf')
            for _ in range(50):
                loss_val    = optimizer.step(closure)
                curr_m_loss = loss_val.item()
                if abs(prev_m_loss - curr_m_loss) < 1e-5: break
                prev_m_loss = curr_m_loss

            if checkpoint_dir:
                self.save_checkpoint(os.path.join(checkpoint_dir, 'latest_model.pt'), it)
                if prev_m_loss < best_loss:
                    best_loss = prev_m_loss
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'), it)

            # ── E-step ────────────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                ll_matrix = torch.zeros((self.C, self.P), device=self.device)

                for r in range(self.R):
                    V_r    = len(self.vocab_clusters_list[r])
                    r_mask = (r_idx_arr == r)
                    if r_mask.sum() == 0:
                        continue

                    c_r   = c_idx_arr[r_mask]
                    m_r   = m_idx_arr[r_mask]
                    w_r   = w_idx_arr[r_mask]
                    cnt_r = counts_arr[r_mask]

                    # Compute [M, P, V_r] log probs for role r
                    gm, gp, gv = torch.meshgrid(
                        torch.arange(self.M, device=self.device),
                        torch.arange(self.P, device=self.device),
                        torch.arange(V_r,    device=self.device),
                        indexing='ij'
                    )
                    gm_f = gm.reshape(-1)
                    gp_f = gp.reshape(-1)
                    gv_f = gv.reshape(-1)
                    paths_all = self.word_paths_list[r][gv_f]  # [M*P*V_r, L_r]
                    signs_all = self.word_signs_list[r][gv_f]

                    chunk = 500 if self.device.type == 'cuda' else 1000
                    lp_flat = torch.zeros(len(gm_f), device=self.device)
                    for s in range(0, len(gm_f), chunk):
                        e = min(s + chunk, len(gm_f))
                        lp_flat[s:e] = self.model.forward_role(
                            r, gm_f[s:e], gp_f[s:e],
                            paths_all[s:e], signs_all[s:e]
                        )
                    all_lp_r = lp_flat.view(self.M, self.P, V_r)  # [M, P, V_r]

                    for p in range(self.P):
                        probs_p = all_lp_r[m_r, p, w_r]
                        ll_matrix[:, p].index_add_(0, c_r, probs_p * cnt_r)

                ll_np    = ll_matrix.cpu().numpy()
                exp_ll   = np.exp(ll_np - np.max(ll_np, axis=1, keepdims=True))
                post_probs = exp_ll / np.sum(exp_ll, axis=1, keepdims=True)
                current_p_std = float(np.std(post_probs))

                # Gibbs sampling for persona assignments
                from joblib import Parallel, delayed
                def sample_single_book(book_id, b_ll, b_assign, b_counts, b_alpha, b_P):
                    char_indices  = np.where(char_to_book == book_id)[0]
                    local_counts  = b_counts[book_id].copy()
                    local_assigns = {}
                    for c in char_indices:
                        old_p = b_assign[c]
                        local_counts[old_p] -= 1
                        logits = np.log(local_counts + b_alpha) + b_ll[c]
                        logits -= logits.max()
                        probs  = np.exp(logits) / np.exp(logits).sum()
                        new_p  = np.random.choice(b_P, p=probs)
                        local_assigns[c] = new_p
                        local_counts[new_p] += 1
                    return book_id, local_assigns, local_counts

                for b in unique_books: _ = self.book_persona_counts[b]
                results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(sample_single_book)(
                        book, ll_np, self.p_assignments,
                        self.book_persona_counts, self.alpha, self.P
                    ) for book in unique_books
                )
                for book_id, local_assigns, local_counts in results:
                    self.book_persona_counts[book_id] = local_counts
                    for c, new_p in local_assigns.items():
                        self.p_assignments[c] = new_p

        self.posterior_probs = post_probs
        return df

    # ------------------------------------------------------------------
    def save_checkpoint(self, path, current_iter):
        checkpoint = {
            'model_weights':              self.model.state_dict(),
            'p_assignments':              self.p_assignments,
            'alpha':                      self.alpha,
            'book_persona_counts':        dict(self.book_persona_counts),
            'current_iter':               current_iter,
            'P':                          self.P,
            'num_internal_nodes_per_role': self.num_internal_nodes_list,
            'vocab_clusters_per_role':    self.vocab_clusters_list,
            'model_type':                 'per_role',
        }
        torch.save(checkpoint, path)

    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(),
                   os.path.join(output_dir, 'sage_model_weights.pt'))
        df_results = self.char_info_df.copy()
        df_results['persona'] = self.p_assignments
        df_results.to_csv(os.path.join(output_dir, 'sage_character_personas.csv'), index=False)
