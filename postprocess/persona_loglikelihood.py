"""
Compute per-persona key word lists via log-likelihood ratio (one-vs-rest).

For each persona p, treat its word frequencies as corpus A and the combined
frequencies of all other personas as corpus B:

  LLR(w, p | r) = log P(w | corpus_A) - log P(w | corpus_B)

with Laplace smoothing k=0.5 applied independently to each corpus.

Two rankings per (persona, role):
  1. Unweighted  : ranked by LLR directly            (empirical freq only)
  2. Eta-weighted: score = eta_norm(p, r, w) * LLR   (model-informed)

Word frequencies : data/processed/female_words_base.csv
Checkpoint       : checkpoints/traditional_W2V-Role_P8_L1.0/best_model.pt
Output           : data/results/traditional_results/W2V-Role/P8_L1.0/
"""
import os, sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
from sage.model_traditional import PerRoleHierarchicalSAGE, ROLES

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH    = "checkpoints/traditional_W2V-Role_P8_L1.0/best_model.pt"
ASSIGN_CSV   = "data/results/traditional_results/W2V-Role/P8_L1.0/char_assignments.csv"
FREQ_CSV     = "data/processed/female_words_base.csv"
CLUSTER_DIR  = "data/sage_cluster_dataset/w2v-role"
OUT_DIR      = "data/results/traditional_results/W2V-Role/P8_L1.0"
ROLE_NAMES   = ['Agent', 'Patient', 'Possessive', 'Predicative']
SMOOTHING    = 0.5   # add-k smoothing for zero-freq words

# ── Load checkpoint ───────────────────────────────────────────────────────────
print("Loading checkpoint ...")
device = torch.device("cpu")
ckpt   = torch.load(CKPT_PATH, map_location=device, weights_only=False)
P      = int(ckpt['P'])
num_internal_nodes_per_role = ckpt['num_internal_nodes_per_role']
vocab_clusters_per_role     = ckpt['vocab_clusters_per_role']      # [R] sorted cluster_ids
model_weights               = ckpt['model_weights']

# Infer M
M = model_weights['eta_meta.0'].shape[0]
sage_model = PerRoleHierarchicalSAGE(M, P, num_internal_nodes_per_role)
sage_model.load_state_dict(model_weights)
sage_model.eval()

# ── Per-role cluster CSV: word → cluster_id ───────────────────────────────────
print("Loading per-role cluster maps ...")
word_to_cluster_per_role = []   # [R] {word: cluster_id}
cluster_to_words_per_role = []  # [R] {cluster_id: [words]}
for role in ROLES:
    df_r = pd.read_csv(os.path.join(CLUSTER_DIR, f"{role}_clusters.csv"))
    word_to_cluster_per_role.append(dict(zip(df_r['word'], df_r['cluster_id'])))
    cluster_to_words_per_role.append(
        df_r.groupby('cluster_id')['word'].apply(list).to_dict()
    )

# ── Rebuild trees to get word_paths / word_signs ─────────────────────────────
# (reuse LiteraryPersonaSAGE._build_balanced_tree via a dummy instance)
from sage.model_traditional import LiteraryPersonaSAGE, _hydrate_cluster_csv
_sage_helper = LiteraryPersonaSAGE(n_personas=P, em_iters=1, l1_lambda=1.0, min_mentions=0)
word_paths_list = []
word_signs_list = []
for r, role in enumerate(ROLES):
    df_r = pd.read_csv(os.path.join(CLUSTER_DIR, f"{role}_clusters.csv"))
    if 'vector' not in df_r.columns:
        df_r = _hydrate_cluster_csv(df_r, role)
    wp, ws, _, _, _, _ = _sage_helper._build_balanced_tree(df_r, role_label=role)
    word_paths_list.append(wp)
    word_signs_list.append(ws)

# ── Compute per-role leaf effects  [P, V_r] ───────────────────────────────────
print("Computing leaf effects ...")
leaf_effects_per_role = []   # [R]  np.ndarray [P, V_r]
with torch.no_grad():
    for r in range(len(ROLES)):
        wp = word_paths_list[r]    # [V_r, L_r]
        ws = word_signs_list[r]
        bg   = (sage_model.eta_bg[r][wp] * ws).sum(dim=1)              # [V_r]
        pers = (sage_model.eta_pers[r][:, wp] * ws).sum(dim=2)         # [P, V_r]
        leaf = (bg.unsqueeze(0) + pers).numpy()                        # [P, V_r]
        leaf_effects_per_role.append(leaf)

# Normalise eta per (p, r): min-max to [0, 1] across vocab
eta_norm_per_role = []
for r in range(len(ROLES)):
    leaf = leaf_effects_per_role[r]      # [P, V_r]
    lo   = leaf.min(axis=1, keepdims=True)
    hi   = leaf.max(axis=1, keepdims=True)
    denom = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    eta_norm_per_role.append((leaf - lo) / denom)   # [P, V_r]  in [0,1]

# ── Load persona assignments ──────────────────────────────────────────────────
print("Loading persona assignments ...")
df_assign = pd.read_csv(ASSIGN_CSV)                      # char_id, persona
char_persona = dict(zip(df_assign['char_id'], df_assign['persona']))

# ── Load word frequencies ─────────────────────────────────────────────────────
print("Loading word frequencies ...")
df_freq = pd.read_csv(FREQ_CSV)                          # character_id, role, word, freq
df_freq['role'] = df_freq['role'].str.lower().str.strip()
df_freq = df_freq[df_freq['role'].isin(ROLES)].copy()
df_freq['persona'] = df_freq['character_id'].map(char_persona)
df_freq = df_freq.dropna(subset=['persona']).copy()
df_freq['persona'] = df_freq['persona'].astype(int)

print(f"  Observations after persona join: {len(df_freq)}")
print(f"  Role distribution:\n{df_freq['role'].value_counts()}")

# ── Aggregate frequencies ─────────────────────────────────────────────────────
# Per (persona, role, word) counts
prf = (df_freq.groupby(['persona', 'role', 'word'])['freq']
       .sum().reset_index().rename(columns={'freq': 'count_p'}))

# Per (role, word) total counts across ALL personas
rf  = (df_freq.groupby(['role', 'word'])['freq']
       .sum().reset_index().rename(columns={'freq': 'count_all'}))

prf = prf.merge(rf, on=['role', 'word'], how='left')

# Role-level totals: per persona and globally
p_role_totals   = df_freq.groupby(['persona', 'role'])['freq'].sum().to_dict()
all_role_totals = df_freq.groupby('role')['freq'].sum().to_dict()

# ── Log-likelihood ratio: one-vs-rest ─────────────────────────────────────────
# corpus A = persona p
# corpus B = all other personas combined  (count_rest = count_all - count_p)
records_unweighted = []
records_weighted   = []

for r_idx, role in enumerate(ROLES):
    r_name    = ROLE_NAMES[r_idx]
    w2c       = word_to_cluster_per_role[r_idx]
    vc        = vocab_clusters_per_role[r_idx]
    c_idx_map = {c: i for i, c in enumerate(vc)}

    sub            = prf[prf['role'] == role].copy()
    all_words_role = sub['word'].unique()
    V_r            = len(all_words_role)

    # word → total count across all personas
    all_counts = dict(zip(sub.drop_duplicates('word')['word'],
                          sub.drop_duplicates('word')['count_all']))
    total_all  = all_role_totals.get(role, 0)

    for p in range(P):
        p_sub    = sub[sub['persona'] == p].copy()
        p_counts = dict(zip(p_sub['word'], p_sub['count_p']))

        total_p    = p_role_totals.get((p, role), 0)
        total_rest = total_all - total_p

        N1 = total_p
        N2 = total_rest

        for word in all_words_role:
            cnt_p    = p_counts.get(word, 0)
            cnt_rest = all_counts.get(word, 0) - cnt_p   # rest = all − p

            a, b = cnt_p, cnt_rest

            # ── LLR (smoothed log probability ratio) ──────────────────────
            log_p_given_p    = np.log((a + SMOOTHING) / (N1 + SMOOTHING * V_r))
            log_p_given_rest = np.log((b + SMOOTHING) / (N2 + SMOOTHING * V_r))
            llr = log_p_given_p - log_p_given_rest

            # ── G² (Dunning log-likelihood, absolute) ─────────────────────
            # E1 = N1*(a+b)/(N1+N2),  E2 = N2*(a+b)/(N1+N2)
            # G² = 2*(a*ln(a/E1) + b*ln(b/E2)),  0*ln(0) = 0 by convention
            # Sign follows direction: + means overrepresented in persona p
            ab   = a + b
            N12  = N1 + N2
            E1   = N1 * ab / N12 if N12 > 0 else 0.0
            E2   = N2 * ab / N12 if N12 > 0 else 0.0
            term1 = a * np.log(a / E1) if (a > 0 and E1 > 0) else 0.0
            term2 = b * np.log(b / E2) if (b > 0 and E2 > 0) else 0.0
            g2_unsigned = 2.0 * (term1 + term2)
            sign = 1.0 if (N1 > 0 and a / N1 >= b / max(N2, 1)) else -1.0
            g2   = sign * g2_unsigned

            # ── Unweighted record ──────────────────────────────────────────
            records_unweighted.append({
                'persona':   p,
                'role':      r_name,
                'word':      word,
                'llr':       llr,
                'g2':        g2,
                'freq_p':    a,
                'freq_rest': b,
            })

            # ── Eta-weighted record ────────────────────────────────────────
            cid = w2c.get(word)
            if cid is not None and cid in c_idx_map:
                eta_w = float(eta_norm_per_role[r_idx][p, c_idx_map[cid]])
            else:
                eta_w = 0.0

            records_weighted.append({
                'persona':   p,
                'role':      r_name,
                'word':      word,
                'llr':       llr,
                'g2':        g2,
                'eta_norm':  eta_w,
                'score':     eta_w * g2,   # weight G² by eta
                'freq_p':    a,
                'freq_rest': b,
            })

# ── Save full tables ──────────────────────────────────────────────────────────
df_uw = pd.DataFrame(records_unweighted)
df_w  = pd.DataFrame(records_weighted)

os.makedirs(OUT_DIR, exist_ok=True)
df_uw.to_csv(os.path.join(OUT_DIR, "llr_unweighted.csv"),  index=False)
df_w.to_csv( os.path.join(OUT_DIR, "llr_eta_weighted.csv"), index=False)
print(f"\nSaved llr_unweighted.csv    ({len(df_uw)} rows)")
print(f"Saved llr_eta_weighted.csv  ({len(df_w)} rows)")

# ── Top-K per (persona, role) ─────────────────────────────────────────────────
K = 20
rows_uw, rows_w = [], []

for r_name in ROLE_NAMES:
    for p in range(P):
        # Unweighted top-K by G² (positive = overrepresented in persona p)
        sub_uw = df_uw[(df_uw['persona']==p) & (df_uw['role']==r_name)]
        top_uw = sub_uw.nlargest(K, 'g2')[['persona','role','word','g2','llr','freq_p','freq_rest']]
        rows_uw.append(top_uw)

        # Eta-weighted top-K by score = eta * G²  (positive G² only)
        sub_w     = df_w[(df_w['persona']==p) & (df_w['role']==r_name)]
        sub_w_pos = sub_w[sub_w['g2'] > 0]
        top_w     = sub_w_pos.nlargest(K, 'score')[['persona','role','word','g2','llr','eta_norm','score','freq_p','freq_rest']]
        rows_w.append(top_w)

df_top_uw = pd.concat(rows_uw, ignore_index=True)
df_top_w  = pd.concat(rows_w,  ignore_index=True)

df_top_uw.to_csv(os.path.join(OUT_DIR, "topK_unweighted.csv"),   index=False)
df_top_w.to_csv( os.path.join(OUT_DIR, "topK_eta_weighted.csv"), index=False)
print(f"Saved topK_unweighted.csv   ({len(df_top_uw)} rows)")
print(f"Saved topK_eta_weighted.csv ({len(df_top_w)} rows)")

# ── Quick preview: top-5 words per persona ────────────────────────────────────
print(f"\n=== Top-5 by G² (unweighted) — Agent role ===")
for p in range(P):
    sub = df_top_uw[(df_top_uw['persona']==p) & (df_top_uw['role']=='Agent')]
    items = [(r['word'], round(r['g2'], 1)) for _, r in sub.head(5).iterrows()]
    print(f"  Persona {p}: {items}")

print(f"\n=== Top-5 by eta*G² (weighted) — Agent role ===")
for p in range(P):
    sub = df_top_w[(df_top_w['persona']==p) & (df_top_w['role']=='Agent')]
    items = [(r['word'], round(r['g2'], 1)) for _, r in sub.head(5).iterrows()]
    print(f"  Persona {p}: {items}")

print(f"\n=== Top-5 by G² (unweighted) — Predicative role ===")
for p in range(P):
    sub = df_top_uw[(df_top_uw['persona']==p) & (df_top_uw['role']=='Predicative')]
    items = [(r['word'], round(r['g2'], 1)) for _, r in sub.head(5).iterrows()]
    print(f"  Persona {p}: {items}")
