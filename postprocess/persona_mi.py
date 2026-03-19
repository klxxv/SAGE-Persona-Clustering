"""
Compute per-persona word importance via PMI and NPMI.

PMI(w, p | r)  = log P(w,p|r) - log P(w|r) - log P(p|r)
               = LLR(w,p|r) - log P(p|r)

NPMI(w, p | r) = PMI / -log P(w,p|r)   in [-1, 1]

Also computes MI(W; P | r) — the aggregate mutual information
between the word distribution and persona assignment per role.

Output: data/results/traditional_results/W2V-Role/P8_L1.0/
    mi_pmi.csv          full (word, persona, role, pmi, npmi)
    mi_npmi_topK.csv    top-K per (persona, role) by NPMI
    mi_aggregate.csv    aggregate MI per role  MI(W;P|r)
"""
import os, sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
ASSIGN_CSV = "data/results/traditional_results/W2V-Role/P8_L1.0/char_assignments.csv"
FREQ_CSV   = "data/processed/female_words_base.csv"
OUT_DIR    = "data/results/traditional_results/W2V-Role/P8_L1.0"
ROLE_NAMES = ['agent', 'patient', 'possessive', 'predicative']
SMOOTHING  = 0.5
K          = 20

# ── Load persona assignments ──────────────────────────────────────────────────
df_assign    = pd.read_csv(ASSIGN_CSV)
char_persona = dict(zip(df_assign['char_id'], df_assign['persona']))
P            = df_assign['persona'].nunique()

# ── Load word frequencies ─────────────────────────────────────────────────────
df_freq = pd.read_csv(FREQ_CSV)
df_freq['role'] = df_freq['role'].str.lower().str.strip()
df_freq = df_freq[df_freq['role'].isin(ROLE_NAMES)].copy()
df_freq['persona'] = df_freq['character_id'].map(char_persona)
df_freq = df_freq.dropna(subset=['persona']).copy()
df_freq['persona'] = df_freq['persona'].astype(int)

print(f"Observations after persona join: {len(df_freq)}")
print(df_freq['role'].value_counts())

# ── Aggregate counts ──────────────────────────────────────────────────────────
# N(w, p, r)
wprf = (df_freq.groupby(['word', 'persona', 'role'])['freq']
        .sum().reset_index().rename(columns={'freq': 'n_wpr'}))

# N(w, r)
wrf  = (df_freq.groupby(['word', 'role'])['freq']
        .sum().reset_index().rename(columns={'freq': 'n_wr'}))

# N(p, r)
prf  = (df_freq.groupby(['persona', 'role'])['freq']
        .sum().reset_index().rename(columns={'freq': 'n_pr'}))

# N(r)
rf   = (df_freq.groupby('role')['freq']
        .sum().reset_index().rename(columns={'freq': 'n_r'}))

# Unique words per role (for smoothing denominator)
words_per_role = df_freq.groupby('role')['word'].nunique().to_dict()

# ── Merge all counts onto wprf ────────────────────────────────────────────────
wprf = wprf.merge(wrf,  on=['word', 'role'],         how='left')
wprf = wprf.merge(prf,  on=['persona', 'role'],      how='left')
wprf = wprf.merge(rf,   on='role',                   how='left')

# ── Smoothed probabilities ────────────────────────────────────────────────────
wprf['V_r'] = wprf['role'].map(words_per_role)

# P(w, p | r) = (N(w,p,r) + k) / (N(r) + k * V_r * P)
wprf['p_wpr'] = (wprf['n_wpr'] + SMOOTHING) / (
    wprf['n_r'] + SMOOTHING * wprf['V_r'] * P
)

# P(w | r) = (N(w,r) + k*P) / (N(r) + k * V_r * P)
wprf['p_wr']  = (wprf['n_wr']  + SMOOTHING * P) / (
    wprf['n_r'] + SMOOTHING * wprf['V_r'] * P
)

# P(p | r) = (N(p,r) + k*V_r) / (N(r) + k * V_r * P)
wprf['p_pr']  = (wprf['n_pr']  + SMOOTHING * wprf['V_r']) / (
    wprf['n_r'] + SMOOTHING * wprf['V_r'] * P
)

# ── PMI and NPMI ──────────────────────────────────────────────────────────────
wprf['pmi']  = (np.log(wprf['p_wpr'])
                - np.log(wprf['p_wr'])
                - np.log(wprf['p_pr']))

# NPMI = PMI / -log P(w,p|r),  clipped to [-1, 1]
wprf['npmi'] = (wprf['pmi'] / (-np.log(wprf['p_wpr']))).clip(-1, 1)

# ── Save full PMI table ───────────────────────────────────────────────────────
out_cols = ['word', 'persona', 'role', 'n_wpr', 'n_wr', 'pmi', 'npmi']
wprf[out_cols].to_csv(os.path.join(OUT_DIR, "mi_pmi.csv"), index=False)
print(f"\nSaved mi_pmi.csv ({len(wprf)} rows)")

# ── Top-K per (persona, role) by NPMI ────────────────────────────────────────
rows = []
for role in ROLE_NAMES:
    role_cap = role.capitalize()
    for p in range(P):
        sub = wprf[(wprf['role'] == role) & (wprf['persona'] == p)]
        top = sub.nlargest(K, 'npmi')[out_cols].copy()
        top['role'] = role_cap
        rows.append(top)

df_topk = pd.concat(rows, ignore_index=True)
df_topk.to_csv(os.path.join(OUT_DIR, "mi_npmi_topK.csv"), index=False)
print(f"Saved mi_npmi_topK.csv ({len(df_topk)} rows)")

# ── Aggregate MI(W; P | r) per role ──────────────────────────────────────────
# MI = sum_{w,p} P(w,p|r) * PMI(w,p|r)
# Only positive PMI terms contribute positively; sum all.
agg_rows = []
for role in ROLE_NAMES:
    sub = wprf[wprf['role'] == role]
    mi  = (sub['p_wpr'] * sub['pmi']).sum()
    # Normalised: NMI = MI / H(W|r)  where H(W|r) = -sum_w P(w|r)*log P(w|r)
    wr_sub  = sub.drop_duplicates('word')
    h_w     = -(wr_sub['p_wr'] * np.log(wr_sub['p_wr'])).sum()
    h_p     = -(sub.drop_duplicates('persona')['p_pr'] * np.log(sub.drop_duplicates('persona')['p_pr'])).sum()
    nmi     = mi / (0.5 * (h_w + h_p)) if (h_w + h_p) > 0 else 0.0
    agg_rows.append({'role': role, 'MI': mi, 'H_W': h_w, 'H_P': h_p, 'NMI': nmi})
    print(f"  {role:12s}: MI={mi:.4f}, H(W)={h_w:.4f}, H(P)={h_p:.4f}, NMI={nmi:.4f}")

pd.DataFrame(agg_rows).to_csv(
    os.path.join(OUT_DIR, "mi_aggregate.csv"), index=False
)
print(f"Saved mi_aggregate.csv")

# ── Preview top-5 NPMI per persona per role ───────────────────────────────────
for role in ROLE_NAMES:
    role_cap = role.capitalize()
    print(f"\n=== {role_cap} - NPMI top-5 ===")
    for p in range(P):
        sub   = df_topk[(df_topk['role'] == role_cap) & (df_topk['persona'] == p)]
        items = [(r['word'], round(r['npmi'], 3), int(r['n_wpr']))
                 for _, r in sub.head(5).iterrows()]
        print(f"  P{p}: {items}")
