import os
import argparse
import pandas as pd
import numpy as np
import torch
from sage.model_traditional import (
    LiteraryPersonaSAGE, PerRoleHierarchicalSAGE, ROLES
)

# ── Per-role cluster directory ────────────────────────────────────────────────
PER_ROLE_CLUSTER_DIR = "data/sage_cluster_dataset/w2v-role"

ROLE_NAMES = ['Agent', 'Patient', 'Possessive', 'Predicative']


# ─────────────────────────────────────────────────────────────────────────────
def load_name_maps():
    df_book = pd.read_csv("data/processed/book_id_name.csv")
    df_book.columns = df_book.columns.str.strip('\ufeff')
    book_map = dict(zip(df_book['book_id'], df_book['book_name']))

    df_auth = pd.read_csv("data/processed/author_id_name.csv")
    df_auth.columns = df_auth.columns.str.strip('\ufeff')
    author_map = dict(zip(df_auth['author_id'], df_auth['author_name']))

    df_names = pd.read_csv("data/processed/character_id_name.csv")
    for col in ['book', 'name']:
        df_names[col] = df_names[col].astype(str).str.strip('"')
    df_names['character_id'] = pd.to_numeric(
        df_names['character_id'].astype(str).str.strip('"'), errors='coerce'
    ).astype('Int64')
    return book_map, author_map, df_names


# ─────────────────────────────────────────────────────────────────────────────
def eval_one_per_role(label, n_personas, l1_lambda, data_file, ckpt_base, res_base):
    """Evaluate a per-role (W2V-Role) checkpoint."""
    ckpt_path = os.path.join(
        ckpt_base, f"traditional_{label}_P{n_personas}_L{l1_lambda}", "best_model.pt"
    )
    res_dir = os.path.join(res_base, label, f"P{n_personas}_L{l1_lambda}")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f">>> Evaluating {label}  P={n_personas}  L1={l1_lambda}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    ckpt          = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_weights = ckpt['model_weights']
    p_assignments = ckpt['p_assignments']
    P             = int(ckpt.get('P', n_personas))
    num_internal_nodes_per_role = ckpt['num_internal_nodes_per_role']   # [R] ints
    vocab_clusters_per_role     = ckpt['vocab_clusters_per_role']       # [R] lists

    # Infer M from saved weights  (eta_meta[r] shape: [M, N_r+1])
    M = model_weights['eta_meta.0'].shape[0]
    print(f"  Detected M={M} authors, P={P}, R={len(ROLES)}")

    # ── 2. Rebuild per-role trees (deterministic) ─────────────────────────────
    role_cluster_csvs = {
        role: os.path.join(PER_ROLE_CLUSTER_DIR, f"{role}_clusters.csv")
        for role in ROLES
    }
    sage = LiteraryPersonaSAGE(n_personas=P, em_iters=1, l1_lambda=l1_lambda, min_mentions=0)
    df_processed, _ = sage.load_and_preprocess_data(data_file, role_cluster_csvs)
    print(f"  Role counts after preprocessing:\n{df_processed['role'].value_counts()}")

    # ── 3. Reconstruct char ordering (must match training) ────────────────────
    char_keys      = sorted(df_processed['char_key'].unique())
    local_char_map = {ck: i for i, ck in enumerate(char_keys)}
    df_processed['c_idx'] = df_processed['char_key'].map(local_char_map)

    C         = len(char_keys)
    char_info = (
        df_processed.groupby('c_idx')[['author', 'book', 'char_id']]
        .first().reindex(range(C))
    )
    char_info['persona'] = p_assignments[:C]

    # ── 4. Enrich with human-readable names ───────────────────────────────────
    book_map, author_map, df_names = load_name_maps()
    char_info = char_info.reset_index()
    char_info['book_name']   = char_info['book'].map(book_map)
    char_info['author_name'] = char_info['author'].map(author_map)
    char_info['char_id']     = pd.to_numeric(char_info['char_id'], errors='coerce').astype('Int64')

    df_assign = char_info.merge(
        df_names,
        left_on=['book_name', 'char_id'],
        right_on=['book', 'character_id'],
        how='left'
    )
    os.makedirs(res_dir, exist_ok=True)
    assign_path = os.path.join(res_dir, "char_assignments.csv")
    df_assign[['book_name', 'name', 'author_name', 'char_id', 'persona']].rename(
        columns={'book_name': 'book', 'author_name': 'author'}
    ).to_csv(assign_path, index=False)
    print(f"  char_assignments saved  ({len(df_assign)} chars) -> {assign_path}")

    # ── 5. Reconstruct model and compute per-role leaf effects ────────────────
    sage_model = PerRoleHierarchicalSAGE(M, P, num_internal_nodes_per_role).to(device)
    sage_model.load_state_dict(model_weights)
    sage_model.eval()

    # leaf_effects_per_role[r] = np.ndarray [P, V_r]
    leaf_effects_per_role = []
    with torch.no_grad():
        for r, role in enumerate(ROLES):
            wp = sage.word_paths_list[r].to(device)   # [V_r, L_r]
            ws = sage.word_signs_list[r].to(device)   # [V_r, L_r]
            V_r = wp.shape[0]

            # bg_effects [V_r]
            bg = (sage_model.eta_bg[r][wp] * ws).sum(dim=1)

            # pers_effects [P, V_r]
            pers = (sage_model.eta_pers[r][:, wp] * ws).sum(dim=2)

            leaf = (bg.unsqueeze(0) + pers).cpu().numpy()   # [P, V_r]
            leaf_effects_per_role.append(leaf)

            print(f"  Role {ROLE_NAMES[r]:12s} | "
                  f"max={leaf.max():.4e}  min={leaf.min():.4e}  "
                  f"nonzero={np.count_nonzero(leaf)}/{leaf.size}")

    # ── 6. Per-role: cluster_id → words ───────────────────────────────────────
    cluster_to_words_per_role = []
    for role in ROLES:
        df_r = pd.read_csv(role_cluster_csvs[role])
        c2w  = df_r.groupby('cluster_id')['word'].apply(list).to_dict()
        cluster_to_words_per_role.append(c2w)

    # ── 7. Extract raw keywords ───────────────────────────────────────────────
    kw_records = []
    for r, r_name in enumerate(ROLE_NAMES):
        leaf    = leaf_effects_per_role[r]            # [P, V_r]
        vc      = vocab_clusters_per_role[r]          # sorted cluster_ids
        c2w     = cluster_to_words_per_role[r]

        for p in range(P):
            sorted_idx = np.argsort(leaf[p])[::-1]
            for rank, v_idx in enumerate(sorted_idx):
                if leaf[p, v_idx] <= 0:
                    break
                cid   = vc[v_idx]
                words = c2w.get(cid, [])
                for word in words:
                    kw_records.append({
                        'persona':      p,
                        'role':         r_name,
                        'cluster_id':   cid,
                        'cluster_rank': rank + 1,
                        'word':         word,
                        'weight':       float(leaf[p, v_idx]),
                    })

    kw_path = os.path.join(res_dir, "keywords.csv")
    pd.DataFrame(kw_records).to_csv(kw_path, index=False)
    print(f"  keywords saved          ({len(kw_records)} rows)  -> {kw_path}")

    # ── 8. Filtered keywords (remove words in >80% of personas per role) ──────
    global_hits   = {}
    all_word_entries = []   # (p, r_name, word, weight, cluster_id)

    for r, r_name in enumerate(ROLE_NAMES):
        leaf = leaf_effects_per_role[r]
        vc   = vocab_clusters_per_role[r]
        c2w  = cluster_to_words_per_role[r]

        for p in range(P):
            top10_idx  = np.argsort(leaf[p])[::-1][:10]
            seen_words: set = set()
            for v_idx in top10_idx:
                if leaf[p, v_idx] <= 0:
                    continue
                cid   = vc[v_idx]
                words = c2w.get(cid, [])
                for word in words:
                    if word not in seen_words:
                        seen_words.add(word)
                        key = (r_name, word)
                        global_hits[key] = global_hits.get(key, 0) + 1
                        all_word_entries.append(
                            (p, r_name, word, float(leaf[p, v_idx]), cid)
                        )

    threshold = max(1, P * 0.8)
    filtered = [
        {'persona': p, 'role': r, 'word': w, 'weight': wt, 'cluster_id': cid}
        for p, r, w, wt, cid in all_word_entries
        if global_hits[(r, w)] <= threshold
    ]
    fk_path = os.path.join(res_dir, "filtered-keywords.csv")
    pd.DataFrame(filtered).to_csv(fk_path, index=False)
    print(f"  filtered-keywords saved ({len(filtered)} rows)  -> {fk_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-role Traditional SAGE models"
    )
    parser.add_argument(
        "--labels", nargs='+', default=["W2V-Role"],
        help="Labels to evaluate (must match training label, e.g. W2V-Role)"
    )
    parser.add_argument("--start_p",   type=int,   default=8)
    parser.add_argument("--end_p",     type=int,   default=8)
    parser.add_argument("--l1",        type=float, default=1.0)
    parser.add_argument("--data",      type=str,
                        default="data/processed/female_words_with_ids.csv")
    parser.add_argument("--ckpt_base", type=str,   default="checkpoints")
    parser.add_argument("--res_base",  type=str,
                        default="data/results/traditional_results")
    args = parser.parse_args()

    for label in args.labels:
        for p in range(args.start_p, args.end_p + 1):
            eval_one_per_role(
                label, p, args.l1,
                args.data, args.ckpt_base, args.res_base
            )

    print("\n" + "#" * 60)
    print(">>> All evaluations complete.")
    print("#" * 60)


if __name__ == "__main__":
    main()
