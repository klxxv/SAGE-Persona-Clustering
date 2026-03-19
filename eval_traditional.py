import os
import argparse
import pandas as pd
import numpy as np
import torch
from sage.model_traditional import LiteraryPersonaSAGE, HierarchicalSAGE


CLUSTER_LABELS = {
    "BERT-256": {
        "word_csv":  "data/sage_cluster_dataset/bert_256/word2vec_clusters.csv",
        "temp_vecs": "data/processed/temp_vectors_BERT-256.csv",
    },
    "BERT-512": {
        "word_csv":  "data/sage_cluster_dataset/bert_512/word2vec_clusters.csv",
        "temp_vecs": "data/processed/temp_vectors_BERT-512.csv",
    },
    "W2V-256": {
        "word_csv":  "data/sage_cluster_dataset/w2v_256/word2vec_clusters.csv",
        "temp_vecs": "data/processed/temp_vectors_W2V-256.csv",
    },
    "W2V-512": {
        "word_csv":  "data/sage_cluster_dataset/w2v_512/word2vec_clusters.csv",
        "temp_vecs": "data/processed/temp_vectors_W2V-512.csv",
    },
}

ROLE_NAMES = ['Agent', 'Patient', 'Possessive', 'Predicative']


def load_name_maps():
    """Load book_id → book_name and author_id → author_name mappings."""
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


def eval_one_traditional(label, n_personas, l1_lambda, data_file, ckpt_base, res_base):
    ckpt_path = os.path.join(
        ckpt_base, f"traditional_{label}_P{n_personas}_L{l1_lambda}", "best_model.pt"
    )
    res_dir = os.path.join(res_base, label, f"P{n_personas}_L{l1_lambda}")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return

    cfg = CLUSTER_LABELS[label]
    temp_vecs = cfg["temp_vecs"]
    orig_cluster_csv = cfg["word_csv"]

    if not os.path.exists(temp_vecs):
        print(f"  [SKIP] Temp vectors not found: {temp_vecs}")
        return

    print(f"\n{'='*60}")
    print(f">>> Evaluating Traditional SAGE  label={label}  P={n_personas}  L1={l1_lambda}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_weights = ckpt['model_weights']
    p_assignments = ckpt['p_assignments']          # np.ndarray [C]
    P = int(ckpt.get('P', n_personas))

    # Infer architecture dimensions from saved weights
    eta_bg   = model_weights['eta_bg']             # [R, num_internal_nodes+1]
    eta_meta = model_weights['eta_meta']            # [M, R, num_internal_nodes+1]
    eta_pers = model_weights['eta_pers']            # [P, R, num_internal_nodes+1]
    R   = eta_bg.shape[0]
    M   = eta_meta.shape[0]
    num_internal_nodes = eta_bg.shape[1] - 1       # padding_idx = num_internal_nodes

    # ── 2. Rebuild balanced binary tree (deterministic from same cluster CSV) ─
    sage = LiteraryPersonaSAGE(n_personas=P, em_iters=1, l1_lambda=l1_lambda, min_mentions=0)
    df_processed, _ = sage.load_and_preprocess_data(data_file, temp_vecs)
    # word_paths and word_signs are now set on `sage`

    # ── 3. Reconstruct char ordering (must match training) ───────────────────
    df_processed["char_key"] = (
        df_processed["book"].astype(str) + "_" + df_processed["char_id"].astype(str)
    )
    char_keys = sorted(df_processed["char_key"].unique())
    local_char_map = {ck: i for i, ck in enumerate(char_keys)}
    df_processed['c_idx'] = df_processed['char_key'].map(local_char_map)
    df_processed['m_idx'] = df_processed['author'].map(
        {a: i for i, a in enumerate(sorted(df_processed['author'].unique()))}
    )

    C = len(char_keys)
    char_info = (
        df_processed.groupby("c_idx")[["author", "book", "char_id"]]
        .first()
        .reindex(range(C))
    )
    char_info["persona"] = p_assignments[:C]

    # ── 4. Enrich assignments with human-readable names ───────────────────────
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

    # ── 5. Reconstruct model and compute leaf effects [P, R, V] ───────────────
    sage_model = HierarchicalSAGE(M, P, R, num_internal_nodes).to(device)
    sage_model.load_state_dict(model_weights)
    sage_model.eval()

    wp = sage.word_paths.to(device)     # [V, max_path_len]
    ws = sage.word_signs.to(device)     # [V, max_path_len]
    V = wp.shape[0]

    with torch.no_grad():
        # eta_pers[:, :, wp]  → [P, R, V, max_path_len]
        leaf_effects = (
            sage_model.eta_pers[:, :, wp] * ws
        ).sum(dim=3).cpu().numpy()   # [P, R, V]

    # ── 6. Build cluster_id → list-of-words mapping from original CSV ─────────
    df_orig = pd.read_csv(orig_cluster_csv)
    # cluster column may be named 'cluster' or 'cluster_id'
    clust_col = 'cluster_id' if 'cluster_id' in df_orig.columns else 'cluster'
    cluster_to_words: dict[int, list[str]] = (
        df_orig.groupby(clust_col)['word'].apply(list).to_dict()
    )

    vocab_clusters = sage.vocab_clusters   # sorted list of cluster_ids, length V

    # ── 7. Extract raw keywords ───────────────────────────────────────────────
    # For each (persona, role): top-10 cluster indices → expand to words
    kw_records = []
    for p in range(P):
        for r_idx, r_name in enumerate(ROLE_NAMES):
            if r_idx >= R:
                continue
            weights = leaf_effects[p, r_idx]                    # [V]
            top10_idx = np.argsort(weights)[::-1][:10]
            for rank, v_idx in enumerate(top10_idx):
                if weights[v_idx] <= 0:
                    continue
                cluster_id = vocab_clusters[v_idx]
                words = cluster_to_words.get(cluster_id, [])
                for word in words:
                    kw_records.append({
                        'persona':    p,
                        'role':       r_name,
                        'cluster_id': cluster_id,
                        'cluster_rank': rank + 1,
                        'word':       word,
                        'weight':     float(weights[v_idx]),
                    })

    kw_path = os.path.join(res_dir, "keywords.csv")
    pd.DataFrame(kw_records).to_csv(kw_path, index=False)
    print(f"  keywords saved          ({len(kw_records)} rows)  -> {kw_path}")

    # ── 8. Filtered keywords: remove words in >80% of personas per role ────────
    # Count how many personas contain each (role, word) in their top-10 clusters
    global_hits: dict[tuple, int] = {}
    all_word_entries = []   # (p, r_name, word, weight, cluster_id)

    for p in range(P):
        for r_idx, r_name in enumerate(ROLE_NAMES):
            if r_idx >= R:
                continue
            weights = leaf_effects[p, r_idx]
            top10_idx = np.argsort(weights)[::-1][:10]
            seen_words: set[str] = set()   # deduplicate within same persona-role
            for v_idx in top10_idx:
                if weights[v_idx] <= 0:
                    continue
                cluster_id = vocab_clusters[v_idx]
                words = cluster_to_words.get(cluster_id, [])
                for word in words:
                    if word not in seen_words:
                        seen_words.add(word)
                        key = (r_name, word)
                        global_hits[key] = global_hits.get(key, 0) + 1
                        all_word_entries.append((p, r_name, word, float(weights[v_idx]), cluster_id))

    threshold = max(1, P * 0.8)
    filtered = [
        {'persona': p, 'role': r, 'word': w, 'weight': wt, 'cluster_id': cid}
        for p, r, w, wt, cid in all_word_entries
        if global_hits[(r, w)] <= threshold
    ]
    fk_path = os.path.join(res_dir, "filtered-keywords.csv")
    pd.DataFrame(filtered).to_csv(fk_path, index=False)
    print(f"  filtered-keywords saved ({len(filtered)} rows)  -> {fk_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Traditional SAGE models: regenerate char_assignments, keywords, filtered-keywords"
    )
    parser.add_argument(
        "--labels", nargs='+',
        default=["BERT-256", "BERT-512", "W2V-256", "W2V-512"],
        help="Which cluster-type labels to evaluate"
    )
    parser.add_argument("--start_p",  type=int,   default=5)
    parser.add_argument("--end_p",    type=int,   default=10)
    parser.add_argument("--l1",       type=float, default=1.0)
    parser.add_argument("--data",     type=str,   default="data/processed/sage_input_bert512.csv")
    parser.add_argument("--ckpt_base",type=str,   default="checkpoints")
    parser.add_argument("--res_base", type=str,   default="data/results/traditional_results")
    args = parser.parse_args()

    for label in args.labels:
        if label not in CLUSTER_LABELS:
            print(f"  [WARN] Unknown label '{label}', valid: {list(CLUSTER_LABELS)}")
            continue
        for p in range(args.start_p, args.end_p + 1):
            eval_one_traditional(label, p, args.l1, args.data, args.ckpt_base, args.res_base)

    print("\n" + "#" * 60)
    print(">>> All Traditional SAGE evaluations complete.")
    print("#" * 60)


if __name__ == "__main__":
    main()
