"""
Build per-role W2V clusters for SAGE.
Each role (agent, patient, possessive, predicative) gets its own independent
K-Means clustering and saves TWO files per role:

  {role}_clusters.csv       word, cluster_id only  (lightweight, tracked in git)
  {role}_clusters_full.csv  word, cluster_id, vector (large, gitignored, used for training)

n_clusters = 2^round(log2(n_unique_words / 10))   (nearest power of 2)
Output: data/sage_cluster_dataset/w2v-role/
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROLES = ['agent', 'patient', 'possessive', 'predicative']

DATA_FILE   = "data/processed/female_words_with_ids.csv"
VOCAB_FILE  = "data/processed/female_vocab_map.csv"
EMB_FILE    = "data/processed/female_word2vec_embedding.csv"
OUT_DIR     = "data/sage_cluster_dataset/w2v-role"


def nearest_power_of_2(n):
    if n < 1:
        return 1
    lo = 1 << (n.bit_length() - 1)   # largest power <= n
    hi = lo << 1                       # smallest power >= n
    return lo if (n - lo) <= (hi - n) else hi


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load training data ────────────────────────────────────────────────────
    print("Loading training data...")
    df = pd.read_csv(DATA_FILE)
    df['role'] = df['role'].astype(str).str.lower().str.strip()
    df = df[df['role'].isin(ROLES)].copy()

    # ── Load W2V embeddings ───────────────────────────────────────────────────
    print("Loading W2V vocab map...")
    df_vocab = pd.read_csv(VOCAB_FILE)
    df_vocab.columns = df_vocab.columns.str.lstrip('\ufeff')
    word_id_map = dict(zip(df_vocab['word_id'], df_vocab['word']))   # id → word

    print("Loading W2V embeddings...")
    df_emb = pd.read_csv(EMB_FILE)
    df_emb.columns = df_emb.columns.str.lstrip('\ufeff')
    emb_col = [c for c in df_emb.columns if c != 'word_id'][0]
    print(f"  Embedding column: '{emb_col}'")

    df_emb['word'] = df_emb['word_id'].map(word_id_map)
    df_emb = df_emb.dropna(subset=['word'])
    df_emb['vec'] = df_emb[emb_col].apply(
        lambda x: np.fromstring(x, sep=',')
    )
    word_to_vec = dict(zip(df_emb['word'], df_emb['vec']))
    print(f"  Vocabulary size: {len(word_to_vec)}")

    # ── Per-role clustering ───────────────────────────────────────────────────
    for role in ROLES:
        print(f"\n{'='*60}")
        print(f"Role: {role.upper()}")

        role_words = df[df['role'] == role]['word'].unique()
        covered    = [w for w in role_words if w in word_to_vec]
        print(f"  Unique words in training data : {len(role_words)}")
        print(f"  Words with W2V embedding      : {len(covered)}")

        if len(covered) < 4:
            print(f"  [SKIP] Too few words to cluster.")
            continue

        n_clusters = nearest_power_of_2(len(covered) // 10)
        n_clusters = max(n_clusters, 2)   # at least 2 clusters
        print(f"  n_unique/10 = {len(covered)//10}  →  n_clusters = {n_clusters}")

        words  = np.array(covered)
        vecs   = np.vstack([word_to_vec[w] for w in words])  # [N, D]

        print(f"  Running K-Means (k={n_clusters})...")
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(vecs)

        # Cluster centroid vectors
        centroids = km.cluster_centers_   # [n_clusters, D]

        # Build per-word centroid string (vector = centroid of its cluster)
        centroid_strs = [
            ','.join(f'{v:.6f}' for v in centroids[c]) for c in labels
        ]

        df_full = pd.DataFrame({
            'word':       words,
            'cluster_id': labels.astype(int),
            'vector':     centroid_strs,
        })
        df_lite = df_full[['word', 'cluster_id']]

        # Full version (with vectors) – gitignored, used at training time
        full_path = os.path.join(OUT_DIR, f"{role}_clusters_full.csv")
        df_full.to_csv(full_path, index=False)
        print(f"  Saved full ({len(df_full)} words) → {full_path}")

        # Lightweight version (no vectors) – tracked in git
        lite_path = os.path.join(OUT_DIR, f"{role}_clusters.csv")
        df_lite.to_csv(lite_path, index=False)
        print(f"  Saved lite ({len(df_lite)} words) → {lite_path}")

        # Summary per cluster
        sizes = pd.Series(labels).value_counts()
        print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
              f"mean={sizes.mean():.1f}, median={sizes.median():.0f}")

    print(f"\n{'#'*60}")
    print(">>> Per-role W2V clustering complete.")
    print(f">>> Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
