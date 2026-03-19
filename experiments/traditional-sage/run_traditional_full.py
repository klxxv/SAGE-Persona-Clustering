import os
import time
import pandas as pd
import numpy as np
import torch
import argparse
from sage.model_traditional import LiteraryPersonaSAGE
from sage.metrics import calculate_distribution_silhouette

def run_traditional_full(n_personas=8, em_iters=100, l1_lambda=1.0, subset_chars=None, 
                         data_file=None, word_csv=None, label="Traditional"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cluster_type = "W2V" if "w2v" in word_csv.lower() else "BERT"
    
    print(f">>> {label} SAGE ({cluster_type}-based): P={n_personas}, iters={em_iters}, l1={l1_lambda}")

    if word_csv is None: word_csv = "data/sage_cluster_dataset/bert_512/word2vec_clusters.csv"
    if data_file is None: data_file = "data/processed/sage_input_bert512.csv"

    df_c = pd.read_csv(word_csv)
    if 'vector' not in df_c.columns:
        print(f">>> Computing {cluster_type} centroids for semantic tree...")
        df_vocab = pd.read_csv("data/processed/female_vocab_map.csv")
        emb_file = "data/processed/female_bert_pca100_embedding.csv" if cluster_type == "BERT" else "data/processed/female_word2vec_embedding.csv"
        
        if not os.path.exists(emb_file): raise FileNotFoundError(f"Embedding file not found: {emb_file}")
        df_emb = pd.read_csv(emb_file)
        
        # Robust column detection for embeddings
        emb_col = None
        for col in ['bert_pca_100', 'word2vec_embedding', 'vector']:
            if col in df_emb.columns:
                emb_col = col
                break
        if emb_col is None: raise KeyError(f"Could not find embedding column in {emb_file}. Columns: {df_emb.columns.tolist()}")
        
        df_emb['vec_arr'] = df_emb[emb_col].apply(lambda x: np.array([float(v) for v in str(x).split(',')]))
        df_joined = df_vocab.merge(df_emb[['word_id', 'vec_arr']], on='word_id').merge(df_c[['word', 'cluster']], on='word')
        centroids = df_joined.groupby('cluster')['vec_arr'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0)).to_dict()
        df_c['cluster_id'] = df_c['cluster']
        df_c['vector'] = df_c['cluster'].map(lambda c: ','.join(f'{v:.6f}' for v in centroids[c]))
        
        word_csv = f"data/processed/temp_vectors_{label}.csv"
        df_c.to_csv(word_csv, index=False)

    model = LiteraryPersonaSAGE(n_personas=n_personas, em_iters=em_iters, l1_lambda=l1_lambda, min_mentions=0)
    df_processed, num_nodes = model.load_and_preprocess_data(data_file, word_csv)
    
    if subset_chars is not None:
        unique_chars = df_processed['char_key'].unique()
        np.random.seed(42)
        subset_keys = np.random.choice(unique_chars, size=min(subset_chars, len(unique_chars)), replace=False)
        df_processed = df_processed[df_processed['char_key'].isin(subset_keys)].copy()

    # Organized results folder
    res_dir = os.path.abspath(f"data/results/traditional_results/{label}/P{n_personas}_L{l1_lambda}")
    ckpt_dir = os.path.abspath(f"checkpoints/traditional_{label}_P{n_personas}_L{l1_lambda}")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_time = time.time()
    df_fit_mapped = model.fit(df_processed, num_nodes, checkpoint_dir=ckpt_dir)
    print(f">>> Training completed in {time.time() - start_time:.2f}s")

    df_assign = model.char_info_df.copy()
    df_assign["persona"] = model.p_assignments
    df_assign.to_csv(os.path.join(res_dir, "char_assignments.csv"), index=False)

    s_score = calculate_distribution_silhouette(df_fit_mapped, model.p_assignments, len(model.vocab_clusters))
    print(f"    Silhouette Score: {s_score:.4f}")

    all_word_weights = []
    with torch.no_grad():
        for p in range(n_personas):
            for r in range(model.R):
                logits = model.model.eta_pers[p, r, model.word_paths] 
                path_weights = (logits * model.word_signs).sum(dim=1) 
                top_idx = torch.argsort(path_weights, descending=True)[:15]
                for idx in top_idx:
                    w = path_weights[idx].item()
                    if w > 0:
                        all_word_weights.append({
                            'persona': p, 'role': ['Agent','Patient','Possessive','Predicative'][r],
                            'cluster_id': model.vocab_clusters[idx], 'weight': w
                        })
    pd.DataFrame(all_word_weights).to_csv(os.path.join(res_dir, "keywords.csv"), index=False)
    print(f">>> Results saved to {res_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_personas", type=int, default=8)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--l1", type=float, default=1.0)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--word_csv", type=str, default=None)
    parser.add_argument("--label", type=str, default="Traditional")
    args = parser.parse_args()
    run_traditional_full(n_personas=args.n_personas, em_iters=args.iters, l1_lambda=args.l1, 
                         subset_chars=args.subset, data_file=args.data_file, 
                         word_csv=args.word_csv, label=args.label)
