import os
import sys
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from sage.model_traditional import LiteraryPersonaSAGE

def run_p_task(p_val, data_file, word_csv_file, output_root):
    print(f"\n[Process P={p_val}] Initializing model...")
    # Explicitly set device to CPU or single GPU per process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LiteraryPersonaSAGE(
        n_personas=p_val,
        em_iters=100, # Gibbs EM rounds
        l1_lambda=1.0, # Match paper (Laplace prior scale=1)
        min_mentions=0
    )
    
    # Load and Preprocess
    # Note: LiteraryPersonaSAGE._build_balanced_tree needs a vector column in word_csv
    # We will ensure the word_csv passed has BERT vectors
    df_processed, num_nodes = model.load_and_preprocess_data(data_file, word_csv_file)
    
    # Fit
    ckpt_dir = os.path.join(output_root, f"traditional_P{p_val}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    model.fit(df_processed, num_nodes, checkpoint_dir=ckpt_dir)
    
    # Save Final
    model.save_results(ckpt_dir)
    print(f"[Process P={p_val}] COMPLETED. Results saved to {ckpt_dir}")

def main():
    data_file = "data/processed/sage_input_bert512.csv"
    word_csv_file = "data/sage_cluster_dataset/bert_512/word2vec_clusters.csv"
    output_root = "checkpoints/traditional_search"
    os.makedirs(output_root, exist_ok=True)
    
    # Step 0: LiteraryPersonaSAGE expects word_csv to have 'cluster_id' and 'vector' columns.
    # Compute real BERT PCA-100 cluster centroids from tracked embedding files.
    df_c = pd.read_csv(word_csv_file)
    if 'vector' not in df_c.columns:
        print(">>> Computing real BERT PCA-100 cluster centroids for semantic tree building...")
        vocab_file = "data/processed/female_vocab_map.csv"
        emb_file   = "data/processed/female_bert_pca100_embedding.csv"
        df_vocab = pd.read_csv(vocab_file)                          # word_id, word
        df_emb   = pd.read_csv(emb_file)                            # word_id, bert_pca_100
        df_emb['vec_arr'] = df_emb['bert_pca_100'].apply(
            lambda x: np.array([float(v) for v in x.split(',')])
        )
        # Join: word_id -> word -> cluster
        df_joined = (df_vocab
                     .merge(df_emb[['word_id', 'vec_arr']], on='word_id')
                     .merge(df_c[['word', 'cluster']], on='word'))
        # Cluster centroid = mean of all member word vectors
        centroids = (df_joined
                     .groupby('cluster')['vec_arr']
                     .apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0))
                     .to_dict())
        df_c['cluster_id'] = df_c['cluster']
        df_c['vector'] = df_c['cluster'].map(
            lambda c: ','.join(f'{v:.6f}' for v in centroids[c])
        )
        word_csv_temp = "data/processed/temp_cluster_vectors.csv"
        df_c.to_csv(word_csv_temp, index=False)
        word_csv_file = word_csv_temp
        print(f"    Real centroids computed for {len(centroids)} clusters (dim=100). Saved to {word_csv_temp}")

    p_list = [8]
    output_root = "checkpoints/test_traditional_fix"
    os.makedirs(output_root, exist_ok=True)
    
    for p in p_list:
        run_p_task(p, data_file, word_csv_file, output_root)

if __name__ == "__main__":
    main()
