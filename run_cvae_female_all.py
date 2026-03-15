import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sage.model import AdvancedLiterarySAGE
from sage.metrics import calculate_silhouette, calculate_latent_silhouette, calculate_flat_perplexity

def check_cuda_environment():
    if torch.cuda.is_available():
        return True
    return False

def run_female_all():
    has_gpu = check_cuda_environment()
    word_csv = "fullset_data/word2vec_clusters.csv" 
    data_file = "fullset_data/all_words.csv"
    if not os.path.exists(word_csv):
        word_csv = "data/processed/word2vec_clusters.csv"
        data_file = "data/processed/all_words.csv"

    n_personas = 8
    iters = 1000 if has_gpu else 50
    out_dir = 'data/results/female_all'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f">>> Initializing CVAE-SAGE for All Female Characters (iters={iters})")
    trainer = AdvancedLiterarySAGE(n_personas=n_personas, mode='cvae_flat', iters=iters, l1_lambda=1e-6)
    
    df_full = trainer.load_data(data_file, word_csv)
    
    meta = pd.read_csv('data/raw/all_characters_metadata.csv')
    meta['char_key'] = meta['book'] + '_' + meta['char_id'].astype(str)
    females = meta[meta['gender'].str.contains('she/her', na=False, case=False)]
    meta_dict = dict(zip(females['char_key'], females['best_name']))
    
    female_keys = females['char_key'].tolist()
    df_full = df_full[df_full['char_key'].isin(female_keys)].copy()
    print(f"    Total characters loaded: {len(df_full['char_key'].unique())}")
    
    char_keys = list(df_full['char_key'].unique())
    np.random.seed(42)
    np.random.shuffle(char_keys)
    split_idx = int(len(char_keys) * 0.9)
    train_keys = char_keys[:split_idx]
    test_keys = char_keys[split_idx:]
    
    train_df = df_full[df_full['char_key'].isin(train_keys)].copy()
    test_df = df_full[df_full['char_key'].isin(test_keys)].copy()
    
    start_time = time.time()
    trainer.fit(train_df, batch_size=8192, checkpoint_dir=f'{out_dir}/checkpoints')
    end_time = time.time()
    print(f">>> Training completed in {end_time - start_time:.2f}s")
    
    torch.save(trainer.model.state_dict(), f'{out_dir}/model.pt')

    labels = trainer.p_assignments
    unique_personas = np.unique(labels)
    
    # 评估
    C = trainer.C
    V = trainer.V
    char_word_counts = train_df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_feats = np.zeros((C, V))
    char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
    row_sums = char_feats.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    char_dists = char_feats / row_sums
    
    raw_s_score = calculate_silhouette(char_dists, labels)
    latent_s_score, _ = calculate_latent_silhouette(trainer.model, train_df, trainer.device)
    perp = calculate_flat_perplexity(trainer.model, test_df, trainer.device)
    
    print(f"    [Raw Space] Silhouette Score   : {raw_s_score:.4f}")
    print(f"    [Latent Space] Silhouette Score: {latent_s_score:.4f}")
    print(f"    [Test Set] Perplexity          : {perp:.4f}")

    # 导出角色
    inv_char_map = {v: k for k, v in trainer.char_map.items()}
    cluster_assignments = []
    
    for p in unique_personas:
        chars_in_cluster = []
        for i in range(len(labels)):
            if labels[i] == p:
                char_key = inv_char_map[i]
                char_name = meta_dict.get(char_key, char_key)
                chars_in_cluster.append(char_name)
                cluster_assignments.append({'persona': p, 'char_key': char_key, 'character_name': char_name})
        print(f"\n  Persona {p} Characters (sample): {', '.join(chars_in_cluster[:10])} ... ({len(chars_in_cluster)} total)")

    pd.DataFrame(cluster_assignments).to_csv(f'{out_dir}/cluster_assignments.csv', index=False)

    # 导出特征词
    vocab = trainer.vocab
    eta_persona = trainer.model.decoder.eta_persona.detach().cpu().numpy()
    role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}
    records = []
    for p in unique_personas:
        for r_idx, r_name in role_names.items():
            if r_idx < trainer.R:
                weights = eta_persona[p, r_idx, :]
                top_indices = np.argsort(weights)[-10:][::-1]
                for i in top_indices:
                    if weights[i] > 0:
                        records.append({'persona': p, 'role': r_name, 'word': vocab[i], 'weight': weights[i]})
    pd.DataFrame(records).to_csv(f'{out_dir}/keywords.csv', index=False)

if __name__ == "__main__":
    run_female_all()