import pandas as pd
import numpy as np
import os
import re
import torch

def smart_normalize(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def create_split_tables():
    # 1. Paths
    all_words_path = "fullset_data/all_words.csv"
    targets_path = "data/processed/target_female_ids.csv"
    w2v_path = "fullset_data/word2vec_clusters.csv"
    bert_path = "fullset_data/bert_clusters.csv"
    ckpt_path = "data/results/cvae_flat_full_model.pt"
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print(">>> [Finalizer] Loading data...")
    df_all = pd.read_csv(all_words_path)
    df_targets = pd.read_csv(targets_path)
    
    # 2. Book Mapping
    real_books = df_all['book'].unique()
    real_books_norm = {smart_normalize(b): b for b in real_books}
    manual_map = {"peachblossompavilion": "Peach_Blossom_Pavillion", "don'tcrytailake": "Dont_Cry_Tai_Lake"}

    def get_real_book(excel_book):
        norm = smart_normalize(excel_book)
        if norm in manual_map: return manual_map[norm]
        if norm in real_books_norm: return real_books_norm[norm]
        for nb, full in real_books_norm.items():
            if norm in nb or nb in norm: return full
        return None

    df_targets['book_mapped'] = df_targets['book'].apply(get_real_book)
    
    # 3. Merge and Aggregate
    print(">>> Aggregating character data...")
    df_merged = pd.merge(df_all, df_targets, left_on=['book', 'char_id'], right_on=['book_mapped', 'char_id'], suffixes=('', '_target'))
    df_base = df_merged.groupby(['book_target', 'author', 'name', 'role', 'word'])['count'].sum().reset_index()
    df_base = df_base.rename(columns={'book_target': 'book'})

    # 4. Add Logits to Base table
    df_base['logits'] = 0.0
    if os.path.exists(ckpt_path):
        try:
            print(">>> Enriching Base table with Logits...")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if 'decoder.eta_bg' in state_dict:
                eta_bg = state_dict['decoder.eta_bg'].numpy()
                vocab_list = pd.read_csv(w2v_path)['word'].tolist()
                w_to_idx = {w: i for i, w in enumerate(vocab_list)}
                r_to_idx = {'agent': 0, 'patient': 1, 'possessive': 2, 'predicative': 3}
                def fetch_logit(row):
                    wi, ri = w_to_idx.get(row['word']), r_to_idx.get(row['role'])
                    return eta_bg[ri, wi] if wi is not None and ri is not None else 0.0
                df_base['logits'] = df_base.apply(fetch_logit, axis=1)
        except: pass

    # 5. Save Base Table
    base_out = os.path.join(output_dir, "female_words_base.csv")
    df_base.to_csv(base_out, index=False, encoding='utf-8-sig')
    print(f"Saved Base table: {base_out}")

    # 6. Create and Save Embedding Tables (Separately to avoid large files)
    unique_female_words = df_base['word'].unique()
    
    if os.path.exists(w2v_path):
        print(">>> Creating W2V Embedding Table...")
        df_w2v_all = pd.read_csv(w2v_path)[['word', 'vector']]
        df_w2v_subset = df_w2v_all[df_w2v_all['word'].isin(unique_female_words)]
        w2v_out = os.path.join(output_dir, "female_embeddings_w2v.csv")
        df_w2v_subset.to_csv(w2v_out, index=False, encoding='utf-8-sig')
        print(f"Saved W2V table: {w2v_out}")

    if os.path.exists(bert_path):
        print(">>> Creating BERT Embedding Table...")
        df_bert_all = pd.read_csv(bert_path)[['word', 'vector']]
        df_bert_subset = df_bert_all[df_bert_all['word'].isin(unique_female_words)]
        bert_out = os.path.join(output_dir, "female_embeddings_bert.csv")
        df_bert_subset.to_csv(bert_out, index=False, encoding='utf-8-sig')
        print(f"Saved BERT table: {bert_out}")

    print("\n>>> All split tables created successfully.")

if __name__ == "__main__":
    create_split_tables()
