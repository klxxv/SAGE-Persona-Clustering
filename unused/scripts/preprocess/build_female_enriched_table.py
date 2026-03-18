import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F

def build_enriched_table():
    # 1. 路径设置
    all_words_path = "fullset_data/all_words.csv"
    targets_path = "data/processed/target_female_ids.csv"
    w2v_path = "fullset_data/word2vec_clusters.csv"
    bert_path = "fullset_data/bert_clusters.csv"
    ckpt_path = "checkpoints/cvae_flat_full/cvae_flat_full_model.pt"
    out_path = "data/processed/female_words_enriched.csv"

    print(">>> [Builder] Loading raw data and targets...")
    if not os.path.exists(all_words_path) or not os.path.exists(targets_path):
        print("Error: Required CSV files missing.")
        return

    df_all = pd.read_csv(all_words_path)
    df_targets = pd.read_csv(targets_path)

    # 2. 标准化书名以确保匹配
    def normalize_book(name):
        return "".join(str(name).replace("_", "").split()).lower()

    df_all['book_norm'] = df_all['book'].apply(normalize_book)
    df_targets['book_norm'] = df_targets['book'].apply(normalize_book)

    # 3. 关联 ID 并按 Name 合并
    print(">>> Merging and aggregating words by character name...")
    # Merge targets onto all words
    df_merged = pd.merge(df_all, df_targets, on=['book_norm', 'char_id'], suffixes=('', '_target'))
    
    # Aggregate counts by Name
    df_agg = df_merged.groupby(['book', 'author', 'name', 'role', 'word'])['count'].sum().reset_index()
    
    # 4. 加载嵌入向量 (Embeddings)
    print(">>> Enriching with Word2Vec and BERT embeddings...")
    if os.path.exists(w2v_path):
        df_w2v = pd.read_csv(w2v_path)[['word', 'vector']].rename(columns={'vector': 'w2v_embedding'})
        df_agg = pd.merge(df_agg, df_w2v, on='word', how='left')
    
    if os.path.exists(bert_path):
        df_bert = pd.read_csv(bert_path)[['word', 'vector']].rename(columns={'vector': 'bert_embedding'})
        df_agg = pd.merge(df_agg, df_bert, on='word', how='left')

    # 5. 加载 Logits (背景偏置 eta_bg)
    # 注意：这反映了词汇在全局背景下的基础权重
    df_agg['logits'] = 0.0
    if os.path.exists(ckpt_path):
        try:
            print(">>> Extracting logits from model checkpoints...")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            # 假设 eta_bg 存在于 state_dict 中 (Shape: [R, V])
            if 'decoder.eta_bg' in state_dict:
                eta_bg = state_dict['decoder.eta_bg'].numpy() # [4, V]
                # 我们需要建立词到索引的映射来对齐 Logits
                # 使用 w2v 文件中的词表顺序作为索引参考
                df_vocab = pd.read_csv(w2v_path)
                word_to_idx = {w: i for i, w in enumerate(df_vocab['word'])}
                
                role_map = {'agent': 0, 'patient': 1, 'possessive': 2, 'predicative': 3}
                
                def get_logit(row):
                    w_idx = word_to_idx.get(row['word'])
                    r_idx = role_map.get(row['role'])
                    if w_idx is not None and r_idx is not None:
                        return eta_bg[r_idx, w_idx]
                    return 0.0
                
                df_agg['logits'] = df_agg.apply(get_logit, axis=1)
        except Exception as e:
            print(f"Warning: Could not load logits from model: {e}")

    # 6. 保存最终大表
    print(f">>> Final table size: {len(df_agg)} rows.")
    df_agg.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f">>> Enriched female feature table saved to: {out_path}")

    # Preview
    print("\nSample Output:")
    print(df_agg[['book', 'name', 'role', 'word', 'count', 'logits']].head(10))

if __name__ == "__main__":
    build_enriched_table()
