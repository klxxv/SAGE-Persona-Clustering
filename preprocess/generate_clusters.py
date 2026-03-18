import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time

def parse_vector(s):
    # 解析逗号分隔的字符串向量
    return np.fromstring(s, sep=',')

def load_data_and_vectors(embedding_paths, vocab_map_path, name):
    print(f">>> Loading {name} embeddings...")
    # 1. 加载词汇表
    df_vocab = pd.read_csv(vocab_map_path)
    vocab_map = dict(zip(df_vocab['word_id'], df_vocab['word']))
    
    # 2. 加载嵌入分片并合并
    df_list = []
    if isinstance(embedding_paths, list):
        for p in embedding_paths:
            df_list.append(pd.read_csv(p))
        df_emb = pd.concat(df_list, axis=0)
    else:
        df_emb = pd.read_csv(embedding_paths)
    
    # 3. 解析向量
    emb_col = 'word2vec_embedding' if 'word2vec' in name.lower() else 'bert_embedding'
    print(f"    Parsing vectors from {emb_col}...")
    
    vectors = np.stack(df_emb[emb_col].apply(parse_vector).values)
    words = df_emb['word_id'].map(vocab_map).values
    
    print(f"    Loaded {name}: {vectors.shape} vectors.")
    return words, vectors

def perform_clustering(words, vectors, n_clusters, output_dir, name_prefix):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f">>> Running K-Means for {name_prefix} with {n_clusters} clusters...")
    start_time = time.time()
    
    # 设置 n_init 为 10 以保证聚类稳定性
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.2f}s")
    
    # 保存结果
    res_df = pd.DataFrame({
        'word': words,
        'cluster': cluster_labels
    })
    
    # 统一命名为 word2vec_clusters.csv 方便 run_cvae_full.py 读取
    output_path = os.path.join(output_dir, 'word2vec_clusters.csv')
    res_df.to_csv(output_path, index=False)
    print(f"    Saved to: {output_path}")

def run_all():
    vocab_map_path = 'data/processed/female_vocab_map.csv'
    
    # Word2Vec 路径
    w2v_path = 'data/processed/female_word2vec_embedding.csv'
    # BERT 分片路径
    bert_paths = [
        'data/processed/female_bert_embedding_part1.csv',
        'data/processed/female_bert_embedding_part2.csv',
        'data/processed/female_bert_embedding_part3.csv'
    ]
    
    # 加载数据
    w2v_words, w2v_vectors = load_data_and_vectors(w2v_path, vocab_map_path, 'Word2Vec')
    bert_words, bert_vectors = load_data_and_vectors(bert_paths, vocab_map_path, 'BERT')
    
    # 配置
    configs = [
        (w2v_words, w2v_vectors, 256, 'data/results/clusters/w2v_256', 'W2V'),
        (w2v_words, w2v_vectors, 512, 'data/results/clusters/w2v_512', 'W2V'),
        (bert_words, bert_vectors, 256, 'data/results/clusters/bert_256', 'BERT'),
        (bert_words, bert_vectors, 512, 'data/results/clusters/bert_512', 'BERT'),
    ]
    
    for words, vectors, n, path, name in configs:
        perform_clustering(words, vectors, n, path, name)
    
    print("\n>>> ALL CLUSTERING TASKS FINISHED.")

if __name__ == "__main__":
    run_all()
