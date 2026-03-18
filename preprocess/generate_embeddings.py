import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import gensim.downloader as api

# Note: Requires 'transformers' and 'gensim'
try:
    from transformers import BertTokenizer, BertModel
except ImportError:
    print("Error: transformers not found. Please install it in the conda environment.")

def generate_embeddings(test_mode=False):
    processed_dir = 'data/processed'
    vocab_path = os.path.join(processed_dir, 'female_vocab_map.csv')
    output_path = os.path.join(processed_dir, 'female_words_embedding.csv')
    
    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} not found.")
        return

    df_vocab = pd.read_csv(vocab_path)
    if test_mode:
        # Select two specific words for testing
        df_vocab = df_vocab.head(5).copy()
        print(f">>> Running in TEST MODE with words: {df_vocab['word'].tolist()}")

    # 1. Initialize BERT
    print(">>> Initializing BERT (bert-base-uncased)...")
    # Explicitly using CPU as requested, though code handles CUDA if available
    device = torch.device("cpu")
    print(f">>> Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    # 2. Initialize Word2Vec
    print(">>> Loading Word2Vec (word2vec-google-news-300)... This may take a while if downloading for the first time.")
    try:
        w2v_model = api.load('word2vec-google-news-300')
    except Exception as e:
        print(f"Error loading Word2Vec: {e}")
        return

    results = []

    print(f">>> Processing {len(df_vocab)} words...")
    for _, row in tqdm(df_vocab.iterrows(), total=len(df_vocab)):
        word = str(row['word']).lower()
        word_id = row['word_id']
        
        # A. BERT Embedding
        inputs = tokenizer(word, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # If word is split into multiple subwords, we average them or take the first
            # shape: [1, seq_len, 768]
            # seq_len includes [CLS] at index 0 and [SEP] at last index
            embeddings = outputs.last_hidden_state[0]
            if embeddings.shape[0] > 2:
                # Average the embeddings of the subwords (excluding [CLS] and [SEP])
                bert_vec = torch.mean(embeddings[1:-1], dim=0).cpu().numpy()
            else:
                # If only one token (+ special tokens)
                bert_vec = embeddings[1].cpu().numpy()
        
        # B. Word2Vec Embedding
        if word in w2v_model:
            w2v_vec = w2v_model[word]
        else:
            # Check for capitalized version if lowercase not found
            if word.capitalize() in w2v_model:
                w2v_vec = w2v_model[word.capitalize()]
            else:
                w2v_vec = np.zeros(300)

        results.append({
            'word_id': word_id,
            'word2vec_embedding': ",".join(map(str, w2v_vec.tolist())),
            'bert_embedding': ",".join(map(str, bert_vec.tolist()))
        })

    df_emb = pd.DataFrame(results)
    df_emb.to_csv(output_path, index=False)
    print(f"\nSUCCESS: Saved embeddings to {output_path}")

if __name__ == '__main__':
    # Full embedding generation as requested
    generate_embeddings(test_mode=False)
