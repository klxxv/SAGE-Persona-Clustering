import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

# =================================================================================
# IMPORTANT: This script requires the following libraries to be installed:
# pip install pandas gensim torch scikit-learn transformers
# =================================================================================

# Try to import necessary libraries and provide helpful error messages
try:
    from gensim.models import Word2Vec
except ImportError:
    print("Error: gensim is not installed. Please run 'pip install gensim'")
    exit()

try:
    import torch
    from transformers import BertTokenizer, BertModel
except ImportError:
    print("Error: transformers or torch is not installed. Please run 'pip install transformers torch'")
    exit()

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except ImportError:
    print("Error: scikit-learn is not installed. Please run 'pip install scikit-learn'")
    exit()


def get_text_corpus(df):
    """Convert the dataframe of words into a list of sentences for Word2Vec."""
    print(">>> Building corpus for Word2Vec...")
    char_word_groups = df.groupby(['book', 'char_id'])['word'].apply(list)
    corpus = [words for words in char_word_groups if len(words) > 1]
    print(f"Corpus built with {len(corpus)} documents.")
    return corpus

def train_word2vec_embeddings(words, corpus, embedding_dim=100, workers=4):
    """Train Word2Vec model and return a dictionary of word vectors."""
    print(">>> Training Word2Vec model...")
    w2v_model = Word2Vec(
        corpus,
        vector_size=embedding_dim,
        window=5,
        min_count=5,
        sg=1,
        workers=workers
    )
    print("Word2Vec model trained.")
    embedding_dict = {word: w2v_model.wv[word] for word in w2v_model.wv.index_to_key if word in words}
    print(f"Extracted {len(embedding_dict)} word vectors.")
    return embedding_dict

def get_bert_embeddings(words, model_name='bert-base-uncased', batch_size=32):
    """Generate BERT embeddings for a list of words."""
    print(f">>> Loading BERT model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Generating BERT embeddings on device: {device}...")
    word_vectors = {}
    word_list = list(words)

    with torch.no_grad():
        for i in tqdm(range(0, len(word_list), batch_size), desc="BERT Embedding"):
            batch = word_list[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            for word, embedding in zip(batch, embeddings):
                word_vectors[word] = embedding
    
    print(f"Generated {len(word_vectors)} BERT embeddings.")
    return word_vectors

def perform_kmeans(embedding_dict, n_clusters=1000, use_pca=False, pca_components=100):
    """
    Performs K-Means and adds the final vector representation to the output.
    """
    print(f">>> Performing K-Means clustering with {n_clusters} clusters...")
    
    words = list(embedding_dict.keys())
    vectors = np.array(list(embedding_dict.values()))

    if use_pca:
        print(f"Applying PCA to reduce dimensions from {vectors.shape[1]} to {pca_components}...")
        pca = PCA(n_components=pca_components)
        vectors = pca.fit_transform(vectors)
        print("PCA complete.")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    # Convert final vectors to a string format for CSV compatibility
    vector_strings = [','.join(map(str, vec)) for vec in vectors]
    
    df_clusters = pd.DataFrame({
        "word": words,
        "cluster_id": cluster_labels,
        "vector": vector_strings
    })
    print("K-Means clustering complete.")
    return df_clusters


if __name__ == "__main__":
    # --- Configuration ---
    DATA_FILE = os.path.join("fullset_data", "all_words.csv")
    OUTPUT_DIR = "fullset_data"
    BERT_MODEL_PATH = "bert-base-uncased"
    N_CLUSTERS = 1000
    PCA_COMPONENTS = 100
    
    # --- Main Logic ---
    print("--- Starting Vocabulary Preprocessing ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        exit()
        
    df = pd.read_csv(DATA_FILE)
    unique_words = set(df['word'].dropna().unique())
    print(f"Found {len(unique_words)} unique words.")

    # --- Method 1: Word2Vec + K-Means ---
    print("\n--- Processing: Word2Vec ---")
    corpus = get_text_corpus(df)
    w2v_embeddings = train_word2vec_embeddings(unique_words, corpus)
    df_w2v_clusters = perform_kmeans(w2v_embeddings, n_clusters=N_CLUSTERS, use_pca=False)
    
    w2v_output_path = os.path.join(OUTPUT_DIR, "word2vec_clusters.csv")
    df_w2v_clusters.to_csv(w2v_output_path, index=False)
    print(f"Word2Vec cluster map saved to: {w2v_output_path}")

    # --- Method 2: BERT + K-Means with PCA ---
    print("\n--- Processing: BERT with PCA ---")
    bert_embeddings = get_bert_embeddings(unique_words, model_name=BERT_MODEL_PATH)
    df_bert_clusters = perform_kmeans(
        bert_embeddings, 
        n_clusters=N_CLUSTERS, 
        use_pca=True, 
        pca_components=PCA_COMPONENTS
    )
    
    bert_output_path = os.path.join(OUTPUT_DIR, "bert_clusters.csv")
    df_bert_clusters.to_csv(bert_output_path, index=False)
    print(f"BERT cluster map saved to: {bert_output_path}")
    
    print("\n--- Preprocessing Complete! ---")
    print("You can now run the main SAGE script with the generated cluster files.")
