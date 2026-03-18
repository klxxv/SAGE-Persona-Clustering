import pandas as pd
import os

def debug_missing():
    enriched_path = "data/processed/female_words_enriched.csv"
    w2v_path = "fullset_data/word2vec_clusters.csv"
    
    if not os.path.exists(enriched_path): return
    
    df = pd.read_csv(enriched_path)
    # Check rows where w2v_embedding is NaN
    missing_w2v = df[df['w2v_embedding'].isna()]
    
    print(f"Total rows in enriched table: {len(df)}")
    print(f"Rows with missing Word2Vec: {len(missing_w2v)} ({len(missing_w2v)/len(df)*100:.2f}%)")
    
    if len(missing_w2v) > 0:
        print("\nSample of missing words:")
        # Look at the most frequent missing words
        top_missing = missing_w2v.groupby('word')['count'].sum().sort_values(ascending=False).head(20)
        print(top_missing)
        
        # Check if these words exist in the raw cluster file at all
        if os.path.exists(w2v_path):
            df_w2v = pd.read_csv(w2v_path)
            sample_missing_word = top_missing.index[0]
            exists = sample_missing_word in df_w2v['word'].values
            print(f"\nDoes '{sample_missing_word}' exist in word2vec_clusters.csv? {exists}")

if __name__ == "__main__":
    debug_missing()
