import pandas as pd
import json
import os

def analyze_lda_results(results_dir, clusters_file, assignment_file):
    print(f"\n{'='*50}")
    print(f">>> LDA-SAGE Persona Semantic Analysis")
    print(f"{'='*50}\n")
    
    # 1. Load data
    clusters_df = pd.read_csv(clusters_file)
    with open(os.path.join(results_dir, "lda_keywords.json"), 'r') as f:
        keywords = json.load(f)
    
    # Create id to example words mapping
    # Sort each cluster to get shorter/more common words first if possible
    id_to_words = clusters_df.groupby('cluster_id')['word'].apply(
        lambda x: ", ".join(list(x)[:6]) # Top 6 words per cluster
    ).to_dict()
    
    # 2. Print Persona Meanings
    for p_id, cluster_ids in keywords.items():
        print(f"[{p_id}] Top Features:")
        for cid in cluster_ids[:10]: # Top 10 clusters
            words = id_to_words.get(int(cid), "Unknown")
            print(f"  - Cluster {cid:4}: {words}")
        print("-" * 30)
    
    # 3. Distribution Analysis
    df_assign = pd.read_csv(assignment_file)
    dist = df_assign['persona_label'].value_counts(normalize=True).sort_index()
    
    print("\n>>> Dominant Persona Distribution among Characters:")
    for p_label, ratio in dist.items():
        print(f"  Persona {p_label}: {ratio:6.2%}")
    
    print(f"\nTotal characters analyzed: {len(df_assign)}")

if __name__ == "__main__":
    analyze_lda_results(
        results_dir="data/results/lda_results",
        clusters_file="data/processed/word2vec_clusters.csv",
        assignment_file="data/results/lda_results/lda_persona_assignments.csv"
    )
