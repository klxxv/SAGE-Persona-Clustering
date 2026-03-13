
import torch
import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

def extract_p4():
    checkpoint_path = 'checkpoints/P4/checkpoint_it1000.pt' # it200 or it1000? Log showed it reached 1000? Wait, log tail showed it was at 1000 earlier?
    # Let me check which one is the latest.
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/P4/checkpoint_it200.pt'
    
    print(f">>> Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    p_assignments = checkpoint['p_assignments']
    eta_pers = checkpoint['model_weights']['eta_pers'] # [P, num_internal_nodes]
    
    # Load data for mapping
    print(">>> Loading data for mapping...")
    word_csv = 'data/processed/word2vec_clusters.csv'
    data_file = 'data/processed/all_words.csv'
    
    df_words = pd.read_csv(word_csv)
    df = pd.read_csv(data_file)
    
    # Replicate mapping logic from model.py
    roles = ['agent', 'patient', 'possessive', 'predicative']
    df = df[df['role'].isin(roles)].copy()
    
    # Map cluster_id (important: this happens BEFORE filtering char_keys in model.py)
    word_to_cluster = dict(zip(df_words.word, df_words.cluster_id))
    df['cluster_id'] = df['word'].map(word_to_cluster)
    df.dropna(subset=['cluster_id'], inplace=True)
    
    df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
    
    # Filter min_mentions (default 10)
    char_totals = df.groupby("char_key")["count"].sum()
    valid_keys = char_totals[char_totals >= 10].index
    df = df[df["char_key"].isin(valid_keys)].copy()
    
    char_keys = sorted(df["char_key"].unique())
    char_map = {ck: i for i, ck in enumerate(char_keys)}
    
    # Check if p_assignments length matches char_keys
    print(f"p_assignments: {len(p_assignments)}, char_keys: {len(char_keys)}")
    
    # Get character info
    char_info = df.groupby("char_key")[["book", "author", "char_id"]].first().reset_index()
    char_info['persona'] = char_info['char_key'].map(lambda x: p_assignments[char_map[x]] if x in char_map else -1)
    
    # Save sage_character_personas.csv as expected by visualization scripts
    output_csv = os.path.join(os.path.dirname(checkpoint_path), "sage_character_personas.csv")
    # Save sage_model_weights.pt as expected by visualization scripts
    weights_path = os.path.join(os.path.dirname(checkpoint_path), "sage_model_weights.pt")
    torch.save(checkpoint['model_weights'], weights_path)
    print(f">>> Saved {weights_path}")
    
    char_info[['char_key', 'book', 'char_id', 'persona']].to_csv(output_csv, index=False)
    print(f">>> Saved {output_csv}")
    
    # Load metadata for names and genders
    metadata_file = 'data/results/sage_personas_with_metadata.csv' # Use existing merged file for convenience
    if os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        # Create a map from (book, char_id) to (best_name, gender)
        meta_map = {}
        for _, row in meta_df.iterrows():
            key = (row['book'], str(row['char_id']))
            meta_map[key] = (row['best_name'], row['gender'])
            
        def get_meta(char_key):
            parts = char_key.split('_')
            c_id = parts[-1]
            b_name = "_".join(parts[:-1])
            # Try variations
            candidates = [(b_name, c_id), (b_name.replace('_', ' '), c_id)]
            for cand in candidates:
                if cand in meta_map:
                    return meta_map[cand]
            return ("Unknown", "Unknown")
            
        char_info['name_gender'] = char_info['char_key'].apply(get_meta)
        char_info['name'] = char_info['name_gender'].apply(lambda x: x[0])
        char_info['gender'] = char_info['name_gender'].apply(lambda x: x[1])
    
    # Get counts and top characters
    results = []
    for p in range(checkpoint['P']):
        p_chars = char_info[char_info['persona'] == p]
        male = len(p_chars[p_chars['gender'] == 'he/him/his'])
        female = len(p_chars[p_chars['gender'] == 'she/her'])
        neutral = len(p_chars) - male - female
        
        # Get top characters by frequency (need to join with total counts)
        char_counts = df.groupby("char_key")["count"].sum().reset_index()
        p_chars = p_chars.merge(char_counts, on="char_key")
        top_chars = p_chars.sort_values("count", ascending=False).head(5)
        
        results.append({
            "id": p,
            "count": len(p_chars),
            "male": round(male / len(p_chars) * 100, 1) if len(p_chars) > 0 else 0,
            "female": round(female / len(p_chars) * 100, 1) if len(p_chars) > 0 else 0,
            "neutral": round(neutral / len(p_chars) * 100, 1) if len(p_chars) > 0 else 0,
            "top_chars": [f"{row['name']} ({row['book']})" for _, row in top_chars.iterrows()],
            "avg_count": int(p_chars['count'].mean()) if len(p_chars) > 0 else 0,
            "med_count": int(p_chars['count'].median()) if len(p_chars) > 0 else 0,
        })
        
    print("\n--- Persona Summary (N=4) ---")
    for res in results:
        print(f"P{res['id']}: Count={res['count']}, Female={res['female']}%, Male={res['male']}%, MedFreq={res['med_count']}")
        print(f"     Top: {', '.join(res['top_chars'])}")

    # Get keywords for each persona
    # eta_pers has shape [P, num_internal_nodes]
    # In this version of SAGE, eta_pers is added to eta_bg for each node in the tree.
    # To get cluster-level importance, we look at the weights assigned to the internal nodes.
    # However, visualize_personas.py uses eta_pers.mean(dim=1) if it's [P, R, V+1]
    
    # Let's check the shape of eta_pers in the checkpoint
    print(f"eta_pers shape: {eta_pers.shape}")
    
    # If eta_pers is [P, num_internal_nodes], we need to map to words.
    # For simplicity, let's just get the most frequent words for each persona from the assignments
    print(">>> Extracting top keywords from character assignments...")
    df_merged = df.merge(char_info[['char_key', 'persona']], on='char_key')
    
    persona_keywords = []
    for p in range(checkpoint['P']):
        p_df = df_merged[df_merged['persona'] == p]
        top_words = p_df.groupby('word')['count'].sum().sort_values(ascending=False).head(50).reset_index()
        top_words['persona'] = p
        persona_keywords.append(top_words)
    
    keywords_df = pd.concat(persona_keywords)
    keywords_csv = 'data/results/p4_keywords.csv'
    keywords_df.to_csv(keywords_csv, index=False)
    print(f">>> Saved {keywords_csv}")
    
    # Save to JSON for report
    with open('data/results/p4_summary.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n>>> Results saved to data/results/p4_summary.json")

if __name__ == "__main__":
    extract_p4()
