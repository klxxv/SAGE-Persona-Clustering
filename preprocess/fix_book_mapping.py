import pandas as pd
import numpy as np
import os
import re
from difflib import get_close_matches

def smart_normalize(s):
    # Remove all non-alphanumeric and lowercase
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def update_mapping_and_diagnostic():
    all_words_path = "fullset_data/all_words.csv"
    targets_path = "data/processed/target_female_ids.csv"
    
    df_all = pd.read_csv(all_words_path)
    df_targets = pd.read_csv(targets_path)
    
    # 1. Get unique real book names from data
    real_books = df_all['book'].unique()
    real_books_norm = {smart_normalize(b): b for b in real_books}
    
    # 2. Map Excel books to real books
    def find_best_match(excel_book):
        norm_excel = smart_normalize(excel_book)
        
        # Manual Overrides for tricky cases
        manual_map = {
            "peachblossompavilion": "Peach_Blossom_Pavillion",
            "don'tcrytailake": "Dont_Cry_Tai_Lake"
        }
        if norm_excel in manual_map:
            return manual_map[norm_excel]

        # Direct match
        if norm_excel in real_books_norm:
            return real_books_norm[norm_excel]
        
        # Substring or fuzzy match
        for norm_real, real_full in real_books_norm.items():
            if norm_excel in norm_real or norm_real in norm_excel:
                return real_full
        
        return None

    # Apply mapping
    df_targets['book_mapped'] = df_targets['book'].apply(find_best_match)
    
    # Check what's still missing
    still_missing = df_targets[df_targets['book_mapped'].isna()]['book'].unique()
    if len(still_missing) > 0:
        print(f">>> Still could not map these books: {still_missing}")

    # 3. Final Merge
    df_merged = pd.merge(df_all, df_targets, left_on=['book', 'char_id'], right_on=['book_mapped', 'char_id'], suffixes=('', '_target'))
    
    # Aggregate
    df_female = df_merged.groupby(['book_target', 'name', 'role', 'word'])['count'].sum().reset_index()
    char_keys = (df_female['book_target'] + "_" + df_female['name']).unique()
    
    print(f"\n>>> FINAL VERIFICATION <<<")
    print(f"Total Unique Female Characters Matched: {len(char_keys)}")
    print(f"Goal: Close to 151")
    
    # 4. Save enriched data
    df_female.to_csv("data/processed/female_words_enriched_final.csv", index=False, encoding='utf-8-sig')
    print(">>> Final aggregated table saved to data/processed/female_words_enriched_final.csv")

if __name__ == "__main__":
    update_mapping_and_diagnostic()
