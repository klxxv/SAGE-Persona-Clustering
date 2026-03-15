import pandas as pd
import numpy as np
import os
import re

def debug_the_gap():
    excel_path = "target_female.xlsx"
    all_words_path = "fullset_data/all_words.csv"
    
    # 1. Read Excel Raw
    df_raw = pd.read_excel(excel_path, header=None)
    df_raw.iloc[:, 0] = df_raw.iloc[:, 0].ffill()
    df_raw.iloc[:, 1] = df_raw.iloc[:, 1].ffill()
    
    # Exclude header
    df_rows = df_raw.iloc[1:]
    print(f"Total Physical Rows in Excel (excl header): {len(df_rows)}")
    
    # 2. Re-extract without any filtering to see what we have
    excel_entries = []
    for idx, row in df_rows.iterrows():
        book = str(row[0]).strip()
        name = str(row[1]).strip()
        
        # Look for any ID in row
        ids = []
        for v in row[2:]:
            if pd.notna(v):
                m = re.findall(r'\d+', str(v))
                ids.extend([int(i) for i in m])
        
        if ids:
            excel_entries.append({'book': book, 'name': name, 'ids': ids, 'row': idx+1})
        else:
            print(f"Row {idx+1} has NO ID detected: {book} | {name}")

    print(f"\nRows with at least one ID: {len(excel_entries)}")
    
    # 3. Check matching with all_words.csv
    df_all = pd.read_csv(all_words_path)
    def normalize(s): return "".join(str(s).replace("_", "").split()).lower()
    
    all_books_norm = set(df_all['book'].apply(normalize).unique())
    
    matched = []
    unmatched_book = []
    unmatched_id = []
    
    for entry in excel_entries:
        b_norm = normalize(entry['book'])
        if b_norm not in all_books_norm:
            unmatched_book.append(entry)
            continue
            
        # Check if IDs exist in that book
        book_data = df_all[df_all['book'].apply(normalize) == b_norm]
        ids_in_data = set(book_data['char_id'].unique())
        
        found_any_id = False
        for cid in entry['ids']:
            if cid in ids_in_data:
                found_any_id = True
                break
        
        if found_any_id:
            matched.append(entry)
        else:
            unmatched_id.append(entry)

    print(f"\nMatch Results:")
    print(f"- Successfully matched (Book & ID found in all_words): {len(matched)}")
    print(f"- Unmatched due to BOOK name: {len(unmatched_book)}")
    print(f"- Unmatched due to ID not in that book: {len(unmatched_id)}")
    
    if unmatched_book:
        print("\nBooks that failed to match (Samples):")
        for e in unmatched_book[:5]:
            print(f"  Excel Book: '{e['book']}' (Normalized: {normalize(e['book'])})")
        print(f"  Available books in all_words (First 5): {list(all_books_norm)[:5]}")

    if unmatched_id:
        print("\nRoles where ID was not found in the book data (Samples):")
        for e in unmatched_id[:5]:
            print(f"  Book: {e['book']} | Name: {e['name']} | IDs: {e['ids']}")

if __name__ == "__main__":
    debug_the_gap()
