import pandas as pd
import glob
import os
import re

def extract_all_ids_per_row():
    file_path = "target_female.xlsx"
    print(f"Processing: {file_path}")
    
    # Read without headers
    df = pd.read_excel(file_path, header=None)
    
    # ffill for Book and Name
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    df.iloc[:, 1] = df.iloc[:, 1].ffill()
    
    all_pairs = []
    
    for idx, row in df.iterrows():
        book = str(row[0]).strip()
        name = str(row[1]).strip()
        
        # Skip headers and empty names
        if name.lower() in ['name', 'character', 'nan', '閸忔湹绮', '婵挸鎮'] or len(name) < 1: 
            continue
        
        # Collect ALL numeric IDs in this row (from Column 2 onwards)
        for col_idx in range(2, len(row)):
            val = row[col_idx]
            if pd.notna(val):
                # Use regex to find all numbers in the cell
                matches = re.findall(r'\d+', str(val))
                for m in matches:
                    cid = int(m)
                    if cid < 10000: # Realistic range
                        all_pairs.append({
                            'book': book,
                            'name': name,
                            'char_id': cid
                        })

    # Create final dataframe
    output_df = pd.DataFrame(all_pairs)
    
    # Drop duplicates where (book, name, char_id) are exactly the same
    output_df = output_df.drop_duplicates()
    
    # Final check: user wants all 245 unique (Book, ID) cases
    unique_book_id = output_df.drop_duplicates(subset=['book', 'char_id'])

    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/target_female_ids.csv'
    output_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"\n>>> RE-EXTRACTION COMPLETE <<<")
    print(f"Total rows in CSV: {len(output_df)}")
    print(f"Total Unique (Book, ID) entities: {len(unique_book_id)}")
    print(f"Data saved to {out_path}")

if __name__ == "__main__":
    extract_all_ids_per_row()
