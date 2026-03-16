import pandas as pd
import os
import re

def extract_all_ids_per_row():
    file_path = "target_female.xlsx"
    print(f"Processing: {file_path}")
    
    # Read without headers
    df = pd.read_excel(file_path, header=None)
    
    # ffill for Book
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    
    all_pairs = []
    
    for idx, row in df.iterrows():
        book = str(row[0]).strip()
        name = str(row[1]).strip()
        
        # Skip headers and empty rows
        if name.lower() in ['name', 'character', 'nan', '閸忔湹绮', '婵挸鎮'] or len(name) < 1: 
            continue
        
        found_id = False
        # Collect ALL numeric IDs in this row (from Column 2 onwards)
        for col_idx in range(2, len(row)):
            val = row[col_idx]
            if pd.notna(val):
                # Use regex to find all numbers in the cell
                matches = re.findall(r'\d+', str(val))
                for m in matches:
                    cid = int(m)
                    if cid < 20000: # Realistic range
                        all_pairs.append({
                            'book': book,
                            'name': name,
                            'char_id': cid
                        })
                        found_id = True
        
        # If no ID was found, still add a record with a placeholder or just the name
        # to ensure the book is represented (though it might not match any words)
        if not found_id:
            all_pairs.append({
                'book': book,
                'name': name,
                'char_id': -1 # Use -1 as "No ID found"
            })

    # Create final dataframe
    output_df = pd.DataFrame(all_pairs)
    output_df = output_df.drop_duplicates()

    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/target_female_ids.csv'
    output_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    unique_books = output_df['book'].unique()
    unique_chars = output_df[['book', 'name']].drop_duplicates()

    print(f"\n>>> EXTRACTION COMPLETE <<<")
    print(f"Total Unique Books found: {len(unique_books)}")
    print(f"Total Unique Characters (Book+Name): {len(unique_chars)}")
    print(f"Data saved to {out_path}")

if __name__ == "__main__":
    extract_all_ids_per_row()
