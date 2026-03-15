import pandas as pd
import glob
import os
import re

def check_multi_id_relationship():
    file_path = "target_female.xlsx"
    print(f"Analyzing relationship in: {file_path}")
    
    # Read without headers
    df = pd.read_excel(file_path, header=None)
    
    # ffill for Book and Name
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    df.iloc[:, 1] = df.iloc[:, 1].ffill()
    
    raw_data = []
    
    for idx, row in df.iterrows():
        book = str(row[0]).strip()
        name = str(row[1]).strip()
        
        if name.lower() in ['name', 'nan', 'character', '閸忔湹绮', '婵挸鎮']: continue
        
        # Look for ALL potential IDs in the row
        found_ids_in_row = []
        for col_idx in range(2, len(row)):
            val = row[col_idx]
            if pd.notna(val):
                match = re.search(r'\d+', str(val))
                if match:
                    found_ids_in_row.append(int(match.group()))
        
        for cid in found_ids_in_row:
            raw_data.append({'book': book, 'name': name, 'char_id': cid, 'row': idx})

    full_df = pd.DataFrame(raw_data)
    
    # 1. Check if one NAME has multiple IDs
    name_id_counts = full_df.groupby(['book', 'name'])['char_id'].nunique()
    multi_id_characters = name_id_counts[name_id_counts > 1]
    
    print("\n" + "="*50)
    print("1. Cases where ONE NAME has MULTIPLE IDs:")
    print("="*50)
    if not multi_id_characters.empty:
        print(multi_id_characters)
        print("\nDetail details:")
        for (book, name), count in multi_id_characters.items():
            ids = full_df[(full_df['book']==book) & (full_df['name']==name)]['char_id'].unique()
            print(f"Book: {book} | Name: {name} | IDs: {list(ids)}")
    else:
        print("None. Every character name has exactly one ID.")

    # 2. Check if one ID has multiple NAMES (Alias/Duplicates)
    id_name_counts = full_df.groupby(['book', 'char_id'])['name'].nunique()
    multi_name_ids = id_name_counts[id_name_counts > 1]
    
    print("\n" + "="*50)
    print("2. Cases where ONE ID has MULTIPLE NAMES (Duplicates/Aliases):")
    print("="*50)
    if not multi_name_ids.empty:
        print(multi_name_ids)
        print("\nDetail details:")
        for (book, cid), count in multi_name_ids.items():
            names = full_df[(full_df['book']==book) & (full_df['char_id']==cid)]['name'].unique()
            print(f"Book: {book} | ID: {cid} | Names: {list(names)}")
    else:
        print("None. Every ID belongs to exactly one character name.")

    # 3. Verify against target_female_ids.csv
    csv_path = 'data/processed/target_female_ids.csv'
    if os.path.exists(csv_path):
        csv_df = pd.read_csv(csv_path)
        print(f"\nCSV total rows: {len(csv_df)}")
        print(f"Raw data unique (Book, ID) pairs: {len(full_df.drop_duplicates(subset=['book', 'char_id']))}")
        if len(csv_df) == len(full_df.drop_duplicates(subset=['book', 'char_id'])):
            print(">>> VERIFIED: All unique Book-ID pairs correctly extracted to CSV.")
        else:
            print(">>> WARNING: Mismatch in row counts!")

if __name__ == "__main__":
    check_multi_id_relationship()
