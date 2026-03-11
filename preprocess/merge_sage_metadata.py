
import csv
import os

metadata_file = 'all_characters_metadata.csv'
sage_file = r'sage_l1_w2v\sage_character_personas.csv'
output_file = 'sage_personas_with_metadata.csv'

# 1. Load metadata into a lookup map: (book, char_id) -> metadata
metadata_map = {}
with open(metadata_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['book'], row['char_id'])
        metadata_map[key] = row

print(f"Loaded {len(metadata_map)} characters from metadata.")

# 2. Process SAGE file and join with metadata
merged_data = []
missing_count = 0

with open(sage_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # char_key is usually "BookName_CharID"
        char_key = row['char_key']
        
        # Split char_key to get book and char_id
        # We need to handle cases where BookName contains underscores
        # The char_id is the last part
        parts = char_key.split('_')
        char_id = parts[-1]
        book_name_from_key = "_".join(parts[:-1])
        
        # Use book from 'book' column if available, otherwise from key
        book_col = row.get('book', book_name_from_key)
        
        # Try a few variations of the book name to match metadata
        # Metadata 'book' is from the folder name output_...
        # SAGE 'book' might have underscores instead of spaces or be slightly different
        
        match = None
        # Possible candidates in metadata
        candidates = [book_col, book_name_from_key, book_col.replace(' ', '_'), book_name_from_key.replace('_', ' ')]
        
        for cand in candidates:
            if (cand, char_id) in metadata_map:
                match = metadata_map[(cand, char_id)]
                break
        
        if match:
            new_row = {**row, **match}
            merged_data.append(new_row)
        else:
            missing_count += 1
            # print(f"Missing metadata for: {char_key} (Book: {book_col}, ID: {char_id})")

print(f"Merged {len(merged_data)} characters. Missing metadata for {missing_count} characters.")

# 3. Write results
if merged_data:
    fieldnames = list(merged_data[0].keys())
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)
    print(f"Saved merged data to {output_file}.")
else:
    print("No data merged.")
