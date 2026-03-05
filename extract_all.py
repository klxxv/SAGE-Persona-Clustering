import json
import csv
import os
import glob
import re

txt_dir = r"原始数据\上海_小说_txt_3"
data_dir = r"原始数据\上海小说_Data"
out_dir = r"SAGE\fullset_data"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Map book name to author
book_to_author = {}
for file in os.listdir(txt_dir):
    if file.endswith('.txt'):
        # e.g., A_Case_of_Two_Cities_-_Qiu_Xiaolong.txt
        # The_Painter_From_Shanghai_-_Jennifer_Cody_Epstein (1).txt
        match = re.match(r"(.*?)_-_(.*?)(?: \(\d+\))?\.txt", file)
        if match:
            book_name = match.group(1)
            author = match.group(2)
            book_to_author[book_name] = author
        else:
            print(f"Could not parse filename: {file}")

all_rows = []

for folder in os.listdir(data_dir):
    if not folder.startswith("output_"):
        continue
    book_name = folder[7:] # remove 'output_'
    author = book_to_author.get(book_name, "Unknown")
    
    book_dir = os.path.join(data_dir, folder)
    book_files = glob.glob(os.path.join(book_dir, "*.book"))
    
    if not book_files:
        print(f"No .book file in {folder}")
        continue
        
    book_file = book_files[0]
    
    with open(book_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading JSON from {book_file}")
            continue
            
    characters = data.get('characters', [])
    
    book_out_dir = os.path.join(out_dir, book_name)
    if not os.path.exists(book_out_dir):
        os.makedirs(book_out_dir)
        
    book_rows = []
    
    for char in characters:
        char_id = char.get('id')
        for role in ['agent', 'patient', 'poss', 'mod']:
            words = char.get(role, [])
            word_counts = {}
            for entry in words:
                word = entry.get('w', '').lower()
                if word:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            for word, count in word_counts.items():
                row = {
                    'author': author,
                    'book': book_name,
                    'char_id': char_id,
                    'role': role,
                    'word': word,
                    'count': count
                }
                book_rows.append(row)
                all_rows.append(row)
                
    # Save book-specific csv
    if book_rows:
        book_csv_path = os.path.join(book_out_dir, 'character_features.csv')
        with open(book_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['author', 'book', 'char_id', 'role', 'word', 'count'])
            writer.writeheader()
            writer.writerows(book_rows)

# Save all_words.csv
all_csv_path = os.path.join(out_dir, 'all_words.csv')
with open(all_csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['author', 'book', 'char_id', 'role', 'word', 'count'])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Extracted {len(all_rows)} total word entries across {len(book_to_author)} books.")
