
import json
import os
import csv
import glob

base_dir = r'..\原始数据\上海小说_Data'
output_file = 'all_characters_metadata.csv'

# Character mapping: book_name -> {char_id -> {name, gender, other_names}}
all_data = []

folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('output_')]

for folder in folders:
    book_short_name = folder.replace('output_', '')
    folder_path = os.path.join(base_dir, folder)
    
    # Find .book file
    book_files = glob.glob(os.path.join(folder_path, '*.book'))
    if not book_files:
        continue
    
    book_file = book_files[0]
    print(f"Processing {book_short_name}...")
    
    try:
        with open(book_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for char in data.get('characters', []):
            char_id = char['id']
            
            # Gender
            gender_info = char.get('g', {})
            if isinstance(gender_info, dict):
                gender = gender_info.get('argmax', 'UNKNOWN')
            else:
                gender = str(gender_info)
            
            # Proper names
            proper_mentions = char.get('mentions', {}).get('proper', [])
            all_names = [m['n'] for m in proper_mentions]
            
            if proper_mentions:
                proper_mentions.sort(key=lambda x: x['c'], reverse=True)
                best_name = proper_mentions[0]['n']
            else:
                common_mentions = char.get('mentions', {}).get('common', [])
                if common_mentions:
                    common_mentions.sort(key=lambda x: x['c'], reverse=True)
                    best_name = common_mentions[0]['n']
                else:
                    best_name = f"Character_{char_id}"
            
            all_data.append({
                'book': book_short_name,
                'char_id': char_id,
                'best_name': best_name,
                'gender': gender,
                'count': char.get('count', 0),
                'other_names': "|".join(all_names)
            })
    except Exception as e:
        print(f"Error processing {book_file}: {e}")

# Write to CSV
with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['book', 'char_id', 'best_name', 'gender', 'count', 'other_names'])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Successfully extracted metadata for {len(all_data)} characters to {output_file}.")
