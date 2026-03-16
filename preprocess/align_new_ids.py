import pandas as pd
import json
import os
import glob
import ast
from difflib import SequenceMatcher

def msg(text):
    print(f">>> [Aligner] {text}")

def fuzzy_match(s1, s2):
    if not s1 or not s2: return 0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def align_ids():
    processed_dir = 'data/processed'
    data_dir = 'original_data'
    
    # 1. Load current mappings
    msg("Loading mapping files...")
    df_targets = pd.read_csv(os.path.join(processed_dir, 'target_female_ids.csv'))
    df_char_names = pd.read_csv(os.path.join(processed_dir, 'character_id_name.csv'))
    df_book_map = pd.read_csv(os.path.join(processed_dir, 'book_id_name.csv'))
    
    book_id_to_name = dict(zip(df_book_map['book_id'], df_book_map['book_name']))
    
    # Create a map of character_id -> {book_name, char_name}
    char_info = {}
    for _, row in df_char_names.iterrows():
        char_info[row['character_id']] = {
            'book': row['book'],
            'name': row['name']
        }

    # 2. Iterate through characters
    updated_rows = []
    
    for idx, row in df_targets.iterrows():
        c_id = row['character_id']
        b_id = row['book_id']
        b_name = book_id_to_name.get(b_id)
        target_name = char_info.get(c_id, {}).get('name', 'Unknown')
        
        # Determine how many IDs we need to find (N)
        try:
            old_ids = ast.literal_eval(row['booknlp_cha_id'])
            if not isinstance(old_ids, list): old_ids = [old_ids]
        except:
            old_ids = [-1]
        
        N = len(old_ids)
        
        # Check if this book has been re-processed
        # Strategy: Look for the .book file in original_data
        folder_name = f"output_{b_name.replace(' ', '_')}"
        book_json_paths = glob.glob(os.path.join(data_dir, folder_name, "*.book"))
        
        if not book_json_paths:
            # Fallback
            folder_name_alt = f"output_{b_name}"
            book_json_paths = glob.glob(os.path.join(data_dir, folder_name_alt, "*.book"))

        if book_json_paths:
            # Found new analysis data!
            book_json_path = book_json_paths[0]
            
            with open(book_json_path, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            
            # Find candidate characters based on name similarity
            candidates = []
            for char in book_data.get('characters', []):
                # Extract all possible names for this character in the new analysis
                mentions = char.get('mentions', {})
                proper = [m.get('n') for m in mentions.get('proper', [])]
                common = [m.get('n') for m in mentions.get('common', [])]
                all_possible_names = proper + common
                
                if not all_possible_names: continue
                
                # Get max similarity score
                max_score = 0
                for name_variant in all_possible_names:
                    # Handle multi-names separated by pipe if they exist
                    for sub_name in str(name_variant).split('|'):
                        score = fuzzy_match(target_name, sub_name)
                        if score > max_score:
                            max_score = score
                
                if max_score > 0.5: # Loose threshold to catch all potentials
                    candidates.append({
                        'id': char.get('id'),
                        'score': max_score,
                        'count': char.get('count', 0)
                    })
            
            if candidates:
                # Sort primarily by score, secondarily by occurrence count (importance)
                candidates.sort(key=lambda x: (x['score'], x['count']), reverse=True)
                
                # Take top N
                new_ids = [c['id'] for c in candidates[:N]]
                
                # If we found fewer than N, fill with -1 or repeat? 
                # Directive says number should be same.
                while len(new_ids) < N:
                    new_ids.append(-1)
                
                msg(f"Aligned '{target_name}' in '{b_name}': Old count {N} -> New IDs {new_ids}")
                row['booknlp_cha_id'] = str(new_ids)
            else:
                msg(f"Warning: No candidates found for '{target_name}' in '{b_name}'")
                # Keep as is or mark -1
        else:
            # No new book file found, skip alignment for this book
            pass
            
        updated_rows.append(row)

    # 3. Save updated targets
    df_new_targets = pd.DataFrame(updated_rows)
    df_new_targets.to_csv(os.path.join(processed_dir, 'target_female_ids.csv'), index=False, encoding='utf-8-sig')
    msg("Alignment complete.")

if __name__ == '__main__':
    align_ids()
