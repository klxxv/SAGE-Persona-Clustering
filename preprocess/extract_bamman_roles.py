import pandas as pd
import os
import ast
from collections import defaultdict
from tqdm import tqdm
import re

def sn(s):
    if not s: return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def main():
    processed_dir = 'data/processed'
    orig_data_dir = 'original_data'
    
    # 1. Load Mappings
    df_books = pd.read_csv(os.path.join(processed_dir, 'book_id_name.csv'))
    book_id_to_name = dict(zip(df_books['book_id'], df_books['book_name']))
    
    df_targets = pd.read_csv(os.path.join(processed_dir, 'target_female_ids.csv'))
    
    # Map: book_name -> { booknlp_coref_id -> character_id }
    book_to_char_map = defaultdict(dict)
    for _, row in df_targets.iterrows():
        b_name = book_id_to_name.get(row['book_id'])
        c_id = row['character_id']
        try:
            bnlp_ids = ast.literal_eval(row['booknlp_cha_id'])
            if not isinstance(bnlp_ids, list): bnlp_ids = [bnlp_ids]
        except: continue
        for bid in bnlp_ids:
            if bid != -1:
                book_to_char_map[b_name][bid] = c_id

    all_extracted_words = []
    folder_list = os.listdir(orig_data_dir)
    folder_map = {sn(f.replace('output_', '')): f for f in folder_list}

    # 2. Process each book
    for book_name, bnlp_mapping in tqdm(book_to_char_map.items(), desc="Processing Books"):
        folder_name = folder_map.get(sn(book_name))
        if not folder_name:
            # Substring match
            for f_sn, f in folder_map.items():
                if sn(book_name) in f_sn or f_sn in sn(book_name):
                    folder_name = f; break
        if not folder_name: continue
            
        folder_path = os.path.join(orig_data_dir, folder_name)
        tokens_file = next((f for f in os.listdir(folder_path) if f.endswith('.tokens')), None)
        entities_file = next((f for f in os.listdir(folder_path) if f.endswith('.entities')), None)
        if not tokens_file or not entities_file: continue
            
        # A. Map token_ID_within_document -> character_id
        token_to_char_id = {}
        with open(os.path.join(folder_path, entities_file), 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 3:
                    try:
                        coref_id = int(p[0])
                        if coref_id in bnlp_mapping:
                            cid = bnlp_mapping[coref_id]
                            for t_id in range(int(p[1]), int(p[2]) + 1):
                                token_to_char_id[t_id] = cid
                    except: pass

        if not token_to_char_id: continue

        # B. Extract Roles from tokens using Bamman (2014) logic
        current_s_id = -1
        s_tokens = {} # tok_in_doc -> data
        head_to_children_rels = defaultdict(set)

        def process_sentence():
            for t_id, t in s_tokens.items():
                cid = token_to_char_id.get(t_id)
                if cid is None: continue
                
                rel = t['dep_rel']
                head_id = t['head_id']
                head_tok = s_tokens.get(head_id)
                if not head_tok: continue
                
                role, word = None, None
                h_lemma = head_tok['lemma']
                
                # Bamman et al. (2014) Logic:
                if rel == 'poss':
                    role, word = 'possessive', h_lemma
                elif rel in ['dobj', 'nsubjpass']:
                    role, word = 'patient', h_lemma
                elif rel == 'agent':
                    role, word = 'agent', h_lemma
                elif rel == 'nsubj':
                    # Differentiate agent and predicative using copula child
                    if 'cop' in head_to_children_rels[head_id]:
                        role, word = 'predicative', h_lemma
                    else:
                        role, word = 'agent', h_lemma
                
                if role and word and word.isalpha():
                    all_extracted_words.append({'character_id': cid, 'role': role, 'word': word.lower()})

        with open(os.path.join(folder_path, tokens_file), 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) < 12: continue
                try:
                    sid, tid_doc, lemma, rel, head = int(p[1]), int(p[3]), p[5], p[10], int(p[11])
                    if sid != current_s_id:
                        if current_s_id != -1: process_sentence()
                        current_s_id = sid
                        s_tokens = {}
                        head_to_children_rels = defaultdict(set)
                    
                    s_tokens[tid_doc] = {'lemma': lemma, 'dep_rel': rel, 'head_id': head}
                    head_to_children_rels[head].add(rel)
                except: pass
            process_sentence()

    # 3. Finalize
    if not all_extracted_words:
        print("!!! No data found."); return

    df_base = pd.DataFrame(all_extracted_words).groupby(['character_id', 'role', 'word']).size().reset_index(name='freq')
    unique_words = sorted(df_base['word'].unique())
    word_to_id = {word: i for i, word in enumerate(unique_words)}
    df_base['word_id'] = df_base['word'].map(word_to_id)
    
    out_path = os.path.join(processed_dir, 'female_words_base.csv')
    df_base[['character_id', 'role', 'word', 'word_id', 'freq']].to_csv(out_path, index=False, encoding='utf-8-sig')
    
    vocab_path = os.path.join(processed_dir, 'female_vocab_map.csv')
    pd.DataFrame({'word_id': range(len(unique_words)), 'word': unique_words}).to_csv(vocab_path, index=False, encoding='utf-8-sig')
    
    print(f"\nSUCCESS: Generated {out_path} with {len(df_base)} rows.")
    print(f"Unique characters covered: {df_base['character_id'].nunique()}")

if __name__ == '__main__':
    main()
