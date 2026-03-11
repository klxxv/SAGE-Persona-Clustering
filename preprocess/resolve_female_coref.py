
import json
import csv

book_file = r'..\原始数据\上海小说_Data\output_A_Case_of_Two_Cities\two_cities.book'
entities_file = r'..\原始数据\上海小说_Data\output_A_Case_of_Two_Cities\two_cities.entities'
output_csv = 'female_characters_resolved.csv'

# 1. Load book data for character metadata (gender and best name)
with open(book_file, 'r', encoding='utf-8') as f:
    book_data = json.load(f)

character_map = {}
for char in book_data['characters']:
    gender_info = char.get('g', {})
    if isinstance(gender_info, dict):
        gender_label = gender_info.get('argmax', 'UNKNOWN')
    else:
        gender_label = str(gender_info)
    
    if gender_label == 'she/her':
        # Determine best name
        proper_mentions = char.get('mentions', {}).get('proper', [])
        if proper_mentions:
            proper_mentions.sort(key=lambda x: x['c'], reverse=True)
            best_name = proper_mentions[0]['n']
        else:
            common_mentions = char.get('mentions', {}).get('common', [])
            if common_mentions:
                common_mentions.sort(key=lambda x: x['c'], reverse=True)
                best_name = common_mentions[0]['n']
            else:
                best_name = f"Character_{char['id']}"
        
        character_map[char['id']] = best_name

print(f"Loaded {len(character_map)} female characters.")

# 2. Extract all mentions from entities file for these characters
with open(entities_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    mentions = []
    for row in reader:
        coref_id = int(row['COREF'])
        if coref_id in character_map:
            mentions.append({
                'Character_ID': coref_id,
                'Character_Name': character_map[coref_id],
                'Mention_Text': row['text'],
                'Mention_Type': row['prop'], # NOM, PROP, PRON
                'Category': row['cat'],      # PER, FAC, etc.
                'Start_Token': row['start_token'],
                'End_Token': row['end_token']
            })

print(f"Found {len(mentions)} mentions for female characters.")

# 3. Write results to CSV
with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Character_ID', 'Character_Name', 'Mention_Text', 'Mention_Type', 'Category', 'Start_Token', 'End_Token'])
    writer.writeheader()
    writer.writerows(mentions)

print(f"Results saved to {output_csv}.")
