
import json
import os

book_file = r'..\原始数据\上海小说_Data\output_A_Case_of_Two_Cities\two_cities.book'

with open(book_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

female_characters = []
for char in data['characters']:
    gender = char.get('g', {})
    if isinstance(gender, dict):
        gender_label = gender.get('argmax', 'UNKNOWN')
    else:
        gender_label = str(gender)
    
    if gender_label == 'she/her':
        # Get best name
        proper_mentions = char.get('mentions', {}).get('proper', [])
        if proper_mentions:
            # Sort by count
            proper_mentions.sort(key=lambda x: x['c'], reverse=True)
            best_name = proper_mentions[0]['n']
        else:
            # Try common mentions
            common_mentions = char.get('mentions', {}).get('common', [])
            if common_mentions:
                common_mentions.sort(key=lambda x: x['c'], reverse=True)
                best_name = common_mentions[0]['n']
            else:
                best_name = f"Character_{char['id']}"
        
        female_characters.append({
            'id': char['id'],
            'name': best_name,
            'count': char.get('count', 0),
            'mentions': char.get('mentions', {})
        })

print(f"Total female characters found: {len(female_characters)}")
# Save to a file for the user
output_file = 'female_characters_two_cities.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(female_characters, f, ensure_ascii=False, indent=2)

for char in female_characters[:20]:
    print(f"ID: {char['id']}, Name: {char['name']}, Count: {char['count']}")
