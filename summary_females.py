
import json

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
            proper_mentions.sort(key=lambda x: x['c'], reverse=True)
            best_name = proper_mentions[0]['n']
        else:
            common_mentions = char.get('mentions', {}).get('common', [])
            if common_mentions:
                common_mentions.sort(key=lambda x: x['c'], reverse=True)
                best_name = common_mentions[0]['n']
            else:
                best_name = f"Character_{char['id']}"
        
        female_characters.append((char['id'], best_name, char.get('count', 0)))

# Sort by count
female_characters.sort(key=lambda x: x[2], reverse=True)

print(f"{'ID':<8} | {'Best Name':<40} | {'Mentions':<10}")
print("-" * 65)
for char_id, name, count in female_characters[:40]:
    print(f"{char_id:<8} | {name:<40} | {count:<10}")
