
import csv
from collections import defaultdict

merged_file = 'sage_personas_with_metadata.csv'

persona_gender_counts = defaultdict(lambda: defaultdict(int))
persona_names = defaultdict(list)

with open(merged_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = row['persona']
        g = row['gender']
        name = row['best_name']
        persona_gender_counts[p][g] += 1
        if g == 'she/her':
            persona_names[p].append(name)

print(f"{'Persona':<10} | {'Female':<10} | {'Male':<10} | {'Top Female Names'}")
print("-" * 70)
for p in sorted(persona_gender_counts.keys()):
    counts = persona_gender_counts[p]
    female = counts.get('she/her', 0)
    male = counts.get('he/him/his', 0)
    top_names = ", ".join(list(set(persona_names[p]))[:5])
    print(f"{p:<10} | {female:<10} | {male:<10} | {top_names}")
