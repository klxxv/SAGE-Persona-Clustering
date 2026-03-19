"""
Generate female_words_base.csv from female_words_with_ids.csv.
Columns: character_id, role, word, freq
(freq = count from original data)
"""
import pandas as pd

df = pd.read_csv("data/processed/female_words_with_ids.csv")
df['role'] = df['role'].str.lower().str.strip()

out = (df[['character_id', 'role', 'word', 'count']]
       .rename(columns={'count': 'freq'})
       .reset_index(drop=True))

print("Role distribution:")
print(out['role'].value_counts())
print(f"Total rows: {len(out)}")

out.to_csv("data/processed/female_words_base.csv", index=False)
print("Saved -> data/processed/female_words_base.csv")
