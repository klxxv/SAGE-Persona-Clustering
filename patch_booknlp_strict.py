import os
from pathlib import Path

booknlp_dir = Path(r'C:\Users\klxxv\miniconda3\envs\semantics\lib\site-packages\booknlp')

for p in booknlp_dir.rglob('*.py'):
    try:
        text = p.read_text('utf-8')
        if 'load_state_dict' in text and 'strict=False' not in text:
            # We want to replace .load_state_dict(X) with .load_state_dict(X, strict=False)
            # A simple regex can do this.
            import re
            new_text = re.sub(r'load_state_dict\((.*?)\)', r'load_state_dict(\1, strict=False)', text)
            if new_text != text:
                p.write_text(new_text, 'utf-8')
                print(f"Patched strict=False in: {p}")
    except Exception as e:
        print(f"Failed {p}: {e}")
