import os
from pathlib import Path

booknlp_dir = Path(r'C:\Users\klxxv\miniconda3\envs\semantics\lib\site-packages\booknlp')

for p in booknlp_dir.rglob('*.py'):
    try:
        text = p.read_text('utf-8')
        if '.split("/")' in text or ".split('/')" in text:
            new_text = text.replace('.split("/")', '.replace("\\\\", "/").split("/")').replace(".split('/')", ".replace('\\\\', '/').split('/')")
            p.write_text(new_text, 'utf-8')
            print(f"Patched: {p}")
    except Exception as e:
        print(f"Failed {p}: {e}")
