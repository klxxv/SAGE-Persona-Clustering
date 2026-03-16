import os
import re
from pathlib import Path

booknlp_dir = Path(r'C:\Users\klxxv\miniconda3\envs\semantics\lib\site-packages\booknlp')

for p in booknlp_dir.rglob('*.py'):
    try:
        text = p.read_text('utf-8')
        if 'strict=False' in text:
            # We want to remove strict=False from torch.load if it's there
            # and append it correctly to load_state_dict
            
            # Revert the mistake:
            text = text.replace(', strict=False', '')
            
            # Now correctly add strict=False to load_state_dict calls
            # Usually it's like: self.model.load_state_dict(torch.load(model_file, map_location=device))
            
            lines = text.split('\n')
            new_lines = []
            for line in lines:
                if 'load_state_dict(' in line and 'strict=False' not in line:
                    if line.endswith(')'):
                        line = line[:-1] + ', strict=False)'
                    elif line.endswith('))'):
                        line = line[:-1] + ', strict=False)'
                new_lines.append(line)
            
            new_text = '\n'.join(new_lines)
            
            p.write_text(new_text, 'utf-8')
            print(f"Fixed strict=False in: {p}")
    except Exception as e:
        print(f"Failed {p}: {e}")
