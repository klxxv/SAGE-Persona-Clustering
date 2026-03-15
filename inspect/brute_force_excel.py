import pandas as pd
import glob
import os

def brute_force_inspect():
    excel_files = [f for f in glob.glob("*.xlsx") if not os.path.basename(f).startswith('~$')]
    if not excel_files: return
    
    df = pd.read_excel(excel_files[0], header=None)
    
    print(f"Total Rows in Excel: {len(df)}")
    
    found_rows = []
    for idx, row in df.iterrows():
        # Check if ANY cell in this row contains a number that could be an ID
        has_id = False
        potential_id = None
        for col_idx in range(len(row)):
            val = row[col_idx]
            if pd.notna(val):
                try:
                    s_val = str(val).split('.')[0].strip()
                    if s_val.isdigit() and len(s_val) < 5: # IDs are usually not huge
                        potential_id = s_val
                        has_id = True
                        break
                except: continue
        
        if has_id:
            # Print row content for debugging
            found_rows.append((idx, potential_id, list(row)[:5])) # First 5 columns

    print(f"\nRows containing a potential ID: {len(found_rows)}")
    for r in found_rows[:20]:
        print(f"Row {r[0]}: ID={r[1]} | Data={r[2]}")

if __name__ == "__main__":
    brute_force_inspect()
