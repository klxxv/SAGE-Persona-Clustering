import pandas as pd
import glob
import os

def check_excel():
    excel_files = glob.glob("*.xlsx")
    if not excel_files:
        print("No Excel files found.")
        return
    
    file_path = excel_files[0]
    print(f"Reading file: {file_path}")
    
    try:
        # Check first few rows and headers
        df = pd.read_excel(file_path, nrows=5)
        print("Columns found:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"Error reading Excel: {e}")

if __name__ == "__main__":
    check_excel()
