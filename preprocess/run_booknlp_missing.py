import os
import glob
import shutil
from datetime import datetime
from booknlp.booknlp import BookNLP
from tqdm import tqdm

# Missing books list (simplified names to match original_text files)
# Note: I will use partial matching to find the actual filenames
MISSING_BOOKS = [
    "These_Violent_Delights",
    "The_Risk_Agent",
    "The_Shanghai_Factor",
    "The_Shanghai_Moon",
    "The_Song_of_the_Jade_Lily",
    "The_Valley_of_Amazement",
    "What_We_Were_Promised",
    "When_Red_Is_Black"
]

def run_booknlp():
    text_dir = "original_text"
    output_base_dir = "original_data"
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Get all txt files
    all_txt_files = glob.glob(os.path.join(text_dir, "*.txt"))

    # Initialize BookNLP
    model_params = {
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "small"
    }
    print(">>> Initializing BookNLP (small model)...")
    booknlp = BookNLP("en", model_params)
    
    print("\n>>> Starting book processing...")
    for book_pattern in tqdm(MISSING_BOOKS, desc="Processing Books"):
        # Find the specific file
        target_file = None
        for txt_file in all_txt_files:
            if book_pattern in os.path.basename(txt_file):
                target_file = txt_file
                break
        
        if not target_file:
            print(f"\n!!! Could not find text file for pattern: {book_pattern}")
            continue
            
        # Create output directory: original_data/output_BookName
        book_name = book_pattern # or extract from filename
        output_dir = os.path.join(output_base_dir, f"output_{book_name}")
        
        # BACKUP LOGIC: Backup the entire existing folder
        if os.path.exists(output_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(output_base_dir, "old_backups", f"output_{book_name}_{timestamp}")
            os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
            print(f"\n>>> Backing up existing output to: {backup_dir}")
            shutil.move(output_dir, backup_dir)

        os.makedirs(output_dir, exist_ok=True)
            
        print(f"\n>>> Processing: {target_file}")
        print(f">>> Output to: {output_dir}")
        
        try:
            booknlp.process(target_file, output_dir, book_name)
            print(f">>> Successfully processed {book_name}")
        except Exception as e:
            print(f"\n!!! Error processing {book_name}: {e}")

if __name__ == "__main__":
    run_booknlp()
