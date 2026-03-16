import os
import subprocess
import glob

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
    
    for book_pattern in MISSING_BOOKS:
        # Find the specific file
        target_file = None
        for txt_file in all_txt_files:
            if book_pattern in os.path.basename(txt_file):
                target_file = txt_file
                break
        
        if not target_file:
            print(f"!!! Could not find text file for pattern: {book_pattern}")
            continue
            
        # Create output directory: original_data/output_BookName
        book_name = book_pattern # or extract from filename
        output_dir = os.path.join(output_base_dir, f"output_{book_name}")
        
        # BACKUP LOGIC: Move existing .html to backup folder
        if os.path.exists(output_dir):
            backup_dir = os.path.join(output_base_dir, "old_html_backups", f"output_{book_name}")
            os.makedirs(backup_dir, exist_ok=True)
            for html_file in glob.glob(os.path.join(output_dir, "*.html")):
                import shutil
                shutil.copy(html_file, backup_dir)
                print(f"Backed up: {os.path.basename(html_file)} to {backup_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n>>> Processing: {target_file}")
        print(f">>> Output to: {output_dir}")
        
        # Construct the command
        # booknlp --target_file <FILE> --output_dir <DIR> --model_size small --id <ID>
        # Using subprocess to call the CLI version
        cmd = [
            "booknlp",
            "--target_file", target_file,
            "--output_dir", output_dir,
            "--model_size", "small",
            "--id", book_name
        ]
        
        try:
            # Note: This requires booknlp to be installed in the environment
            subprocess.run(cmd, check=True)
            print(f">>> Successfully processed {book_name}")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error processing {book_name}: {e}")
        except FileNotFoundError:
            print("!!! Error: 'booknlp' command not found. Please ensure it is installed and in your PATH.")
            break

if __name__ == "__main__":
    run_booknlp()
