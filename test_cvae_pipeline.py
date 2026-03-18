import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# Fix path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sage.model import AdvancedLiterarySAGE

def run_small_test():
    print(">>> Starting Small Batch CVAE-Flat Test...")
    
    # 1. Setup paths
    data_file = 'data/processed/female_words_with_ids.csv'
    word_csv = 'data/processed/female_vocab_map.csv' # We use this for the word list
    
    # 2. Initialize Trainer (Small settings)
    # Using P=4 personas, 10 iterations for a quick smoke test
    trainer = AdvancedLiterarySAGE(n_personas=4, mode='cvae_flat', iters=10, l1_lambda=1e-6)
    
    # 3. Load Data
    print(">>> Loading data...")
    df = trainer.load_data(data_file, word_csv)
    
    # 4. Subset Data for quick test (Take first 20 characters)
    unique_chars = df['char_key'].unique()
    test_chars = unique_chars[:20]
    df_small = df[df['char_key'].isin(test_chars)].copy()
    print(f">>> Testing with {len(test_chars)} characters and {len(df_small)} samples.")
    
    # 5. Fit model
    # Note: checkpoint_dir will store temporary models
    checkpoint_dir = 'checkpoints/test_cvae_output'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n>>> Starting training loop...")
    trainer.fit(df_small, batch_size=256, checkpoint_dir=checkpoint_dir)
    
    # 6. Verify Output
    print("\n>>> Test Results:")
    labels = trainer.p_assignments
    unique_p = np.unique(labels)
    print(f"    Personas discovered: {len(unique_p)}")
    print(f"    Assignment counts: {np.bincount(labels, minlength=4)}")
    
    print("\n>>> CVAE-Flat Pipeline Test: SUCCESS")

if __name__ == '__main__':
    run_small_test()
